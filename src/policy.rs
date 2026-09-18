//! The questions the model answers each decision, and how answers bind to buttons.

use crate::scorer::{Scored, Scorer, classify_multi};
use anyhow::Result;

pub fn schema(can_move: bool) -> Vec<(String, Vec<String>)> {
    let t = |n: &str, l: &[&str]| (n.to_string(), l.iter().map(|s| s.to_string()).collect());
    let mut v = vec![
        t("side", &["LEFT", "RIGHT", "CENTER", "NONE"]),
        t("offset", &["SMALL", "LARGE", "NONE"]),
    ];
    if can_move {
        v.push(t("ahead", &["OPEN", "WALL"]));
    }
    v
}

#[derive(Debug, Clone)]
pub struct Decision {
    /// Display name of the chosen move.
    pub action: &'static str,
    /// Buttons to press for `hold_tics`.
    pub buttons: Vec<&'static str>,
    /// Engine tics to hold `action` before releasing the buttons and waiting
    /// for the next verdict. `u32::MAX` = hold until the next verdict.
    pub hold_tics: u32,
    pub side: Vec<Scored>,
    pub offset: Vec<Scored>,
    pub ahead: Vec<Scored>,
    /// Direct policy: probability per option (sorted, descending).
    pub options: Vec<Scored>,
    pub confidence: f32,
}

/// Minimum probability on the winning side label before we trust it; below
/// this we scan (turn right) instead of acting on noise.
pub const MIN_CONF: f32 = 0.45;

/// `buttons` are the engine's available buttons; movement is used only when present.
/// `far`: the nearest enemy is beyond close range (exact distance, computed in Rust).
/// `deeper_left`: which side has more room, computed in Rust from the depth buffer; used only to pick the veer direction.
pub fn decide_with(sc: &mut Scorer, text: &str, ammo: i32, buttons: &[String], far: bool, deeper_left: bool) -> Result<Decision> {
    let can_move = buttons.iter().any(|b| b == "MOVE_FORWARD");
    let res = classify_multi(sc, text, &schema(can_move))?;
    let side = res[0].1.clone();
    let offset = res[1].1.clone();
    let ahead = res.get(2).map(|r| r.1.clone()).unwrap_or_default();
    let top = &side[0];
    let fine = offset[0].label == "SMALL";
    let open = ahead.first().map(|a| a.label == "OPEN").unwrap_or(false);
    // One binary turn is ~1.8 deg/tic. Long pulse ~7 deg, short pulse ~1.8 deg,
    // both under the 12 / 3 deg bucket edges so a stale verdict cannot overshoot
    // past the next bucket.
    let (action, buttons, hold_tics): (&'static str, Vec<&'static str>, u32) = if top.prob < MIN_CONF {
        ("scan", vec!["TURN_RIGHT"], u32::MAX)
    } else {
        match top.label.as_str() {
            "LEFT" => ("turn left", vec!["TURN_LEFT"], if fine { 1 } else { 4 }),
            "RIGHT" => ("turn right", vec!["TURN_RIGHT"], if fine { 1 } else { 4 }),
            "CENTER" if ammo > 0 && can_move && far && open => ("advance+fire", vec!["ATTACK", "MOVE_FORWARD"], u32::MAX),
            "CENTER" if ammo > 0 => ("attack", vec!["ATTACK"], u32::MAX),
            "CENTER" => ("scan", vec!["TURN_RIGHT"], u32::MAX),
            // NONE: explore when we can move, scan otherwise.
            _ if can_move && open => ("explore", vec!["MOVE_FORWARD"], u32::MAX),
            _ if can_move => ("wall: veer", vec![if deeper_left { "TURN_LEFT" } else { "TURN_RIGHT" }], 10),
            _ => ("scan", vec!["TURN_RIGHT"], u32::MAX),
        }
    };
    Ok(Decision { action, buttons, hold_tics, confidence: top.prob, side, offset, ahead, options: vec![] })
}

/// Controller-level stuck recovery shared by sync and real-time play (the
/// openjev reproduction's "veer"): when the policy is pushing forward but the
/// player has not displaced, use the wall (doors), back off, then turn.
#[derive(Default)]
pub struct Controller {
    hist: std::collections::VecDeque<(f32, f32)>,
    recover_left: u32,
    flips: u32,
}

impl Controller {
    /// `tics` = engine tics the returned buttons will be held for.
    pub fn apply(&mut self, buttons: Vec<&'static str>, pos: &[f32], tics: u32) -> (Vec<&'static str>, bool) {
        if self.recover_left > 0 {
            // 8 tics use+push, 8 tics back off, then a ~50-100 deg turn alternating direction.
            let phase = if self.recover_left > 44 { 0 } else if self.recover_left > 36 { 1 } else { 2 };
            self.recover_left = self.recover_left.saturating_sub(tics.max(1));
            let turn = if self.flips % 2 == 0 { "TURN_RIGHT" } else { "TURN_LEFT" };
            let b = match phase { 0 => vec!["USE", "MOVE_FORWARD"], 1 => vec!["MOVE_BACKWARD"], _ => vec![turn] };
            return (b, true);
        }
        if buttons.contains(&"MOVE_FORWARD") && pos.len() >= 2 {
            for _ in 0..tics.max(1) {
                self.hist.push_back((pos[0], pos[1]));
            }
            while self.hist.len() > 15 {
                self.hist.pop_front();
            }
            if self.hist.len() == 15 {
                let (x0, y0) = self.hist[0];
                if ((pos[0] - x0).powi(2) + (pos[1] - y0).powi(2)).sqrt() < 4.0 {
                    self.hist.clear();
                    self.recover_left = 52;
                    self.flips += 1;
                    return (vec!["USE", "MOVE_FORWARD"], true);
                }
            }
        } else {
            self.hist.clear();
        }
        (buttons, false)
    }
}

/// Which policy drives: the keyword schema over gliner25-rs, or the trained head over the raw state.
pub enum Brain {
    Keyword(Scorer),
    Direct { brain: crate::direct::DirectBrain, opts: Vec<crate::options::OptionDef> },
}

impl Brain {
    pub fn keyword(sc: Scorer) -> Self {
        Brain::Keyword(sc)
    }
    pub fn direct(head: &std::path::Path, buttons: &[String]) -> Result<Self> {
        let table = crate::options::load_table(None)?;
        let opts = crate::options::available(&table, buttons);
        Ok(Brain::Direct { brain: crate::direct::DirectBrain::load(head)?, opts })
    }
    pub fn is_direct(&self) -> bool {
        matches!(self, Brain::Direct { .. })
    }
    /// One decision. Returns the decision and the text the model read.
    pub fn decide(&mut self, o: &crate::bridge::Obs, buttons: &[String]) -> Result<(Decision, String)> {
        match self {
            Brain::Keyword(sc) => {
                let sit = crate::describe::describe(o);
                let far = sit.nearest.as_ref().map(|a| a.dist > 450.0).unwrap_or(false);
                let d = decide_with(sc, &sit.text, o.ammo, buttons, far, o.depth.near_left > o.depth.near_right)?;
                Ok((d, sit.text))
            }
            Brain::Direct { brain, opts } => {
                let ctx = crate::state::serialize(o, o.moved);
                let p = brain.score(&ctx, opts)?;
                let (i, _) = p.iter().enumerate().fold((0, -1f32), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc });
                let mut options: Vec<Scored> = opts.iter().zip(&p).map(|(o, &prob)| Scored { label: o.name.clone(), prob }).collect();
                options.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
                let leaked: &'static str = Box::leak(opts[i].name.clone().into_boxed_str());
                let btns: Vec<&'static str> = opts[i].buttons.iter().map(|b| -> &'static str { Box::leak(b.clone().into_boxed_str()) }).collect();
                Ok((Decision { action: leaked, buttons: btns, hold_tics: opts[i].hold, confidence: p[i], side: vec![], offset: vec![], ahead: vec![], options }, ctx))
            }
        }
    }
}
