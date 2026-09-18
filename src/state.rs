//! Context serialization (engine state -> canonical JSON, the model's only
//! input) and the scripted oracle (label source for training, never a runtime
//! policy in direct mode).

use crate::bridge::Obs;
use crate::describe::{is_alive, is_monster, is_pickup, pretty_name};
use crate::options::{OptionDef, index_of};
use std::collections::VecDeque;

/// Displacement over the last `WINDOW` tics, so the model can see "I am not
/// moving" without any rule deciding what that means.
pub const WINDOW: usize = 15;

#[derive(Default)]
pub struct Tracker {
    hist: VecDeque<(f32, f32)>,
}

impl Tracker {
    pub fn reset(&mut self) {
        self.hist.clear();
    }
    /// Push the current position `tics` times and return the displacement over
    /// the window, or -1 while the window is not yet full.
    pub fn update(&mut self, pos: &[f32], tics: u32) -> f32 {
        if pos.len() < 2 {
            return -1.0;
        }
        for _ in 0..tics.max(1) {
            self.hist.push_back((pos[0], pos[1]));
        }
        while self.hist.len() > WINDOW {
            self.hist.pop_front();
        }
        if self.hist.len() < WINDOW {
            return -1.0;
        }
        let (x0, y0) = self.hist[0];
        ((pos[0] - x0).powi(2) + (pos[1] - y0).powi(2)).sqrt()
    }
}

/// Fixed key order, integers, nearest three of each. Angle: left positive.
pub fn serialize(o: &Obs, moved: f32) -> String {
    let mut enemies = vec![];
    let mut items = vec![];
    for a in &o.actors {
        let e = format!("{{\"t\":\"{}\",\"d\":{},\"a\":{}}}", pretty_name(&a.name), a.dist.round() as i32, a.rel_angle.round() as i32);
        if is_monster(&a.name) {
            if is_alive(a) && enemies.len() < 3 { enemies.push(e); }
        } else if is_pickup(&a.name) && items.len() < 3 {
            items.push(e);
        }
    }
    format!(
        "{{\"hp\":{},\"ammo\":{},\"moved\":{},\"depth\":{{\"l\":{},\"c\":{},\"r\":{}}},\"enemies\":[{}],\"items\":[{}]}}",
        o.health, o.ammo, moved.round() as i32,
        o.depth.near_left.round() as i32, o.depth.near.round() as i32, o.depth.near_right.round() as i32,
        enemies.join(","), items.join(",")
    )
}

/// Scripted oracle over the exact state. Returns an index into `opts`.
pub fn oracle(o: &Obs, moved: f32, opts: &[OptionDef]) -> usize {
    let pick = |names: &[&str]| -> usize {
        for n in names {
            if let Some(i) = index_of(opts, n) { return i; }
        }
        0
    };
    let can_move = index_of(opts, "move forward").is_some();
    let turn_toward = |a: f32| -> usize {
        let left = a > 0.0;
        if a.abs() <= 12.0 {
            if left { pick(&["turn left a little", "turn left"]) } else { pick(&["turn right a little", "turn right"]) }
        } else if left { pick(&["turn left"]) } else { pick(&["turn right"]) }
    };
    // 1. Live enemies: face them, fire only when centered and in pistol range.
    if let Some(m) = o.actors.iter().find(|a| is_monster(&a.name) && is_alive(a)) {
        let a = m.rel_angle;
        if a.abs() <= 3.0 {
            if o.ammo > 0 && m.dist < 1500.0 {
                return pick(&["attack"]);
            }
            return if can_move && m.dist >= 1500.0 { pick(&["move forward"]) } else { pick(&["move backward", "turn right"]) };
        }
        return turn_toward(a);
    }
    // 2. Pickups: walk to the nearest one.
    if can_move {
        if let Some(it) = o.actors.iter().find(|a| is_pickup(&a.name)) {
            let a = it.rel_angle;
            if a.abs() <= 6.0 {
                return pick(&["move forward"]);
            }
            return turn_toward(a);
        }
    }
    if !can_move {
        return pick(&["turn right"]);
    }
    let wall = o.depth.near < 12.0;
    let stuck = moved >= 0.0 && moved < 4.0;
    if stuck {
        return if wall { pick(&["use", "turn right"]) } else if o.depth.near_left > o.depth.near_right { pick(&["turn left"]) } else { pick(&["turn right"]) };
    }
    if wall {
        return if o.depth.near_left > o.depth.near_right { pick(&["turn left"]) } else { pick(&["turn right"]) };
    }
    pick(&["move forward"])
}
