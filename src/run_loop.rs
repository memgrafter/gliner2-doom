//! Real-time play. The engine runs at its own ticrate in ASYNC_PLAYER mode and
//! the main thread feeds it one tic at a time with the most recent decision.
//! The scorer lives on a worker thread that always takes the newest
//! observation and drops anything older, so a slow decision never queues up
//! stale premises (the pattern from the openjev reproduction's doom_live.py).

use crate::bridge::{Bridge, InitOpts, Obs};
use crate::policy::{self, Brain, Decision};
use crate::tui::{Tui, bar};
use anyhow::Result;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use crossterm::event::KeyCode;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[derive(Clone)]
pub struct Verdict {
    pub seq: u64,
    pub text: String,
    pub decision: Decision,
    pub ms: f64,
}

#[derive(Default)]
struct Shared {
    obs: Option<(u64, Obs)>,
    verdict: Option<Verdict>,
    quit: bool,
    decisions: u64,
    ready: bool,
}

pub struct LiveOpts {
    pub scenario: String,
    pub episodes: u32,
    pub tui: bool,
    pub cols: usize,
    pub wad: Option<String>,
    pub map: Option<String>,
    /// Show ViZDoom's own game window (at 640x480) instead of / in addition to the TUI.
    pub window: bool,
    /// Direct policy: no controller-level recovery, the head owns every press.
    pub direct: bool,
}

pub struct Summary {
    pub rewards: Vec<f32>,
    pub kills: Vec<i32>,
    pub items: Vec<i32>,
    pub ammo_spent: Vec<i32>,
    /// Path length walked per episode, in map units (a Doom corridor is ~128 wide).
    pub travelled: Vec<f32>,
    pub decisions: u64,
    pub tics: u64,
    pub secs: f64,
    pub lat_p50: f64,
}

/// `make_scorer` runs on the worker thread: the ORT-backed engine is not
/// `Send`, so it is built where it is used and never crosses a thread.
pub fn run(make_brain: impl FnOnce(&[String]) -> Result<Brain> + Send + 'static, opts: LiveOpts) -> Result<Summary> {
    let shared = Arc::new((Mutex::new(Shared::default()), Condvar::new()));
    let lat = Arc::new(Mutex::new(VecDeque::<f64>::with_capacity(512)));

    // Worker: newest observation in, verdict out.
    let buttons_slot: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(vec![]));
    let worker = {
        let shared = shared.clone();
        let lat = lat.clone();
        let buttons_slot = buttons_slot.clone();
        std::thread::spawn(move || -> Result<()> {
            let buttons = loop {
                let b = buttons_slot.lock().unwrap().clone();
                if !b.is_empty() { break b; }
                std::thread::sleep(Duration::from_millis(20));
            };
            let mut brain = make_brain(&buttons)?;
            {
                let (m, cv) = &*shared;
                m.lock().unwrap().ready = true;
                cv.notify_all();
            }
            let mut last_seq = 0u64;
            loop {
                let (seq, obs) = {
                    let (m, cv) = &*shared;
                    let mut g = m.lock().unwrap();
                    while !g.quit && g.obs.as_ref().map(|(s, _)| *s <= last_seq).unwrap_or(true) {
                        g = cv.wait(g).unwrap();
                    }
                    if g.quit {
                        return Ok(());
                    }
                    g.obs.take().unwrap()
                };
                last_seq = seq;
                let t = Instant::now();
                let (decision, text) = brain.decide(&obs, &buttons)?;
                let ms = t.elapsed().as_secs_f64() * 1000.0;
                {
                    let mut l = lat.lock().unwrap();
                    if l.len() == 512 {
                        l.pop_front();
                    }
                    l.push_back(ms);
                }
                let mut g = shared.0.lock().unwrap();
                g.verdict = Some(Verdict { seq, text, decision, ms });
                g.decisions += 1;
            }
        })
    };

    let init = InitOpts {
        scenario: opts.scenario.clone(),
        mode: "async".into(),
        width: if opts.window { 640 } else { 160 },
        height: if opts.window { 480 } else { 120 },
        visible: opts.window,
        ticrate: Some(35),
        wad: opts.wad.clone(),
        map: opts.map.clone(),
        ..Default::default()
    };
    let mut b = Bridge::spawn(&init)?;
    *buttons_slot.lock().unwrap() = b.buttons.clone();
    let b_buttons = b.buttons.clone();
    // Do not start the clock until the model can answer.
    {
        let (m, cv) = &*shared;
        let mut g = m.lock().unwrap();
        while !g.ready {
            g = cv.wait(g).unwrap();
        }
    }
    let mut tui = if opts.tui { Some(Tui::new(opts.cols)?) } else { None };
    let encoder_label = if opts.direct {
        match crate::encoder::kind() {
            crate::encoder::Kind::Metal => if crate::head::device().is_metal() { "Metal, f32" } else { "DeBERTa port on CPU" },
            crate::encoder::Kind::Onnx => "ONNX Runtime, CPU, fp16",
        }
    } else {
        "gliner25-rs, ONNX Runtime, CPU, fp16"
    };

    let t_start = Instant::now();
    let mut rewards = vec![];
    let mut kills: Vec<i32> = vec![];
    let mut items: Vec<i32> = vec![];
    let mut ammo_spent: Vec<i32> = vec![];
    let mut tics_total = 0u64;
    let mut seq = 0u64;
    let mut paused = false;
    let mut manual = false;
    let mut step_once = false;
    let mut quit = false;
    let mut current: Vec<&'static str> = vec!["TURN_RIGHT"];
    let mut hold_left: u32 = u32::MAX;
    let mut kills_last = 0i32;
    let mut ctl = policy::Controller::default();
    let mut recovering;
    let mut last_verdict: Option<Verdict> = None;
    let mut fps_win: VecDeque<Instant> = VecDeque::new();
    let mut dec_win: VecDeque<(Instant, u64)> = VecDeque::new();

    let mut travelled: Vec<f32> = vec![];
    let mut tracker = crate::state::Tracker::default();
    'episodes: for ep in 0..opts.episodes {
        let mut o = b.reset()?;
        tracker.reset();
        o.moved = -1.0;
        let start = (o.pos.first().copied().unwrap_or(0.0), o.pos.get(1).copied().unwrap_or(0.0));
        let mut path_len = 0f32;
        let mut last_pos = start;
        let mut spent = 0i32;
        let mut last_ammo = o.ammo;
        let mut manual_btn: Option<&'static str> = None;
        while !o.finished {
            // Keys.
            if let Some(t) = tui.as_mut() {
                while let Some(k) = t.key()? {
                    match k {
                        KeyCode::Char('q') | KeyCode::Esc => { quit = true; break 'episodes; }
                        KeyCode::Char('p') => paused = !paused,
                        KeyCode::Char(' ') => step_once = true,
                        KeyCode::Char('m') => manual = !manual,
                        KeyCode::Char('a') => manual_btn = Some("TURN_LEFT"),
                        KeyCode::Char('d') => manual_btn = Some("TURN_RIGHT"),
                        KeyCode::Char('w') => manual_btn = Some("MOVE_FORWARD"),
                        KeyCode::Char('s') => manual_btn = Some("MOVE_BACKWARD"),
                        KeyCode::Char('f') => manual_btn = Some("ATTACK"),
                        _ => {}
                    }
                }
            }
            // Latest verdict -> current action.
            {
                let g = shared.0.lock().unwrap();
                if let Some(v) = &g.verdict {
                    if last_verdict.as_ref().map(|l| l.seq < v.seq).unwrap_or(true) {
                        current = v.decision.buttons.clone();
                        hold_left = v.decision.hold_tics;
                        last_verdict = Some(v.clone());
                    }
                }
            }
            recovering = false;
            let mut buttons: Vec<&'static str> = if manual {
                manual_btn.take().into_iter().collect()
            } else if hold_left == 0 {
                vec![] // pulse spent: release the buttons until the next verdict
            } else {
                current.clone()
            };
            if !manual && !opts.direct {
                let (b2, r) = ctl.apply(buttons, &o.pos, 1);
                buttons = b2;
                recovering = r;
            }
            let buttons: Vec<&str> = buttons.into_iter().filter(|b| b_buttons.iter().any(|x| x == b)).collect();
            let advance = !paused || step_once;
            step_once = false;
            if advance {
                o = b.act(&buttons, 1)?;
                o.moved = tracker.update(&o.pos, 1);
                if o.ammo < last_ammo { spent += last_ammo - o.ammo; }
                last_ammo = o.ammo;
                if o.pos.len() >= 2 {
                    path_len += ((o.pos[0] - last_pos.0).powi(2) + (o.pos[1] - last_pos.1).powi(2)).sqrt();
                    last_pos = (o.pos[0], o.pos[1]);
                }
                if hold_left != u32::MAX && hold_left > 0 && !manual {
                    hold_left -= 1;
                }
                tics_total += 1;
                seq += 1;
                {
                    let (m, cv) = &*shared;
                    let mut g = m.lock().unwrap();
                    g.obs = Some((seq, o.clone()));
                    cv.notify_one();
                }
            } else {
                std::thread::sleep(Duration::from_millis(28));
            }
            // Draw every other tic: rendering competes with the scorer for CPU.
            if let Some(t) = tui.as_mut().filter(|_| seq % 2 == 0 || paused) {
                let now = Instant::now();
                fps_win.push_back(now);
                while fps_win.front().map(|f| now.duration_since(*f) > Duration::from_secs(1)).unwrap_or(false) {
                    fps_win.pop_front();
                }
                let dec = shared.0.lock().unwrap().decisions;
                dec_win.push_back((now, dec));
                while dec_win.front().map(|f| now.duration_since(f.0) > Duration::from_secs(1)).unwrap_or(false) {
                    dec_win.pop_front();
                }
                let dps = dec_win.front().map(|f| (dec - f.1) as f64).unwrap_or(0.0);
                let fo = b.obs(true)?;
                if let (Some(fr), Some(shape)) = (&fo.frame, &fo.frame_shape) {
                    let rgb = STANDARD.decode(fr)?;
                    let (h, w) = (shape[0] as usize, shape[1] as usize);
                    let p50 = {
                        let l = lat.lock().unwrap();
                        let mut v: Vec<f64> = l.iter().copied().collect();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        v.get(v.len() / 2).copied().unwrap_or(0.0)
                    };
                    let mut panel = vec![
                        format!("\x1b[1mgliner2-doom\x1b[0m  GLiNER2.5-multi encoder ({})  {}  policy: {}", encoder_label, opts.scenario, if opts.direct { "direct (trained head)" } else { "keyword schema" }),
                        format!("episode {}/{}  tic {}  reward {:.0}  kills {}  health {}  ammo {}", ep + 1, opts.episodes, o.tic, o.total_reward, o.kills, o.health, o.ammo),
                        format!("{:.0} tics/s  {dps:.0} decisions/s  latency p50 {p50:.0} ms  {}{}", fps_win.len(),
                            if paused { "[PAUSED] " } else { "" }, if manual { "[MANUAL] " } else { "" }),
                        String::new(),
                    ];
                    if let Some(v) = &last_verdict {
                        panel.push(format!("\x1b[2mmodel read:\x1b[0m {}", v.text)); // Tui::draw wraps every panel line
                        panel.push(String::new());
                        panel.push(format!("\x1b[1maction: {}\x1b[0m  [{}]  ({:.0} ms){}", v.decision.action, v.decision.buttons.join("+"), v.ms,
                            if recovering { "  RECOVERING" } else { "" }));
                        panel.push(String::new());
                        if !v.decision.options.is_empty() {
                            panel.push("options".to_string());
                            for s in &v.decision.options {
                                panel.push(format!("  {:<20}{} {:>5.1}%", s.label, bar(s.prob, 20), s.prob * 100.0));
                            }
                        }
                        if !v.decision.side.is_empty() { panel.push("side".to_string()); }
                        for s in &v.decision.side {
                            panel.push(format!("  {:<11}{} {:>5.1}%", s.label, bar(s.prob, 20), s.prob * 100.0));
                        }
                        if !v.decision.offset.is_empty() { panel.push("offset".to_string()); }
                        for s in &v.decision.offset {
                            panel.push(format!("  {:<11}{} {:>5.1}%", s.label, bar(s.prob, 20), s.prob * 100.0));
                        }
                        if !v.decision.ahead.is_empty() {
                            panel.push("ahead".to_string());
                            for s in &v.decision.ahead {
                                panel.push(format!("  {:<11}{} {:>5.1}%", s.label, bar(s.prob, 20), s.prob * 100.0));
                            }
                        }
                    }
                    panel.push(String::new());
                    panel.push("\x1b[2mq quit  p pause  space step  m manual (a/d turn, f fire, w/s move)\x1b[0m".into());
                    t.draw(&rgb, w, h, &panel)?;
                }
            }
        }
        rewards.push(o.total_reward);
        let _ = start;
        travelled.push(path_len);
        items.push(o.items);
        ammo_spent.push(spent);
        kills_last = o.kills;
        kills.push(kills_last);
        if tui.is_some() && ep + 1 < opts.episodes {
            std::thread::sleep(Duration::from_millis(800));
        }
    }
    let _ = quit;
    {
        let (m, cv) = &*shared;
        m.lock().unwrap().quit = true;
        cv.notify_all();
    }
    drop(tui);
    let secs = t_start.elapsed().as_secs_f64();
    let decisions = shared.0.lock().unwrap().decisions;
    let _ = worker.join();
    let lat_p50 = {
        let l = lat.lock().unwrap();
        let mut v: Vec<f64> = l.iter().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v.get(v.len() / 2).copied().unwrap_or(0.0)
    };
    let _ = kills_last;
    Ok(Summary { rewards, kills, items, ammo_spent, travelled, decisions, tics: tics_total, secs, lat_p50 })
}
