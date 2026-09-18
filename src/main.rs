mod bridge;
mod debertav2;
mod describe;
mod direct;
mod encoder;
mod encoder_metal;
mod head;
mod options;
mod state;
mod policy;
mod run_loop;
mod scorer;
mod serve;
mod systemone;
mod tui;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[derive(Parser)]
#[command(name = "gliner2-doom", version, about)]
struct Cli {
    /// Encoder for the direct policy: metal (vendored DeBERTa-v2 on the Apple GPU, default) | onnx (ONNX Runtime on the CPU).
    #[arg(long, global = true, default_value = "metal")]
    encoder: String,
    /// Device for the trained head and the metal encoder: auto (Metal when available, default) | metal | cpu.
    #[arg(long, global = true, default_value = "auto")]
    device: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Latency and RSS of classification-only decisions (the S1 gate).
    Bench {
        /// Directory holding an ONNX export; fetched from the Hub into the HF cache if empty.
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        /// fp32 | fp16 | fp16_iobinding
        #[arg(long, default_value = "fp16")]
        precision: String,
        #[arg(long, default_value_t = 100)]
        runs: usize,
        /// Label counts to sweep, comma separated.
        #[arg(long, default_value = "3,5,8,12")]
        label_counts: String,
    },
    /// Run a scripted oracle policy through the bridge (S2 check + kills baseline).
    Smoke {
        #[arg(long, default_value = "defend_the_center")]
        scenario: String,
        #[arg(long, default_value_t = 3)]
        episodes: u32,
        #[arg(long, default_value_t = 4)]
        frame_skip: u32,
    },
    /// GLiNER2.5 plays, synchronous (the engine waits for each decision). Default policy: the trained head.
    Play {
        #[arg(long, default_value = "defend_the_center")]
        scenario: String,
        #[arg(long, default_value_t = 3)]
        episodes: u32,
        /// Engine tics each decision is held for.
        #[arg(long, default_value_t = 4)]
        frame_skip: u32,
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        #[arg(long, default_value = "fp16")]
        precision: String,
        /// Append one JSON line per decision (jevlike training format).
        #[arg(long)]
        record: Option<PathBuf>,
        /// Print every decision, not just a summary.
        #[arg(long)]
        verbose: bool,
        /// Episode length cap in tics for level play (default 120 s).
        #[arg(long)]
        timeout_tics: Option<u32>,
        /// direct (trained head, default) | keyword (the zero-shot schema)
        #[arg(long, default_value = "direct")]
        policy: String,
        /// Trained head for --policy direct.
        #[arg(long, default_value = "heads/v3.safetensors")]
        head: PathBuf,
        /// IWAD for a real map (implies scenario=level).
        #[arg(long)]
        wad: Option<String>,
        #[arg(long)]
        map: Option<String>,
    },
    /// Real-time play: engine at 35 tics/s, model deciding as fast as it can, terminal UI. Default policy: the trained head.
    Live {
        #[arg(long, default_value = "defend_the_center")]
        scenario: String,
        #[arg(long, default_value_t = 3)]
        episodes: u32,
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        #[arg(long, default_value = "fp16")]
        precision: String,
        /// Run the same loop without drawing (for non-TTY checks).
        #[arg(long)]
        no_tui: bool,
        /// Width of the rendered frame in terminal cells.
        #[arg(long, default_value_t = 80)]
        cols: usize,
        /// Play a real map from this IWAD instead of a scenario (implies scenario=level).
        #[arg(long)]
        wad: Option<String>,
        #[arg(long)]
        map: Option<String>,
        /// Open ViZDoom's game window (watchable without a TTY; combine with --no-tui).
        #[arg(long)]
        window: bool,
        /// direct (trained head, default) | keyword (the zero-shot schema)
        #[arg(long, default_value = "direct")]
        policy: String,
        /// Trained head for --policy direct.
        #[arg(long, default_value = "heads/v3.safetensors")]
        head: PathBuf,
    },
    /// HTTP server speaking the System One contract (POST /v1/systemone).
    Serve {
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        #[arg(long, default_value = "fp16")]
        precision: String,
    },
    /// Business-task twin: route / score / yes-no over support messages with the same model, no HTTP.
    Triage {
        /// JSONL of System One requests; built-in examples when omitted.
        #[arg(long)]
        file: Option<PathBuf>,
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        #[arg(long, default_value = "fp16")]
        precision: String,
    },
    /// Direct policy, step 1: run the oracle (with epsilon-random actions) and record (context, options, label) rows.
    Collect {
        #[arg(long, default_value = "defend_the_center")]
        scenario: String,
        #[arg(long, default_value_t = 40)]
        episodes: u32,
        #[arg(long, default_value_t = 0.2)]
        epsilon: f64,
        #[arg(long, default_value = "data")]
        out_dir: PathBuf,
        #[arg(long)]
        timeout_tics: Option<u32>,
        /// Option table JSON (default: built-in).
        #[arg(long)]
        table: Option<PathBuf>,
        /// DAgger: let this head drive while the oracle labels.
        #[arg(long)]
        head: Option<PathBuf>,
        /// IWAD for a real map (implies scenario=level).
        #[arg(long)]
        wad: Option<String>,
        #[arg(long)]
        map: Option<String>,
        /// Open ViZDoom's game window to watch the rollout (the oracle, or the driving head).
        #[arg(long)]
        window: bool,
    },
    /// Direct policy, step 2: train the option-attention head on data/train.jsonl, select on val, fit temperature.
    Train {
        #[arg(long, default_value = "data")]
        data_dir: PathBuf,
        #[arg(long, default_value = "head.safetensors")]
        out: PathBuf,
        #[arg(long, default_value_t = 10)]
        epochs: usize,
        #[arg(long, default_value_t = 64)]
        batch: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        /// Ablation: permute contexts across rows so only the label prior is learnable.
        #[arg(long)]
        shuffle_contexts: bool,
        #[arg(long, default_value = "")]
        notes: String,
        /// Reward-weighted regression on the actions taken (no oracle labels). Use with data collected by a driving head.
        #[arg(long)]
        rwr: bool,
        #[arg(long, default_value_t = 5.0)]
        rwr_beta: f32,
        /// Warm start weights.
        #[arg(long)]
        init_from: Option<PathBuf>,
    },
    /// Direct policy, step 3: offline metrics of a head on a JSONL split (top-1, NLL, ECE, shuffled-context control).
    Eval {
        #[arg(long, default_value = "head.safetensors")]
        head: PathBuf,
        #[arg(long, default_value = "data/test.jsonl")]
        data: PathBuf,
    },
    /// Sanity check of the raw encoder path (token count, latency, batch).
    EncoderCheck {
        #[arg(long, default_value = "{\"hp\":72,\"ammo\":18,\"moved\":0,\"depth\":{\"l\":37,\"c\":49,\"r\":19},\"enemies\":[{\"t\":\"imp\",\"d\":210,\"a\":-14}],\"items\":[]}")]
        text: String,
    },
    /// Metal encoder vs the ONNX states already cached on disk: max/mean abs diff of hidden states (ticket gd-ngu7).
    EncoderParity {
        #[arg(long, default_value = "data2/states.metal.bin")]
        states: PathBuf,
        #[arg(long, default_value_t = 500)]
        n: usize,
        /// Contexts per forward pass; 1 tests the single-context path play uses.
        #[arg(long, default_value_t = 32)]
        batch: usize,
        /// cache (f16 values on disk) | onnx-fp16 | onnx-fp32 (recomputed through ONNX Runtime now; the clean reference).
        #[arg(long, default_value = "onnx-fp32")]
        reference: String,
    },
    /// Encoder throughput and latency on real contexts for the selected --encoder (ticket gd-ngu7 step 4).
    EncoderBench {
        #[arg(long, default_value = "data2/states.metal.bin")]
        states: PathBuf,
        #[arg(long, default_value_t = 512)]
        n: usize,
        /// Batch sizes to sweep, comma separated. 1 is the play-time path.
        #[arg(long, default_value = "1,8,32,64")]
        batches: String,
        #[arg(long, default_value_t = 3)]
        warmup: usize,
        /// Sort the sample by token count first so batches pad less (what build_cache should do too).
        #[arg(long)]
        bucket: bool,
    },
    /// Score a few canned Doom situations against a multi-task schema (S3 label design probe).
    Probe {
        #[arg(long, default_value = "models/gliner2.5-multi-v1-onnx")]
        models_dir: PathBuf,
        #[arg(long, default_value = "fp16")]
        precision: String,
        /// Extra situation lines to score, in addition to the canned ones.
        #[arg(long)]
        situation: Vec<String>,
        /// Override the schema: repeatable "task name=label one;label two;...".
        #[arg(long)]
        task: Vec<String>,
        /// Skip the canned situations.
        #[arg(long)]
        no_canned: bool,
    },
}

/// Candidate play-time schema: several small questions answered in one pass.
fn play_schema() -> Vec<(String, Vec<String>)> {
    let t = |name: &str, labels: &[&str]| (name.to_string(), labels.iter().map(|s| s.to_string()).collect());
    vec![
        t("enemy side", &["an enemy is left of the crosshair", "an enemy is right of the crosshair", "an enemy is centered in the crosshair", "no enemy is visible"]),
        t("enemy distance", &["the nearest enemy is close", "the nearest enemy is far away", "no enemy is visible"]),
        t("ammo", &["ammo is low", "ammo is fine"]),
        t("health", &["health is low", "health is fine"]),
        t("path ahead", &["a wall is close ahead", "open space ahead"]),
    ]
}

const CANNED: [&str; 5] = [
    "Situation: A demon slightly left of center, close. No items in view. Straight ahead: open space. Left: open space. Right: open space. Health 100, ammo 26, kills 0.",
    "Situation: An imp far right, far away. No items in view. Straight ahead: open space. Left: open space. Right: wall close. Health 100, ammo 26, kills 1.",
    "Situation: A zombie dead center, very close. No items in view. Straight ahead: open space. Left: open space. Right: open space. Health 45, ammo 3, kills 4.",
    "Situation: No monsters in view. No items in view. Straight ahead: wall close. Left: open space. Right: open space. Health 12, ammo 26, kills 2.",
    "Situation: A chainsaw marine slightly right of center, close. A demon far left, far away. No items in view. Straight ahead: open space. Health 80, ammo 0, kills 6.",
];

fn probe(models_dir: PathBuf, precision: String, extra: Vec<String>, task: Vec<String>, no_canned: bool) -> Result<()> {
    let prec = scorer::parse_precision(&precision)?;
    let mut sc = scorer::Scorer::new(&models_dir, prec)?;
    let schema = if task.is_empty() {
        play_schema()
    } else {
        task.iter()
            .map(|t| {
                let (name, labels) = t.split_once('=').expect("--task NAME=l1;l2;...");
                (name.trim().to_string(), labels.split(';').map(|l| l.trim().to_string()).collect())
            })
            .collect()
    };
    let n_labels: usize = schema.iter().map(|(_, l)| l.len()).sum();
    println!("schema: {} tasks, {} labels total", schema.len(), n_labels);
    let canned: Vec<String> = if no_canned { vec![] } else { CANNED.iter().map(|s| s.to_string()).collect() };
    let sits: Vec<String> = canned.into_iter().chain(extra).collect();
    for _ in 0..3 { let _ = scorer::classify_multi(&mut sc, &sits[0], &schema)?; }
    for s in &sits {
        let t = Instant::now();
        let res = scorer::classify_multi(&mut sc, s, &schema)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("\n{s}\n  [{ms:.0} ms]");
        for (task, rows) in res {
            let line: Vec<String> = rows.iter().map(|r| format!("{} {:.0}%", r.label, r.prob * 100.0)).collect();
            println!("  {task:<15} {}", line.join(" | "));
        }
    }
    Ok(())
}

/// A realistic situation line, the shape `describe` will emit at play time.
const SITUATION: &str = "Situation: A demon slightly left of center, close. A zombie far right, far away. \
No items in view. Straight ahead: open space. Left: wall close. Right: open space. \
Health 72, ammo 18, kills 3. Last action: turn right.";

/// World-statement labels, never action names (see ticket pro-2tst).
const LABELS: [&str; 12] = [
    "an enemy is left of the crosshair",
    "an enemy is right of the crosshair",
    "an enemy is centered in the crosshair",
    "no enemy is visible",
    "ammo is low",
    "health is low",
    "an enemy is very close",
    "a wall is close ahead",
    "open space ahead",
    "a health pickup is visible",
    "an ammo pickup is visible",
    "the player is stuck",
];

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok();
    out.and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(f64::NAN)
}

fn bench(models_dir: PathBuf, precision: String, runs: usize, label_counts: String) -> Result<()> {
    let prec = scorer::parse_precision(&precision)?;
    let rss0 = rss_mb();
    let mut sc = scorer::Scorer::new(&models_dir, prec)?;
    println!("load: {:.2}s  precision={precision}  rss_after_load={:.0} MB (was {:.0} MB)", sc.load_secs, rss_mb(), rss0);

    for n in label_counts.split(',').filter_map(|s| s.trim().parse::<usize>().ok()) {
        let labels: Vec<String> = LABELS.iter().take(n).map(|s| s.to_string()).collect();
        let mut top = String::new();
        for _ in 0..5 {
            top = sc.classify("situation", SITUATION, &labels)?[0].label.clone();
        }
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let t = Instant::now();
            let _ = sc.classify("situation", SITUATION, &labels)?;
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
            std::thread::sleep(Duration::from_millis(2));
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = samples[samples.len() / 2];
        let p95 = samples[((samples.len() as f64 * 0.95) as usize).min(samples.len() - 1)];
        println!(
            "labels={n:>2}  p50={p50:>7.1} ms  p95={p95:>7.1} ms  min={:>7.1} ms  -> {:>5.1} decisions/s  rss={:.0} MB  top={top:?}",
            samples[0], 1000.0 / p50, rss_mb()
        );
    }
    let full: Vec<String> = LABELS.iter().map(|s| s.to_string()).collect();
    println!("--- full distribution on the sample situation:");
    for s in sc.classify("situation", SITUATION, &full)? {
        println!("  {:5.1}%  {}", s.prob * 100.0, s.label);
    }
    Ok(())
}

/// Turn toward the nearest actor, shoot when it is inside the hit window.
/// Mirrors the oracle policy the openjev reproduction used (18.8 kills on defend_the_center).
fn oracle(o: &bridge::Obs) -> &'static str {
    match o.actors.first() {
        None => "TURN_RIGHT",
        Some(a) if a.rel_angle > 3.0 => "TURN_LEFT",
        Some(a) if a.rel_angle < -3.0 => "TURN_RIGHT",
        Some(_) => "ATTACK",
    }
}

fn smoke(scenario: String, episodes: u32, frame_skip: u32) -> Result<()> {
    let opts = bridge::InitOpts { scenario, mode: "sync".into(), width: 160, height: 120, ..Default::default() };
    let mut b = bridge::Bridge::spawn(&opts)?;
    println!("bridge up, buttons={:?}", b.buttons);
    let mut kills_all = vec![];
    for ep in 0..episodes {
        let t0 = Instant::now();
        let mut o = b.reset()?;
        let mut steps = 0u32;
        while !o.finished {
            let a = oracle(&o);
            o = b.act(&[a], frame_skip)?;
            steps += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        println!("episode {ep}: reward={:.0} kills={} steps={steps} tics={}  {:.1}s ({:.0} decisions/s)",
            o.total_reward, o.kills, o.tic, secs, steps as f64 / secs);
        kills_all.push(o.total_reward as i32);
    }
    println!("mean kills {:.2}", kills_all.iter().sum::<i32>() as f64 / kills_all.len() as f64);
    Ok(())
}

fn play(scenario: String, episodes: u32, frame_skip: u32, models_dir: PathBuf, precision: String,
        record: Option<PathBuf>, verbose: bool, timeout_tics: Option<u32>, policy_name: String, head: PathBuf,
        wad: Option<String>, map: Option<String>) -> Result<()> {
    use std::io::Write;
    let scenario = if wad.is_some() || map.is_some() { "level".to_string() } else { scenario };
    let opts = bridge::InitOpts { scenario, mode: "sync".into(), width: 160, height: 120, timeout_tics, wad, map, ..Default::default() };
    let mut b = bridge::Bridge::spawn(&opts)?;
    let mut brain = match policy_name.as_str() {
        "direct" => policy::Brain::direct(&head, &b.buttons)?,
        _ => policy::Brain::keyword(scorer::Scorer::new(&models_dir, scorer::parse_precision(&precision)?)?),
    };
    let direct = brain.is_direct();
    let mut tracker = state::Tracker::default();
    let mut rec = match &record {
        Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?),
        None => None,
    };
    let mut kills_all = vec![];
    let mut lat_all: Vec<f64> = vec![];
    for ep in 0..episodes {
        let t0 = Instant::now();
        let mut o = b.reset()?;
        let mut steps = 0u32;
        let mut ctl = policy::Controller::default();
        tracker.reset();
        o.moved = -1.0;
        while !o.finished {
            let t = Instant::now();
            let (d, text) = brain.decide(&o, &b.buttons)?;
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            lat_all.push(ms);
            let oracle_action = oracle(&o);
            if verbose {
                println!("[{:>4}] {:<20} conf={:.2} ({} oracle) {:.0}ms  {}", o.tic, d.action, d.confidence,
                    if d.buttons.first().copied() == Some(oracle_action) { "=" } else { "!=" }, ms, text);
            }
            if let Some(f) = rec.as_mut() {
                let line = serde_json::json!({
                    "context": text,
                    "options": if direct { d.options.iter().map(|s| s.label.clone()).collect::<Vec<_>>() } else { d.side.iter().map(|s| s.label.clone()).collect::<Vec<_>>() },
                    "probabilities": if direct { d.options.iter().map(|s| s.prob).collect::<Vec<_>>() } else { d.side.iter().map(|s| s.prob).collect::<Vec<_>>() },
                    "offset": d.offset.iter().map(|s| (s.label.clone(), s.prob)).collect::<Vec<_>>(),
                    "action": d.action, "oracle": oracle_action, "ms": ms,
                    "tic": o.tic, "health": o.health, "ammo": o.ammo, "kills": o.kills,
                });
                writeln!(f, "{line}")?;
            }
            let hold = d.hold_tics.min(frame_skip).max(1);
            let (buttons, recovering) = if direct { (d.buttons.clone(), false) } else { ctl.apply(d.buttons.clone(), &o.pos, hold) };
            if verbose && recovering {
                println!("        recovering: {:?} at pos ({:.0},{:.0})", buttons, o.pos.first().unwrap_or(&0.0), o.pos.get(1).unwrap_or(&0.0));
            }
            let buttons: Vec<&str> = buttons.into_iter().filter(|x| b.buttons.iter().any(|y| y == x)).collect();
            o = b.act(&buttons, hold)?;
            o.moved = tracker.update(&o.pos, hold);
            steps += 1;
        }
        let secs = t0.elapsed().as_secs_f64();
        println!("episode {ep}: reward={:.0} (=kills in scenarios) KILLCOUNT={} items={} ammo_left={} hp={} steps={steps} tics={}  {:.1}s ({:.1} decisions/s)",
            o.total_reward, o.kills, o.items, o.ammo, o.health, o.tic, secs, steps as f64 / secs);
        kills_all.push(o.total_reward as i32);
    }
    lat_all.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("mean reward {:.2} over {} episodes; decision latency p50 {:.0} ms p95 {:.0} ms",
        kills_all.iter().sum::<i32>() as f64 / kills_all.len() as f64, kills_all.len(),
        lat_all[lat_all.len() / 2], lat_all[(lat_all.len() as f64 * 0.95) as usize]);
    Ok(())
}

fn live(scenario: String, episodes: u32, models_dir: PathBuf, precision: String, no_tui: bool, cols: usize,
        wad: Option<String>, map: Option<String>, window: bool, policy_name: String, head: PathBuf) -> Result<()> {
    let prec = scorer::parse_precision(&precision)?;
    let direct = policy_name == "direct";
    let make = move |buttons: &[String]| -> Result<policy::Brain> {
        if direct { policy::Brain::direct(&head, buttons) } else { Ok(policy::Brain::keyword(scorer::Scorer::new(&models_dir, prec)?)) }
    };
    let scenario = if wad.is_some() || map.is_some() { "level".to_string() } else { scenario };
    let s = run_loop::run(make, run_loop::LiveOpts { scenario, episodes, tui: !no_tui, cols, wad, map, window, direct })?;
    println!("episodes: rewards {:?} (=kills in scenarios)  mean {:.2};  KILLCOUNT {:?};  items {:?};  ammo spent {:?};  path length {:?} map units", s.rewards,
        s.rewards.iter().sum::<f32>() / s.rewards.len().max(1) as f32, s.kills, s.items, s.ammo_spent, s.travelled.iter().map(|t| t.round()).collect::<Vec<_>>());
    println!("{} tics in {:.1}s = {:.1} tics/s; {} decisions = {:.1} decisions/s; decision latency p50 {:.0} ms",
        s.tics, s.secs, s.tics as f64 / s.secs, s.decisions, s.decisions as f64 / s.secs, s.lat_p50);
    Ok(())
}

fn triage(file: Option<PathBuf>, models_dir: PathBuf, precision: String) -> Result<()> {
    let prec = scorer::parse_precision(&precision)?;
    let mut sc = scorer::Scorer::new(&models_dir, prec)?;
    let reqs: Vec<systemone::Request> = match file {
        Some(p) => std::fs::read_to_string(p)?.lines().filter(|l| !l.trim().is_empty())
            .map(serde_json::from_str).collect::<std::result::Result<_, _>>()?,
        None => systemone::examples(),
    };
    for r in &reqs {
        let resp = systemone::answer(&mut sc, r)?;
        println!("state: {}", serde_json::to_string(&r.state)?);
        println!("  {:.0} ms, {} passes", resp.timing_ms, resp.usage["passes"]);
        for (qid, a) in &resp.answers {
            println!("  {qid:<12} {}", serde_json::to_string(a)?);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    encoder::set_kind(encoder::parse_kind(&cli.encoder)?);
    head::set_device(head::parse_device(&cli.device)?);
    match cli.cmd {
        Cmd::Bench { models_dir, precision, runs, label_counts } => bench(models_dir, precision, runs, label_counts),
        Cmd::Smoke { scenario, episodes, frame_skip } => smoke(scenario, episodes, frame_skip),
        Cmd::Play { scenario, episodes, frame_skip, models_dir, precision, record, verbose, timeout_tics, policy, head, wad, map } =>
            play(scenario, episodes, frame_skip, models_dir, precision, record, verbose, timeout_tics, policy, head, wad, map),
        Cmd::Live { scenario, episodes, models_dir, precision, no_tui, cols, wad, map, window, policy, head } =>
            live(scenario, episodes, models_dir, precision, no_tui, cols, wad, map, window, policy, head),
        Cmd::Serve { port, models_dir, precision } => {
            let prec = scorer::parse_precision(&precision)?;
            serve::run(move || scorer::Scorer::new(&models_dir, prec), port)
        }
        Cmd::Triage { file, models_dir, precision } => triage(file, models_dir, precision),
        Cmd::Collect { scenario, episodes, epsilon, out_dir, timeout_tics, table, head, wad, map, window } => {
            let scenario = if wad.is_some() || map.is_some() { "level".to_string() } else { scenario };
            let tag = match (&wad, &map) {
                (Some(w), Some(m)) => format!("{}:{m}", std::path::Path::new(w).file_stem().and_then(|x| x.to_str()).unwrap_or("wad")),
                (None, Some(m)) => format!("freedoom2:{m}"),
                _ => scenario.clone(),
            };
            let driver: Option<Box<dyn FnMut(&str, &[options::OptionDef]) -> Result<usize>>> = match head {
                Some(h) => Some(direct::driver_from_head(&h)?),
                None => None,
            };
            let (rows, rewards) = direct::collect(direct::CollectOpts { scenario, episodes, epsilon, out_dir: out_dir.clone(), timeout_tics: timeout_tics.or(Some(2100)), table, wad, map, driver, tag, window })?;
            println!("collected {rows} rows into {}; episode rewards mean {:.2}; totals train={} val={} test={}", out_dir.display(),
                rewards.iter().sum::<f32>() / rewards.len().max(1) as f32,
                direct::count_rows(&out_dir.join("train.jsonl")), direct::count_rows(&out_dir.join("val.jsonl")), direct::count_rows(&out_dir.join("test.jsonl")));
            Ok(())
        }
        Cmd::EncoderCheck { text } => direct::encoder_check(&text),
        Cmd::EncoderParity { states, n, batch, reference } => direct::encoder_parity(&states, n, batch, &reference),
        Cmd::EncoderBench { states, n, batches, warmup, bucket } => direct::encoder_bench(&states, n, &batches, warmup, bucket),
        Cmd::Train { data_dir, out, epochs, batch, lr, shuffle_contexts, notes, rwr, rwr_beta, init_from } =>
            direct::train(direct::TrainOpts { data_dir, out, epochs, batch, lr, shuffle_contexts, notes, rwr, rwr_beta, init_from }),
        Cmd::Eval { head, data } => direct::eval(&head, &data),
        Cmd::Probe { models_dir, precision, situation, task, no_canned } => probe(models_dir, precision, situation, task, no_canned),
    }
}
