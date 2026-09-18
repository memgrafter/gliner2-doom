//! The direct policy pipeline: collect -> train -> eval -> play.
//! Context JSON in, option probabilities out. No describer, no rules.

use crate::bridge::{Bridge, InitOpts};
use crate::encoder::Encoder;
use crate::options::{OptionDef, available, load_table};
use crate::state::{Tracker, oracle, serialize};
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub context: String,
    pub options: Vec<String>,
    /// Oracle's choice (the training label).
    pub label: usize,
    /// What was actually pressed (differs from label under epsilon or a driving head).
    pub action: usize,
    pub reward: f32,
    pub ep: u32,
    pub tic: u32,
    pub scenario: String,
    #[serde(default)]
    pub kills: i32,
    #[serde(default)]
    pub items: i32,
    #[serde(default)]
    pub health: i32,
    #[serde(default)]
    pub ammo: i32,
}

pub struct CollectOpts {
    pub scenario: String,
    pub episodes: u32,
    pub epsilon: f64,
    pub out_dir: PathBuf,
    pub timeout_tics: Option<u32>,
    pub table: Option<PathBuf>,
    pub wad: Option<String>,
    pub map: Option<String>,
    /// When set, this head drives and the oracle only labels (DAgger).
    pub driver: Option<Box<dyn FnMut(&str, &[OptionDef]) -> Result<usize>>>,
    pub tag: String,
    /// Open ViZDoom's game window so the rollout can be watched.
    pub window: bool,
}

pub fn collect(mut o: CollectOpts) -> Result<(usize, Vec<f32>)> {
    let table = load_table(o.table.as_deref())?;
    std::fs::create_dir_all(&o.out_dir)?;
    let mut files: Vec<std::fs::File> = ["train", "val", "test"]
        .iter()
        .map(|n| std::fs::OpenOptions::new().create(true).append(true).open(o.out_dir.join(format!("{n}.jsonl"))))
        .collect::<std::io::Result<_>>()?;
    let init = InitOpts { scenario: o.scenario.clone(), mode: "sync".into(), width: if o.window { 640 } else { 160 }, height: if o.window { 480 } else { 120 }, visible: o.window, timeout_tics: o.timeout_tics, wad: o.wad.clone(), map: o.map.clone(), ..Default::default() };
    let mut b = Bridge::spawn(&init)?;
    let opts = available(&table, &b.buttons);
    let names: Vec<String> = opts.iter().map(|x| x.name.clone()).collect();
    let mut rng = rand::rng();
    let mut rows = 0usize;
    let mut rewards = vec![];
    let t0 = Instant::now();
    for ep in 0..o.episodes {
        let mut obs = b.reset()?;
        let mut tracker = Tracker::default();
        let mut moved = -1.0f32;
        // Of every 8 episodes, the 7th goes to val and the 8th to test, so 8-episode
        // map collections contribute to both; 4-episode DAgger runs stay train-only.
        let split = match ep % 8 { 6 => 1, 7 => 2, _ => 0 };
        while !obs.finished {
            let context = serialize(&obs, moved);
            let label = oracle(&obs, moved, &opts);
            let action = if rng.random::<f64>() < o.epsilon {
                rng.random_range(0..opts.len())
            } else if let Some(d) = o.driver.as_mut() {
                d(&context, &opts)?
            } else {
                label
            };
            let row = Row { context, options: names.clone(), label, action, reward: obs.total_reward, ep, tic: obs.tic, scenario: o.tag.clone(),
                kills: obs.kills, items: obs.items, health: obs.health, ammo: obs.ammo };
            writeln!(files[split], "{}", serde_json::to_string(&row)?)?;
            rows += 1;
            let hold = opts[action].hold;
            let buttons: Vec<&str> = opts[action].buttons.iter().map(String::as_str).collect();
            obs = b.act(&buttons, hold)?;
            moved = tracker.update(&obs.pos, hold);
        }
        rewards.push(obs.total_reward);
        if ep % 4 == 3 {
            eprintln!("  ep {ep}: rows so far {rows}, reward {:.0}, kills {}, items {}, ammo {}, {:.0}s", obs.total_reward, obs.kills, obs.items, obs.ammo, t0.elapsed().as_secs_f64());
        }
    }
    Ok((rows, rewards))
}

pub fn encoder_check(text: &str) -> Result<()> {
    let mut enc = Encoder::load(gliner25_rs::Precision::Fp16, 4)?;
    println!("encoder backend: {}", enc.name());
    let t = Instant::now();
    let e = enc.encode_batch(&[text])?;
    println!("1 text: {} tokens, hidden {}x{}x768, {:.0} ms (first call)", e.s, e.b, e.s, t.elapsed().as_secs_f64() * 1000.0);
    for _ in 0..3 { let _ = enc.encode_batch(&[text])?; }
    let t = Instant::now();
    let _ = enc.encode_batch(&[text])?;
    println!("1 text warm: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    let many: Vec<&str> = std::iter::repeat_n(text, 32).collect();
    let t = Instant::now();
    let e = enc.encode_batch(&many)?;
    println!("32 texts: {}x{}x768, {:.0} ms", e.b, e.s, t.elapsed().as_secs_f64() * 1000.0);
    let p = enc.pool("turn left a little")?;
    println!("pooled option vector: len {} first {:.3} {:.3}", p.len(), p[0], p[1]);
    Ok(())
}

pub fn count_rows(p: &Path) -> usize {
    // used by the collect summary

    std::fs::read_to_string(p).map(|s| s.lines().filter(|l| !l.trim().is_empty()).count()).unwrap_or(0)
}


// ---------------------------------------------------------------------------
// Dataset -> tensors
// ---------------------------------------------------------------------------

use crate::head::{self, Head, HeadMeta};
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap, ops};
use half::f16;
use std::collections::HashMap;

pub fn read_rows(p: &Path) -> Result<Vec<Row>> {
    Ok(std::fs::read_to_string(p)?.lines().filter(|l| !l.trim().is_empty()).map(serde_json::from_str).collect::<std::result::Result<_, _>>()?)
}

/// Hidden states for every distinct context, kept as f16 to fit in RAM.
pub struct Cache {
    pub states: HashMap<String, (usize, Vec<f16>)>,
    pub option_vecs: HashMap<String, Vec<f32>>,
}

/// On-disk cache of encoder states (`<data_dir>/states.bin`, or `states.metal.bin`
/// for the Metal encoder; append-only:
/// u32 ctx_len, ctx bytes, u16 n_tokens, n_tokens*768 f16). Encoding 20k
/// contexts costs ~8 minutes; re-reading them costs a second.
fn read_state_file(p: &Path) -> HashMap<String, (usize, Vec<f16>)> {
    let mut out = HashMap::new();
    let Ok(bytes) = std::fs::read(p) else { return out };
    let mut i = 0usize;
    while i + 6 <= bytes.len() {
        let cl = u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        if i + cl + 2 > bytes.len() { break; }
        let ctx = String::from_utf8_lossy(&bytes[i..i + cl]).to_string();
        i += cl;
        let n = u16::from_le_bytes(bytes[i..i + 2].try_into().unwrap()) as usize;
        i += 2;
        let nb = n * HIDDEN * 2;
        if i + nb > bytes.len() { break; }
        let v: Vec<f16> = bytes[i..i + nb].chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])).collect();
        i += nb;
        out.insert(ctx, (n, v));
    }
    out
}

fn append_state(f: &mut std::fs::File, ctx: &str, n: usize, v: &[f16]) -> Result<()> {
    f.write_all(&(ctx.len() as u32).to_le_bytes())?;
    f.write_all(ctx.as_bytes())?;
    f.write_all(&(n as u16).to_le_bytes())?;
    let mut buf = Vec::with_capacity(v.len() * 2);
    for x in v { buf.extend_from_slice(&x.to_le_bytes()); }
    f.write_all(&buf)?;
    Ok(())
}

pub fn build_cache(enc: &mut Encoder, rows: &[Row], batch: usize, disk: Option<&Path>) -> Result<Cache> {
    let mut states = disk.map(read_state_file).unwrap_or_default();
    let mut file = match disk {
        Some(p) => Some(std::fs::OpenOptions::new().create(true).append(true).open(p)?),
        None => None,
    };
    let mut uniq: Vec<&str> = rows.iter().map(|r| r.context.as_str()).filter(|c| !states.contains_key(*c)).collect();
    // gd-eok7: batches pad to their longest context, so sort by length first (byte length tracks token count).
    uniq.sort_by(|a, b| a.len().cmp(&b.len()).then_with(|| a.cmp(b)));
    uniq.dedup();
    let t0 = Instant::now();
    eprintln!("cache: {} contexts on disk, {} to encode", states.len(), uniq.len());
    for (bi, chunk) in uniq.chunks(batch).enumerate() {
        let e = enc.encode_batch(chunk)?;
        for (i, ctx) in chunk.iter().enumerate() {
            let len = e.mask[i * e.s..(i + 1) * e.s].iter().filter(|&&m| m > 0.0).count();
            let v: Vec<f16> = e.hidden[i * e.s * HIDDEN..(i * e.s + len) * HIDDEN].iter().map(|&x| f16::from_f32(x)).collect();
            if let Some(f) = file.as_mut() { append_state(f, ctx, len, &v)?; }
            states.insert(ctx.to_string(), (len, v));
        }
        if bi % 50 == 0 {
            eprintln!("  encoded {}/{} contexts, {:.0}s", ((bi + 1) * batch).min(uniq.len()), uniq.len(), t0.elapsed().as_secs_f64());
        }
    }
    let mut option_vecs = HashMap::new();
    for r in rows {
        for o in &r.options {
            if !option_vecs.contains_key(o) {
                option_vecs.insert(o.clone(), enc.pool(o)?);
            }
        }
    }
    eprintln!("cache: {} contexts, {} options, {:.0}s", states.len(), option_vecs.len(), t0.elapsed().as_secs_f64());
    Ok(Cache { states, option_vecs })
}

use crate::encoder::HIDDEN;

/// Tensors for a batch of rows (padded).
pub fn batch_tensors(cache: &Cache, rows: &[&Row], dev: &Device) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let b = rows.len();
    let s = rows.iter().map(|r| cache.states[&r.context].0).max().unwrap_or(1);
    let j = rows.iter().map(|r| r.options.len()).max().unwrap_or(1);
    let mut h = vec![0f32; b * s * HIDDEN];
    let mut m = vec![0f32; b * s];
    let mut o = vec![0f32; b * j * HIDDEN];
    let mut om = vec![0f32; b * j];
    let mut y = vec![0u32; b];
    for (i, r) in rows.iter().enumerate() {
        let (len, st) = &cache.states[&r.context];
        for (k, x) in st.iter().enumerate() {
            h[i * s * HIDDEN + k] = x.to_f32();
        }
        for t in 0..*len {
            m[i * s + t] = 1.0;
        }
        for (q, name) in r.options.iter().enumerate() {
            let v = &cache.option_vecs[name];
            o[(i * j + q) * HIDDEN..(i * j + q + 1) * HIDDEN].copy_from_slice(v);
            om[i * j + q] = 1.0;
        }
        y[i] = r.label as u32;
    }
    Ok((
        Tensor::from_vec(h, (b, s, HIDDEN), dev)?,
        Tensor::from_vec(m, (b, s), dev)?,
        Tensor::from_vec(o, (b, j, HIDDEN), dev)?,
        Tensor::from_vec(om, (b, j), dev)?,
        Tensor::from_vec(y, b, dev)?,
    ))
}

/// Logits for every row, in order, batched.
pub fn all_logits(head: &Head, cache: &Cache, rows: &[Row], batch: usize) -> Result<Vec<Vec<f32>>> {
    let dev = head::device();
    let mut out = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(batch) {
        let refs: Vec<&Row> = chunk.iter().collect();
        let (h, m, o, om, _) = batch_tensors(cache, &refs, &dev)?;
        let lg = head.forward(&h, &m, &o, &om)?.to_vec2::<f32>()?;
        for (i, r) in chunk.iter().enumerate() {
            out.push(lg[i][..r.options.len()].to_vec());
        }
    }
    Ok(out)
}

fn softmax_t(logits: &[f32], t: f32) -> Vec<f32> {
    let mx = logits.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = logits.iter().map(|x| ((x - mx) / t).exp()).collect();
    let z: f32 = e.iter().sum();
    e.iter().map(|x| x / z).collect()
}

pub struct Metrics {
    pub top1: f32,
    pub nll: f32,
    pub ece: f32,
}

pub fn metrics(logits: &[Vec<f32>], rows: &[Row], t: f32) -> Metrics {
    let n = rows.len().max(1) as f32;
    let mut correct = 0f32;
    let mut nll = 0f32;
    let mut bins = vec![(0f32, 0f32, 0f32); 10]; // (sum conf, sum correct, count)
    for (lg, r) in logits.iter().zip(rows) {
        let p = softmax_t(lg, t);
        let (am, pm) = p.iter().enumerate().fold((0, -1f32), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
        let ok = if am == r.label { 1.0 } else { 0.0 };
        correct += ok;
        nll += -(p[r.label].max(1e-9)).ln();
        let bi = ((pm * 10.0) as usize).min(9);
        bins[bi].0 += pm;
        bins[bi].1 += ok;
        bins[bi].2 += 1.0;
    }
    let ece = bins.iter().filter(|b| b.2 > 0.0).map(|b| (b.2 / n) * ((b.0 / b.2) - (b.1 / b.2)).abs()).sum();
    Metrics { top1: correct / n, nll: nll / n, ece }
}

pub fn fit_temperature(logits: &[Vec<f32>], rows: &[Row]) -> f32 {
    let mut best = (1.0f32, f32::MAX);
    let mut t = 0.5f32;
    while t <= 3.0 {
        let m = metrics(logits, rows, t);
        if m.nll < best.1 {
            best = (t, m.nll);
        }
        t += 0.05;
    }
    best.0
}

pub struct TrainOpts {
    pub data_dir: PathBuf,
    pub out: PathBuf,
    pub epochs: usize,
    pub batch: usize,
    pub lr: f64,
    pub shuffle_contexts: bool,
    pub notes: String,
    /// Reward-weighted regression: train on the actions actually taken, weighted
    /// by exp(advantage / beta) of the engine reward-to-go. No oracle in the loop.
    pub rwr: bool,
    pub rwr_beta: f32,
    /// Warm start from this head (RWR fine-tunes; from scratch otherwise).
    pub init_from: Option<PathBuf>,
}

/// Shaped per-step reward from engine counters, then discounted reward-to-go
/// within each (scenario, episode). Kills 10, items 3, health delta 0.05 per
/// point, ammo spent -0.05 per round (so shots that do not lead to kills cost).
pub fn reward_to_go(rows: &[Row], gamma: f32) -> Vec<f32> {
    let mut g = vec![0f32; rows.len()];
    // rows are appended in order; group consecutive rows by (scenario, ep)
    let mut i = 0;
    while i < rows.len() {
        let mut j = i;
        while j + 1 < rows.len() && rows[j + 1].scenario == rows[i].scenario && rows[j + 1].ep == rows[i].ep && rows[j + 1].tic > rows[j].tic {
            j += 1;
        }
        // per-step rewards r[t] = signal(t+1) - signal(t); scenario reward (kills in defend) via total_reward delta
        let mut r = vec![0f32; j - i + 1];
        for t in i..j {
            let (a, b) = (&rows[t], &rows[t + 1]);
            r[t - i] = 10.0 * (b.kills - a.kills).max(0) as f32 + 10.0 * (b.reward - a.reward).max(0.0)
                + 3.0 * (b.items - a.items).max(0) as f32 + 0.05 * (b.health - a.health) as f32
                - 0.05 * (a.ammo - b.ammo).max(0) as f32;
        }
        let mut acc = 0f32;
        for t in (i..=j).rev() {
            acc = r[t - i] + gamma * acc;
            g[t] = acc;
        }
        i = j + 1;
    }
    g
}

pub fn train(o: TrainOpts) -> Result<()> {
    let mut train_rows = read_rows(&o.data_dir.join("train.jsonl"))?;
    let val_rows = read_rows(&o.data_dir.join("val.jsonl"))?;
    let mut weights: Vec<f32> = vec![1.0; train_rows.len()];
    if o.rwr {
        let g = reward_to_go(&train_rows, 0.97);
        // advantage against the per-scenario mean, exponentiated and normalised to mean 1
        let mut sum: HashMap<String, (f32, usize)> = HashMap::new();
        for (r, v) in train_rows.iter().zip(&g) { let e = sum.entry(r.scenario.clone()).or_default(); e.0 += v; e.1 += 1; }
        let mut w: Vec<f32> = train_rows.iter().zip(&g).map(|(r, v)| { let (s, n) = sum[&r.scenario]; ((v - s / n as f32) / o.rwr_beta).clamp(-3.0, 3.0).exp() }).collect();
        let mean = w.iter().sum::<f32>() / w.len().max(1) as f32;
        for x in &mut w { *x /= mean; }
        weights = w;
        // the label is what was actually done
        for r in &mut train_rows { r.label = r.action; }
        let hi = weights.iter().filter(|&&x| x > 1.5).count();
        eprintln!("rwr: {} rows, {} with weight > 1.5, beta {}", train_rows.len(), hi, o.rwr_beta);
    }
    if o.shuffle_contexts {
        // Ablation: break the state/label correspondence while keeping both marginals.
        let mut rng = rand::rng();
        let mut by_opts: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
        for (i, r) in train_rows.iter().enumerate() {
            by_opts.entry(r.options.clone()).or_default().push(i);
        }
        for idxs in by_opts.values() {
            let mut ctxs: Vec<String> = idxs.iter().map(|&i| train_rows[i].context.clone()).collect();
            for i in (1..ctxs.len()).rev() {
                let j = rng.random_range(0..=i);
                ctxs.swap(i, j);
            }
            for (k, &i) in idxs.iter().enumerate() {
                train_rows[i].context = ctxs[k].clone();
            }
        }
        eprintln!("ablation: contexts permuted across rows");
    }
    eprintln!("train {} rows, val {} rows", train_rows.len(), val_rows.len());
    let mut enc = Encoder::load(gliner25_rs::Precision::Fp16, 8)?;
    let mut all: Vec<Row> = train_rows.clone();
    all.extend(val_rows.iter().cloned());
    let cache = build_cache(&mut enc, &all, 64, Some(&o.data_dir.join(crate::encoder::states_file_name())))?;
    drop(enc);

    let dev = head::device();
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let model = Head::new(vb)?;
    if let Some(p) = &o.init_from {
        varmap.load(p)?;
        eprintln!("warm start from {}", p.display());
    }
    let mut opt = candle_nn::AdamW::new(varmap.all_vars(), candle_nn::ParamsAdamW { lr: o.lr, weight_decay: 0.01, ..Default::default() })?;
    let mut rng = rand::rng();
    let mut idx: Vec<usize> = (0..train_rows.len()).collect();
    let mut best = (f32::MIN, 0usize);
    let tmp = o.out.with_extension("best.safetensors");
    for epoch in 0..o.epochs {
        let t0 = Instant::now();
        for i in (1..idx.len()).rev() {
            let j = rng.random_range(0..=i);
            idx.swap(i, j);
        }
        let mut loss_sum = 0f32;
        let mut nb = 0usize;
        for chunk in idx.chunks(o.batch) {
            let refs: Vec<&Row> = chunk.iter().map(|&i| &train_rows[i]).collect();
            let (h, m, op, om, y) = batch_tensors(&cache, &refs, &dev)?;
            let logits = model.forward(&h, &m, &op, &om)?;
            let logp = ops::log_softmax(&logits, D::Minus1)?;
            let wt = Tensor::from_vec(chunk.iter().map(|&i| weights[i]).collect::<Vec<f32>>(), chunk.len(), &dev)?;
            let picked = logp.gather(&y.unsqueeze(1)?, 1)?.squeeze(1)?; // [B]
            let nll = (picked * wt)?.mean_all()?.neg()?;
            // label smoothing over the real options only
            let smooth = ((&logp * &om)?.sum_all()? / om.sum_all()?)?.neg()?;
            let loss = ((nll * 0.95)? + (smooth * 0.05)?)?;
            opt.backward_step(&loss)?;
            loss_sum += loss.to_scalar::<f32>()?;
            nb += 1;
        }
        let vl = all_logits(&model, &cache, &val_rows, 256)?;
        let vm = metrics(&vl, &val_rows, 1.0);
        eprintln!("epoch {epoch}: loss {:.4}  val top1 {:.4}  val nll {:.4}  ece {:.4}  {:.0}s", loss_sum / nb.max(1) as f32, vm.top1, vm.nll, vm.ece, t0.elapsed().as_secs_f64());
        if vm.top1 > best.0 {
            best = (vm.top1, epoch);
            varmap.save(&tmp)?;
        }
    }
    // Restore best, fit temperature on val, save.
    let mut varmap2 = VarMap::new();
    let vb2 = VarBuilder::from_varmap(&varmap2, DType::F32, &dev);
    let model2 = Head::new(vb2)?;
    varmap2.load(&tmp)?;
    std::fs::remove_file(&tmp).ok();
    let vl = all_logits(&model2, &cache, &val_rows, 256)?;
    let t = fit_temperature(&vl, &val_rows);
    let vm = metrics(&vl, &val_rows, t);
    let mut names: Vec<String> = cache.option_vecs.keys().cloned().collect();
    names.sort();
    let meta = HeadMeta { temperature: t, d_att: head::D_ATT, options: names, trained_rows: train_rows.len(), notes: o.notes };
    head::save(&varmap2, &meta, &o.out)?;
    println!("saved {} (best epoch {}, val top1 {:.4}, T={t:.2}, val nll {:.4}, ece {:.4})", o.out.display(), best.1, vm.top1, vm.nll, vm.ece);
    Ok(())
}

pub fn eval(head_path: &Path, data: &Path) -> Result<()> {
    let rows = read_rows(data)?;
    let loaded = head::load(head_path)?;
    let mut enc = Encoder::load(gliner25_rs::Precision::Fp16, 8)?;
    let cache = build_cache(&mut enc, &rows, 64, data.parent().map(|d| d.join(crate::encoder::states_file_name())).as_deref())?;
    drop(enc);
    let lg = all_logits(&loaded.head, &cache, &rows, 256)?;
    let t = loaded.meta.temperature;
    let m = metrics(&lg, &rows, t);
    println!("{}: {} rows  top1 {:.4}  nll {:.4}  ece {:.4}  (T={t:.2})", data.display(), rows.len(), m.top1, m.nll, m.ece);
    // Per scenario.
    let mut scen: Vec<String> = rows.iter().map(|r| r.scenario.clone()).collect();
    scen.sort();
    scen.dedup();
    for sc in &scen {
        let (l, r): (Vec<Vec<f32>>, Vec<Row>) = lg.iter().zip(&rows).filter(|(_, r)| &r.scenario == sc).map(|(l, r)| (l.clone(), r.clone())).unzip();
        let mm = metrics(&l, &r, t);
        println!("  {sc:<18} {} rows  top1 {:.4}  ece {:.4}", r.len(), mm.top1, mm.ece);
    }
    // Shuffled-context control: each row scored with another row's context (same option set).
    let mut rng = rand::rng();
    let mut by_opts: HashMap<Vec<String>, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        by_opts.entry(r.options.clone()).or_default().push(i);
    }
    let mut shuffled = rows.clone();
    for idxs in by_opts.values() {
        let mut ctxs: Vec<String> = idxs.iter().map(|&i| rows[i].context.clone()).collect();
        for i in (1..ctxs.len()).rev() {
            let j = rng.random_range(0..=i);
            ctxs.swap(i, j);
        }
        for (k, &i) in idxs.iter().enumerate() {
            shuffled[i].context = ctxs[k].clone();
        }
    }
    let lgs = all_logits(&loaded.head, &cache, &shuffled, 256)?;
    let ms = metrics(&lgs, &shuffled, t);
    let majority = {
        let mut c: HashMap<(Vec<String>, usize), usize> = HashMap::new();
        for r in &rows { *c.entry((r.options.clone(), r.label)).or_default() += 1; }
        let mut best: HashMap<Vec<String>, usize> = HashMap::new();
        for ((o, _), n) in &c { let e = best.entry(o.clone()).or_default(); if *n > *e { *e = *n; } }
        best.values().sum::<usize>() as f32 / rows.len().max(1) as f32
    };
    println!("  shuffled-context control top1 {:.4} (majority-label baseline {:.4}; a real model sits far above both)", ms.top1, majority);
    // Confusions.
    let mut conf: HashMap<(String, String), usize> = HashMap::new();
    for (l, r) in lg.iter().zip(&rows) {
        let p = softmax_t(l, t);
        let am = p.iter().enumerate().fold((0, -1f32), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc }).0;
        if am != r.label {
            *conf.entry((r.options[r.label].clone(), r.options[am].clone())).or_default() += 1;
        }
    }
    let mut cv: Vec<_> = conf.into_iter().collect();
    cv.sort_by(|a, b| b.1.cmp(&a.1));
    for ((t, p), n) in cv.iter().take(8) {
        println!("  confusion: truth={t:<22} pred={p:<22} {n}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Runtime: the head as a policy
// ---------------------------------------------------------------------------

pub struct DirectBrain {
    enc: Encoder,
    loaded: head::Loaded,
}

impl DirectBrain {
    pub fn load(head_path: &Path) -> Result<Self> {
        Ok(Self { enc: Encoder::load(gliner25_rs::Precision::Fp16, 4)?, loaded: head::load(head_path)? })
    }
    /// Probabilities over `opts` for `context`.
    pub fn score(&mut self, context: &str, opts: &[OptionDef]) -> Result<Vec<f32>> {
        let e = self.enc.encode_batch(&[context])?;
        let mut ov = Vec::with_capacity(opts.len() * HIDDEN);
        for o in opts {
            ov.extend_from_slice(&self.enc.pool(&o.name)?);
        }
        head::predict(&self.loaded.head, &e.hidden[..e.s * HIDDEN], e.s, &ov, opts.len(), self.loaded.meta.temperature)
    }
}

pub fn driver_from_head(path: &Path) -> Result<Box<dyn FnMut(&str, &[OptionDef]) -> Result<usize>>> {
    let mut brain = DirectBrain::load(path)?;
    Ok(Box::new(move |ctx: &str, opts: &[OptionDef]| {
        let p = brain.score(ctx, opts)?;
        Ok(p.iter().enumerate().fold((0, -1f32), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc }).0)
    }))
}

// ---------------------------------------------------------------------------
// Metal encoder parity (ticket gd-ngu7 step 3)
// ---------------------------------------------------------------------------

/// Re-encode contexts sampled from an ONNX cache file and compare hidden states.
/// `reference`: cache (the f16 values on disk, from the ONNX fp16 export) | onnx-fp16 |
/// onnx-fp32 (recomputed now through ONNX Runtime, f32 output). The fp32 export is
/// the clean reference; the ticket's bar is max <= 5e-3 and mean <= 5e-4 against it.
pub fn encoder_parity(states: &Path, n: usize, batch: usize, reference: &str) -> Result<()> {
    use crate::encoder::OnnxEncoder;
    let disk = read_state_file(states);
    anyhow::ensure!(!disk.is_empty(), "no cached states in {}", states.display());
    let mut all: Vec<&String> = disk.keys().collect();
    all.sort();
    let step = (all.len() / n.max(1)).max(1);
    let picked: Vec<&String> = all.iter().step_by(step).take(n).copied().collect();
    let mut onnx = match reference {
        "cache" => None,
        "onnx-fp16" => Some(OnnxEncoder::load(gliner25_rs::Precision::Fp16, 4)?),
        "onnx-fp32" => Some(OnnxEncoder::load(gliner25_rs::Precision::Fp32, 4)?),
        _ => anyhow::bail!("--reference must be cache, onnx-fp16 or onnx-fp32"),
    };
    let enc = crate::encoder_metal::MetalEncoder::load(None)?;
    let (mut max, mut sum, mut ref_abs, mut cnt, mut len_mismatch) = (0f32, 0f64, 0f64, 0usize, 0usize);
    let mut worst: Vec<(f32, f32, f32, usize, usize)> = Vec::new(); // (diff, ref, mine, token, dim)
    let (mut over_rel_1pct, mut over_abs_5e3) = (0usize, 0usize);
    let (mut t_metal, mut t_ref) = (0f64, 0f64);
    for chunk in picked.chunks(batch) {
        let texts: Vec<&str> = chunk.iter().map(|s| s.as_str()).collect();
        let t = Instant::now();
        let e = enc.encode_batch(&texts)?;
        t_metal += t.elapsed().as_secs_f64();
        let t = Instant::now();
        let refs: Vec<(usize, Vec<f32>)> = match onnx.as_mut() {
            Some(o) => {
                let r = o.encode_batch(&texts)?;
                (0..chunk.len())
                    .map(|i| {
                        let len = r.mask[i * r.s..(i + 1) * r.s].iter().filter(|&&m| m > 0.0).count();
                        (len, r.hidden[i * r.s * HIDDEN..(i * r.s + len) * HIDDEN].to_vec())
                    })
                    .collect()
            }
            None => chunk.iter().map(|c| { let (l, v) = &disk[*c]; (*l, v.iter().map(|x| x.to_f32()).collect()) }).collect(),
        };
        t_ref += t.elapsed().as_secs_f64();
        for (i, (len, ref_h)) in refs.iter().enumerate() {
            let got = e.mask[i * e.s..(i + 1) * e.s].iter().filter(|&&m| m > 0.0).count();
            if got != *len {
                len_mismatch += 1;
                continue;
            }
            let mine = &e.hidden[i * e.s * HIDDEN..(i * e.s + len) * HIDDEN];
            for (k, (a, b)) in mine.iter().zip(ref_h).enumerate() {
                let d = (a - b).abs();
                if d > max {
                    max = d;
                }
                if d > 5e-3 {
                    over_abs_5e3 += 1;
                }
                if d / (b.abs() + 1.0) > 0.01 {
                    over_rel_1pct += 1;
                }
                if worst.len() < 5 || d > worst[4].0 {
                    worst.push((d, *b, *a, k / HIDDEN, k % HIDDEN));
                    worst.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
                    worst.truncate(5);
                }
                sum += d as f64;
                ref_abs += b.abs() as f64;
                cnt += 1;
            }
        }
    }
    let mean = sum / cnt.max(1) as f64;
    println!(
        "parity vs {reference} ({} sample): {} contexts, batch {batch}, metal {:?}; metal {:.1}s, reference {:.1}s; {} values; max abs diff {max:.2e}; mean abs diff {mean:.2e}; mean |ref| {:.3}; token-count mismatches {len_mismatch}",
        states.display(), picked.len(), enc.dtype, t_metal, t_ref, cnt, ref_abs / cnt.max(1) as f64
    );
    println!("values with abs diff > 5e-3: {over_abs_5e3} ({:.4}%); with relative diff > 1% (|a-b|/(|ref|+1)): {over_rel_1pct} ({:.5}%)",
        100.0 * over_abs_5e3 as f64 / cnt.max(1) as f64, 100.0 * over_rel_1pct as f64 / cnt.max(1) as f64);
    for (d, r, m, t, k) in &worst {
        println!("  worst: diff {d:.3e} at token {t} dim {k}: ref {r:.3} vs metal {m:.3}");
    }
    println!("bar: max <= 5e-3 and mean <= 5e-4 and no mismatches -> {}", if max <= 5e-3 && mean <= 5e-4 && len_mismatch == 0 { "PASS" } else { "FAIL" });
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoder throughput / latency bench (ticket gd-ngu7 step 4)
// ---------------------------------------------------------------------------

/// Encode real contexts sampled from a cache file with the selected backend at each
/// batch size: ms per context (throughput) and p50 per call (latency). Batch 1 is the
/// play-time number; it is dispatch-bound on Metal, so take it on an idle machine.
pub fn encoder_bench(states: &Path, n: usize, batches: &str, warmup: usize, bucket: bool) -> Result<()> {
    let disk = read_state_file(states);
    anyhow::ensure!(!disk.is_empty(), "no cached states in {}", states.display());
    let mut all: Vec<&String> = disk.keys().collect();
    all.sort();
    let step = (all.len() / n.max(1)).max(1);
    let mut picked: Vec<&str> = all.iter().step_by(step).take(n).map(|s| s.as_str()).collect();
    if bucket {
        picked.sort_by_key(|c| disk[*c].0); // length bucketing: batches pad far less
    }
    let tokens: f64 = picked.iter().map(|c| disk[*c].0 as f64).sum::<f64>() / picked.len() as f64;
    let mut enc = Encoder::load(gliner25_rs::Precision::Fp16, 8)?;
    println!("encoder bench: backend {}, {} contexts from {}, mean {tokens:.1} tokens, length-bucketed {bucket}", enc.name(), picked.len(), states.display());
    println!("{:>6} {:>10} {:>12} {:>14} {:>12} {:>12} {:>8}", "batch", "calls", "ms/context", "us/padded-tok", "p50 ms/call", "p95 ms/call", "pad len");
    for b in batches.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).filter(|&b| b > 0) {
        for chunk in picked.chunks(b).take(warmup) {
            let _ = enc.encode_batch(chunk)?;
        }
        let mut per_call: Vec<f64> = Vec::new();
        let (mut ctx, mut pad, mut padded_tokens) = (0usize, 0usize, 0usize);
        for chunk in picked.chunks(b) {
            let t = Instant::now();
            let e = enc.encode_batch(chunk)?;
            per_call.push(t.elapsed().as_secs_f64() * 1000.0);
            ctx += chunk.len();
            pad += e.s;
            padded_tokens += e.b * e.s;
        }
        let total: f64 = per_call.iter().sum();
        per_call.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = per_call[per_call.len() / 2];
        let p95 = per_call[((per_call.len() as f64 * 0.95) as usize).min(per_call.len() - 1)];
        println!("{b:>6} {:>10} {:>12.2} {:>14.1} {:>12.1} {:>12.1} {:>8.1}", per_call.len(), total / ctx as f64, total * 1000.0 / padded_tokens.max(1) as f64, p50, p95, pad as f64 / per_call.len() as f64);
    }
    Ok(())
}
