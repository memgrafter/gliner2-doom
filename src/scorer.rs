//! Thin wrapper over gliner25-rs: one single-label classification task per call,
//! returning every label's probability. Classification-only requests skip the
//! boundary head, so a decision is one encoder pass plus the classifier MLP.

use anyhow::Result;
use gliner25_rs::{BoundaryConfig, BoundaryEngine, ExecutionMode, Precision, SchemaTask, hub};
use std::path::Path;
use std::time::Instant;

pub struct Scorer {
    engine: BoundaryEngine,
    pub load_secs: f64,
}

#[derive(Debug, Clone)]
pub struct Scored {
    pub label: String,
    pub prob: f32,
}

impl Scorer {
    /// `models_dir` is used if it holds an export; otherwise the fp16 export of
    /// gliner2.5-multi-v1 is fetched into the HF cache. `precision` overrides
    /// the crate's choice (on CPU the crate would fetch fp32; fp16 is half the
    /// disk and was faster on CPU in the crate's own benchmarks).
    pub fn new(models_dir: &Path, precision: Precision) -> Result<Self> {
        gliner25_rs::init("gliner2-doom");
        let t0 = Instant::now();
        // Standard, not Auto: on macOS the crate resolves Auto to the bound
        // (IoBinding) path, and that path binds routed_gather's [1,K,768]
        // output straight into classifier.onnx, which expects [K,768]
        // ("Invalid rank for input: choice_states"). The standard path
        // reshapes on the way through and is the right mode on CPU anyway.
        let cfg = BoundaryConfig::new(models_dir)
            .or_download(hub::GLINER25_MULTI_V1)
            .with_precision(precision)
            .with_execution(ExecutionMode::Standard);
        let engine = BoundaryEngine::new(cfg)?;
        eprintln!("engine: execution={:?} device={}", engine.execution(),
            std::env::var("GLINER2_DEVICE").unwrap_or_else(|_| "auto".into()));
        Ok(Self { engine, load_secs: t0.elapsed().as_secs_f64() })
    }

    /// Softmax over `labels` for `text`, sorted by probability, descending.
    pub fn classify(&mut self, task: &str, text: &str, labels: &[String]) -> Result<Vec<Scored>> {
        let tasks = vec![SchemaTask::classification(task, labels.to_vec())];
        let out = self.engine.extract(text, &tasks)?;
        let mut rows: Vec<Scored> = out
            .classifications
            .into_iter()
            .filter(|c| c.task == task)
            .map(|c| Scored { label: c.label, prob: c.score })
            .collect();
        rows.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
        Ok(rows)
    }
}

/// Several single-label tasks scored in ONE encoder pass. Returns, per task,
/// the labels sorted by probability. This is the schema shape for play:
/// small softmaxes per question (enemy side, distance, ammo...) instead of one
/// softmax over a dozen unrelated statements.
pub fn classify_multi(sc: &mut Scorer, text: &str, tasks: &[(String, Vec<String>)]) -> Result<Vec<(String, Vec<Scored>)>> {
    let schema: Vec<SchemaTask> = tasks
        .iter()
        .map(|(name, labels)| SchemaTask::classification(name.clone(), labels.clone()))
        .collect();
    let out = sc.engine.extract(text, &schema)?;
    let mut res = Vec::with_capacity(tasks.len());
    for (name, _) in tasks {
        let mut rows: Vec<Scored> = out
            .classifications
            .iter()
            .filter(|c| &c.task == name)
            .map(|c| Scored { label: c.label.clone(), prob: c.score })
            .collect();
        rows.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
        res.push((name.clone(), rows));
    }
    Ok(res)
}

pub fn parse_precision(s: &str) -> Result<Precision> {
    Ok(match s {
        "fp32" => Precision::Fp32,
        "fp16" => Precision::Fp16,
        "fp16_iobinding" => Precision::Fp16IoBinding,
        other => anyhow::bail!("unknown precision {other}; use fp32 | fp16 | fp16_iobinding"),
    })
}
