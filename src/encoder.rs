//! The frozen GLiNER2.5 encoder: text in, per-token hidden states out. Two
//! backends behind one type: ONNX Runtime on the CPU (the original path) and the
//! DeBERTa-v2 port in candle-transformers on the Apple GPU (`encoder_metal.rs`,
//! ticket gd-ngu7). Shared by training and play so both see the same numbers.

use crate::encoder_metal::MetalEncoder;
use anyhow::{Result, anyhow};
use gliner25_rs::{Precision, hub};
use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub const HIDDEN: usize = 768;

/// Which backend `Encoder::load` builds. Chosen once per process from the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Onnx,
    Metal,
}

static KIND: OnceLock<Kind> = OnceLock::new();

pub fn set_kind(k: Kind) {
    let _ = KIND.set(k);
}

pub fn kind() -> Kind {
    *KIND.get().unwrap_or(&Kind::Onnx)
}

pub fn parse_kind(s: &str) -> Result<Kind> {
    match s {
        "onnx" => Ok(Kind::Onnx),
        "metal" => Ok(Kind::Metal),
        _ => Err(anyhow!("--encoder must be onnx or metal, got {s:?}")),
    }
}

/// Cache file for this backend. The two produce slightly different numbers, so they never share one.
pub fn states_file_name() -> &'static str {
    match kind() {
        Kind::Onnx => "states.bin",
        Kind::Metal => "states.metal.bin",
    }
}

/// One batch of hidden states, padded to `s` tokens.
pub struct Encoded {
    pub hidden: Vec<f32>, // [b, s, HIDDEN]
    pub mask: Vec<f32>,   // [b, s]
    pub b: usize,
    pub s: usize,
}

/// Tokenize and right-pad a batch. Truncation keeps [SEP] (id 2) as the last token.
/// Returns (ids, attention mask, batch, padded length), both row-major [b, s].
pub fn pad_batch(tok: &Tokenizer, texts: &[&str], max_len: usize) -> Result<(Vec<u32>, Vec<u32>, usize, usize)> {
    let b = texts.len();
    let mut seqs = Vec::with_capacity(b);
    for t in texts {
        let enc = tok.encode(*t, true).map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > max_len {
            ids.truncate(max_len - 1);
            ids.push(2);
        }
        seqs.push(ids);
    }
    let s = seqs.iter().map(Vec::len).max().unwrap_or(1);
    let mut ids = vec![0u32; b * s];
    let mut am = vec![0u32; b * s];
    for (i, seq) in seqs.iter().enumerate() {
        for (j, &t) in seq.iter().enumerate() {
            ids[i * s + j] = t;
            am[i * s + j] = 1;
        }
    }
    Ok((ids, am, b, s))
}

/// The original path: the jugaadsrl ONNX export of the encoder through ONNX Runtime.
pub struct OnnxEncoder {
    session: Session,
    tok: Tokenizer,
    pub max_len: usize,
}

impl OnnxEncoder {
    pub fn load(precision: Precision, intra_threads: usize) -> Result<Self> {
        gliner25_rs::init("gliner2-doom");
        let (dir, prec): (PathBuf, Precision) = hub::download(hub::GLINER25_MULTI_V1, precision)?;
        let suffix = match prec { Precision::Fp32 => "fp32", Precision::Fp16 => "fp16", Precision::Fp16IoBinding => "fp16_iobinding" };
        // Grouped Hub layout puts each precision in its own folder (fp16_25/); a flat export has the files at the root.
        let sub = dir.join(format!("{suffix}_25"));
        let base = if sub.join(format!("encoder_{suffix}.onnx")).exists() { sub } else { dir.clone() };
        let model = base.join(format!("encoder_{suffix}.onnx"));
        let session = Session::builder()
            .map_err(|e| anyhow!("session builder: {e}"))?
            .with_intra_threads(intra_threads)
            .map_err(|e| anyhow!("intra threads: {e}"))?
            .commit_from_file(&model)
            .map_err(|e| anyhow!("loading {}: {e}", model.display()))?;
        let tok = Tokenizer::from_file(base.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer: {e}"))?;
        Ok(Self { session, tok, max_len: 128 })
    }

    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Encoded> {
        let (ids, am, b, s) = pad_batch(&self.tok, texts, self.max_len)?;
        let ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
        let am64: Vec<i64> = am.iter().map(|&x| i64::from(x)).collect();
        let out = self.session.run(vec![
            ("input_ids", Tensor::from_array((vec![b as i64, s as i64], ids))?.into_dyn()),
            ("attention_mask", Tensor::from_array((vec![b as i64, s as i64], am64))?.into_dyn()),
        ])?;
        let v = out.get("last_hidden_state").ok_or_else(|| anyhow!("encoder produced no last_hidden_state"))?;
        let hidden: Vec<f32> = match v.try_extract_tensor::<f32>() {
            Ok((_, d)) => d.to_vec(),
            Err(_) => v.try_extract_tensor::<f16>()?.1.iter().map(|x| x.to_f32()).collect(),
        };
        Ok(Encoded { hidden, mask: am.into_iter().map(|x| x as f32).collect(), b, s })
    }
}

enum Backend {
    Onnx(OnnxEncoder),
    Metal(MetalEncoder),
}

pub struct Encoder {
    backend: Backend,
    pool_cache: HashMap<String, Vec<f32>>,
}

impl Encoder {
    /// Builds the backend chosen by `set_kind`. `precision` and `intra_threads` apply to ONNX only.
    pub fn load(precision: Precision, intra_threads: usize) -> Result<Self> {
        let backend = match kind() {
            Kind::Onnx => Backend::Onnx(OnnxEncoder::load(precision, intra_threads)?),
            Kind::Metal => Backend::Metal(MetalEncoder::load(None)?),
        };
        Ok(Self { backend, pool_cache: HashMap::new() })
    }

    pub fn name(&self) -> &'static str {
        match self.backend {
            Backend::Onnx(_) => "onnx",
            Backend::Metal(_) => "metal",
        }
    }

    pub fn encode_batch(&mut self, texts: &[&str]) -> Result<Encoded> {
        match &mut self.backend {
            Backend::Onnx(e) => e.encode_batch(texts),
            Backend::Metal(e) => e.encode_batch(texts),
        }
    }

    /// Mean of the non-special token states for a short phrase (an option name). Cached.
    pub fn pool(&mut self, text: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.pool_cache.get(text) {
            return Ok(v.clone());
        }
        let e = self.encode_batch(&[text])?;
        let n = e.mask.iter().filter(|&&m| m > 0.0).count();
        let inner = n.saturating_sub(2).max(1); // drop [CLS] and [SEP]
        let mut v = vec![0f32; HIDDEN];
        for t in 1..=inner {
            for k in 0..HIDDEN {
                v[k] += e.hidden[t * HIDDEN + k];
            }
        }
        for x in &mut v {
            *x /= inner as f32;
        }
        self.pool_cache.insert(text.to_string(), v.clone());
        Ok(v)
    }
}
