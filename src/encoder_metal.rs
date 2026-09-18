//! GLiNER2.5's encoder (mDeBERTa-v3-base, fine-tuned by Fastino) run through the
//! DeBERTa-v2 port vendored from candle-transformers (src/debertav2.rs) on the Apple GPU (ticket gd-ngu7).
//! Weights come from the original `fastino/gliner2.5-multi-v1` safetensors: the
//! encoder sits under the `encoder.` prefix with Hugging Face's own tensor names,
//! so one prefix call is the whole name map. Only those tensors are read; the
//! boundary, relation and record heads in the same file never leave disk.

use crate::encoder::{Encoded, pad_batch};
use crate::head;
use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use crate::debertav2::{Config, DebertaV2Model};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub const REPO: &str = "fastino/gliner2.5-multi-v1";
const FILES: [&str; 3] = ["encoder_config/config.json", "tokenizer.json", "model.safetensors"];
const LOCAL_DIR: &str = "models/gliner2.5-multi-v1";

pub struct MetalEncoder {
    model: DebertaV2Model,
    tok: Tokenizer,
    dev: Device,
    pub dtype: DType,
    pub max_len: usize,
}

/// The three files, from `dir` (default `models/gliner2.5-multi-v1`) when complete, else the HF cache.
fn locate(dir: Option<&Path>) -> Result<[PathBuf; 3]> {
    let local = dir.map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from(LOCAL_DIR));
    if FILES.iter().all(|f| local.join(f).exists()) {
        return Ok(FILES.map(|f| local.join(f)));
    }
    let repo = hf_hub::api::sync::Api::new()?.model(REPO.to_string());
    let mut got = Vec::with_capacity(3);
    for f in FILES {
        got.push(repo.get(f).with_context(|| format!("downloading {f} from {REPO}"))?);
    }
    Ok([got.remove(0), got.remove(0), got.remove(0)])
}

impl MetalEncoder {
    /// Head device (Metal when available) and f32 weights. `GLINER2_DOOM_ENCODER_DTYPE=f16` opts
    /// into f16: it measured within 10% of f32 for speed and ~6x noisier against the fp32
    /// reference, so f32 stays the default, especially for building training caches.
    pub fn load(dir: Option<&Path>) -> Result<Self> {
        let dtype = match std::env::var("GLINER2_DOOM_ENCODER_DTYPE").as_deref() {
            Ok("f16") => DType::F16,
            _ => DType::F32,
        };
        Self::load_with(dir, &head::device(), dtype)
    }

    pub fn load_with(dir: Option<&Path>, dev: &Device, dtype: DType) -> Result<Self> {
        let [cfg_p, tok_p, w_p] = locate(dir)?;
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(&cfg_p).with_context(|| cfg_p.display().to_string())?)
            .with_context(|| format!("parsing {}", cfg_p.display()))?;
        let t0 = std::time::Instant::now();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&w_p], dtype, dev)? };
        let model = DebertaV2Model::load(vb.pp("encoder"), &cfg).context("building DeBERTa-v2 from the encoder.* tensors")?;
        let tok = Tokenizer::from_file(&tok_p).map_err(|e| anyhow!("tokenizer: {e}"))?;
        eprintln!(
            "metal encoder: {} layers, hidden {}, {dtype:?}, device {}, loaded in {:.1}s",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            if dev.is_metal() { "metal" } else { "cpu" },
            t0.elapsed().as_secs_f64()
        );
        Ok(Self { model, tok, dev: dev.clone(), dtype, max_len: 128 })
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<Encoded> {
        let (ids, am, b, s) = pad_batch(&self.tok, texts, self.max_len)?;
        let ids_t = Tensor::from_vec(ids, (b, s), &self.dev)?;
        let am_t = Tensor::from_vec(am.clone(), (b, s), &self.dev)?;
        let h = self.model.forward(&ids_t, None, Some(am_t))?; // [b, s, 768]
        let hidden = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Encoded { hidden, mask: am.into_iter().map(|x| x as f32).collect(), b, s })
    }
}
