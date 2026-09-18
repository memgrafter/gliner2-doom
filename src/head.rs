//! Option-attention scoring head (jevlike's shape) on top of frozen encoder
//! states. Trained and run with candle; on the Apple GPU when `device()` finds one.

use anyhow::{Result, anyhow};
use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap, linear, linear_no_bias, ops};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::OnceLock;

pub const HIDDEN: usize = 768;
pub const D_ATT: usize = 256;

/// Where the head (and the Metal encoder) run. Chosen once per process from the CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePref {
    Cpu,
    Metal,
    Auto,
}

static PREF: OnceLock<DevicePref> = OnceLock::new();
static DEVICE: OnceLock<Device> = OnceLock::new();

pub fn set_device(p: DevicePref) {
    let _ = PREF.set(p);
}

pub fn parse_device(s: &str) -> Result<DevicePref> {
    match s {
        "cpu" => Ok(DevicePref::Cpu),
        "metal" => Ok(DevicePref::Metal),
        "auto" => Ok(DevicePref::Auto),
        _ => Err(anyhow!("--device must be cpu, metal or auto, got {s:?}")),
    }
}

/// The process-wide compute device: Metal when preferred and available, else CPU. Built once.
pub fn device() -> Device {
    DEVICE
        .get_or_init(|| {
            let pref = *PREF.get().unwrap_or(&DevicePref::Cpu);
            if pref == DevicePref::Cpu {
                return Device::Cpu;
            }
            match Device::new_metal(0) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("metal unavailable ({e}), using CPU");
                    Device::Cpu
                }
            }
        })
        .clone()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadMeta {
    pub temperature: f32,
    pub d_att: usize,
    pub options: Vec<String>,
    pub trained_rows: usize,
    pub notes: String,
}

pub struct Head {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,
    m1: Linear,
    m2: Linear,
}

impl Head {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wq: linear_no_bias(HIDDEN, D_ATT, vb.pp("wq"))?,
            wk: linear_no_bias(HIDDEN, D_ATT, vb.pp("wk"))?,
            wv: linear_no_bias(HIDDEN, D_ATT, vb.pp("wv"))?,
            wo: linear_no_bias(HIDDEN, D_ATT, vb.pp("wo"))?,
            m1: linear(3 * D_ATT, D_ATT, vb.pp("m1"))?,
            m2: linear(D_ATT, 1, vb.pp("m2"))?,
        })
    }

    /// h [B,S,768], mask [B,S] (1 = token), opts [B,J,768], omask [B,J] (1 = real option) -> logits [B,J]
    pub fn forward(&self, h: &Tensor, mask: &Tensor, opts: &Tensor, omask: &Tensor) -> Result<Tensor> {
        let q = self.wq.forward(opts)?; // [B,J,d]
        let k = self.wk.forward(h)?; // [B,S,d]
        let v = self.wv.forward(h)?; // [B,S,d]
        let scale = 1.0 / (D_ATT as f64).sqrt();
        let att = (q.matmul(&k.transpose(1, 2)?.contiguous()?)? * scale)?; // [B,J,S]
        let neg = ((mask.ones_like()? - mask)? * -1e9)?.unsqueeze(1)?; // [B,1,S]
        let att = att.broadcast_add(&neg)?;
        let att = ops::softmax(&att, D::Minus1)?;
        let c = att.matmul(&v)?; // [B,J,d]
        let oo = self.wo.forward(opts)?; // [B,J,d]
        let feat = Tensor::cat(&[&c, &oo, &(&c * &oo)?], 2)?; // [B,J,3d]
        let s = self.m2.forward(&self.m1.forward(&feat)?.relu()?)?.squeeze(2)?; // [B,J]
        let oneg = ((omask.ones_like()? - omask)? * -1e9)?;
        Ok(s.broadcast_add(&oneg)?)
    }
}

pub struct Loaded {
    pub head: Head,
    /// Kept alive: the head's tensors are views into it.
    #[allow(dead_code)]
    pub varmap: VarMap,
    pub meta: HeadMeta,
}

pub fn load(path: &Path) -> Result<Loaded> {
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device());
    let head = Head::new(vb)?;
    varmap.load(path)?;
    let meta_path = path.with_extension("json");
    let meta: HeadMeta = serde_json::from_str(&std::fs::read_to_string(&meta_path).map_err(|e| anyhow!("{}: {e}", meta_path.display()))?)?;
    Ok(Loaded { head, varmap, meta })
}

pub fn save(varmap: &VarMap, meta: &HeadMeta, path: &Path) -> Result<()> {
    varmap.save(path)?;
    std::fs::write(path.with_extension("json"), serde_json::to_string_pretty(meta)?)?;
    Ok(())
}

/// Probabilities over options for one context. `h` is [S,768] flattened, `opts` is [J,768] flattened.
pub fn predict(head: &Head, h: &[f32], s: usize, opts: &[f32], j: usize, temperature: f32) -> Result<Vec<f32>> {
    let dev = device();
    let ht = Tensor::from_slice(h, (1, s, HIDDEN), &dev)?;
    let mt = Tensor::ones((1, s), DType::F32, &dev)?;
    let ot = Tensor::from_slice(opts, (1, j, HIDDEN), &dev)?;
    let om = Tensor::ones((1, j), DType::F32, &dev)?;
    let logits = (head.forward(&ht, &mt, &ot, &om)? / temperature as f64)?;
    Ok(ops::softmax(&logits, D::Minus1)?.squeeze(0)?.to_vec1::<f32>()?)
}
