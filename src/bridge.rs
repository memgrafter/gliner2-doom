//! Typed client for `bridge/vizdoom_bridge.py` (JSON lines over stdio).
//! The Python side owns the engine; this side owns the clock and the policy.

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

#[derive(Debug, Clone, Serialize, Default)]
pub struct InitOpts {
    pub scenario: String,
    pub mode: String, // "sync" | "async"
    pub width: u32,
    pub height: u32,
    pub visible: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wad: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub buttons: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_tics: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ticrate: Option<u32>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)] // protocol fields; not every consumer reads every one
pub struct Actor {
    pub name: String,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub dist: f32,
    /// Degrees from facing to the actor: positive = left, negative = right.
    pub rel_angle: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[allow(dead_code)]
pub struct Depth {
    pub left: f32,
    pub center: f32,
    pub right: f32,
    /// 10th percentile depth in the center third (middle+lower bands): the nearest thing ahead.
    #[serde(default)]
    pub near: f32,
    #[serde(default)]
    pub near_left: f32,
    #[serde(default)]
    pub near_right: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Obs {
    pub ok: bool,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub finished: bool,
    #[serde(default)]
    pub tic: u32,
    #[serde(default)]
    pub reward: f32,
    #[serde(default)]
    pub total_reward: f32,
    #[serde(default)]
    pub health: i32,
    #[serde(default)]
    pub ammo: i32,
    #[serde(default)]
    pub kills: i32,
    /// ITEMCOUNT: pickups collected this episode.
    #[serde(default)]
    pub items: i32,
    #[serde(default)]
    pub pos: Vec<f32>,
    #[serde(default)]
    pub actors: Vec<Actor>,
    #[serde(default)]
    pub depth: Depth,
    /// Displacement over the last 15 tics, filled in by the play loop (-1 = unknown).
    #[serde(default = "neg_one")]
    pub moved: f32,
    #[serde(default)]
    pub frame: Option<String>,
    #[serde(default)]
    pub frame_shape: Option<Vec<u32>>,
}

fn neg_one() -> f32 { -1.0 }

#[derive(Debug, Deserialize)]
struct InitResp {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    buttons: Vec<String>,
}

pub struct Bridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    pub buttons: Vec<String>,
}

fn python() -> String {
    std::env::var("GLINER2_DOOM_PYTHON").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/virtualenvs/gliner2-doom-vizdoom/bin/python")
    })
}

fn script() -> std::path::PathBuf {
    // Next to the Cargo manifest at dev time; next to the binary when installed.
    let dev = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("bridge/vizdoom_bridge.py");
    if dev.exists() {
        return dev;
    }
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("vizdoom_bridge.py")))
        .unwrap_or(dev)
}

impl Bridge {
    pub fn spawn(opts: &InitOpts) -> Result<Self> {
        let mut child = Command::new(python())
            .arg(script())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawning {} (set GLINER2_DOOM_PYTHON)", python()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?);
        let mut b = Self { child, stdin, stdout, buttons: vec![] };
        let mut req = serde_json::to_value(opts)?;
        req["cmd"] = "init".into();
        let resp: InitResp = b.call(&req)?;
        if !resp.ok {
            return Err(anyhow!("bridge init failed: {}", resp.error.unwrap_or_default()));
        }
        b.buttons = resp.buttons;
        Ok(b)
    }

    fn call<T: for<'de> Deserialize<'de>>(&mut self, req: &serde_json::Value) -> Result<T> {
        let line = serde_json::to_string(req)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut resp = String::new();
        let n = self.stdout.read_line(&mut resp)?;
        if n == 0 {
            return Err(anyhow!("bridge closed its stdout"));
        }
        Ok(serde_json::from_str(&resp).with_context(|| format!("bad bridge reply: {resp}"))?)
    }

    fn obs_call(&mut self, req: serde_json::Value) -> Result<Obs> {
        let o: Obs = self.call(&req)?;
        if !o.ok {
            return Err(anyhow!("bridge error: {}", o.error.unwrap_or_default()));
        }
        Ok(o)
    }

    pub fn reset(&mut self) -> Result<Obs> {
        self.obs_call(serde_json::json!({"cmd": "reset"}))
    }

    /// Press `buttons` for `tics` engine tics and return the observation after.
    pub fn act(&mut self, buttons: &[&str], tics: u32) -> Result<Obs> {
        self.obs_call(serde_json::json!({"cmd": "act", "buttons": buttons, "tics": tics}))
    }

    pub fn obs(&mut self, frame: bool) -> Result<Obs> {
        self.obs_call(serde_json::json!({"cmd": "obs", "frame": frame}))
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.stdin.write_all(b"{\"cmd\":\"close\"}\n");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}
