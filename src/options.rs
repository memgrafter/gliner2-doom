//! The option table: what the model may choose between, and what each choice
//! presses. Data, not rules. Filtered to the buttons the engine exposes.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptionDef {
    pub name: String,
    pub buttons: Vec<String>,
    /// Engine tics the buttons are held. Sync play acts for exactly this long;
    /// real-time play holds for this long, then releases until the next verdict.
    pub hold: u32,
}

const DEFAULT_TABLE: &str = r#"[
  {"name": "attack",              "buttons": ["ATTACK"],               "hold": 4},
  {"name": "turn left",           "buttons": ["TURN_LEFT"],            "hold": 4},
  {"name": "turn right",          "buttons": ["TURN_RIGHT"],           "hold": 4},
  {"name": "turn left a little",  "buttons": ["TURN_LEFT"],            "hold": 1},
  {"name": "turn right a little", "buttons": ["TURN_RIGHT"],           "hold": 1},
  {"name": "move forward",        "buttons": ["MOVE_FORWARD"],         "hold": 4},
  {"name": "move backward",       "buttons": ["MOVE_BACKWARD"],        "hold": 4},
  {"name": "strafe left",         "buttons": ["MOVE_LEFT"],            "hold": 4},
  {"name": "strafe right",        "buttons": ["MOVE_RIGHT"],           "hold": 4},
  {"name": "use",                 "buttons": ["USE", "MOVE_FORWARD"],  "hold": 2}
]"#;

pub fn default_table() -> Vec<OptionDef> {
    serde_json::from_str(DEFAULT_TABLE).expect("built-in option table")
}

pub fn load_table(path: Option<&std::path::Path>) -> anyhow::Result<Vec<OptionDef>> {
    match path {
        Some(p) => Ok(serde_json::from_str(&std::fs::read_to_string(p)?)?),
        None => Ok(default_table()),
    }
}

/// Options whose every button the engine exposes.
pub fn available(table: &[OptionDef], buttons: &[String]) -> Vec<OptionDef> {
    table
        .iter()
        .filter(|o| o.buttons.iter().all(|b| buttons.iter().any(|x| x == b)))
        .cloned()
        .collect()
}

pub fn index_of(opts: &[OptionDef], name: &str) -> Option<usize> {
    opts.iter().position(|o| o.name == name)
}
