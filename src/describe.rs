//! Snapshot -> one situation line. Keyword style on purpose: the probe showed
//! GLiNER2.5 answers "Side: RIGHT" style fields at 97-100% but confuses left
//! and right in natural sentences (30-70%). See ticket pro-2tst notes.

use crate::bridge::{Actor, Obs};

const MONSTERS: &[&str] = &[
    "Zombieman", "ShotgunGuy", "ChaingunGuy", "DoomImp", "Demon", "Spectre", "LostSoul", "Cacodemon",
    "HellKnight", "BaronOfHell", "Arachnotron", "PainElemental", "Revenant", "Mancubus", "Archvile",
    "SpiderMastermind", "Cyberdemon", "WolfensteinSS", "MarineChainsawVzd", "MarineChainsaw",
];

pub fn is_monster(name: &str) -> bool {
    if name.starts_with("Dead") || name.starts_with("Gibbed") {
        return false; // DeadMarine etc. are decorations
    }
    MONSTERS.iter().any(|m| name == *m) || name.contains("Marine") || name.contains("Vzd")
}

/// A monster label whose screen box is wider than tall is lying down: Doom
/// corpses keep their class name, so the sprite shape is the only tell.
pub fn is_alive(a: &Actor) -> bool {
    a.h >= a.w
}

const PICKUPS: &[&str] = &[
    "Medikit", "Stimpack", "HealthBonus", "ArmorBonus", "GreenArmor", "BlueArmor", "Soulsphere", "Megasphere",
    "Berserk", "Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox", "Cell", "CellPack", "Backpack",
    "Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle", "Chainsaw", "BFG9000",
    "BlueCard", "RedCard", "YellowCard", "BlueSkull", "RedSkull", "YellowSkull", "InvulnerabilitySphere", "BlurSphere",
    "RadSuit", "Allmap", "Infrared",
];

/// Something worth walking to. Blood, barrels, decorations and corpses are not.
pub fn is_pickup(name: &str) -> bool {
    PICKUPS.iter().any(|p| name == *p)
}

pub fn pretty_name(name: &str) -> String {
    match name {
        "MarineChainsawVzd" | "MarineChainsaw" => "chainsaw marine".into(),
        "Zombieman" => "zombie".into(),
        "ShotgunGuy" => "shotgun guy".into(),
        "ChaingunGuy" => "chaingunner".into(),
        "DoomImp" => "imp".into(),
        "HellKnight" => "hell knight".into(),
        "BaronOfHell" => "baron".into(),
        "LostSoul" => "lost soul".into(),
        "PainElemental" => "pain elemental".into(),
        other => other.trim_end_matches("Vzd").to_lowercase(),
    }
}

/// Hit window: one binary turn step is ~1.8 deg/tic and the crosshair window
/// measured in the openjev reproduction was about +-2.7 deg.
pub const CENTER_DEG: f32 = 3.0;
/// Inside this the target is one short turn pulse away; beyond it, a long one.
pub const SLIGHT_DEG: f32 = 12.0;

pub fn side(a: &Actor) -> &'static str {
    if a.rel_angle.abs() <= CENTER_DEG {
        "CENTER"
    } else if a.rel_angle > 0.0 {
        "LEFT"
    } else {
        "RIGHT"
    }
}

/// Fine/coarse aim as its own field: labels that share tokens with LEFT/RIGHT
/// ("SLIGHTLY LEFT") made the model confuse the sides, separate fields do not.
pub fn offset(a: &Actor) -> &'static str {
    if a.rel_angle.abs() <= CENTER_DEG {
        "NONE"
    } else if a.rel_angle.abs() <= SLIGHT_DEG {
        "SMALL"
    } else {
        "LARGE"
    }
}

pub fn distance(a: &Actor) -> &'static str {
    if a.dist < 160.0 {
        "VERY CLOSE"
    } else if a.dist < 450.0 {
        "CLOSE"
    } else {
        "FAR"
    }
}

/// Depth buffer is 8-bit; on map01 the middle band averages ~48 in an open
/// hall, ~16 one step from a wall, 1-3 when touching it.
pub const WALL_DEPTH: f32 = 12.0;

/// WALL when the nearest surface in the center third is within reach (10th
/// percentile depth, so a corner or a low step counts even if the band mean is high).
pub fn ahead(o: &Obs) -> &'static str {
    if o.depth.near < WALL_DEPTH { "WALL" } else { "OPEN" }
}

#[allow(dead_code)]
pub struct Situation {
    pub text: String,
    pub nearest: Option<Actor>,
    pub enemies: usize,
    pub items: Vec<String>,
}

pub fn describe(o: &Obs) -> Situation {
    let enemies: Vec<&Actor> = o.actors.iter().filter(|a| is_monster(&a.name) && is_alive(a)).collect();
    let items: Vec<String> = o.actors.iter().filter(|a| is_pickup(&a.name)).map(|a| pretty_name(&a.name)).collect();
    let nearest = enemies.first().cloned().cloned();
    let mut text = match &nearest {
        Some(a) => format!("Nearest enemy: {}. Side: {}. Offset: {}. Distance: {}.", pretty_name(&a.name), side(a), offset(a), distance(a)),
        None => "Nearest enemy: NONE. Side: NONE. Offset: NONE. Distance: NONE.".to_string(),
    };
    text.push_str(&format!(" Ahead: {}. Enemies in view: {}.", ahead(o), enemies.len()));
    if !items.is_empty() {
        text.push_str(&format!(" Items: {}.", items.join(", ")));
    }
    text.push_str(&format!(" Health: {}. Ammo: {}.", o.health, o.ammo));
    Situation { text, nearest, enemies: enemies.len(), items }
}
