//! The physics settings as controls. Generated from whatever keys the world has, so
//! a new key in defaults.json appears here with no changes.

use auto_life_gpu::Config;
use auto_life_gpu::config::Scope;
use serde_json::{Map, Value};

pub struct Var {
    pub key: String,
    pub group: &'static str,
    pub scope: Scope,
    pub default: Value,
    pub range: (f64, f64),
    pub log: bool,
    pub whole: bool,
    pub help: &'static str,
}

pub const GROUPS: [&str; 10] = [
    "Sun and ground",
    "Rain and genes",
    "Body and upkeep",
    "Eating and moving",
    "Fighting and helping",
    "Scent and noise",
    "What's switched on",
    "How fields spread",
    "Senses and mating",
    "Other",
];

fn group_of(key: &str) -> &'static str {
    match key {
        k if k.ends_with("_on") => "What's switched on",
        k if k.ends_with("_flow") || k.ends_with("_decay") || k.ends_with("_rot") => {
            "How fields spread"
        }
        "sun" | "season_ticks" | "ground_cap" | "rock" => "Sun and ground",
        k if k.starts_with("rain_") || k.starts_with("mut_") || k.starts_with("gene_") => {
            "Rain and genes"
        }
        "spend" | "commitment" | "womb" | "body" | "max_energy" | "hazard" | "upkeep"
        | "upkeep_e" | "rest_saving" => "Body and upkeep",
        "ground_rate" | "meat_rate" | "meat_yield" | "move_rate" | "enzyme_power"
        | "enzyme_cap" => "Eating and moving",
        k if k.starts_with("bite_")
            || k.starts_with("heat_")
            || k.starts_with("give_")
            || k.starts_with("share_")
            || k == "exposure" =>
        {
            "Fighting and helping"
        }
        "emit_power" | "noise_power" => "Scent and noise",
        "vision_steps" | "compat" => "Senses and mating",
        _ => "Other",
    }
}

fn help_of(key: &str) -> &'static str {
    match key {
        "sun" => "Mean sunlight. Scales how fast ground food grows.",
        "season_ticks" => "Ticks in one year of seasons. 0 means no seasons.",
        "ground_cap" => "The most ground food a square can hold.",
        "rock" => "Whether new worlds have rock ridges.",
        "rain_p" => "How often a random new cell lands.",
        "rain_energy" => "Energy a landing cell starts with.",
        "rain_burst_ticks" => {
            "For this many ticks, cells land much faster until 1% of the world is covered."
        }
        "gene_init" => "Spread of the weights in a landing cell's random genome.",
        "mut_rate" => "Chance each weight mutates at birth.",
        "mut_size" => "How big a mutation is.",
        "spend" => "The most energy a cell can spend per tick.",
        "commitment" => {
            "Spending a little on an action barely works. Bigger wastes more on dabbling."
        }
        "womb" => "Energy needed in the womb to give birth.",
        "body" => "Energy locked in a body, left as meat at death.",
        "max_energy" => "The most energy a cell can hold.",
        "hazard" => "Chance per tick of dying of old age, at any age.",
        "upkeep" => "Energy lost per tick just for being alive.",
        "upkeep_e" => "Extra upkeep per unit of stored energy.",
        "rest_saving" => "How much of its upkeep a fully resting cell saves.",
        "ground_rate" => "How fast spending turns ground food into energy.",
        "meat_rate" => "How fast spending eats meat.",
        "meat_yield" => "Share of meat eaten that becomes energy.",
        "move_rate" => "How much spending buys a move.",
        "bite_power" => "How much a bite takes.",
        "bite_yield" => "Share of a bite the biter keeps.",
        "exposure" => "Cells inside a clump are harder to bite.",
        "heat_power" => "Heat made per unit spent.",
        "heat_burn" => "Energy burned per unit of heat on a square.",
        "give_power" => "Energy passed on per unit spent on giving.",
        "give_yield" => "Share of a gift that arrives.",
        "share_rate" => "How fast two cells that both agree level out their energy.",
        "enzyme_power" => "Enzyme made per unit spent.",
        "enzyme_cap" => "The most enzyme can speed up eating.",
        "emit_power" => "Scent made per unit spent.",
        "noise_power" => "Noise made by each move and bite.",
        "vision_steps" => "How far cells can see, in steps that double the distance each time.",
        "compat" => "Mates need genomes this close. 0 means anyone can mate.",
        "report_tau" => "How close two genomes count as similar, for measuring only.",
        "ledger" => "Keep an exact energy count, for checking the physics. Slower.",
        _ => "",
    }
}

fn range_of(key: &str, default: f64) -> (f64, f64, bool, bool) {
    if key.ends_with("_ticks") {
        return (0.0, (default * 8.0).max(10000.0), false, true);
    }
    if key.ends_with("_steps") {
        return (0.0, 6.0, false, true);
    }
    if key.ends_with("_flow") {
        // Squares trade in pairs, and above one half they'd swap more than the difference.
        return (0.0, 0.5, false, false);
    }
    if key.ends_with("_decay") || key.ends_with("_yield") || key.ends_with("_saving") {
        return (0.0, 1.0, false, false);
    }
    if key.ends_with("_rot") || (key.ends_with("_rate") && default < 1.0) {
        return (0.0, 1.0, true, false);
    }
    if key == "hazard" || key == "rain_p" || key == "upkeep_e" {
        return (0.0, (default * 40.0).max(1e-3), true, false);
    }
    if default <= 0.0 {
        return (0.0, 1.0, false, false);
    }
    (0.0, default * 5.0, default < 0.1, false)
}

/// Builds the list from the defaults, plus anything else the world has.
pub fn all_vars(cfg: &Config, world: &Map<String, Value>) -> Vec<Var> {
    let mut keys: Vec<String> = cfg.physics.keys().cloned().collect();
    for k in world.keys() {
        if !keys.contains(k) {
            keys.push(k.clone());
        }
    }
    let mut out = vec![];
    for key in keys {
        let default = cfg
            .physics
            .get(&key)
            .or_else(|| world.get(&key))
            .cloned()
            .unwrap_or(Value::Null);
        let d = default.as_f64().unwrap_or(0.0);
        let (lo, hi, log, whole) = range_of(&key, d);
        out.push(Var {
            group: group_of(&key),
            scope: cfg.scope(&key),
            default,
            range: (lo, hi),
            log,
            whole,
            help: help_of(&key),
            key,
        });
    }
    out
}
