//! World settings. Physics lives in a JSON file next to the shaders, and every
//! key there becomes a WGSL constant, so an experiment can add a rule by editing
//! text files, without building anything.

use serde_json::{Map, Value};
use std::path::PathBuf;

/// Scent and sensor layout, shared with the shaders. Input and output counts are
/// in blocks of four, because the GPU does best with 4x4 weight blocks.
pub const IN_BLOCKS: u32 = 13;
pub const OUT_ROWS: u32 = 6;
pub const SUN_RANGE: f32 = 8.0;
/// Bytes per square: occ, seen, seen_next, terrain, and three vec4 fields.
pub const SQUARE_BYTES: u64 = 4 + 4 + 4 + 4 + 16 * 3;
pub const CELL_BYTES: u64 = 64;
pub const STAT_BYTES: u64 = 112;
pub const TILE_ROW_BYTES: u64 = 48;
/// A genome sketch per slot: four packed words.
pub const SKETCH_BYTES: u64 = 16;

/// Where each field sits in a `Cell` record, in 32-bit words. Matches `WGSL_STRUCTS` in world.rs.
/// Everything on the CPU side reads cells through these.
pub mod cell {
    pub const WORDS: usize = (super::CELL_BYTES / 4) as usize;
    pub const E: usize = 0;
    pub const WOMB: usize = 1;
    pub const AGE: usize = 2;
    pub const HIT: usize = 3;
    pub const X: usize = 4;
    pub const VOICE: usize = 8;
    pub const POS: usize = 10;
    pub const FLAGS: usize = 11;
    pub const CLADE: usize = 12;
    pub const GEN: usize = 13;
    pub const RGB: usize = 14;
    pub const STAMP: usize = 15;

    // The bits of `FLAGS`, as in common.wgsl.
    pub const DIR_MASK: u32 = 7;
    pub const F_MATE: u32 = 8;
    pub const F_SHARE: u32 = 16;
    pub const F_MOVED: u32 = 32;
    pub const F_BIT: u32 = 64;
    /// What last hurt it: 1 bitten, 2 burned, otherwise starved or old.
    pub const CAUSE_SHIFT: u32 = 8;
    pub const CAUSE_MASK: u32 = 3;
}

/// Where each measurement sits in a `Stat` record, in 32-bit words. Matches `WGSL_STRUCTS` in world.rs.
pub mod stat {
    pub const WORDS: usize = (super::STAT_BYTES / 4) as usize;
    pub const DIET: usize = 0;
    pub const BUDGET: usize = 4;
    pub const CHASE: usize = 16;
    pub const FOLLOW: usize = 20;
    pub const MISC: usize = 24;
}

/// Indexes of the shared counters, as in common.wgsl.
pub mod counter {
    pub const BIRTHS: usize = 3;
    pub const STARVED: usize = 5;
    pub const BITTEN: usize = 6;
    pub const BURNED: usize = 7;
    pub const OLD: usize = 8;
}

/// When a physics setting takes effect.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scope {
    /// The running world follows it from the next tick.
    Live,
    /// Only read while building a world, so it waits for the next new world.
    NewWorld,
    /// No shader reads it.
    Unused,
}

/// A saved world's header.
pub fn read_save_head(path: &str) -> Result<Value, String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).map_err(|e| format!("{path}: {e}"))?;
    let len = u64::from_le_bytes(len);
    if len > 1 << 24 {
        return Err(format!("{path} isn't a saved world"));
    }
    let mut head = vec![0u8; len as usize];
    f.read_exact(&mut head)
        .map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_slice(&head).map_err(|_| format!("{path} isn't a saved world"))
}

#[derive(Clone)]
pub struct Config {
    pub size: u32,
    pub slots: u32,
    pub seed: u32,
    /// Rows of four memory units.
    pub hidden_rows: u32,
    /// Slots that fit in one genome buffer. A buffer can't pass 2 GB here, so bigger worlds use two.
    pub shard_slots: u32,
    pub physics: Map<String, Value>,
    pub shaders: Option<PathBuf>,
}

impl Config {
    pub fn new(size: u32, seed: u32, shaders: Option<PathBuf>) -> Result<Config, String> {
        if !size.is_power_of_two() || size < 32 {
            return Err(format!(
                "world size {size} must be a power of two, at least 32"
            ));
        }
        let text = read_source(&shaders, "defaults.json")?;
        let physics: Value =
            serde_json::from_str(&text).map_err(|e| format!("defaults.json: {e}"))?;
        let Value::Object(physics) = physics else {
            return Err("defaults.json must be an object".into());
        };
        let mut cfg = Config {
            size,
            slots: 0,
            shard_slots: 0,
            seed,
            hidden_rows: 2,
            physics,
            shaders,
        };
        cfg.shard_slots = cfg.max_shard_slots();
        cfg.slots = cfg.default_slots();
        Ok(cfg)
    }

    /// The shape and physics a saved world was made with. Keys it doesn't have keep their defaults.
    pub fn from_save(path: &str, shaders: Option<PathBuf>) -> Result<Config, String> {
        let head = read_save_head(path)?;
        let head_len = {
            use std::io::Read;
            let mut len = [0u8; 8];
            std::fs::File::open(path)
                .and_then(|mut f| f.read_exact(&mut len))
                .map_err(|e| format!("{path}: {e}"))?;
            u64::from_le_bytes(len)
        };
        let int = |k: &str| head[k].as_u64().map(|v| v as u32);
        let size = int("size").ok_or_else(|| format!("{path} has no size"))?;
        let mut cfg = Config::new(size, int("seed").unwrap_or(1), shaders)?;
        cfg.hidden_rows = int("hidden_rows").unwrap_or(cfg.hidden_rows);
        cfg.shard_slots = int("shard_slots").unwrap_or_else(|| cfg.max_shard_slots());
        cfg.slots = int("slots").ok_or_else(|| format!("{path} has no slots"))?;
        // The buffers' sizes depend on the cell and genome layouts, which change between versions.
        let body = crate::world::saved_bytes(
            cfg.size as u64,
            cfg.slots as u64,
            cfg.hidden_rows as u64,
            cfg.shard_slots as u64,
        );
        let on_disk = std::fs::metadata(path)
            .map_err(|e| format!("{path}: {e}"))?
            .len();
        if on_disk != 8 + head_len + body {
            return Err(format!(
                "{path} was saved by a different version of the simulation, or is cut short"
            ));
        }
        if let Value::Object(saved) = &head["physics"] {
            for (k, v) in saved {
                cfg.physics.insert(k.clone(), v.clone());
            }
        }
        Ok(cfg)
    }

    /// When a setting takes effect, found by reading the shaders, so new settings are right on their own.
    pub fn scope(&self, key: &str) -> Scope {
        // The running world takes the sun through its scale, which `World::set_physics` works out.
        if key == "sun" {
            return Scope::Live;
        }
        let name = key.to_uppercase();
        let names: Vec<String> = if key == "rock" {
            vec!["ROCK_ON".into(), "ROCK_ON_F".into()]
        } else if let Some(stem) = key.strip_suffix("_decay") {
            vec![name.clone(), format!("{}_DECAY_KEEP", stem.to_uppercase())]
        } else {
            vec![name.clone(), format!("{name}_F")]
        };
        let used_in = |file: &str| {
            let text = self.source(file).unwrap_or_default();
            names.iter().any(|n| has_word(&text, n))
        };
        const RUNNING: [&str; 6] = [
            "common.wgsl",
            "act.wgsl",
            "births.wgsl",
            "rain.wgsl",
            "sweep.wgsl",
            "tick.wgsl",
        ];
        if RUNNING.iter().any(|f| used_in(f)) {
            Scope::Live
        } else if used_in("terrain.wgsl") {
            Scope::NewWorld
        } else {
            Scope::Unused
        }
    }

    pub fn blocks(&self) -> u32 {
        self.hidden_rows * (IN_BLOCKS + self.hidden_rows + OUT_ROWS)
    }

    pub fn max_shard_slots(&self) -> u32 {
        ((2u64 << 30) / (self.blocks() as u64 * 16)) as u32
    }

    pub fn default_slots(&self) -> u32 {
        let squares = self.size as u64 * self.size as u64;
        (squares / 2)
            .min(self.max_shard_slots() as u64 * 2)
            .max(1024) as u32
    }

    /// Whether the genomes need a second buffer.
    pub fn two_shards(&self) -> bool {
        self.slots > self.shard_slots
    }

    /// Why this world can't be made on a device with this limit per buffer, if it can't.
    pub fn check_buffers(&self, limit: u64) -> Result<(), String> {
        let squares = self.size as u64 * self.size as u64;
        let slots = self.slots as u64;
        let genome = self.blocks() as u64 * 16;
        let near = slots.min(self.shard_slots as u64);
        let largest = (squares * 16)
            .max(slots * STAT_BYTES)
            .max(slots * CELL_BYTES)
            .max(slots * self.hidden_rows as u64 * 16)
            .max(near * genome)
            .max((slots - near) * genome);
        if largest > limit {
            return Err(format!(
                "This GPU allows {:.2} GB per buffer, and this world needs {:.2} GB",
                limit as f64 / 1e9,
                largest as f64 / 1e9
            ));
        }
        Ok(())
    }

    /// The most cells whose genomes fit a device with this limit per buffer.
    pub fn slots_within(&self, limit: u64) -> u32 {
        let genome = self.blocks() as u64 * 16;
        let one_buffer = limit.min(2 << 30) / genome;
        (one_buffer * if limit >= 2 << 30 { 2 } else { 1 }).min(u32::MAX as u64) as u32
    }

    pub fn vram_bytes(&self) -> u64 {
        let squares = self.size as u64 * self.size as u64;
        let slot =
            CELL_BYTES + STAT_BYTES + 16 + self.hidden_rows as u64 * 16 + self.blocks() as u64 * 16;
        squares * SQUARE_BYTES + self.slots as u64 * (slot + 16 + 4) + (1 << 16)
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let Some(old) = self.physics.get(key) else {
            return Err(format!(
                "unknown setting {key}. Add it to defaults.json first"
            ));
        };
        let new = match old {
            Value::Bool(_) => Value::Bool(
                value
                    .parse()
                    .map_err(|_| format!("{key} takes true or false"))?,
            ),
            _ => Value::from(
                value
                    .parse::<f64>()
                    .map_err(|_| format!("{key} takes a number"))?,
            ),
        };
        self.physics.insert(key.to_string(), new);
        Ok(())
    }

    pub fn f(&self, key: &str) -> f32 {
        self.physics
            .get(key)
            .and_then(Value::as_f64)
            .unwrap_or_else(|| panic!("no number {key}")) as f32
    }

    pub fn b(&self, key: &str) -> bool {
        self.physics
            .get(key)
            .and_then(Value::as_bool)
            .unwrap_or_else(|| panic!("no switch {key}"))
    }

    pub fn tile(&self) -> u32 {
        64.min(self.size / 4)
    }

    /// WGSL constants for the world's shape and every physics setting.
    pub fn header(&self) -> String {
        let log = self.size.trailing_zeros();
        let (hr, blocks) = (self.hidden_rows, self.blocks());
        let tile = self.tile();
        let mut s = format!(
            "const W: u32 = {w}u;\nconst LOG_W: u32 = {log}u;\nconst MASK: u32 = {m}u;\nconst SLOTS: u32 = {slots}u;\nconst RING_MASK: u32 = {rm}u;\nconst HR: u32 = {hr}u;\nconst IN_BLOCKS: u32 = {IN_BLOCKS}u;\nconst OUT_ROWS: u32 = {OUT_ROWS}u;\nconst BLOCKS: u32 = {blocks}u;\nconst HH_BASE: u32 = {hh}u;\nconst HO_BASE: u32 = {ho}u;\nconst TILE: u32 = {tile}u;\nconst TILES: u32 = {tiles}u;\nconst SUN_RANGE: f32 = {SUN_RANGE:?};\nconst SHARD_BLOCKS: u32 = {shard_blocks}u;\n",
            w = self.size,
            m = self.size - 1,
            slots = self.slots,
            rm = self.slots.next_power_of_two() - 1,
            hh = hr * IN_BLOCKS,
            ho = hr * IN_BLOCKS + hr * hr,
            tiles = self.size / tile,
            shard_blocks = self.shard_slots as u64 * blocks as u64,
        );
        for (k, v) in &self.physics {
            let name = k.to_uppercase();
            match v {
                Value::Bool(b) => {
                    // ROCK is a marker in the shaders, so the switch has another name.
                    let name = if name == "ROCK" {
                        "ROCK_ON".to_string()
                    } else {
                        name
                    };
                    s += &format!(
                        "const {name}: bool = {b};\nconst {name}_F: f32 = {};\n",
                        *b as u32 as f32
                    );
                }
                Value::Number(n) => {
                    let n = n.as_f64().unwrap_or(0.0) as f32;
                    s += &format!("const {name}: f32 = {n:?};\n");
                    if let Some(stem) = k.strip_suffix("_decay") {
                        // Fields update twice a tick, so each sweep keeps the square root.
                        let keep = (1.0 - n.clamp(0.0, 1.0)).sqrt();
                        s += &format!(
                            "const {}_DECAY_KEEP: f32 = {keep:?};\n",
                            stem.to_uppercase()
                        );
                    }
                }
                _ => {}
            }
        }
        s
    }

    pub fn source(&self, file: &str) -> Result<String, String> {
        read_source(&self.shaders, file)
    }

    /// The think step, spelled out with no arrays, because arrays in a function
    /// go to scratch memory and cost about ten times as much.
    pub fn brain_wgsl(&self) -> (String, String) {
        let hr = self.hidden_rows;
        let mut s = String::from("let base = s * BLOCKS;\n    let ms = s * HR;\n");
        for r in 0..hr {
            s += &format!("    let m{r} = mem[ms + {r}u] * MEMORY_ON_F;\n");
        }
        for r in 0..hr {
            s += &format!("    let h{r} = squash(");
            for k in 0..IN_BLOCKS {
                s += &format!("bt(base + {}u, i{k}) + ", r * IN_BLOCKS + k);
            }
            let terms: Vec<String> = (0..hr)
                .map(|j| format!("bt(base + HH_BASE + {}u, m{j})", r * hr + j))
                .collect();
            s += &terms.join(" + ");
            s += ");\n";
        }
        for r in 0..hr {
            s += &format!("    mem[ms + {r}u] = h{r};\n");
        }
        for o in 0..OUT_ROWS {
            let terms: Vec<String> = (0..hr)
                .map(|j| format!("bt(base + HO_BASE + {}u, h{j})", o * hr + j))
                .collect();
            s += &format!("    let o{o} = {};\n", terms.join(" + "));
        }
        let mut copy = String::new();
        for r in 0..hr {
            copy += &format!("mem[child * HR + {r}u] = h{r};\n                ");
        }
        (s, copy)
    }
}

fn has_word(text: &str, word: &str) -> bool {
    let is_name = |c: char| c.is_ascii_alphanumeric() || c == '_';
    text.match_indices(word).any(|(at, _)| {
        !text[..at].chars().next_back().is_some_and(is_name)
            && !text[at + word.len()..].chars().next().is_some_and(is_name)
    })
}

fn read_source(dir: &Option<PathBuf>, file: &str) -> Result<String, String> {
    if let Some(dir) = dir {
        let path = dir.join(file);
        return std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()));
    }
    Ok(match file {
        "defaults.json" => include_str!("../shaders/defaults.json"),
        "common.wgsl" => include_str!("../shaders/common.wgsl"),
        "act.wgsl" => include_str!("../shaders/act.wgsl"),
        "births.wgsl" => include_str!("../shaders/births.wgsl"),
        "bake.wgsl" => include_str!("../shaders/bake.wgsl"),
        "census.wgsl" => include_str!("../shaders/census.wgsl"),
        "rain.wgsl" => include_str!("../shaders/rain.wgsl"),
        "sweep.wgsl" => include_str!("../shaders/sweep.wgsl"),
        "terrain.wgsl" => include_str!("../shaders/terrain.wgsl"),
        "tick.wgsl" => include_str!("../shaders/tick.wgsl"),
        "view.wgsl" => include_str!("../shaders/view.wgsl"),
        other => return Err(format!("no shader {other}")),
    }
    .to_string())
}
