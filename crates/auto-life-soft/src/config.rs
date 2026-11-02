//! World settings for the soft engine. As in the lattice engine, physics lives in a JSON file
//! next to the shaders, every key becomes a WGSL constant, and an experiment can add a rule by
//! editing text files with no rebuild.

use auto_life_gpu::config::{Scope, has_word, read_save_head};
use serde_json::{Map, Value};
use std::path::PathBuf;

/// Brain inputs and outputs, in rows of four. See think.wgsl.
pub const IN_BLOCKS: u32 = 14;
pub const OUT_ROWS: u32 = 7;
pub const SUN_RANGE: f32 = 8.0;
/// Most cells a bin holds. Bins are 2 squares wide.
pub const BIN_CAP: u32 = 8;
/// Values per square that cells add to: wanted food, wanted meat, meat, heat, enzyme, four scents.
pub const DEP_STRIDE: u32 = 9;
pub const CELL_BYTES: u64 = 96;
pub const PLAN_BYTES: u64 = 80;
pub const SENSE_BYTES: u64 = 96;
pub const ACC_BYTES: u64 = 48;
pub const BODY_BYTES: u64 = 32;
pub const SKETCH_BYTES: u64 = 16;
pub const E_ONE: f32 = 65536.0;

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
        if !size.is_power_of_two() || !(32..=4096).contains(&size) {
            return Err(format!(
                "world size {size} must be a power of two from 32 to 4096"
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
        check_layout(&head, path)?;
        let int = |k: &str| head[k].as_u64().map(|v| v as u32);
        let size = int("size").ok_or_else(|| format!("{path} has no size"))?;
        let mut cfg = Config::new(size, int("seed").unwrap_or(1), shaders)?;
        cfg.hidden_rows = int("hidden_rows").unwrap_or(cfg.hidden_rows);
        cfg.shard_slots = int("shard_slots").unwrap_or_else(|| cfg.max_shard_slots());
        cfg.slots = int("slots").ok_or_else(|| format!("{path} has no slots"))?;
        if let Value::Object(saved) = &head["physics"] {
            for (k, v) in saved {
                cfg.physics.insert(k.clone(), v.clone());
            }
        }
        Ok(cfg)
    }

    pub fn blocks(&self) -> u32 {
        self.hidden_rows * (IN_BLOCKS + self.hidden_rows + OUT_ROWS)
    }

    pub fn max_shard_slots(&self) -> u32 {
        ((2u64 << 30) / (self.blocks() as u64 * 16)) as u32
    }

    /// Half a cell per square, up to what two genome buffers hold.
    pub fn default_slots(&self) -> u32 {
        let squares = self.size as u64 * self.size as u64;
        (squares / 2)
            .min(self.max_shard_slots() as u64 * 2)
            .max(1024) as u32
    }

    pub fn two_shards(&self) -> bool {
        self.slots > self.shard_slots
    }

    pub fn tile(&self) -> u32 {
        64.min(self.size / 4)
    }

    pub fn tiles(&self) -> u32 {
        self.size / self.tile()
    }

    pub fn bins(&self) -> u32 {
        self.size / 2
    }

    /// Words of the bitset that asks for slots. Rounded to whole chunks of 64.
    pub fn claim_words(&self) -> u32 {
        let tiles = self.tiles();
        (self.slots + tiles * tiles).div_ceil(32).next_multiple_of(64)
    }

    pub fn free_words(&self) -> u32 {
        self.slots.div_ceil(32).next_multiple_of(64)
    }

    pub fn bits_words(&self) -> u32 {
        self.claim_words() + self.free_words()
    }

    pub fn vram_bytes(&self) -> u64 {
        let squares = self.size as u64 * self.size as u64;
        let bins = self.bins() as u64 * self.bins() as u64;
        let slot = CELL_BYTES
            + PLAN_BYTES
            + SENSE_BYTES
            + ACC_BYTES
            + SKETCH_BYTES
            + 8 // mail
            + 16 // jobs
            + 4 // ring
            + self.hidden_rows as u64 * 16
            + self.blocks() as u64 * 16;
        squares * (4 + 4 * 16 + DEP_STRIDE as u64 * 4)
            + bins * BIN_CAP as u64 * (4 + BODY_BYTES)
            + self.slots as u64 * slot
            + (1 << 16)
    }

    /// Whether this world could be made on a device with this limit per buffer, and whether its
    /// settings make sense.
    pub fn check(&self, limit: u64) -> Result<(), String> {
        let squares = self.size as u64 * self.size as u64;
        let slots = self.slots as u64;
        let genome = self.blocks() as u64 * 16;
        let near = slots.min(self.shard_slots as u64);
        let largest = (squares * DEP_STRIDE as u64 * 4)
            .max(squares * 16)
            .max(slots * CELL_BYTES)
            .max(near * genome)
            .max((slots - near) * genome);
        if largest > limit {
            return Err(format!(
                "This GPU allows {:.2} GB per buffer, and this world needs {:.2} GB",
                limit as f64 / 1e9,
                largest as f64 / 1e9
            ));
        }
        if slots > (1 << 28) {
            return Err("at most 2^28 slots".into());
        }
        // Everything a cell can touch must be in the 3x3 bins around it.
        let reach = 2.0 * self.f("r_max") + self.f("gap");
        if reach > 2.0 {
            return Err(format!(
                "two discs of r_max {} with gap {} reach {reach} squares, past the 2 squares a bin spans",
                self.f("r_max"),
                self.f("gap")
            ));
        }
        if self.f("gap") <= 0.0 {
            return Err("gap must be above 0".into());
        }
        if self.f("brain_every") < 1.0 {
            return Err("brain_every must be at least 1".into());
        }
        // Jacobi with four neighbours is only stable below a quarter.
        for (k, v) in &self.physics {
            if k.ends_with("_flow") && v.as_f64().unwrap_or(0.0) >= 0.25 {
                return Err(format!("{k} must be below 0.25, or squares would overshoot"));
            }
        }
        let tile = self.f("batch_tile") as u32;
        if tile > 0 {
            if !tile.is_power_of_two() || tile < 32 || tile > self.size || !self.size.is_multiple_of(tile) {
                return Err(format!(
                    "batch_tile {tile} must be a power of two, at least 32, that divides the world size {}",
                    self.size
                ));
            }
            if (self.f("batch_wall") as u32) < 5 {
                return Err("batch_wall must be at least 5".into());
            }
        }
        Ok(())
    }

    /// When a setting takes effect, found by reading the shaders.
    pub fn scope(&self, key: &str) -> Scope {
        // The running world takes the sun through its scale, which `World::set_physics` works out.
        if key == "sun" {
            return Scope::Live;
        }
        let name = if key == "rock" { "ROCK_ON".to_string() } else { key.to_uppercase() };
        let used_in = |file: &str| {
            let text = self.source(file).unwrap_or_default();
            has_word(&text, &name) || has_word(&text, &format!("{name}_F"))
        };
        const RUNNING: [&str; 12] = [
            "common.wgsl",
            "think.wgsl",
            "charge.wgsl",
            "contact.wgsl",
            "emit.wgsl",
            "bite.wgsl",
            "settle.wgsl",
            "rain.wgsl",
            "alloc.wgsl",
            "genes.wgsl",
            "fields.wgsl",
            "gather.wgsl",
        ];
        if key == "brain_every" || key == "food_flow" || key == "meat_flow" {
            return Scope::Live;
        }
        if RUNNING.iter().any(|f| used_in(f)) {
            Scope::Live
        } else if used_in("terrain.wgsl") || used_in("noise.wgsl") {
            Scope::NewWorld
        } else {
            Scope::Unused
        }
    }

    pub fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        let Some(old) = self.physics.get(key) else {
            return Err(format!("unknown setting {key}. Add it to defaults.json first"));
        };
        let new = match old {
            Value::Bool(_) => Value::Bool(
                value.parse().map_err(|_| format!("{key} takes true or false"))?,
            ),
            _ => Value::from(
                value.parse::<f64>().map_err(|_| format!("{key} takes a number"))?,
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

    /// WGSL constants for the world's shape and every physics setting.
    pub fn header(&self) -> String {
        let log = self.size.trailing_zeros();
        let shift = 32 - log;
        let (hr, blocks) = (self.hidden_rows, self.blocks());
        let tile = self.tile();
        let bins = self.bins();
        let free_base = self.claim_words();
        let mut s = format!(
            "const W: u32 = {w}u;\nconst LOG_W: u32 = {log}u;\nconst MASK: u32 = {m}u;\nconst SLOTS: u32 = {slots}u;\nconst RING_MASK: u32 = {rm}u;\nconst HR: u32 = {hr}u;\nconst IN_BLOCKS: u32 = {IN_BLOCKS}u;\nconst OUT_ROWS: u32 = {OUT_ROWS}u;\nconst BLOCKS: u32 = {blocks}u;\nconst HH_BASE: u32 = {hh}u;\nconst HO_BASE: u32 = {ho}u;\nconst TILE: u32 = {tile}u;\nconst LOG_TILE: u32 = {lt}u;\nconst TILES: u32 = {tiles}u;\nconst SUN_RANGE: f32 = {SUN_RANGE:?};\nconst SHARD_BLOCKS: u32 = {shard_blocks}u;\nconst BINS: u32 = {bins}u;\nconst LOG_B: u32 = {lb}u;\nconst BIN_MASK: u32 = {bm}u;\nconst BIN_CAP: u32 = {BIN_CAP}u;\nconst DEP_STRIDE: u32 = {DEP_STRIDE}u;\nconst POS_SHIFT: u32 = {shift}u;\nconst POS_LOW: u32 = {low}u;\nconst POS_TO_SQ: f32 = {to_sq:?};\nconst SQ_TO_POS: f32 = {to_pos:?};\nconst FREE_BASE: u32 = {free_base}u;\nconst TOTAL_WORDS: u32 = {total}u;\nconst CHUNKS: u32 = {chunks}u;\nconst BRAIN_EVERY_U: u32 = {every}u;\nconst FOOD_REACHABLE: f32 = {fr:?};\nconst MEAT_REACHABLE: f32 = {mr:?};\n",
            w = self.size,
            m = self.size - 1,
            slots = self.slots,
            rm = self.slots.next_power_of_two() - 1,
            hh = hr * IN_BLOCKS,
            ho = hr * IN_BLOCKS + hr * hr,
            lt = tile.trailing_zeros(),
            tiles = self.tiles(),
            shard_blocks = self.shard_slots as u64 * blocks as u64,
            lb = bins.trailing_zeros(),
            bm = bins - 1,
            low = ((1u64 << shift) - 1) as u32,
            to_sq = 1.0 / (1u64 << shift) as f32,
            to_pos = (1u64 << shift) as f32,
            total = self.bits_words(),
            chunks = self.bits_words() / 64,
            every = (self.f("brain_every") as u32).max(1),
            fr = 1.0 - 4.0 * self.f("food_flow"),
            mr = 1.0 - 4.0 * self.f("meat_flow"),
        );
        for (k, v) in &self.physics {
            let name = k.to_uppercase();
            match v {
                Value::Bool(b) => {
                    let name = if name == "ROCK" { "ROCK_ON".to_string() } else { name };
                    s += &format!(
                        "const {name}: bool = {b};\nconst {name}_F: f32 = {};\n",
                        *b as u32 as f32
                    );
                }
                Value::Number(n) => {
                    let n = n.as_f64().unwrap_or(0.0) as f32;
                    s += &format!("const {name}: f32 = {n:?};\n");
                }
                _ => {}
            }
        }
        s
    }

    pub fn source(&self, file: &str) -> Result<String, String> {
        read_source(&self.shaders, file)
    }
}

/// What a save or dump records about its layout, so a world from another layout is refused
/// with a clear message instead of being read as garbage.
pub fn layout_json() -> Value {
    serde_json::json!({
        "engine": "soft", "in_blocks": IN_BLOCKS, "out_rows": OUT_ROWS, "cell_bytes": CELL_BYTES,
        "plan_bytes": PLAN_BYTES, "sense_bytes": SENSE_BYTES,
    })
}

pub fn check_layout(head: &Value, path: &str) -> Result<(), String> {
    if head["layout"] == layout_json() {
        return Ok(());
    }
    Err(format!(
        "{path} was saved with {}, and this engine has {}. Rerun it from its seed, or use the binary that made it",
        head["layout"],
        layout_json()
    ))
}

fn read_source(dir: &Option<PathBuf>, file: &str) -> Result<String, String> {
    if let Some(dir) = dir {
        let path = dir.join(file);
        // The land's noise is shared with the lattice engine, so a physics folder needn't copy it.
        if file != "noise.wgsl" || path.exists() {
            return std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()));
        }
    }
    Ok(match file {
        "defaults.json" => include_str!("../shaders/defaults.json"),
        "common.wgsl" => include_str!("../shaders/common.wgsl"),
        "terrain.wgsl" => include_str!("../shaders/terrain.wgsl"),
        "noise.wgsl" => include_str!("../../auto-life-gpu/shaders/noise.wgsl"),
        "think.wgsl" => include_str!("../shaders/think.wgsl"),
        "charge.wgsl" => include_str!("../shaders/charge.wgsl"),
        "gather.wgsl" => include_str!("../shaders/gather.wgsl"),
        "contact.wgsl" => include_str!("../shaders/contact.wgsl"),
        "emit.wgsl" => include_str!("../shaders/emit.wgsl"),
        "bite.wgsl" => include_str!("../shaders/bite.wgsl"),
        "settle.wgsl" => include_str!("../shaders/settle.wgsl"),
        "rain.wgsl" => include_str!("../shaders/rain.wgsl"),
        "alloc.wgsl" => include_str!("../shaders/alloc.wgsl"),
        "genes.wgsl" => include_str!("../shaders/genes.wgsl"),
        "fields.wgsl" => include_str!("../shaders/fields.wgsl"),
        "census.wgsl" => include_str!("../shaders/census.wgsl"),
        "view.wgsl" => include_str!("../shaders/view.wgsl"),
        other => return Err(format!("no shader {other}")),
    }
    .to_string())
}
