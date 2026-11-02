//! World settings. Physics lives in a JSON file next to the shaders, and every
//! key there becomes a WGSL constant, so an experiment can add a rule by editing
//! text files, without building anything.

use serde_json::{Map, Value};
use std::path::PathBuf;

/// Scent and sensor layout, shared with the shaders. Input and output counts are
/// in blocks of four, because the GPU does best with 4x4 weight blocks.
///
/// Inputs 0-11 are the senses every cell has. 12 is waste, 13 tracks, 14 practice and how a
/// place feels, 15 coats, 16 mineral, 17 bites seen. Outputs 0-5 are as act.wgsl describes,
/// and 6 holds the spare budget actions x0 to x3. Weights exist for all of them, but
/// [`Config::input_live`] and [`Config::output_live`] leave out what no rule that is on uses.
pub const IN_BLOCKS: u32 = 18;
pub const OUT_ROWS: u32 = 7;
pub const SUN_RANGE: f32 = 8.0;
/// Bytes per square: occ, seen, seen_next, terrain, and four vec4 fields.
pub const SQUARE_BYTES: u64 = 4 + 4 + 4 + 4 + 16 * 4;
pub const CELL_BYTES: u64 = 96;
pub const CELL_WORDS: usize = CELL_BYTES as usize / 4;
pub const STAT_BYTES: u64 = 144;
pub const STAT_WORDS: usize = STAT_BYTES as usize / 4;
pub const TILE_ROW_BYTES: u64 = 48;

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

    /// Whether a brain input block can be anything but zero in this world. A block
    /// that can't is not computed at all, so a rule that's off costs only its weights.
    /// A rule that adds an input must add its switch here.
    pub fn input_live(&self, block: u32) -> bool {
        match block {
            12 => self.b("waste_on"),
            13 => self.b("roads_on"),
            14 => {
                (self.b("skill_on") && self.b("skill_sense"))
                    || (self.b("biomes_on") && self.b("biome_feel"))
            }
            15 => self.b("coat_on") && self.b("coat_sense"),
            16 => self.b("mineral_on"),
            17 => self.b("gang_on") && self.b("gang_sense"),
            _ => true,
        }
    }

    /// Whether a brain output row is read by anything in this world.
    pub fn output_live(&self, row: u32) -> bool {
        match row {
            6 => ["x0_on", "x1_on", "x2_on", "x3_on", "mineral_on", "birth_choice"]
                .iter()
                .any(|k| self.b(k)),
            _ => true,
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
            for k in (0..IN_BLOCKS).filter(|k| self.input_live(*k)) {
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
            if !self.output_live(o) {
                s += &format!("    let o{o} = vec4f(0.0);\n");
                continue;
            }
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

/// What a save or dump records about its layout, so a world from another layout is
/// refused with a clear message instead of being read as garbage.
pub fn layout_json() -> Value {
    serde_json::json!({
        "in_blocks": IN_BLOCKS, "out_rows": OUT_ROWS, "cell_bytes": CELL_BYTES, "stat_bytes": STAT_BYTES,
    })
}

/// Checks a save's or dump's header against this build's layout.
pub fn check_layout(head: &Value, path: &str) -> Result<(), String> {
    if head["layout"] == layout_json() {
        return Ok(());
    }
    let saved = if head["layout"].is_null() {
        "an older layout, 13 input blocks and 6 output rows with 64-byte cells".to_string()
    } else {
        head["layout"].to_string()
    };
    Err(format!(
        "{path} was saved with {saved}, and this engine has {IN_BLOCKS} input blocks and {OUT_ROWS} output rows with {CELL_BYTES}-byte cells. Old worlds can't be loaded. Rerun it from its seed, or use the binary that made it"
    ))
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
