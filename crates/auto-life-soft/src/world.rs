//! The soft world on the GPU: buffers, passes and a scheduler that never holds the card for long,
//! because the same card draws Erik's desktop.
//!
//! A tick reads the old state and each cell writes only its own record, apart from integer atomics
//! whose order can't change a total. Slots are handed out in a fixed order. So a run on one driver
//! repeats exactly, including when a bin overflows or the slots run out.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use auto_core::{Gpu, GpuOptions};
use auto_life_gpu::config::brain_wgsl_for;
use auto_life_gpu::world::Pacing;
use bytemuck::{Pod, Zeroable};

use crate::config::{
    ACC_BYTES, BIN_CAP, BODY_BYTES, CELL_BYTES, Config, DEP_STRIDE, E_ONE, IN_BLOCKS, OUT_ROWS,
    PLAN_BYTES, SENSE_BYTES, SKETCH_BYTES, SUN_RANGE, check_layout, layout_json,
};

const RING_ENTRIES: usize = 4096;

const WGSL_STRUCTS: &str = r#"
struct Cell {
    x: u32, y: u32, r: f32, hx: f32,
    hy: f32, stick: f32, push: f32, gv: u32,
    e: u32, womb: u32, unit: f32, flags: u32,
    age: u32, felt: f32, shed: u32, bitten: f32,
    uid: u32, parent: u32, gen: u32, clade: u32,
    mu: f32, born: u32, contacts: u32, pad: u32,
}
struct Plan { s0: vec4f, s1: vec4f, s2: vec4f, mix: vec4f, split: vec4f }
struct Sense {
    count: vec4f, gap: vec4f, v0: vec4f, v1: vec4f, mt: vec4f,
    pressure: f32, contacts: f32, pad0: f32, pad1: f32,
}
struct Acc {
    fx: i32, fy: i32, give_out: u32, target_: u32,
    wish: u32, bite_got: u32, fee: f32, contacts: u32,
    wish_food: u32, wish_meat: u32, pad0: u32, pad1: u32,
}
struct BinBody { slot: u32, x: u32, y: u32, r: f32, hd: u32, stick: f32, push: f32, gv: u32 }
struct Frame {
    tick: u32, row0: u32, seed: u32, phase: u32,
    sun_k: f32, drift_x: f32, drift_y: f32, pad: u32,
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Frame {
    pub tick: u32,
    pub row0: u32,
    pub seed: u32,
    pub phase: u32,
    pub sun_k: f32,
    pub drift_x: f32,
    pub drift_y: f32,
    pub pad: u32,
}
const FRAME_BYTES: usize = std::mem::size_of::<Frame>();

/// A cell's record, as the shaders lay it out.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Cell {
    pub x: u32,
    pub y: u32,
    pub r: f32,
    pub hx: f32,
    pub hy: f32,
    pub stick: f32,
    pub push: f32,
    pub gv: u32,
    pub e: u32,
    pub womb: u32,
    pub unit: f32,
    pub flags: u32,
    pub age: u32,
    pub felt: f32,
    pub shed: u32,
    pub bitten: f32,
    pub uid: u32,
    pub parent: u32,
    pub generation: u32,
    pub clade: u32,
    pub mu: f32,
    pub born: u32,
    pub contacts: u32,
    pub pad: u32,
}

/// How a cell divides its budget until it next thinks.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Plan {
    pub s0: [f32; 4],
    pub s1: [f32; 4],
    pub s2: [f32; 4],
    pub mix: [f32; 4],
    pub split: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Sense {
    pub count: [f32; 4],
    pub gap: [f32; 4],
    pub v0: [f32; 4],
    pub v1: [f32; 4],
    pub mt: [f32; 4],
    pub pressure: f32,
    pub contacts: f32,
    pub pad: [f32; 2],
}

/// What the contact pass adds up for a cell, read back before settle clears it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Acc {
    pub fx: i32,
    pub fy: i32,
    pub give_out: u32,
    pub target: u32,
    pub wish: u32,
    pub bite_got: u32,
    pub fee: f32,
    pub contacts: u32,
    pub wish_food: u32,
    pub wish_meat: u32,
    pub pad: [u32; 2],
}

pub const F_ALIVE: u32 = 1;

/// Energy flows, in the order of common.wgsl.
pub const ROUTES: [&str; 18] = [
    "sun", "rain", "upkeep", "spend", "stick", "harm", "eaten", "eat_got", "meat_eaten", "meat_got",
    "bite_paid", "bite_got", "give_out", "give_in", "cap", "rot_lost", "death", "grow",
];
const COUNTERS: [&str; 9] = [
    "births", "rained", "starved", "bitten", "burned", "old", "overflow", "no_slot", "contacts",
];
const CTR_WORDS: u64 = ROUTES.len() as u64 * 2 + COUNTERS.len() as u64;
const L_HEAD: u64 = 0;
const L_TAIL: u64 = 1;
const L_ALIVE: u64 = 2;
const L_OPEN: u64 = 3;
const L_NEXT_UID: u64 = 7;

/// Everything that has happened since the world began.
#[derive(Clone, Default, Debug)]
pub struct Tally {
    /// Each flow, in energy, as counted. See [`Tally::sample`].
    pub flows: Vec<u64>,
    pub births: u64,
    pub rained: u64,
    pub starved: u64,
    pub bitten: u64,
    pub burned: u64,
    pub old: u64,
    pub overflow: u64,
    pub no_slot: u64,
    /// With the ledger off, one flow in this many is counted, so multiply by this to estimate.
    pub sample: u64,
}

impl Tally {
    pub fn flow(&self, name: &str) -> u64 {
        let k = ROUTES.iter().position(|r| *r == name).expect("route");
        self.flows[k]
    }

    /// All energy that has left the world, in units.
    pub fn lost(&self) -> i128 {
        let f = |n: &str| self.flow(n) as i128;
        f("upkeep")
            + f("spend")
            + f("stick")
            + f("harm")
            + (f("eaten") - f("eat_got"))
            + (f("meat_eaten") - f("meat_got"))
            + (f("bite_paid") - f("bite_got"))
            + (f("give_out") - f("give_in"))
            + f("cap")
            + f("rot_lost")
    }
}

/// Where the energy is right now, exactly, in units.
#[derive(Clone, Copy, Default, Debug)]
pub struct Stock {
    pub food: u64,
    pub meat: u64,
    pub heat: u64,
    pub enzyme: u64,
    pub energy: u64,
    pub womb: u64,
    pub alive: u64,
    pub body_each: u64,
}

impl Stock {
    /// Everything that counts as energy, in units. Heat, enzyme and scents don't: the energy
    /// behind them left the ledger when it was spent.
    pub fn total(&self) -> i128 {
        (self.food + self.meat + self.energy + self.womb + self.alive * self.body_each) as i128
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum Access {
    R,
    RW,
}
use Access::{R, RW};

/// A binding: its name in the shader, the element type there, and how it's used.
type Bind = (&'static str, &'static str, Access);

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    /// One bind group for each way round the two field buffers can be.
    groups: [wgpu::BindGroup; 2],
}

#[derive(Clone, Copy)]
enum Op {
    Dispatch {
        pipe: usize,
        flip: usize,
        frame: Frame,
        groups: [u32; 3],
    },
    Indirect {
        pipe: usize,
        flip: usize,
        frame: Frame,
    },
    ClearBits,
}

const P_THINK: usize = 0;
const P_CHARGE: usize = 1;
const P_GATHER: usize = 2;
const P_CONTACT: usize = 3;
const P_EMIT: usize = 4;
const P_BITE: usize = 5;
const P_SETTLE: usize = 6;
const P_RAIN: usize = 7;
const P_COUNT: usize = 8;
const P_PREFIX: usize = 9;
const P_GIVE_BACK: usize = 10;
const P_BIRTH: usize = 11;
const P_GENES: usize = 12;
const P_FIELDS: usize = 13;
const P_CENSUS_SQUARES: usize = 14;
const P_CENSUS_SLOTS: usize = 15;
const P_NAMES: [&str; 16] = [
    "think", "charge", "gather", "contact", "emit", "bite", "settle", "rain", "count", "prefix",
    "give_back", "birth", "genes", "fields", "census", "census",
];

/// GPU time per pass, measured with timestamp queries when asked for.
struct Profile {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    staging: wgpu::Buffer,
    period_ns: f64,
    per_pipe_ns: [f64; 16],
    ticks: u64,
}

pub struct World {
    pub gpu: Arc<Gpu>,
    pub cfg: Config,
    pub tick: u64,
    pub sun_k: f32,
    pub open: u64,
    pub pacing: Pacing,
    pub longest_submit_ms: f64,
    /// Submissions over the limit since the last report. The first few of a run don't count:
    /// they include memory being paged in, and it's sustained holds that make the desktop stutter.
    pub slow_submits: u32,
    pub gpu_seconds: f64,
    pub submits: u64,
    pub buffers: HashMap<&'static str, wgpu::Buffer>,
    pipes: Vec<Pipe>,
    frames: wgpu::Buffer,
    stride: usize,
    pending: Vec<Op>,
    ops_per_submit: usize,
    splits: u32,
    /// Which of the two field buffers holds the current fields.
    flip: usize,
    profile: Option<Profile>,
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The real buffer behind a name in a shader. Field names depend on which way round they are.
fn buffer_for(name: &'static str, flip: usize) -> &'static str {
    match (name, flip) {
        ("fa0", 0) | ("fa1", 1) => "fa_a",
        ("fa0", _) | ("fa1", _) => "fa_b",
        ("fb0", 0) | ("fb1", 1) => "fb_a",
        ("fb0", _) | ("fb1", _) => "fb_b",
        _ => name,
    }
}

impl World {
    pub fn new(cfg: Config, pacing: Pacing) -> Result<World, String> {
        let adapter_name = std::env::var("AUTO_ADAPTER").ok();
        let gpu = pollster::block_on(Gpu::headless(GpuOptions {
            adapter_name,
            ..Default::default()
        }))?;
        Self::with_gpu(Arc::new(gpu), cfg, pacing)
    }

    /// A new world on a device someone else also uses, such as a window.
    pub fn with_gpu(gpu: Arc<Gpu>, cfg: Config, pacing: Pacing) -> Result<World, String> {
        let mut w = Self::allocate(gpu, cfg, pacing)?;
        w.build_terrain()?;
        Ok(w)
    }

    /// Reopens a saved world, with the shape and physics it was saved with.
    pub fn open_saved(
        gpu: Arc<Gpu>,
        path: &str,
        shaders: Option<std::path::PathBuf>,
        pacing: Pacing,
    ) -> Result<World, String> {
        let cfg = Config::from_save(path, shaders)?;
        let mut w = Self::allocate(gpu, cfg, pacing)?;
        w.load(path)?;
        Ok(w)
    }

    fn allocate(gpu: Arc<Gpu>, cfg: Config, pacing: Pacing) -> Result<World, String> {
        let device = &gpu.device;
        let limit = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        cfg.check(limit)?;
        let squares = cfg.size as u64 * cfg.size as u64;
        let bins = cfg.bins() as u64 * cfg.bins() as u64 * BIN_CAP as u64;
        let slots = cfg.slots as u64;
        let blocks = cfg.blocks() as u64;
        let ring_cap = cfg.slots.next_power_of_two() as u64;
        let copy = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let table: Vec<(&'static str, u64, wgpu::BufferUsages)> = vec![
            ("terrain", squares * 4, copy),
            ("fa_a", squares * 16, copy),
            ("fa_b", squares * 16, copy),
            ("fb_a", squares * 16, copy),
            ("fb_b", squares * 16, copy),
            ("dep", squares * DEP_STRIDE as u64 * 4, copy),
            ("bins", bins * 4, copy),
            ("bb", bins * BODY_BYTES, copy),
            ("cell", slots * CELL_BYTES, copy),
            ("plan", slots * PLAN_BYTES, copy),
            ("sense", slots * SENSE_BYTES, copy),
            ("acc", slots * ACC_BYTES, copy),
            ("mail", slots * 8, copy),
            ("sketch", slots * SKETCH_BYTES, copy),
            ("mem", slots * cfg.hidden_rows as u64 * 16, copy),
            ("genome", slots.min(cfg.shard_slots as u64) * blocks * 16, copy),
            (
                "genome1",
                (slots.saturating_sub(cfg.shard_slots as u64) * blocks * 16).max(16),
                copy,
            ),
            ("ring", ring_cap * 4, copy),
            ("jobs", slots * 16, copy),
            ("bits", cfg.bits_words() as u64 * 4, copy),
            ("chunks", (cfg.bits_words() as u64 / 64) * 4, copy),
            ("ctr", CTR_WORDS * 4, copy),
            ("ctl", 16 * 4, copy),
            ("tot", 16 * 4, copy),
            ("args", 3 * 4, copy | wgpu::BufferUsages::INDIRECT),
        ];
        let mut buffers = HashMap::new();
        for (name, bytes, usage) in table {
            if bytes > limit {
                return Err(format!(
                    "{name} needs {bytes} bytes, the GPU allows {limit} per buffer"
                ));
            }
            buffers.insert(
                name,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(name),
                    size: bytes,
                    usage,
                    mapped_at_creation: false,
                }),
            );
        }
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let stride = FRAME_BYTES.next_multiple_of(align);
        let frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frames"),
            size: (stride * RING_ENTRIES) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipes = build_sim_pipes(&gpu, &cfg, &buffers, &frames)?;
        let size_now = cfg.size;
        Ok(World {
            gpu,
            tick: 0,
            sun_k: 0.0,
            open: 0,
            pacing,
            longest_submit_ms: 0.0,
            slow_submits: 0,
            gpu_seconds: 0.0,
            submits: 0,
            buffers,
            pipes,
            frames,
            stride,
            pending: vec![],
            // Start cautious on big worlds. The pacing loop finds the right size from here.
            ops_per_submit: if size_now >= 1024 { 1 } else { 8 },
            splits: (size_now / 512).max(1),
            flip: 0,
            profile: None,
            cfg,
        })
    }

    /// Changes physics on the running world. Everything in the buffers stays as it is, and the
    /// next tick runs by the new rules. If the new shaders don't build, the old rules stay.
    pub fn set_physics(
        &mut self,
        physics: serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), String> {
        self.flush();
        let mut cfg = self.cfg.clone();
        cfg.physics = physics;
        let limit = self.gpu.device.limits().max_storage_buffer_binding_size;
        cfg.check(limit)?;
        let scope = self.gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let built = build_sim_pipes(&self.gpu, &cfg, &self.buffers, &self.frames);
        if let Some(e) = pollster::block_on(scope.pop()) {
            return Err(e.to_string());
        }
        let built = built?;
        let sun_changed = self.cfg.f("sun") != cfg.f("sun");
        self.pipes = built;
        self.cfg = cfg;
        if sun_changed {
            let (_, mean) = self.measure_terrain();
            self.sun_k = (self.cfg.f("sun") as f64 / mean.max(1e-9)) as f32;
        }
        Ok(())
    }

    /// Samples the terrain: the share of open squares, and their mean sun before scaling.
    fn measure_terrain(&self) -> (f64, f64) {
        let w = self.cfg.size;
        let rows_wanted = 256.min(w);
        let step = (w / rows_wanted).max(1);
        let ranges: Vec<(u64, u64)> = (0..rows_wanted)
            .map(|k| ((k * step) as u64 * w as u64 * 4, w as u64 * 4))
            .collect();
        let bytes = self.read_ranges("terrain", &ranges);
        let (mut base, mut open, mut seen) = (0.0f64, 0u64, 0u64);
        for v in bytemuck::cast_slice::<u8, u32>(&bytes) {
            seen += 1;
            if v & (1 << 24) == 0 {
                open += 1;
                base += (v & 0xFFFF) as f64 / 65535.0 * SUN_RANGE as f64;
            }
        }
        (open as f64 / seen.max(1) as f64, base / open.max(1) as f64)
    }

    fn write_frame(&self, frame: Frame) {
        let mut bytes = vec![0u8; self.stride];
        bytes[..FRAME_BYTES].copy_from_slice(bytemuck::bytes_of(&frame));
        self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
    }

    fn build_terrain(&mut self) -> Result<(), String> {
        let binds: &[Bind] = &[
            ("terrain", "u32", RW),
            ("fa0", "vec4i", RW),
            ("ring", "u32", RW),
        ];
        let mk = |entry: &str| -> Result<Pipe, String> {
            build_pipe(&self.gpu, &self.cfg, &self.buffers, &self.frames, "terrain.wgsl", entry, binds)
        };
        let main = mk("main")?;
        let finish = mk("finish")?;
        let fill = mk("fill_ring")?;
        let w = self.cfg.size;
        let band = 256.min(w);
        let run_rows = |this: &Self, pipe: &Pipe, sun_k: f32| {
            let mut row = 0;
            while row < w {
                this.write_frame(Frame {
                    row0: row,
                    seed: this.cfg.seed,
                    sun_k,
                    ..Default::default()
                });
                let mut enc = this.gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipe.pipeline);
                    pass.set_bind_group(0, &pipe.groups[0], &[0]);
                    pass.dispatch_workgroups(w.div_ceil(64), band.min(w - row), 1);
                }
                this.gpu.queue.submit([enc.finish()]);
                this.wait_gpu();
                row += band;
            }
        };
        run_rows(self, &main, 0.0);
        let (open_frac, mean_base) = self.measure_terrain();
        self.open = (w as f64 * w as f64 * open_frac) as u64;
        self.sun_k = (self.cfg.f("sun") as f64 / mean_base.max(1e-9)) as f32;
        run_rows(self, &finish, self.sun_k);

        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&fill.pipeline);
            pass.set_bind_group(0, &fill.groups[0], &[0]);
            pass.dispatch_workgroups(1024, self.cfg.slots.div_ceil(65536), 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.wait_gpu();

        // The ring holds every slot, so head starts one lap ahead of tail.
        let mut ctl = [0u32; 16];
        ctl[L_HEAD as usize] = self.cfg.slots;
        ctl[L_TAIL as usize] = 0;
        ctl[L_OPEN as usize] = self.open.min(u32::MAX as u64) as u32;
        self.gpu
            .queue
            .write_buffer(&self.buffers["ctl"], 0, bytemuck::cast_slice(&ctl));
        self.gpu.queue.submit([]);
        self.wait_gpu();
        Ok(())
    }

    fn base_frame(&self) -> Frame {
        let every = (self.cfg.f("brain_every") as u64).max(1);
        let sx = self.tick as f64 * self.cfg.f("sun_drift") as f64;
        let wrap = self.cfg.size as f64;
        Frame {
            tick: self.tick as u32,
            seed: self.cfg.seed,
            phase: (self.tick % every) as u32,
            sun_k: self.sun_k,
            drift_x: sx.rem_euclid(wrap) as f32,
            drift_y: (sx * 0.618034).rem_euclid(wrap) as f32,
            ..Default::default()
        }
    }

    /// Runs a pass over the slots, or some stretch of them, in bands.
    fn push_slots(&mut self, pipe: usize, total: u32, wg: u32, base: Frame) {
        let groups = total.div_ceil(wg);
        for (first, n) in bands(groups, self.splits) {
            self.push(Op::Dispatch {
                pipe,
                flip: self.flip,
                frame: Frame { row0: first * wg, ..base },
                groups: [n, 1, 1],
            });
        }
    }

    /// Runs a pass over rows of something `width` wide.
    fn push_rows(&mut self, pipe: usize, width: u32, wg: u32, rows: u32, base: Frame) {
        for (first, n) in bands(rows, self.splits) {
            self.push(Op::Dispatch {
                pipe,
                flip: self.flip,
                frame: Frame { row0: first, ..base },
                groups: [width.div_ceil(wg), n, 1],
            });
        }
    }

    /// Queues one tick. Work reaches the GPU in short submissions.
    pub fn step(&mut self) {
        self.tick += 1;
        let base = self.base_frame();
        let every = (self.cfg.f("brain_every") as u32).max(1);
        let slots = self.cfg.slots;
        let bins = self.cfg.bins();
        let size = self.cfg.size;
        let tiles = self.cfg.tiles();

        self.push_slots(P_THINK, slots.div_ceil(every), 64, base);
        self.push_slots(P_CHARGE, slots, 64, base);
        self.push_rows(P_GATHER, bins, 32, bins, base);
        self.push_rows(P_CONTACT, bins, 32, bins, base);
        self.push_rows(P_EMIT, bins, 32, bins, base);
        if self.cfg.b("bite_on") {
            self.push_slots(P_BITE, slots, 64, base);
        }
        self.push_slots(P_SETTLE, slots, 64, base);
        self.push(Op::Dispatch {
            pipe: P_RAIN,
            flip: self.flip,
            frame: base,
            groups: [tiles.div_ceil(8), tiles.div_ceil(8), 1],
        });
        let chunks = self.cfg.bits_words() / 64;
        self.push(Op::Dispatch {
            pipe: P_COUNT,
            flip: self.flip,
            frame: base,
            groups: [chunks.div_ceil(16), 1, 1],
        });
        self.push(Op::Dispatch {
            pipe: P_PREFIX,
            flip: self.flip,
            frame: base,
            groups: [1, 1, 1],
        });
        self.push(Op::Dispatch {
            pipe: P_GIVE_BACK,
            flip: self.flip,
            frame: base,
            groups: [self.cfg.free_words().div_ceil(32), 1, 1],
        });
        self.push(Op::Dispatch {
            pipe: P_BIRTH,
            flip: self.flip,
            frame: base,
            groups: [self.cfg.claim_words().div_ceil(32), 1, 1],
        });
        self.push(Op::Indirect {
            pipe: P_GENES,
            flip: self.flip,
            frame: base,
        });
        self.push(Op::ClearBits);
        self.push_rows(P_FIELDS, size, 64, size, base);
        self.flip = 1 - self.flip;
        if let Some(p) = &mut self.profile {
            p.ticks += 1;
        }
    }

    fn push(&mut self, op: Op) {
        self.pending.push(op);
        if self.pending.len() >= self.ops_per_submit.min(RING_ENTRIES / 2) {
            self.flush();
        }
    }

    /// Waits for everything submitted. On a device shared with a window this polls, because a
    /// blocking wait also blocks the window from submitting anything until it ends.
    fn wait_gpu(&self) {
        if Arc::strong_count(&self.gpu) == 1 {
            self.gpu.wait();
            return;
        }
        let done = Arc::new(AtomicBool::new(false));
        let flag = done.clone();
        self.gpu
            .queue
            .on_submitted_work_done(move || flag.store(true, Ordering::Release));
        while !done.load(Ordering::Acquire) {
            let _ = self.gpu.device.poll(wgpu::PollType::Poll);
            std::thread::sleep(Duration::from_micros(100));
        }
    }

    /// Sends everything queued, waits for it, then adjusts the size of the next submission so
    /// each stays near the target length.
    pub fn flush(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let ops = std::mem::take(&mut self.pending);
        let mut bytes = vec![0u8; self.stride * ops.len().min(RING_ENTRIES)];
        let mut slot = 0usize;
        let mut offsets = Vec::with_capacity(ops.len());
        for op in &ops {
            match op {
                Op::Dispatch { frame, .. } | Op::Indirect { frame, .. } => {
                    bytes[slot * self.stride..slot * self.stride + FRAME_BYTES]
                        .copy_from_slice(bytemuck::bytes_of(frame));
                    offsets.push((slot * self.stride) as u32);
                    slot += 1;
                }
                Op::ClearBits => offsets.push(0),
            }
        }
        self.gpu
            .queue
            .write_buffer(&self.frames, 0, &bytes[..slot.max(1) * self.stride]);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut stamped: Vec<usize> = vec![];
        if self.profile.is_some() {
            self.encode_profiled(&mut enc, &ops, &offsets, &mut stamped);
        } else {
            let mut i = 0;
            while i < ops.len() {
                if let Op::ClearBits = ops[i] {
                    enc.clear_buffer(&self.buffers["bits"], 0, None);
                    i += 1;
                    continue;
                }
                let mut pass = enc.begin_compute_pass(&Default::default());
                let mut current = usize::MAX;
                while i < ops.len() {
                    match ops[i] {
                        Op::Dispatch { pipe, flip, groups, .. } => {
                            if current != pipe {
                                pass.set_pipeline(&self.pipes[pipe].pipeline);
                                current = pipe;
                            }
                            pass.set_bind_group(0, &self.pipes[pipe].groups[flip], &[offsets[i]]);
                            pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                        }
                        Op::Indirect { pipe, flip, .. } => {
                            pass.set_pipeline(&self.pipes[pipe].pipeline);
                            current = pipe;
                            pass.set_bind_group(0, &self.pipes[pipe].groups[flip], &[offsets[i]]);
                            pass.dispatch_workgroups_indirect(&self.buffers["args"], 0);
                        }
                        Op::ClearBits => break,
                    }
                    i += 1;
                }
            }
        }
        let started = Instant::now();
        self.gpu.queue.submit([enc.finish()]);
        self.wait_gpu();
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        if !stamped.is_empty() {
            self.read_profile(&stamped);
        }
        self.submits += 1;
        self.gpu_seconds += ms / 1000.0;
        if self.submits > 32 {
            self.longest_submit_ms = self.longest_submit_ms.max(ms);
            if ms > 80.0 {
                self.slow_submits += 1;
            }
        }

        let target = self.pacing.target_ms;
        if ms > target * 1.25 {
            if self.ops_per_submit > 1 {
                self.ops_per_submit = (self.ops_per_submit / 2).max(1);
            } else {
                self.splits = (self.splits * 2).min(self.cfg.size / 4);
            }
        } else if ms < target * 0.4 {
            self.ops_per_submit = (self.ops_per_submit * 2).min(RING_ENTRIES / 2);
        }
        if self.pacing.duty < 1.0 {
            let rest = ms * (1.0 / self.pacing.duty.max(0.05) - 1.0);
            std::thread::sleep(Duration::from_secs_f64(rest / 1000.0));
        }
    }

    /// Turns on timing of each pass. Slower, because every dispatch gets its own compute pass.
    pub fn enable_profile(&mut self) -> bool {
        if !self.gpu.device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return false;
        }
        self.flush();
        let n = RING_ENTRIES as u32 / 2;
        let device = &self.gpu.device;
        let bytes = n as u64 * 2 * 8;
        self.profile = Some(Profile {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("passes"),
                ty: wgpu::QueryType::Timestamp,
                count: n * 2,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("resolved"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            period_ns: self.gpu.queue.get_timestamp_period() as f64,
            per_pipe_ns: [0.0; 16],
            ticks: 0,
        });
        true
    }

    fn encode_profiled(
        &self,
        enc: &mut wgpu::CommandEncoder,
        ops: &[Op],
        offsets: &[u32],
        stamped: &mut Vec<usize>,
    ) {
        let p = self.profile.as_ref().expect("profile");
        for (i, op) in ops.iter().enumerate() {
            let k = stamped.len() as u32;
            let writes = wgpu::ComputePassTimestampWrites {
                query_set: &p.set,
                beginning_of_pass_write_index: Some(k * 2),
                end_of_pass_write_index: Some(k * 2 + 1),
            };
            match *op {
                Op::ClearBits => enc.clear_buffer(&self.buffers["bits"], 0, None),
                Op::Dispatch { pipe, flip, groups, .. } => {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(writes),
                    });
                    pass.set_pipeline(&self.pipes[pipe].pipeline);
                    pass.set_bind_group(0, &self.pipes[pipe].groups[flip], &[offsets[i]]);
                    pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                    stamped.push(pipe);
                }
                Op::Indirect { pipe, flip, .. } => {
                    let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(writes),
                    });
                    pass.set_pipeline(&self.pipes[pipe].pipeline);
                    pass.set_bind_group(0, &self.pipes[pipe].groups[flip], &[offsets[i]]);
                    pass.dispatch_workgroups_indirect(&self.buffers["args"], 0);
                    stamped.push(pipe);
                }
            }
        }
        let n = stamped.len() as u32;
        if n > 0 {
            enc.resolve_query_set(&p.set, 0..n * 2, &p.resolve, 0);
            enc.copy_buffer_to_buffer(&p.resolve, 0, &p.staging, 0, n as u64 * 16);
        }
    }

    fn read_profile(&mut self, stamped: &[usize]) {
        let Some(p) = &mut self.profile else { return };
        let bytes = stamped.len() as u64 * 16;
        p.staging
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        let _ = self.gpu.device.poll(wgpu::PollType::wait_indefinitely());
        {
            let view = p.staging.get_mapped_range(..bytes).expect("mapped");
            let times: &[u64] = bytemuck::cast_slice(&view);
            for (k, pipe) in stamped.iter().enumerate() {
                let dt = times[k * 2 + 1].wrapping_sub(times[k * 2]) as f64 * p.period_ns;
                p.per_pipe_ns[*pipe] += dt;
            }
        }
        p.staging.unmap();
    }

    /// Milliseconds of GPU per tick in each pass, since profiling began.
    pub fn profile_report(&self) -> Option<Vec<(&'static str, f64)>> {
        let p = self.profile.as_ref()?;
        let ticks = p.ticks.max(1) as f64;
        Some(
            P_NAMES
                .iter()
                .zip(p.per_pipe_ns.iter())
                .take(14)
                .map(|(n, t)| (*n, t / ticks / 1e6))
                .collect(),
        )
    }

    /// Copies byte ranges of a buffer back to the CPU.
    pub fn read_ranges(&self, name: &str, ranges: &[(u64, u64)]) -> Vec<u8> {
        let total: u64 = ranges.iter().map(|r| r.1).sum();
        let staging = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("read back"),
            size: total.max(4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut at = 0;
        for (offset, size) in ranges {
            enc.copy_buffer_to_buffer(&self.buffers[name], *offset, &staging, at, *size);
            at += size;
        }
        self.gpu.queue.submit([enc.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        self.wait_gpu();
        let out = staging.get_mapped_range(..).expect("mapped").to_vec();
        staging.unmap();
        out
    }

    fn read_all(&self, name: &str) -> Vec<u8> {
        let size = self.buffers[name].size();
        let mut out = Vec::with_capacity(size as usize);
        let mut at = 0;
        while at < size {
            let n = (size - at).min(16 << 20);
            out.extend_from_slice(&self.read_ranges(name, &[(at, n)]));
            at += n;
        }
        out
    }

    /// Exact flows and counts so far.
    pub fn tally(&self) -> Tally {
        let b = self.read_ranges("ctr", &[(0, CTR_WORDS * 4)]);
        let w: &[u32] = bytemuck::cast_slice(&b);
        let flows = (0..ROUTES.len())
            .map(|k| w[k * 2] as u64 | (w[k * 2 + 1] as u64) << 32)
            .collect();
        let c = &w[ROUTES.len() * 2..];
        Tally {
            flows,
            births: c[0] as u64,
            rained: c[1] as u64,
            starved: c[2] as u64,
            bitten: c[3] as u64,
            burned: c[4] as u64,
            old: c[5] as u64,
            overflow: c[6] as u64,
            no_slot: c[7] as u64,
            sample: if self.cfg.b("ledger") { 1 } else { 64 },
        }
    }

    pub fn alive(&self) -> u32 {
        let b = self.read_ranges("ctl", &[(L_ALIVE * 4, 4)]);
        u32::from_le_bytes(b[..4].try_into().unwrap())
    }

    /// Where the energy is, added up on the GPU in short pieces.
    pub fn stock(&mut self) -> Stock {
        self.flush();
        self.gpu.queue.write_buffer(&self.buffers["tot"], 0, &[0u8; 64]);
        let size = self.cfg.size;
        let slots_runs = self.cfg.slots.div_ceil(256);
        let run = |pipe: usize, total: u32, this: &mut Self| {
            let splits = (total / 256).max(1).max(this.splits);
            for (first, n) in bands(total.div_ceil(8), splits) {
                this.write_frame(Frame {
                    row0: first * 8,
                    ..this.base_frame()
                });
                let mut enc = this.gpu.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&this.pipes[pipe].pipeline);
                    pass.set_bind_group(0, &this.pipes[pipe].groups[this.flip], &[0]);
                    pass.dispatch_workgroups(n, 1, 1);
                }
                this.gpu.queue.submit([enc.finish()]);
                this.wait_gpu();
            }
        };
        run(P_CENSUS_SQUARES, size, self);
        run(P_CENSUS_SLOTS, slots_runs, self);
        let b = self.read_ranges("tot", &[(0, 64)]);
        let w: &[u32] = bytemuck::cast_slice(&b);
        let at = |k: usize| w[k * 2] as u64 | (w[k * 2 + 1] as u64) << 32;
        Stock {
            food: at(0),
            meat: at(1),
            heat: at(2),
            enzyme: at(3),
            energy: at(4),
            womb: at(5),
            alive: at(6),
            body_each: (self.cfg.f("body") * E_ONE) as u64,
        }
    }

    /// Every cell slot, living or not.
    pub fn read_cells(&self) -> Vec<Cell> {
        bytemuck::pod_collect_to_vec(&self.read_all("cell"))
    }

    pub fn read_plans(&self) -> Vec<Plan> {
        bytemuck::pod_collect_to_vec(&self.read_all("plan"))
    }

    pub fn read_senses(&self) -> Vec<Sense> {
        bytemuck::pod_collect_to_vec(&self.read_all("sense"))
    }

    pub fn read_acc(&self) -> Vec<Acc> {
        bytemuck::pod_collect_to_vec(&self.read_all("acc"))
    }

    pub fn read_sketches(&self) -> Vec<[u32; 4]> {
        bytemuck::pod_collect_to_vec(&self.read_all("sketch"))
    }

    pub fn read_mem(&self) -> Vec<[f32; 4]> {
        bytemuck::pod_collect_to_vec(&self.read_all("mem"))
    }

    pub fn read_terrain(&self) -> Vec<u32> {
        bytemuck::pod_collect_to_vec(&self.read_all("terrain"))
    }

    /// The current food, meat, heat and enzyme of every square, in units.
    pub fn read_fa(&self) -> Vec<[i32; 4]> {
        bytemuck::pod_collect_to_vec(&self.read_all(if self.flip == 0 { "fa_a" } else { "fa_b" }))
    }

    /// The current scents of every square, in units.
    pub fn read_fb(&self) -> Vec<[i32; 4]> {
        bytemuck::pod_collect_to_vec(&self.read_all(if self.flip == 0 { "fb_a" } else { "fb_b" }))
    }

    /// Writes one square's fields, in energy. For tests and tools.
    pub fn write_fa(&self, x: u32, y: u32, v: [f32; 4]) {
        let p = (y << self.cfg.size.trailing_zeros() | x) as u64;
        let units = v.map(|a| (a * E_ONE) as i32);
        for name in ["fa_a", "fa_b"] {
            if (name == "fa_a") == (self.flip == 0) {
                self.gpu.queue.write_buffer(
                    &self.buffers[name],
                    p * 16,
                    bytemuck::cast_slice(&units),
                );
            }
        }
        self.gpu.queue.submit([]);
        self.wait_gpu();
    }

    /// Writes a whole field layer, in units. For tests and tools.
    pub fn write_fa_all(&self, field: &[[i32; 4]]) {
        let name = if self.flip == 0 { "fa_a" } else { "fa_b" };
        self.gpu
            .queue
            .write_buffer(&self.buffers[name], 0, bytemuck::cast_slice(field));
        self.gpu.queue.submit([]);
        self.wait_gpu();
    }

    /// Puts a cell in by hand. Slots are handed out in order, so only use this on a fresh world.
    /// Position is in squares. It rests until it first thinks.
    pub fn plant(&mut self, x: f32, y: f32, e: f32, womb: f32, heading: f32, genome: &[u8]) -> u32 {
        self.flush();
        let c = self.read_ranges("ctl", &[(0, 64)]);
        let ctl: Vec<u32> = bytemuck::pod_collect_to_vec(&c);
        let slot = ctl[L_TAIL as usize];
        let uid = ctl[L_NEXT_UID as usize];
        let to_pos = (1u64 << (32 - self.cfg.size.trailing_zeros())) as f64;
        let (h_sin, h_cos) = heading.sin_cos();
        let e_u = (e * E_ONE) as u32;
        let cell = Cell {
            x: (x as f64 * to_pos) as u32,
            y: (y as f64 * to_pos) as u32,
            r: (self.cfg.f("r0") * (1.0 + e / self.cfg.f("e_size")).sqrt()).min(self.cfg.f("r_max")),
            hx: h_cos,
            hy: h_sin,
            e: e_u,
            womb: (womb * E_ONE) as u32,
            flags: F_ALIVE,
            uid,
            parent: u32::MAX,
            clade: uid,
            mu: self.cfg.f("mut_rate"),
            ..Default::default()
        };
        let plan = Plan {
            s0: [1.0, 0.0, 0.0, 0.0],
            mix: [0.25; 4],
            ..Default::default()
        };
        let q = &self.gpu.queue;
        q.write_buffer(
            &self.buffers["cell"],
            slot as u64 * CELL_BYTES,
            bytemuck::bytes_of(&cell),
        );
        q.write_buffer(
            &self.buffers["plan"],
            slot as u64 * PLAN_BYTES,
            bytemuck::bytes_of(&plan),
        );
        q.write_buffer(&self.buffers["ctl"], L_TAIL * 4, bytemuck::bytes_of(&(slot + 1)));
        q.write_buffer(&self.buffers["ctl"], L_ALIVE * 4, bytemuck::bytes_of(&(ctl[L_ALIVE as usize] + 1)));
        q.write_buffer(&self.buffers["ctl"], L_NEXT_UID * 4, bytemuck::bytes_of(&(uid + 1)));
        let g = self.cfg.blocks() as u64 * 16;
        assert_eq!(genome.len() as u64, g);
        self.write_genomes(slot as u64, genome);
        q.submit([]);
        self.wait_gpu();
        slot
    }

    /// Writes the genomes of consecutive slots, across both buffers if needed.
    pub fn write_genomes(&self, start: u64, bytes: &[u8]) {
        let g = self.cfg.blocks() as u64 * 16;
        let shard = self.cfg.shard_slots as u64;
        let slots = bytes.len() as u64 / g;
        let first = shard.saturating_sub(start).min(slots);
        let (a, b) = bytes.split_at((first * g) as usize);
        let q = &self.gpu.queue;
        if !a.is_empty() {
            q.write_buffer(&self.buffers["genome"], start * g, a);
        }
        if !b.is_empty() {
            q.write_buffer(&self.buffers["genome1"], (start + first - shard) * g, b);
        }
    }

    pub fn read_genome(&self, slot: u32) -> Vec<u8> {
        let g = self.cfg.blocks() as u64 * 16;
        let shard = self.cfg.shard_slots as u64;
        let s = slot as u64;
        if s < shard {
            self.read_ranges("genome", &[(s * g, g)])
        } else {
            self.read_ranges("genome1", &[((s - shard) * g, g)])
        }
    }

    /// The buffers a save holds, in file order, with the current fields under their plain names.
    fn saved_names(&self) -> Vec<(&'static str, &'static str)> {
        let (fa, fb) = if self.flip == 0 { ("fa_a", "fb_a") } else { ("fa_b", "fb_b") };
        let mut names = vec![
            ("terrain", "terrain"),
            ("fa", fa),
            ("fb", fb),
            ("cell", "cell"),
            ("plan", "plan"),
            ("sense", "sense"),
            ("sketch", "sketch"),
            ("mem", "mem"),
            ("genome", "genome"),
        ];
        if self.cfg.two_shards() {
            names.push(("genome1", "genome1"));
        }
        names.extend([("ring", "ring"), ("ctr", "ctr"), ("ctl", "ctl")]);
        names
    }

    /// Writes every buffer that defines the world to a file, in chunks, so RAM stays free.
    pub fn save(&mut self, path: &str) -> Result<(), String> {
        use std::io::Write;
        self.flush();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
        let head = serde_json::json!({
            "size": self.cfg.size, "slots": self.cfg.slots, "shard_slots": self.cfg.shard_slots,
            "hidden_rows": self.cfg.hidden_rows, "seed": self.cfg.seed, "tick": self.tick,
            "sun_k": self.sun_k, "open": self.open, "physics": self.cfg.physics, "layout": layout_json(),
        })
        .to_string();
        f.write_all(&(head.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
        f.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        for (_, real) in self.saved_names() {
            let size = self.buffers[real].size();
            let mut at = 0;
            while at < size {
                let n = (size - at).min(16 << 20);
                let bytes = self.read_ranges(real, &[(at, n)]);
                f.write_all(&bytes).map_err(|e| e.to_string())?;
                at += n;
            }
        }
        Ok(())
    }

    /// Reads a saved world's state. The world must have the same shape.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        use std::io::Read;
        self.flush();
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?);
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| e.to_string())?;
        let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut head).map_err(|e| e.to_string())?;
        let head: serde_json::Value = serde_json::from_slice(&head).map_err(|e| e.to_string())?;
        check_layout(&head, path)?;
        if head["size"] != self.cfg.size
            || head["slots"] != self.cfg.slots
            || head["hidden_rows"] != self.cfg.hidden_rows
        {
            return Err(format!(
                "{path} is a world of another shape: {} squares wide, {} slots",
                head["size"], head["slots"]
            ));
        }
        self.tick = head["tick"].as_u64().unwrap_or(0);
        self.sun_k = head["sun_k"].as_f64().unwrap_or(0.0) as f32;
        self.open = head["open"].as_u64().unwrap_or(0);
        self.flip = 0;
        let mut chunk = vec![0u8; 16 << 20];
        for (_, real) in self.saved_names() {
            let size = self.buffers[real].size();
            let mut at = 0;
            while at < size {
                let n = (size - at).min(16 << 20);
                f.read_exact(&mut chunk[..n as usize])
                    .map_err(|e| format!("{path} is cut short: {e}"))?;
                self.gpu.queue.write_buffer(&self.buffers[real], at, &chunk[..n as usize]);
                self.gpu.queue.submit([]);
                self.wait_gpu();
                at += n;
            }
        }
        Ok(())
    }

    /// Copies the living cells of saved smaller worlds into this fresh world, as a grid of tiles.
    /// Neighbouring tiles come from different files, so populations that evolved apart meet at
    /// the borders. Returns how many cells were planted.
    pub fn seed_from(&mut self, paths: &[String], keep: f64) -> Result<u64, String> {
        self.flush();
        let hr = self.cfg.hidden_rows as u64;
        let genome_bytes = self.cfg.blocks() as usize * 16;
        let mem_bytes = hr as usize * 16;
        let terrain = self.read_terrain();
        let log = self.cfg.size.trailing_zeros();
        let shift = 32 - log;
        let ctl: Vec<u32> = bytemuck::pod_collect_to_vec(&self.read_ranges("ctl", &[(0, 64)]));
        let mut slot = ctl[L_TAIL as usize] as u64;
        let first_slot = slot;
        let mut uid = ctl[L_NEXT_UID as usize];
        let mut rng = self.cfg.seed as u64 ^ 0x7777_1234;
        let mut roll = move || {
            rng = rng.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = rng;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            (z ^ (z >> 31)) as f64 / u64::MAX as f64
        };
        let per_side_of = |src: u64| self.cfg.size as u64 / src;
        let mut sources = vec![];
        for path in paths {
            sources.push(SavedWorld::open(path)?);
        }
        let src_size = sources.first().map(|s| s.size).ok_or("no worlds to seed from")?;
        if sources.iter().any(|s| s.size != src_size) || src_size > self.cfg.size as u64 {
            return Err("seed worlds must all be one size, no wider than this world".into());
        }
        if sources.iter().any(|s| s.hidden_rows != hr) {
            return Err("seed worlds must have this world's rows of memory".into());
        }
        let per_side = per_side_of(src_size);
        let src_shift = 32 - src_size.trailing_zeros();
        for ty in 0..per_side {
            for tx in 0..per_side {
                let which = ((tx + ty) % sources.len() as u64) as usize;
                let source = &mut sources[which];
                for s in 0..source.slots {
                    let mut c = source.cell(s);
                    if c.flags & F_ALIVE == 0 {
                        continue;
                    }
                    if keep < 1.0 && roll() >= keep {
                        continue;
                    }
                    let x = ((c.x >> src_shift) as u64 + tx * src_size) as u32;
                    let y = ((c.y >> src_shift) as u64 + ty * src_size) as u32;
                    if terrain[((y as usize) << log) | x as usize] & (1 << 24) != 0 {
                        continue;
                    }
                    if slot >= self.cfg.slots as u64 {
                        break;
                    }
                    c.x = (x << shift) | ((c.x & ((1u32 << src_shift).wrapping_sub(1))) >> (src_shift - shift).min(31));
                    c.y = (y << shift) | ((c.y & ((1u32 << src_shift).wrapping_sub(1))) >> (src_shift - shift).min(31));
                    c.uid = uid;
                    uid += 1;
                    let q = &self.gpu.queue;
                    q.write_buffer(&self.buffers["cell"], slot * CELL_BYTES, bytemuck::bytes_of(&c));
                    q.write_buffer(&self.buffers["plan"], slot * PLAN_BYTES, &source.read("plan", s, PLAN_BYTES as usize)?);
                    q.write_buffer(&self.buffers["sketch"], slot * SKETCH_BYTES, &source.read("sketch", s, SKETCH_BYTES as usize)?);
                    q.write_buffer(&self.buffers["mem"], slot * hr * 16, &source.read("mem", s, mem_bytes)?);
                    let g = source.read_genome(s, genome_bytes)?;
                    self.write_genomes(slot, &g);
                    slot += 1;
                    if slot.is_multiple_of(4096) {
                        q.submit([]);
                        self.wait_gpu();
                    }
                }
            }
        }
        let planted = slot - first_slot;
        let q = &self.gpu.queue;
        q.write_buffer(&self.buffers["ctl"], L_TAIL * 4, bytemuck::bytes_of(&(slot as u32)));
        q.write_buffer(&self.buffers["ctl"], L_ALIVE * 4, bytemuck::bytes_of(&(ctl[L_ALIVE as usize] + planted as u32)));
        q.write_buffer(&self.buffers["ctl"], L_NEXT_UID * 4, bytemuck::bytes_of(&uid));
        q.submit([]);
        self.wait_gpu();
        Ok(planted)
    }
}

/// A saved world on disk, to read cells from without loading it.
struct SavedWorld {
    file: std::fs::File,
    offsets: HashMap<&'static str, u64>,
    cells: Vec<Cell>,
    size: u64,
    slots: u64,
    hidden_rows: u64,
    shard: u64,
}

impl SavedWorld {
    fn open(path: &str) -> Result<SavedWorld, String> {
        use std::io::{Read, Seek, SeekFrom};
        let mut file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(|e| e.to_string())?;
        let head_len = u64::from_le_bytes(len);
        let mut head = vec![0u8; head_len as usize];
        file.read_exact(&mut head).map_err(|e| e.to_string())?;
        let head: serde_json::Value = serde_json::from_slice(&head).map_err(|e| e.to_string())?;
        check_layout(&head, path)?;
        let int = |k: &str| head[k].as_u64().unwrap_or(0);
        let (size, slots, hidden_rows, shard) = (int("size"), int("slots"), int("hidden_rows"), int("shard_slots"));
        let blocks = hidden_rows * (IN_BLOCKS as u64 + hidden_rows + OUT_ROWS as u64);
        let squares = size * size;
        let mut at = 8 + head_len;
        let mut offsets = HashMap::new();
        let mut table: Vec<(&'static str, u64)> = vec![
            ("terrain", squares * 4),
            ("fa", squares * 16),
            ("fb", squares * 16),
            ("cell", slots * CELL_BYTES),
            ("plan", slots * PLAN_BYTES),
            ("sense", slots * SENSE_BYTES),
            ("sketch", slots * SKETCH_BYTES),
            ("mem", slots * hidden_rows * 16),
            ("genome", slots.min(shard) * blocks * 16),
        ];
        if slots > shard {
            table.push(("genome1", (slots - shard) * blocks * 16));
        }
        for (name, bytes) in table {
            offsets.insert(name, at);
            at += bytes;
        }
        file.seek(SeekFrom::Start(offsets["cell"])).map_err(|e| e.to_string())?;
        let mut raw = vec![0u8; (slots * CELL_BYTES) as usize];
        file.read_exact(&mut raw).map_err(|e| format!("{path} is cut short: {e}"))?;
        Ok(SavedWorld {
            file,
            offsets,
            cells: bytemuck::pod_collect_to_vec(&raw),
            size,
            slots,
            hidden_rows,
            shard,
        })
    }

    fn cell(&self, s: u64) -> Cell {
        self.cells[s as usize]
    }

    fn read(&mut self, name: &str, s: u64, bytes: usize) -> Result<Vec<u8>, String> {
        use std::io::{Read, Seek, SeekFrom};
        self.file
            .seek(SeekFrom::Start(self.offsets[name] + s * bytes as u64))
            .map_err(|e| e.to_string())?;
        let mut v = vec![0u8; bytes];
        self.file.read_exact(&mut v).map_err(|e| e.to_string())?;
        Ok(v)
    }

    fn read_genome(&mut self, s: u64, bytes: usize) -> Result<Vec<u8>, String> {
        if s < self.shard {
            self.read("genome", s, bytes)
        } else {
            self.read("genome1", s - self.shard, bytes)
        }
    }
}

pub fn bands(total: u32, splits: u32) -> Vec<(u32, u32)> {
    let splits = splits.clamp(1, total.max(1));
    (0..splits)
        .map(|k| {
            let a = total as u64 * k as u64 / splits as u64;
            let b = total as u64 * (k as u64 + 1) / splits as u64;
            (a as u32, (b - a) as u32)
        })
        .collect()
}

/// Every pipeline that steps the world. Physics settings are constants in the shaders, so a
/// change of physics means building these again.
fn build_sim_pipes(
    gpu: &Gpu,
    cfg: &Config,
    buffers: &HashMap<&'static str, wgpu::Buffer>,
    frames: &wgpu::Buffer,
) -> Result<Vec<Pipe>, String> {
    const A32: &str = "atomic<u32>";
    let mk = |file: &str, entry: &str, binds: &[Bind]| -> Result<Pipe, String> {
        build_pipe(gpu, cfg, buffers, frames, file, entry, binds)
    };
    let alloc: &[Bind] = &[
        ("bits", "u32", R),
        ("chunks", "u32", RW),
        ("ctl", "u32", RW),
        ("args", "u32", RW),
        ("ctr", A32, RW),
        ("ring", "u32", RW),
        ("cell", "Cell", RW),
        ("plan", "Plan", RW),
        ("sense", "Sense", RW),
        ("mem", "vec4f", RW),
        ("jobs", "vec4u", RW),
        ("terrain", "u32", R),
    ];
    let census: &[Bind] = &[("fa0", "vec4i", R), ("cell", "Cell", R), ("tot", A32, RW)];
    Ok(vec![
        mk(
            "think.wgsl",
            "main",
            &[
                ("cell", "Cell", RW),
                ("plan", "Plan", RW),
                ("sense", "Sense", R),
                ("mem", "vec4f", RW),
                ("genome", "vec4u", R),
                ("terrain", "u32", R),
                ("fa0", "vec4i", R),
                ("fb0", "vec4i", R),
            ],
        )?,
        mk(
            "charge.wgsl",
            "main",
            &[
                ("cell", "Cell", RW),
                ("plan", "Plan", R),
                ("bins", A32, RW),
                ("ctr", A32, RW),
            ],
        )?,
        mk(
            "gather.wgsl",
            "main",
            &[("bins", "u32", RW), ("cell", "Cell", R), ("bb", "BinBody", RW)],
        )?,
        mk(
            "contact.wgsl",
            "main",
            &[
                ("bb", "BinBody", R),
                ("cell", "Cell", R),
                ("plan", "Plan", R),
                ("acc", "Acc", RW),
                ("sense", "Sense", RW),
                ("mail", A32, RW),
                ("ctr", A32, RW),
            ],
        )?,
        mk(
            "emit.wgsl",
            "main",
            &[
                ("bb", "BinBody", R),
                ("cell", "Cell", R),
                ("plan", "Plan", R),
                ("acc", "Acc", RW),
                ("fa0", "vec4i", R),
                ("dep", A32, RW),
            ],
        )?,
        mk(
            "bite.wgsl",
            "main",
            &[
                ("acc", "Acc", RW),
                ("cell", "Cell", R),
                ("mail", "u32", R),
                ("ctr", A32, RW),
            ],
        )?,
        mk(
            "settle.wgsl",
            "main",
            &[
                ("cell", "Cell", RW),
                ("plan", "Plan", R),
                ("acc", "Acc", RW),
                ("mail", "u32", RW),
                ("fa0", "vec4i", R),
                ("dep", A32, RW),
                ("terrain", "u32", R),
                ("bits", A32, RW),
                ("ctr", A32, RW),
            ],
        )?,
        mk("rain.wgsl", "main", &[("ctl", "u32", R), ("bits", A32, RW)])?,
        mk("alloc.wgsl", "count", alloc)?,
        mk("alloc.wgsl", "prefix", alloc)?,
        mk("alloc.wgsl", "give_back", alloc)?,
        mk("alloc.wgsl", "birth", alloc)?,
        mk(
            "genes.wgsl",
            "main",
            &[
                ("jobs", "vec4u", R),
                ("genome", "vec4u", RW),
                ("sketch", "vec4u", RW),
                ("cell", "Cell", R),
                ("ctl", "u32", R),
            ],
        )?,
        mk(
            "fields.wgsl",
            "main",
            &[
                ("terrain", "u32", R),
                ("fa0", "vec4i", R),
                ("fa1", "vec4i", RW),
                ("fb0", "vec4i", R),
                ("fb1", "vec4i", RW),
                ("dep", "u32", RW),
                ("ctr", A32, RW),
            ],
        )?,
        mk("census.wgsl", "squares", census)?,
        mk("census.wgsl", "slots", census)?,
    ])
}

fn build_pipe(
    gpu: &Gpu,
    cfg: &Config,
    buffers: &HashMap<&'static str, wgpu::Buffer>,
    frames: &wgpu::Buffer,
    file: &str,
    entry: &str,
    binds: &[Bind],
) -> Result<Pipe, String> {
    let body = cfg.source(file)?;
    build_pipe_source(gpu, cfg, buffers, frames, file, &body, entry, binds, None)
}

#[allow(clippy::too_many_arguments)]
fn build_pipe_source(
    gpu: &Gpu,
    cfg: &Config,
    buffers: &HashMap<&'static str, wgpu::Buffer>,
    frames: &wgpu::Buffer,
    file: &str,
    body: &str,
    entry: &str,
    binds: &[Bind],
    group1: Option<&wgpu::BindGroupLayout>,
) -> Result<Pipe, String> {
    let device = &gpu.device;
    let body = body.to_string();
    // A world with two genome buffers binds the second wherever the first is bound.
    let mut expanded: Vec<Bind> = binds.to_vec();
    let genome_access = binds.iter().find(|(n, _, _)| *n == "genome").map(|b| b.2);
    if let Some(access) = genome_access
        && cfg.two_shards()
    {
        expanded.push(("genome1", "vec4u", access));
    }
    let binds = &expanded[..];
    let mut src = cfg.header();
    src += WGSL_STRUCTS;
    src += "@group(0) @binding(0) var<uniform> F: Frame;\n";
    let mut entries = vec![wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: wgpu::BufferSize::new(FRAME_BYTES as u64),
        },
        count: None,
    }];
    for (i, (name, elem, access)) in binds.iter().enumerate() {
        let mode = if *access == R { "read" } else { "read_write" };
        src += &format!(
            "@group(0) @binding({}) var<storage, {mode}> {name}: array<{elem}>;\n",
            i + 1
        );
        entries.push(storage_entry(i as u32 + 1, *access == R));
    }
    src += &cfg.source("common.wgsl")?;
    if let Some(access) = genome_access {
        src += if cfg.two_shards() {
            "fn gene(i: u32) -> vec4u { if (i < SHARD_BLOCKS) { return genome[i]; } return genome1[i - SHARD_BLOCKS]; }\n"
        } else {
            "fn gene(i: u32) -> vec4u { return genome[i]; }\n"
        };
        if access == RW {
            src += if cfg.two_shards() {
                "fn set_gene(i: u32, g: vec4u) { if (i < SHARD_BLOCKS) { genome[i] = g; } else { genome1[i - SHARD_BLOCKS] = g; } }\n"
            } else {
                "fn set_gene(i: u32, g: vec4u) { genome[i] = g; }\n"
            };
        }
    }
    let mut body = body;
    if body.contains("//@BRAIN") {
        let (brain, _) = brain_wgsl_for(cfg.hidden_rows, IN_BLOCKS, OUT_ROWS, |_| true, |_| true);
        body = body.replace("//@BRAIN", &brain);
    }
    if body.contains("//@NOISE") {
        body = body.replace("//@NOISE", &cfg.source("noise.wgsl")?);
    }
    src += &body;
    let label = format!("soft {file}:{entry}");
    let module = gpu.shader(&label, &src);
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&label),
        entries: &entries,
    });
    let mut layouts = vec![Some(&layout)];
    if let Some(extra) = group1 {
        layouts.push(Some(extra));
    }
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = |flip: usize| {
        let mut group_entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: frames,
                offset: 0,
                size: wgpu::BufferSize::new(FRAME_BYTES as u64),
            }),
        }];
        for (i, (name, _, _)) in binds.iter().enumerate() {
            group_entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource: buffers[buffer_for(name, flip)].as_entire_binding(),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &layout,
            entries: &group_entries,
        })
    };
    Ok(Pipe {
        pipeline,
        groups: [group(0), group(1)],
    })
}

/// Reads a saved world's living cells, as a dump, with no GPU.
pub fn read_saved_cells(path: &str) -> Result<crate::dump::Dump, String> {
    let mut s = SavedWorld::open(path)?;
    let head = auto_life_gpu::config::read_save_head(path)?;
    let to_sq = 1.0 / (1u64 << (32 - (s.size as u32).trailing_zeros())) as f32;
    let mut cells = vec![];
    for k in 0..s.slots {
        let c = s.cell(k);
        if c.flags & F_ALIVE == 0 {
            continue;
        }
        let plan: Plan = bytemuck::pod_read_unaligned(&s.read("plan", k, PLAN_BYTES as usize)?);
        let sketch: [u32; 4] = bytemuck::pod_read_unaligned(&s.read("sketch", k, SKETCH_BYTES as usize)?);
        cells.push(crate::dump::from_records(&c, &plan, sketch, to_sq));
    }
    Ok(crate::dump::Dump {
        tick: head["tick"].as_u64().unwrap_or(0),
        size: s.size as u32,
        physics: head["physics"].clone(),
        cells,
    })
}

impl World {
    /// A pipeline over the world's buffers, for something that only looks at them. Returns a bind
    /// group for each way round the field buffers can be.
    pub fn extra_pipe(
        &self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(&'static str, &'static str, Access)],
        group1: Option<&wgpu::BindGroupLayout>,
    ) -> Result<(wgpu::ComputePipeline, [wgpu::BindGroup; 2]), String> {
        let p = build_pipe_source(&self.gpu, &self.cfg, &self.buffers, &self.frames, label, source, entry, binds, group1)?;
        Ok((p.pipeline, p.groups))
    }

    /// Which of the two field buffers is current. Changes with every tick queued.
    pub fn flip(&self) -> usize {
        self.flip
    }

    /// A field's top of range, from a sample of squares: near the highest, so dense places still
    /// show detail, but not thrown off by one hot square. Food, meat, heat, enzyme, then scents.
    pub fn field_tops(&self) -> [f32; 8] {
        let size = self.cfg.size as u64;
        let step = (size / 32).max(1);
        let ranges: Vec<(u64, u64)> = (0..size.min(32)).map(|k| (k * step * size * 16, size * 16)).collect();
        let (fa, fb) = if self.flip == 0 { ("fa_a", "fb_a") } else { ("fa_b", "fb_b") };
        let mut out = [1e-3f32; 8];
        for (layer, name) in [fa, fb].iter().enumerate() {
            let bytes = self.read_ranges(name, &ranges);
            let values: &[[i32; 4]] = bytemuck::cast_slice(&bytes);
            for k in 0..4 {
                let mut v: Vec<i32> = values.iter().map(|s| s[k]).filter(|x| *x > 0).collect();
                if v.is_empty() {
                    continue;
                }
                v.sort_unstable();
                let top = v[((v.len() - 1) as f32 * 0.995) as usize];
                out[layer * 4 + k] = (top as f32 / E_ONE).max(1e-3);
            }
        }
        out
    }
}
