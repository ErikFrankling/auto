//! The world on the GPU: buffers, passes and a scheduler that never holds the
//! card for long, because the same card draws Erik's desktop.
//!
//! Each square has at most one cell, and cells run on a 4x4 lattice, so cells that
//! run at the same time can't touch the same data. There are no locks, and a run is
//! the same every time as long as free slots remain. When they run out, which births
//! get the last ones depends on timing.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use auto_core::{Gpu, GpuOptions};
use bytemuck::{Pod, Zeroable};

use crate::config::{CELL_BYTES, CELL_WORDS, Config, STAT_BYTES, check_layout, layout_json};

const RING_ENTRIES: usize = 4096;
const WGSL_STRUCTS: &str = r#"
struct Cell {
    e: f32, womb: f32, age: f32, hit: f32,
    hard: f32, wound: f32, skill_g: f32, skill_f: f32,
    voice: vec2f, pos: u32, flags: u32,
    clade: u32, gen: u32, rgb: u32, stamp: u32,
    feel_u: f32, feel_d: f32, mate_near: f32, mate_kids: f32,
    x0: f32, x1: f32, x2: f32, x3: f32,
}
struct Stat {
    diet: vec4f,
    b0: vec4f, b1: vec4f, b2: vec4f,
    chase: vec4f,
    follow: vec4f,
    misc: vec4f,
    b3: vec4f,
    measures: vec4f,
}
struct Frame {
    tick: u32, px: u32, py: u32, row0: u32,
    axis: u32, off: u32, season: f32, sun_k: f32,
    seed: u32, a: u32, b: u32, c: u32,
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct Frame {
    pub tick: u32,
    pub px: u32,
    pub py: u32,
    pub row0: u32,
    pub axis: u32,
    pub off: u32,
    pub season: f32,
    pub sun_k: f32,
    pub seed: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Copy, PartialEq)]
enum Access {
    R,
    RW,
}
use Access::{R, RW};

struct Pipe {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
}

#[derive(Clone, Copy)]
enum Op {
    Dispatch {
        pipe: usize,
        frame: Frame,
        groups: [u32; 3],
    },
    Indirect {
        pipe: usize,
        frame: Frame,
    },
    CopySeen {
        offset: u64,
        size: u64,
    },
}

const P_START: usize = 0;
const P_FINISH: usize = 1;
const P_RAIN: usize = 2;
const P_ACT: usize = 3;
const P_BIRTHS: usize = 4;
const P_SWEEP: usize = 5;

pub struct Pacing {
    /// Aim for submissions about this long, so the desktop never waits for more.
    pub target_ms: f64,
    /// Share of time the GPU may be busy with this job.
    pub duty: f64,
}

pub struct World {
    pub gpu: Gpu,
    pub cfg: Config,
    pub tick: u64,
    pub sun_k: f32,
    pub open: u64,
    pub start_ground: f64,
    pub pacing: Pacing,
    /// Longest and total time the GPU spent on a submission.
    pub longest_submit_ms: f64,
    /// Submissions over the limit since the last report. The first few of a run don't count:
    /// they include memory being paged in, and it's sustained holds that make the desktop stutter.
    pub slow_submits: u32,
    pub gpu_seconds: f64,
    pub submits: u64,
    pub buffers: HashMap<&'static str, wgpu::Buffer>,
    elems: HashMap<&'static str, &'static str>,
    pipes: Vec<Pipe>,
    frames: wgpu::Buffer,
    stride: usize,
    pending: Vec<Op>,
    // Cheap and expensive work never share a submission on a big world, so each kind
    // has its own size. Small worlds use one size and batch many ticks together.
    ops_per_submit: [usize; 2],
    pending_kind: usize,
    act_splits: u32,
    sweep_splits: u32,
    rng: u64,
    pub bake_pipe: wgpu::ComputePipeline,
    pub bake_layout: wgpu::BindGroupLayout,
    bake_group0: wgpu::BindGroup,
    pub census_pipe: wgpu::ComputePipeline,
    census_group0: wgpu::BindGroup,
}

fn next_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
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

impl World {
    pub fn new(cfg: Config, pacing: Pacing) -> Result<World, String> {
        let adapter_name = std::env::var("AUTO_ADAPTER").ok();
        let gpu = pollster::block_on(Gpu::headless(GpuOptions {
            adapter_name,
            ..Default::default()
        }))?;
        let device = &gpu.device;
        let squares = cfg.size as u64 * cfg.size as u64;
        let slots = cfg.slots as u64;
        let limit = device
            .limits()
            .max_storage_buffer_binding_size
            .min(device.limits().max_buffer_size);
        let blocks = cfg.blocks() as u64;
        let tiles = (cfg.size / cfg.tile()) as u64;
        let ring_cap = cfg.slots.next_power_of_two() as u64;
        let copy = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let table: Vec<(&'static str, &'static str, u64, wgpu::BufferUsages)> = vec![
            ("occ", "u32", squares * 4, copy),
            ("seen", "u32", squares * 4, copy),
            ("seen_next", "u32", squares * 4, copy),
            ("terrain", "u32", squares * 4, copy),
            ("fa", "vec4f", squares * 16, copy),
            ("fb", "vec4f", squares * 16, copy),
            ("fc", "vec4f", squares * 16, copy),
            ("fd", "vec4f", squares * 16, copy),
            ("cell", "Cell", slots * CELL_BYTES, copy),
            ("stat", "Stat", slots * STAT_BYTES, copy),
            ("sketch", "vec4u", slots * 16, copy),
            ("mem", "vec4f", slots * cfg.hidden_rows as u64 * 16, copy),
            (
                "genome",
                "vec4u",
                slots.min(cfg.shard_slots as u64) * blocks * 16,
                copy,
            ),
            (
                "genome1",
                "vec4u",
                (slots.saturating_sub(cfg.shard_slots as u64) * blocks * 16).max(16),
                copy,
            ),
            ("ring", "u32", ring_cap * 4, copy),
            ("jobs", "vec4u", slots * 16, copy),
            ("ctr", "atomic<u32>", 32 * 4, copy),
            ("ctl", "u32", 8 * 4, copy),
            ("args", "u32", 3 * 4, copy | wgpu::BufferUsages::INDIRECT),
            ("tiles", "vec4f", (tiles * tiles * 3 * 16).max(16), copy),
        ];
        let mut buffers = HashMap::new();
        let mut elems = HashMap::new();
        for (name, elem, bytes, usage) in table {
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
            elems.insert(name, elem);
        }
        let align = device.limits().min_uniform_buffer_offset_alignment as usize;
        let stride = 48usize.next_multiple_of(align);
        let frames = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frames"),
            size: (stride * RING_ENTRIES) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mk =
            |file: &str, entry: &str, binds: &[(&'static str, Access)]| -> Result<Pipe, String> {
                let (pipeline, group, _) = build_pipe(
                    &gpu, &cfg, &buffers, &elems, &frames, file, entry, binds, None,
                )?;
                Ok(Pipe { pipeline, group })
            };
        let tick_binds: &[(&'static str, Access)] = &[("ctr", RW), ("ctl", RW), ("args", RW)];
        let pipes = vec![
            mk("tick.wgsl", "start", tick_binds)?,
            mk("tick.wgsl", "finish", tick_binds)?,
            mk(
                "rain.wgsl",
                "main",
                &[
                    ("occ", RW),
                    ("seen_next", RW),
                    ("cell", RW),
                    ("stat", RW),
                    ("mem", RW),
                    ("ring", R),
                    ("ctr", RW),
                    ("ctl", R),
                    ("jobs", RW),
                ],
            )?,
            mk(
                "act.wgsl",
                "main",
                &[
                    ("occ", RW),
                    ("seen", R),
                    ("seen_next", RW),
                    ("terrain", R),
                    ("fa", RW),
                    ("fb", RW),
                    ("fc", RW),
                    ("fd", RW),
                    ("cell", RW),
                    ("stat", RW),
                    ("sketch", R),
                    ("mem", RW),
                    ("genome", R),
                    ("ring", RW),
                    ("ctr", RW),
                    ("ctl", R),
                    ("jobs", RW),
                ],
            )?,
            mk(
                "births.wgsl",
                "main",
                &[
                    ("jobs", R),
                    ("genome", RW),
                    ("sketch", RW),
                    ("ctr", RW),
                    ("ctl", R),
                ],
            )?,
            mk(
                "sweep.wgsl",
                "main",
                &[
                    ("terrain", R),
                    ("fa", RW),
                    ("fb", RW),
                    ("fc", RW),
                    ("fd", RW),
                    ("ctr", RW),
                ],
            )?,
        ];
        let bake_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bake uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let (bake_pipe, bake_group0, _) = build_pipe(
            &gpu,
            &cfg,
            &buffers,
            &elems,
            &frames,
            "bake.wgsl",
            "main",
            &[("cell", RW), ("sketch", R), ("ctr", RW)],
            Some(&bake_layout),
        )?;
        let (census_pipe, census_group0, _) = build_pipe(
            &gpu,
            &cfg,
            &buffers,
            &elems,
            &frames,
            "census.wgsl",
            "main",
            &[
                ("occ", R),
                ("cell", R),
                ("stat", R),
                ("tiles", RW),
                ("ctr", RW),
            ],
            None,
        )?;

        let rng = cfg.seed as u64 ^ 0xA5A5_5A5A_1234_5678;
        let size_now = cfg.size;
        let mut w = World {
            gpu,
            sun_k: 0.0,
            open: 0,
            start_ground: 0.0,
            tick: 0,
            pacing,
            longest_submit_ms: 0.0,
            slow_submits: 0,
            gpu_seconds: 0.0,
            submits: 0,
            buffers,
            elems,
            pipes,
            frames,
            stride,
            pending: vec![],
            // Start cautious on big worlds. The pacing loop finds the right size from here.
            ops_per_submit: if size_now >= 1024 { [1, 1] } else { [8, 8] },
            pending_kind: 0,
            act_splits: (size_now / 4096).max(1),
            sweep_splits: (size_now / 1024).max(1),
            rng,
            bake_pipe,
            bake_layout,
            bake_group0,
            census_pipe,
            census_group0,
            cfg,
        };
        w.build_terrain()?;
        Ok(w)
    }

    /// A pipeline over the world's buffers. Used here and by the video recorder.
    #[allow(clippy::type_complexity)]
    pub fn extra_pipe(
        &self,
        file: &str,
        entry: &str,
        binds: &[(&'static str, Access2)],
        group1: Option<&wgpu::BindGroupLayout>,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroup), String> {
        let binds: Vec<(&'static str, Access)> = binds
            .iter()
            .map(|(n, a)| (*n, if matches!(a, Access2::Read) { R } else { RW }))
            .collect();
        let (p, g, _) = build_pipe(
            &self.gpu,
            &self.cfg,
            &self.buffers,
            &self.elems,
            &self.frames,
            file,
            entry,
            &binds,
            group1,
        )?;
        Ok((p, g))
    }

    fn build_terrain(&mut self) -> Result<(), String> {
        let binds: &[(&'static str, Access)] = &[
            ("terrain", RW),
            ("occ", RW),
            ("seen", RW),
            ("seen_next", RW),
            ("fa", RW),
            ("fb", RW),
            ("fc", RW),
            ("fd", RW),
            ("ring", RW),
            ("cell", RW),
            ("ctr", RW),
        ];
        let (terrain, group, _) = build_pipe(
            &self.gpu,
            &self.cfg,
            &self.buffers,
            &self.elems,
            &self.frames,
            "terrain.wgsl",
            "main",
            binds,
            None,
        )?;
        let (fill, fill_group, _) = build_pipe(
            &self.gpu,
            &self.cfg,
            &self.buffers,
            &self.elems,
            &self.frames,
            "terrain.wgsl",
            "fill_ring",
            binds,
            None,
        )?;
        let w = self.cfg.size;
        let band = 256.min(w);
        let mut row = 0;
        while row < w {
            let frame = Frame {
                row0: row,
                seed: self.cfg.seed,
                ..Default::default()
            };
            let mut bytes = vec![0u8; self.stride];
            bytes[..48].copy_from_slice(bytemuck::bytes_of(&frame));
            self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&terrain);
                pass.set_bind_group(0, &group, &[0]);
                pass.dispatch_workgroups(w.div_ceil(64), band.min(w - row), 1);
            }
            self.gpu.queue.submit([enc.finish()]);
            self.gpu.wait();
            row += band;
        }
        let frame = Frame {
            seed: self.cfg.seed,
            ..Default::default()
        };
        let mut bytes = vec![0u8; self.stride];
        bytes[..48].copy_from_slice(bytemuck::bytes_of(&frame));
        self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
        // Slots start free, in order.
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&fill);
            pass.set_bind_group(0, &fill_group, &[0]);
            pass.dispatch_workgroups(1024, self.cfg.slots.div_ceil(65536), 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.wait();

        // The ring holds every slot, so head starts one lap ahead of tail.
        let mut ctr = [0u32; 32];
        ctr[0] = self.cfg.slots;
        self.gpu
            .queue
            .write_buffer(&self.buffers["ctr"], 0, bytemuck::cast_slice(&ctr));

        // Sample the terrain to scale the sun so its mean is what was asked for.
        let (mut base, mut open, mut seen) = (0.0f64, 0u64, 0u64);
        let rows_wanted = 256.min(w);
        let step = (w / rows_wanted).max(1);
        let ranges: Vec<(u64, u64)> = (0..rows_wanted)
            .map(|k| ((k * step) as u64 * w as u64 * 4, w as u64 * 4))
            .collect();
        let bytes = self.read_ranges("terrain", &ranges);
        let mut ground_sum = 0.0f64;
        for v in bytemuck::cast_slice::<u8, u32>(&bytes) {
            seen += 1;
            if v & (1 << 24) == 0 {
                open += 1;
                base += (v & 0xFFFF) as f64 / 65535.0 * crate::config::SUN_RANGE as f64;
                let b = (v & 0xFFFF) as f32 / 65535.0 * crate::config::SUN_RANGE;
                ground_sum += self
                    .cfg
                    .f("ground_cap")
                    .min(b * 2.0 * self.cfg.f("sun") / 0.05) as f64;
            }
        }
        let squares = w as f64 * w as f64;
        let open_frac = open as f64 / seen.max(1) as f64;
        self.open = (squares * open_frac) as u64;
        self.start_ground = ground_sum * squares / seen.max(1) as f64;
        let mean_base = base / open.max(1) as f64;
        let season_mean = if self.cfg.f("season_ticks") > 0.0 {
            0.55
        } else {
            1.0
        };
        self.sun_k = (self.cfg.f("sun") as f64 / (mean_base * season_mean).max(1e-9)) as f32;
        let ctl = [
            self.cfg.slots,
            0,
            0,
            self.open.min(u32::MAX as u64) as u32,
            0,
            0,
            0,
            0,
        ];
        self.gpu
            .queue
            .write_buffer(&self.buffers["ctl"], 0, bytemuck::cast_slice(&ctl));
        self.gpu.queue.submit([]);
        self.gpu.wait();
        Ok(())
    }

    fn base_frame(&self) -> Frame {
        let season_ticks = self.cfg.f("season_ticks");
        let season = if season_ticks > 0.0 {
            (self.tick as f64 / season_ticks as f64 * std::f64::consts::TAU)
                .rem_euclid(std::f64::consts::TAU) as f32
        } else {
            0.0
        };
        Frame {
            tick: self.tick as u32,
            seed: self.cfg.seed,
            season,
            sun_k: self.sun_k,
            ..Default::default()
        }
    }

    /// Queues one tick. Work reaches the GPU in short submissions.
    pub fn step(&mut self) {
        self.tick += 1;
        let base = self.base_frame();
        let size = self.cfg.size;
        self.push(Op::Dispatch {
            pipe: P_START,
            frame: base,
            groups: [1, 1, 1],
        });
        let tiles = size / self.cfg.tile();
        self.push(Op::Dispatch {
            pipe: P_RAIN,
            frame: base,
            groups: [tiles.div_ceil(8), tiles.div_ceil(8), 1],
        });

        // A new order every tick, so no part of the lattice always goes first.
        let mut order: [u32; 16] = std::array::from_fn(|i| i as u32);
        for i in (1..16).rev() {
            let j = (next_u64(&mut self.rng) % (i as u64 + 1)) as usize;
            order.swap(i, j);
        }
        let lattice_rows = size / 4;
        let act_groups_x = lattice_rows.div_ceil(32);
        for phase in order {
            for (row0, rows) in bands(lattice_rows, self.act_splits) {
                let frame = Frame {
                    px: phase % 4,
                    py: phase / 4,
                    row0,
                    ..base
                };
                self.push(Op::Dispatch {
                    pipe: P_ACT,
                    frame,
                    groups: [act_groups_x, rows, 1],
                });
            }
        }
        self.push(Op::Dispatch {
            pipe: P_FINISH,
            frame: base,
            groups: [1, 1, 1],
        });
        self.push(Op::Indirect {
            pipe: P_BIRTHS,
            frame: base,
        });

        let total = size as u64 * size as u64 * 4;
        let splits = self.sweep_splits as u64;
        for k in 0..splits {
            let offset = total * k / splits;
            let end = total * (k + 1) / splits;
            self.push(Op::CopySeen {
                offset,
                size: end - offset,
            });
        }

        let off = (self.tick & 1) as u32;
        for (row0, rows) in bands(size, self.sweep_splits) {
            let frame = Frame {
                axis: 0,
                off,
                row0,
                ..base
            };
            self.push(Op::Dispatch {
                pipe: P_SWEEP,
                frame,
                groups: [(size / 2).div_ceil(64), rows, 1],
            });
        }
        for (row0, rows) in bands(size / 2, self.sweep_splits) {
            let frame = Frame {
                axis: 1,
                off,
                row0,
                ..base
            };
            self.push(Op::Dispatch {
                pipe: P_SWEEP,
                frame,
                groups: [size.div_ceil(64), rows, 1],
            });
        }
    }

    fn push(&mut self, op: Op) {
        // Sweeps and the snapshot copy touch every square. Cell passes touch only cells.
        let kind = match op {
            _ if self.cfg.size < 2048 => 0,
            Op::CopySeen { .. } => 1,
            Op::Dispatch { pipe, .. } if pipe == P_SWEEP => 1,
            _ => 0,
        };
        if kind != self.pending_kind {
            self.flush();
            self.pending_kind = kind;
        }
        self.pending.push(op);
        if self.pending.len() >= self.ops_per_submit[kind] {
            self.flush();
        }
    }

    /// Sends everything queued, waits for it, then adjusts the size of the next
    /// submission so each stays near the target length.
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
                    bytes[slot * self.stride..slot * self.stride + 48]
                        .copy_from_slice(bytemuck::bytes_of(frame));
                    offsets.push((slot * self.stride) as u32);
                    slot += 1;
                }
                Op::CopySeen { .. } => offsets.push(0),
            }
        }
        self.gpu
            .queue
            .write_buffer(&self.frames, 0, &bytes[..slot.max(1) * self.stride]);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        let mut i = 0;
        while i < ops.len() {
            if let Op::CopySeen { offset, size } = ops[i] {
                enc.copy_buffer_to_buffer(
                    &self.buffers["seen_next"],
                    offset,
                    &self.buffers["seen"],
                    offset,
                    size,
                );
                i += 1;
                continue;
            }
            let mut pass = enc.begin_compute_pass(&Default::default());
            let mut current = usize::MAX;
            while i < ops.len() {
                match ops[i] {
                    Op::Dispatch { pipe, groups, .. } => {
                        if current != pipe {
                            pass.set_pipeline(&self.pipes[pipe].pipeline);
                            pass.set_bind_group(0, &self.pipes[pipe].group, &[offsets[i]]);
                            current = pipe;
                        } else {
                            pass.set_bind_group(0, &self.pipes[pipe].group, &[offsets[i]]);
                        }
                        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
                    }
                    Op::Indirect { pipe, .. } => {
                        pass.set_pipeline(&self.pipes[pipe].pipeline);
                        current = pipe;
                        pass.set_bind_group(0, &self.pipes[pipe].group, &[offsets[i]]);
                        pass.dispatch_workgroups_indirect(&self.buffers["args"], 0);
                    }
                    Op::CopySeen { .. } => break,
                }
                i += 1;
            }
        }
        let started = Instant::now();
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.wait();
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        self.submits += 1;
        self.gpu_seconds += ms / 1000.0;
        if self.submits > 32 {
            self.longest_submit_ms = self.longest_submit_ms.max(ms);
            if ms > 80.0 {
                self.slow_submits += 1;
            }
        }

        let target = self.pacing.target_ms;
        let kind = self.pending_kind;
        if ms > target * 1.25 {
            if self.ops_per_submit[kind] > 1 {
                self.ops_per_submit[kind] = (self.ops_per_submit[kind] / 2).max(1);
            } else if kind == 0 {
                self.act_splits = (self.act_splits * 2).min(self.cfg.size / 4);
            } else {
                self.sweep_splits = (self.sweep_splits * 2).min(self.cfg.size / 2);
            }
        } else if ms < target * 0.4 {
            self.ops_per_submit[kind] = (self.ops_per_submit[kind] * 2).min(RING_ENTRIES / 2);
        }
        if self.pacing.duty < 1.0 {
            let rest = ms * (1.0 / self.pacing.duty.max(0.05) - 1.0);
            std::thread::sleep(Duration::from_secs_f64(rest / 1000.0));
        }
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
        self.gpu.wait();
        let out = staging.get_mapped_range(..).expect("mapped").to_vec();
        staging.unmap();
        out
    }

    pub fn counters(&self) -> [u32; 32] {
        let b = self.read_ranges("ctr", &[(0, 128)]);
        let mut out = [0u32; 32];
        out.copy_from_slice(bytemuck::cast_slice(&b));
        out
    }

    pub fn alive(&self) -> u32 {
        let c = self.counters();
        // Failed births may have pushed tail past what was free. Those took nothing.
        let ctl = self.read_ranges("ctl", &[(0, 4)]);
        let avail = u32::from_le_bytes(ctl[..4].try_into().unwrap());
        let tail = if (c[1].wrapping_sub(avail) as i32) > 0 {
            avail
        } else {
            c[1]
        };
        self.cfg.slots - c[0].wrapping_sub(tail)
    }

    /// Runs the census pass over all tiles, in short submissions. Colours must be baked first.
    pub fn census(&mut self) -> Vec<[f32; 12]> {
        self.flush();
        let tiles = self.cfg.size / self.cfg.tile();
        let band = (tiles / 8).max(1);
        let mut row = 0;
        while row < tiles {
            let frame = Frame {
                row0: row,
                ..self.base_frame()
            };
            let mut bytes = vec![0u8; self.stride];
            bytes[..48].copy_from_slice(bytemuck::bytes_of(&frame));
            self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.census_pipe);
                pass.set_bind_group(0, &self.census_group0, &[0]);
                pass.dispatch_workgroups(tiles.div_ceil(16), band.min(tiles - row), 1);
            }
            self.gpu.queue.submit([enc.finish()]);
            self.gpu.wait();
            row += band;
        }
        let n = tiles as u64 * tiles as u64 * 48;
        let b = self.read_ranges("tiles", &[(0, n)]);
        bytemuck::cast_slice::<u8, f32>(&b)
            .chunks_exact(12)
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    /// Bakes colours with the given uniform bytes.
    pub fn bake(&mut self, uniform: &wgpu::Buffer) {
        self.flush();
        let group1 = self
            .gpu
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bake"),
                layout: &self.bake_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                }],
            });
        let frame = self.base_frame();
        let mut bytes = vec![0u8; self.stride];
        bytes[..48].copy_from_slice(bytemuck::bytes_of(&frame));
        self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.bake_pipe);
            pass.set_bind_group(0, &self.bake_group0, &[0]);
            pass.set_bind_group(1, &group1, &[]);
            pass.dispatch_workgroups(1024, self.cfg.slots.div_ceil(65536), 1);
        }
        self.gpu.queue.submit([enc.finish()]);
        self.gpu.wait();
    }

    /// Writes one frame uniform and returns nothing else. For the recorder's own passes.
    pub fn write_frame(&self, frame: Frame) {
        let mut bytes = vec![0u8; self.stride];
        bytes[..48].copy_from_slice(bytemuck::bytes_of(&frame));
        self.gpu.queue.write_buffer(&self.frames, 0, &bytes);
    }

    pub fn current_frame(&self) -> Frame {
        self.base_frame()
    }

    /// Writes every buffer that defines the world to a file, in chunks, so RAM stays free.
    pub fn save(&mut self, path: &str) -> Result<(), String> {
        use std::io::Write;
        self.flush();
        let mut f =
            std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
        let head = serde_json::json!({
            "size": self.cfg.size, "slots": self.cfg.slots, "shard_slots": self.cfg.shard_slots, "hidden_rows": self.cfg.hidden_rows,
            "seed": self.cfg.seed, "tick": self.tick, "rng": self.rng, "sun_k": self.sun_k, "open": self.open,
            "start_ground": self.start_ground, "physics": self.cfg.physics, "layout": layout_json(),
        })
        .to_string();
        f.write_all(&(head.len() as u64).to_le_bytes())
            .map_err(|e| e.to_string())?;
        f.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
        for name in self.saved_names() {
            let size = self.buffers[name].size();
            let mut at = 0;
            while at < size {
                let n = (size - at).min(64 << 20);
                let bytes = self.read_ranges(name, &[(at, n)]);
                f.write_all(&bytes).map_err(|e| e.to_string())?;
                at += n;
            }
        }
        Ok(())
    }

    /// Reads a saved world's state. The world must have the same shape.
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(|e| e.to_string())?);
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(|e| e.to_string())?;
        let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
        f.read_exact(&mut head).map_err(|e| e.to_string())?;
        let head: serde_json::Value = serde_json::from_slice(&head).map_err(|e| e.to_string())?;
        check_layout(&head, path)?;
        let shard = head["shard_slots"]
            .as_u64()
            .unwrap_or(self.cfg.max_shard_slots() as u64);
        if shard != self.cfg.shard_slots as u64
            || head["size"] != self.cfg.size
            || head["slots"] != self.cfg.slots
            || head["hidden_rows"] != self.cfg.hidden_rows
        {
            return Err(format!(
                "{path} is a world of another shape: {} squares wide, {} slots",
                head["size"], head["slots"]
            ));
        }
        self.tick = head["tick"].as_u64().unwrap_or(0);
        self.rng = head["rng"].as_u64().unwrap_or(0);
        self.sun_k = head["sun_k"].as_f64().unwrap_or(0.0) as f32;
        self.open = head["open"].as_u64().unwrap_or(0);
        self.start_ground = head["start_ground"].as_f64().unwrap_or(0.0);
        let mut chunk = vec![0u8; 64 << 20];
        for name in self.saved_names() {
            let size = self.buffers[name].size();
            let mut at = 0;
            while at < size {
                let n = (size - at).min(64 << 20);
                f.read_exact(&mut chunk[..n as usize])
                    .map_err(|e| e.to_string())?;
                self.gpu
                    .queue
                    .write_buffer(&self.buffers[name], at, &chunk[..n as usize]);
                self.gpu.queue.submit([]);
                self.gpu.wait();
                at += n;
            }
        }
        Ok(())
    }
}

const SAVED: [&str; 17] = [
    "occ",
    "seen",
    "seen_next",
    "terrain",
    "fa",
    "fb",
    "fc",
    "fd",
    "cell",
    "stat",
    "sketch",
    "mem",
    "genome",
    "ring",
    "jobs",
    "ctr",
    "ctl",
];

pub enum Access2 {
    Read,
    ReadWrite,
}

fn bands(total: u32, splits: u32) -> Vec<(u32, u32)> {
    let splits = splits.clamp(1, total.max(1));
    (0..splits)
        .map(|k| {
            let a = total as u64 * k as u64 / splits as u64;
            let b = total as u64 * (k as u64 + 1) / splits as u64;
            (a as u32, (b - a) as u32)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_pipe(
    gpu: &Gpu,
    cfg: &Config,
    buffers: &HashMap<&'static str, wgpu::Buffer>,
    elems: &HashMap<&'static str, &'static str>,
    frames: &wgpu::Buffer,
    file: &str,
    entry: &str,
    binds: &[(&'static str, Access)],
    group1: Option<&wgpu::BindGroupLayout>,
) -> Result<
    (
        wgpu::ComputePipeline,
        wgpu::BindGroup,
        wgpu::BindGroupLayout,
    ),
    String,
> {
    let device = &gpu.device;
    // A world with two genome buffers binds the second wherever the first is bound.
    let mut expanded: Vec<(&'static str, Access)> = binds.to_vec();
    let genome_access = binds.iter().find(|(n, _)| *n == "genome").map(|(_, a)| *a);
    if let Some(access) = genome_access
        && cfg.two_shards()
    {
        expanded.push(("genome1", access));
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
            min_binding_size: wgpu::BufferSize::new(48),
        },
        count: None,
    }];
    for (i, (name, access)) in binds.iter().enumerate() {
        let mode = if *access == R { "read" } else { "read_write" };
        src += &format!(
            "@group(0) @binding({}) var<storage, {mode}> {name}: array<{}>;\n",
            i + 1,
            elems[name]
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
    let mut body = cfg.source(file)?;
    if body.contains("//@BRAIN") {
        let (brain, copy) = cfg.brain_wgsl();
        body = body
            .replace("//@BRAIN", &brain)
            .replace("//@MEMCOPY", &copy);
    }
    src += &body;
    let label = format!("{file}:{entry}");
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
    let mut group_entries = vec![wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: frames,
            offset: 0,
            size: wgpu::BufferSize::new(48),
        }),
    }];
    for (i, (name, _)) in binds.iter().enumerate() {
        group_entries.push(wgpu::BindGroupEntry {
            binding: i as u32 + 1,
            resource: buffers[name].as_entire_binding(),
        });
    }
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&label),
        layout: &layout,
        entries: &group_entries,
    });
    Ok((pipeline, group, layout))
}

/// Test and tooling access to a world's state.
impl World {
    /// Puts a cell on the grid by hand. Slots are handed out in order, so only use this on a fresh world.
    pub fn plant(&mut self, x: u32, y: u32, e: f32, womb: f32, dir: u32, genome: &[u8]) -> u32 {
        self.flush();
        let c = self.counters();
        let slot = c[1];
        let p = ((y << self.cfg.size.trailing_zeros()) | x) as u64;
        let q = &self.gpu.queue;
        q.write_buffer(&self.buffers["ctr"], 4, bytemuck::bytes_of(&(slot + 1)));
        let mut cell = [0u32; CELL_WORDS];
        cell[0] = e.to_bits();
        cell[1] = womb.to_bits();
        cell[10] = p as u32;
        cell[11] = dir;
        cell[12] = slot;
        q.write_buffer(
            &self.buffers["cell"],
            slot as u64 * CELL_BYTES,
            bytemuck::cast_slice(&cell),
        );
        q.write_buffer(&self.buffers["occ"], p * 4, bytemuck::bytes_of(&(slot + 1)));
        let word = 0x80u32 | dir;
        q.write_buffer(&self.buffers["seen"], p * 4, bytemuck::bytes_of(&word));
        q.write_buffer(&self.buffers["seen_next"], p * 4, bytemuck::bytes_of(&word));
        let g = self.cfg.blocks() as u64 * 16;
        assert_eq!(genome.len() as u64, g);
        self.write_genomes(slot as u64, genome);
        q.submit([]);
        self.gpu.wait();
        slot
    }

    pub fn write_field(&mut self, name: &str, x: u32, y: u32, v: [f32; 4]) {
        let p = ((y << self.cfg.size.trailing_zeros()) | x) as u64;
        self.gpu
            .queue
            .write_buffer(&self.buffers[name], p * 16, bytemuck::cast_slice(&v));
        self.gpu.queue.submit([]);
        self.gpu.wait();
    }

    pub fn read_vec4s(&self, name: &str) -> Vec<[f32; 4]> {
        let b = self.read_ranges(name, &[(0, self.buffers[name].size())]);
        bytemuck::cast_slice::<u8, f32>(&b)
            .chunks_exact(4)
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    pub fn read_u32s(&self, name: &str) -> Vec<u32> {
        let b = self.read_ranges(name, &[(0, self.buffers[name].size())]);
        bytemuck::cast_slice::<u8, u32>(&b).to_vec()
    }

    /// Every cell slot: the first 16 words of each record, which hold everything but
    /// the words of a few rules. [`World::read_cells_full`] has those too.
    pub fn read_cells(&self) -> Vec<[u32; 16]> {
        self.read_cells_full()
            .iter()
            .map(|c| c[..16].try_into().unwrap())
            .collect()
    }

    /// Every cell slot, as raw whole records.
    pub fn read_cells_full(&self) -> Vec<[u32; CELL_WORDS]> {
        let b = self.read_ranges("cell", &[(0, self.buffers["cell"].size())]);
        bytemuck::cast_slice::<u8, u32>(&b)
            .chunks_exact(CELL_WORDS)
            .map(|c| c.try_into().unwrap())
            .collect()
    }

    pub fn read_genome(&self, slot: u32) -> Vec<u8> {
        let g = self.cfg.blocks() as u64 * 16;
        self.read_genomes(&[(slot as u64, 1)], g)
    }

    pub fn read_mem(&self, slot: u32) -> Vec<f32> {
        let m = self.cfg.hidden_rows as u64 * 16;
        let b = self.read_ranges("mem", &[(slot as u64 * m, m)]);
        bytemuck::cast_slice::<u8, f32>(&b).to_vec()
    }
}

/// A saved world's buffers, in file order, with their sizes.
fn saved_layout(size: u64, slots: u64, hidden_rows: u64, shard: u64) -> Vec<(&'static str, u64)> {
    let squares = size * size;
    let blocks = hidden_rows
        * (crate::config::IN_BLOCKS as u64 + hidden_rows + crate::config::OUT_ROWS as u64);
    let mut names: Vec<&'static str> = SAVED.to_vec();
    if slots > shard {
        names.insert(
            names.iter().position(|n| *n == "genome").unwrap() + 1,
            "genome1",
        );
    }
    names
        .iter()
        .map(|name| {
            let bytes = match *name {
                "occ" | "seen" | "seen_next" | "terrain" => squares * 4,
                "fa" | "fb" | "fc" | "fd" => squares * 16,
                "cell" => slots * CELL_BYTES,
                "stat" => slots * STAT_BYTES,
                "sketch" | "jobs" => slots * 16,
                "mem" => slots * hidden_rows * 16,
                "genome" => slots.min(shard) * blocks * 16,
                "genome1" => (slots - shard) * blocks * 16,
                "ring" => (slots as u32).next_power_of_two() as u64 * 4,
                "ctr" => 32 * 4,
                "ctl" => 8 * 4,
                _ => unreachable!(),
            };
            (*name, bytes)
        })
        .collect()
}

impl World {
    /// Copies the living cells of saved smaller worlds into this fresh world, as a
    /// grid of tiles. Neighbouring tiles come from different files, so populations
    /// that evolved apart meet at the borders. Returns how many cells were planted.
    pub fn seed_from(&mut self, paths: &[String], keep: f64) -> Result<u64, String> {
        use std::io::{Read, Seek, SeekFrom};
        self.flush();
        let size = self.cfg.size as u64;
        let hr = self.cfg.hidden_rows as u64;
        let blocks = self.cfg.blocks() as u64;
        let genome_bytes = (blocks * 16) as usize;
        let mem_bytes = (hr * 16) as usize;
        let mut occ = self.read_u32s("occ");

        struct Source {
            file: std::fs::File,
            occ: Vec<u32>,
            cell: Vec<u8>,
            sketch: Vec<u8>,
            mem_at: u64,
            genome_at: u64,
            genome1_at: u64,
            shard: u64,
            size: u64,
        }
        let mut sources = vec![];
        for path in paths {
            let mut f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len).map_err(|e| e.to_string())?;
            let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
            f.read_exact(&mut head).map_err(|e| e.to_string())?;
            let head: serde_json::Value =
                serde_json::from_slice(&head).map_err(|e| e.to_string())?;
            check_layout(&head, path)?;
            let (src, slots, src_hr) = (
                head["size"].as_u64().unwrap_or(0),
                head["slots"].as_u64().unwrap_or(0),
                head["hidden_rows"].as_u64().unwrap_or(0),
            );
            if src_hr != hr || src == 0 || src > size || !src.is_power_of_two() {
                return Err(format!(
                    "{path}: a world {src} wide with {src_hr} rows of memory can't seed this one"
                ));
            }
            let mut at = 8 + u64::from_le_bytes(len);
            let mut offsets = std::collections::HashMap::new();
            let src_shard = head["shard_slots"]
                .as_u64()
                .unwrap_or(self.cfg.max_shard_slots() as u64);
            for (name, bytes) in saved_layout(src, slots, hr, src_shard) {
                offsets.insert(name, (at, bytes));
                at += bytes;
            }
            let mut section = |name: &str| -> Result<Vec<u8>, String> {
                let (from, bytes) = offsets[name];
                f.seek(SeekFrom::Start(from)).map_err(|e| e.to_string())?;
                let mut v = vec![0u8; bytes as usize];
                f.read_exact(&mut v).map_err(|e| e.to_string())?;
                Ok(v)
            };
            let occ_bytes = section("occ")?;
            let cell = section("cell")?;
            let sketch = section("sketch")?;
            sources.push(Source {
                occ: bytemuck::pod_collect_to_vec(&occ_bytes),
                cell,
                sketch,
                mem_at: offsets["mem"].0,
                genome_at: offsets["genome"].0,
                genome1_at: offsets.get("genome1").map_or(0, |o| o.0),
                shard: src_shard,
                file: f,
                size: src,
            });
        }
        let src_size = sources
            .first()
            .map(|s| s.size)
            .ok_or("no worlds to seed from")?;
        if sources.iter().any(|s| s.size != src_size) {
            return Err("seed worlds must all be the same size".into());
        }
        let per_side = size / src_size;
        let log = self.cfg.size.trailing_zeros();
        let mut slot = self.counters()[1] as u64;
        let first_slot = slot;
        let max_slots = self.cfg.slots as u64;
        // Filled, then written in chunks, so memory stays small.
        const CHUNK: usize = 16384;
        let (mut cells, mut sketches, mut mems, mut genomes) = (vec![], vec![], vec![], vec![]);
        let mut chunk_start = slot;
        let mut rng = (self.cfg.seed as u64) ^ 0x7777_1234;
        'outer: for ty in 0..per_side {
            for tx in 0..per_side {
                let which = ((tx + ty) % sources.len() as u64) as usize;
                let source = &mut sources[which];
                for sy in 0..src_size {
                    for sx in 0..src_size {
                        let o = source.occ[(sy * src_size + sx) as usize];
                        if o == 0 || o == u32::MAX {
                            continue;
                        }
                        let s = (o - 1) as u64;
                        let p = (((sy + ty * src_size) << log) | (sx + tx * src_size)) as usize;
                        if occ[p] != 0 {
                            continue;
                        }
                        if keep < 1.0 && (next_u64(&mut rng) as f64 / u64::MAX as f64) >= keep {
                            continue;
                        }
                        if slot >= max_slots {
                            break 'outer;
                        }
                        let cell_bytes = CELL_BYTES as usize;
                        let mut rec: [u32; CELL_WORDS] = bytemuck::pod_read_unaligned(
                            &source.cell[s as usize * cell_bytes..(s as usize + 1) * cell_bytes],
                        );
                        rec[10] = p as u32;
                        rec[15] = 0;
                        cells.extend_from_slice(bytemuck::cast_slice(&rec));
                        sketches.extend_from_slice(
                            &source.sketch[s as usize * 16..(s as usize + 1) * 16],
                        );
                        let at_mem = mems.len();
                        mems.resize(at_mem + mem_bytes, 0u8);
                        source
                            .file
                            .seek(SeekFrom::Start(source.mem_at + s * mem_bytes as u64))
                            .map_err(|e| e.to_string())?;
                        source
                            .file
                            .read_exact(&mut mems[at_mem..])
                            .map_err(|e| e.to_string())?;
                        let at_gen = genomes.len();
                        genomes.resize(at_gen + genome_bytes, 0u8);
                        source
                            .file
                            .seek(SeekFrom::Start(if s < source.shard {
                                source.genome_at + s * genome_bytes as u64
                            } else {
                                source.genome1_at + (s - source.shard) * genome_bytes as u64
                            }))
                            .map_err(|e| e.to_string())?;
                        source
                            .file
                            .read_exact(&mut genomes[at_gen..])
                            .map_err(|e| e.to_string())?;
                        occ[p] = slot as u32 + 1;
                        slot += 1;
                        if (slot - chunk_start) as usize >= CHUNK {
                            self.write_slots(chunk_start, &cells, &sketches, &mems, &genomes);
                            cells.clear();
                            sketches.clear();
                            mems.clear();
                            genomes.clear();
                            chunk_start = slot;
                        }
                    }
                }
            }
        }
        self.write_slots(chunk_start, &cells, &sketches, &mems, &genomes);

        // Rock stays rock, and cells get a plain body word until they first act.
        let mut at = 0usize;
        let chunk_words = 8 << 20;
        while at < occ.len() {
            let end = (at + chunk_words).min(occ.len());
            let part = &occ[at..end];
            let seen: Vec<u32> = part
                .iter()
                .map(|o| match *o {
                    0 => 0,
                    u32::MAX => 1,
                    _ => 0x80,
                })
                .collect();
            let q = &self.gpu.queue;
            q.write_buffer(
                &self.buffers["occ"],
                at as u64 * 4,
                bytemuck::cast_slice(part),
            );
            q.write_buffer(
                &self.buffers["seen"],
                at as u64 * 4,
                bytemuck::cast_slice(&seen),
            );
            q.write_buffer(
                &self.buffers["seen_next"],
                at as u64 * 4,
                bytemuck::cast_slice(&seen),
            );
            q.submit([]);
            self.gpu.wait();
            at = end;
        }
        let tail = slot as u32;
        self.gpu
            .queue
            .write_buffer(&self.buffers["ctr"], 4, bytemuck::bytes_of(&tail));
        self.gpu.queue.submit([]);
        self.gpu.wait();
        Ok(slot - first_slot)
    }

    /// The buffers a saved world holds, in file order.
    fn saved_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = SAVED.to_vec();
        if self.cfg.two_shards() {
            names.insert(
                names.iter().position(|n| *n == "genome").unwrap() + 1,
                "genome1",
            );
        }
        names
    }

    /// Writes the genomes of consecutive slots, across both buffers if needed.
    fn write_genomes(&self, start: u64, bytes: &[u8]) {
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

    /// Reads the genomes of runs of slots, as (first slot, how many), in the order given.
    pub fn read_genomes(&self, runs: &[(u64, u64)], g: u64) -> Vec<u8> {
        let shard = self.cfg.shard_slots as u64;
        let (mut near, mut far) = (vec![], vec![]);
        for (start, len) in runs {
            let first = shard.saturating_sub(*start).min(*len);
            if first > 0 {
                near.push((*start * g, first * g));
            }
            if first < *len {
                far.push(((*start + first - shard) * g, (*len - first) * g));
            }
        }
        let a = if near.is_empty() {
            vec![]
        } else {
            self.read_ranges("genome", &near)
        };
        let b = if far.is_empty() {
            vec![]
        } else {
            self.read_ranges("genome1", &far)
        };
        let mut out = Vec::with_capacity(a.len() + b.len());
        let (mut at_a, mut at_b) = (0usize, 0usize);
        for (start, len) in runs {
            let first = shard.saturating_sub(*start).min(*len);
            let n = (first * g) as usize;
            out.extend_from_slice(&a[at_a..at_a + n]);
            at_a += n;
            let m = ((*len - first) * g) as usize;
            out.extend_from_slice(&b[at_b..at_b + m]);
            at_b += m;
        }
        out
    }

    fn write_slots(&self, start: u64, cells: &[u8], sketches: &[u8], mems: &[u8], genomes: &[u8]) {
        if cells.is_empty() {
            return;
        }
        let hr = self.cfg.hidden_rows as u64;
        let q = &self.gpu.queue;
        q.write_buffer(&self.buffers["cell"], start * CELL_BYTES, cells);
        q.write_buffer(&self.buffers["sketch"], start * 16, sketches);
        q.write_buffer(&self.buffers["mem"], start * hr * 16, mems);
        self.write_genomes(start, genomes);
        q.submit([]);
        self.gpu.wait();
    }
}
