//! The thread that owns the world. Everything that waits for the GPU happens here,
//! so the window stays responsive whatever the world is doing. Ticks are paced by
//! `World::flush`, which keeps each submission near its target length.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use auto_core::Gpu;
use auto_life_gpu::config::{CELL_BYTES, SKETCH_BYTES, STAT_BYTES, cell, counter, stat};
use auto_life_gpu::explore::{self, CELL_VIEWS, ViewPipe};
use auto_life_gpu::report::{Palette, Sample, sample_with};
use auto_life_gpu::{Config, Pacing, World, gpu_busy_elsewhere};
use serde_json::{Map, Value};

pub const TARGET_MS: f64 = 8.0;
pub const DUTY: f64 = 0.7;
/// While another GPU job holds the locks, this window takes a smaller share.
const DUTY_SHARING: f64 = 0.35;

#[derive(Clone)]
pub struct NewWorld {
    pub size: u32,
    pub seed: u32,
    pub slots: u32,
    pub hidden_rows: u32,
    pub physics: Map<String, Value>,
    pub seed_from: Vec<String>,
    pub seed_keep: f64,
    /// Something to say once it's running.
    pub note: Option<String>,
}

pub enum Cmd {
    Run(bool),
    /// Ticks per second. None is as fast as the GPU can go.
    Speed(Option<f64>),
    Step,
    Physics(Map<String, Value>),
    New(Box<NewWorld>),
    Load(String),
    Save(String),
    /// A square was clicked.
    Pick(u32, u32),
    Unpick,
    /// What the window is showing, so the right things get measured.
    Showing {
        cell_mode: usize,
        needs_bake: bool,
    },
    Quit,
}

/// One point on the charts.
#[derive(Clone, Copy, Default)]
pub struct Point {
    pub tick: f64,
    pub cells: f64,
    /// Ground, meat, biting, gifts.
    pub diet: [f64; 4],
    /// Starved, bitten, burned, old, per tick.
    pub deaths: [f64; 4],
    pub births: f64,
}

/// What's known about the picked cell or square.
#[derive(Clone)]
pub struct Picked {
    pub x: u32,
    pub y: u32,
    pub slot: Option<u32>,
    pub note: String,
    /// The raw records, laid out as in `config::cell` and `config::stat`.
    pub cell: Vec<u32>,
    pub stat: Vec<f32>,
    pub sketch: [u32; 4],
    pub fa: [f32; 4],
    pub fb: [f32; 4],
    pub fc: [f32; 4],
    pub fd: [f32; 4],
    pub rock: bool,
    /// Rough share of the population within the similarity limit.
    pub similar_share: Option<f32>,
}

#[derive(Clone, Default)]
pub struct Status {
    pub world_id: u64,
    pub view: Option<ViewPipe>,
    pub size: u32,
    pub seed: u32,
    pub slots: u32,
    pub hidden_rows: u32,
    pub vram: u64,
    pub path: Option<String>,
    pub tick: u64,
    pub cells: u32,
    pub running: bool,
    pub tps: f64,
    pub gpu_ms_per_tick: f64,
    pub longest_ms: f64,
    pub physics: Map<String, Value>,
    pub physics_applied: u64,
    pub message: Option<(bool, String)>,
    pub busy: Option<String>,
    pub sharing: bool,
    pub adapter: String,
    pub software: bool,
    pub history: Arc<Vec<Point>>,
    pub picked: Option<Picked>,
    /// The range fitted to the population, and the view it was fitted for.
    pub cell_range: Option<(usize, (f32, f32))>,
    pub hi: [f32; explore::HI_COUNT],
    pub sun_k: f32,
    pub season: f32,
    pub season_on: bool,
    pub ground_cap: f32,
    pub sun_keep: f32,
    pub drift: [f32; 2],
}

pub struct Sim {
    pub gpu: Arc<Gpu>,
    pub status: Arc<Mutex<Status>>,
    pub progress: Arc<AtomicU32>,
    pub view_layout: wgpu::BindGroupLayout,
    pub shaders: Option<PathBuf>,
    pub rx: Receiver<Cmd>,
    /// The similarity limit the window is using, as bits of an f32.
    pub tau: Arc<AtomicU32>,
    pub repaint: Arc<dyn Fn() + Send + Sync>,
}

struct Live {
    world: World,
    palette: Palette,
    bake_uniform: wgpu::Buffer,
    last: [u32; 32],
    last_chart: Instant,
    chart_tick: u64,
    history: Vec<Point>,
    last_palette: Instant,
    last_bake: Instant,
    baked_tick: u64,
    sample_cells: Vec<Sample>,
}

/// The cell being watched. Its founder and age tell it apart from whatever takes over its slot.
struct Watch {
    x: u32,
    y: u32,
    slot: Option<u32>,
    clade: u32,
    age: f32,
    note: String,
}

const LONG_AGO: Duration = Duration::from_secs(3600);

impl Sim {
    pub fn run(self, first: Cmd) {
        let mut live: Option<Live> = None;
        let mut running = true;
        let mut speed: Option<f64> = None;
        let mut credit = 0.0f64;
        let mut steps = 0u32;
        let mut watch: Option<Watch> = None;
        let mut cell_mode = 0usize;
        let mut needs_bake = true;
        let start = Instant::now();
        let ago = |d: Duration| start.checked_sub(d).unwrap_or(start);
        let mut last_loop = Instant::now();
        let mut last_flush = Instant::now();
        let mut last_publish = ago(LONG_AGO);
        let mut last_fields = ago(LONG_AGO);
        let mut last_pick = ago(LONG_AGO);
        let mut last_lock_check = ago(LONG_AGO);
        let (mut rate_tick, mut rate_at, mut rate_gpu) = (0u64, Instant::now(), 0.0f64);
        let mut sharing = false;
        let mut hi = [1.0f32; explore::HI_COUNT];
        let mut cell_range = None;
        let mut next = Some(first);
        loop {
            // Commands first, so a click never waits for more than the tick in progress.
            loop {
                let cmd = match next.take() {
                    Some(c) => c,
                    None => match self.rx.try_recv() {
                        Ok(c) => c,
                        Err(_) => break,
                    },
                };
                match cmd {
                    Cmd::Quit => return,
                    Cmd::Run(on) => {
                        running = on;
                        credit = 0.0;
                        last_publish = ago(LONG_AGO);
                    }
                    Cmd::Speed(s) => {
                        speed = s;
                        credit = 0.0;
                    }
                    Cmd::Step => {
                        running = false;
                        steps += 1;
                    }
                    Cmd::Showing {
                        cell_mode: m,
                        needs_bake: b,
                    } => {
                        cell_mode = m;
                        needs_bake = b;
                        if let Some(l) = &live {
                            cell_range = fit_for(m, &l.sample_cells);
                            last_publish = ago(LONG_AGO);
                        }
                    }
                    Cmd::Pick(x, y) => {
                        if let Some(l) = &mut live {
                            l.world.flush();
                            watch = Some(first_look(&l.world, x, y));
                            last_pick = ago(LONG_AGO);
                        }
                    }
                    Cmd::Unpick => {
                        watch = None;
                        self.status.lock().unwrap().picked = None;
                    }
                    Cmd::Physics(p) => {
                        if let Some(l) = &mut live {
                            let result = l.world.set_physics(p);
                            let mut st = self.status.lock().unwrap();
                            match result {
                                Ok(()) => {
                                    st.physics = l.world.cfg.physics.clone();
                                    st.physics_applied += 1;
                                    st.message = None;
                                }
                                Err(e) => {
                                    st.message =
                                        Some((true, format!("Those settings didn't build: {e}")));
                                }
                            }
                        }
                    }
                    Cmd::Save(path) => {
                        if let Some(l) = &mut live {
                            self.set_busy(format!("Saving {path}"));
                            l.world.progress = self.progress.clone();
                            if let Some(dir) = std::path::Path::new(&path).parent() {
                                let _ = std::fs::create_dir_all(dir);
                            }
                            let result = l.world.save(&path);
                            let mut st = self.status.lock().unwrap();
                            st.busy = None;
                            st.message = Some(match result {
                                Ok(()) => (false, format!("Saved {path}")),
                                Err(e) => (true, format!("Couldn't save {path}: {e}")),
                            });
                        }
                    }
                    Cmd::Load(path) => {
                        // A bad file shouldn't cost the world that's running.
                        let limit = buffer_limit(&self.gpu);
                        let saved = Config::from_save(&path, self.shaders.clone());
                        if let Err(e) = saved.and_then(|c| c.check_buffers(limit)) {
                            self.status.lock().unwrap().message = Some((true, e));
                            continue;
                        }
                        let began = Instant::now();
                        self.release_world(&mut live, &mut watch);
                        debug(&format!(
                            "released the old world after {:?}",
                            began.elapsed()
                        ));
                        self.set_busy(format!("Loading {path}"));
                        let gpu = self.gpu.clone();
                        let shaders = self.shaders.clone();
                        let progress = self.progress.clone();
                        let result = guarded(&gpu, || {
                            World::open_saved(
                                gpu.clone(),
                                &path,
                                shaders,
                                Pacing {
                                    target_ms: TARGET_MS,
                                    duty: DUTY,
                                },
                                progress,
                            )
                        });
                        debug(&format!("opened it after {:?}", began.elapsed()));
                        self.adopt(result, &mut live, Some(path));
                        debug(&format!("ready after {:?}", began.elapsed()));
                        running = live.is_some();
                        last_fields = ago(LONG_AGO);
                        (rate_tick, rate_at, rate_gpu) = fresh_rate(&live);
                    }
                    Cmd::New(spec) => {
                        // Bad seed worlds shouldn't cost the world that's running.
                        let fits = self
                            .world_config(&spec)
                            .and_then(|c| c.check_buffers(buffer_limit(&self.gpu)));
                        if let Err(e) = fits.and_then(|()| check_seeds(&spec, &self.shaders)) {
                            self.status.lock().unwrap().message = Some((true, e));
                            continue;
                        }
                        let began = Instant::now();
                        self.release_world(&mut live, &mut watch);
                        debug(&format!(
                            "released the old world after {:?}",
                            began.elapsed()
                        ));
                        self.set_busy(format!("Building a world {0} x {0}", spec.size));
                        let result = self.make_world(&spec);
                        debug(&format!("built it after {:?}", began.elapsed()));
                        self.adopt(result, &mut live, None);
                        debug(&format!("ready after {:?}", began.elapsed()));
                        if let Some(note) = &spec.note {
                            self.status.lock().unwrap().message = Some((true, note.clone()));
                        }
                        if let Some(l) = &live
                            && !spec.seed_from.is_empty()
                            && l.world.alive() >= l.world.cfg.slots
                        {
                            self.status.lock().unwrap().message = Some((
                                false,
                                "The seed worlds have more cells than this world has room for, so the last tiles are partly empty.".into(),
                            ));
                        }
                        running = live.is_some();
                        last_fields = ago(LONG_AGO);
                        (rate_tick, rate_at, rate_gpu) = fresh_rate(&live);
                    }
                }
            }

            let Some(l) = &mut live else {
                match self.rx.recv_timeout(Duration::from_millis(50)) {
                    Ok(c) => next = Some(c),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
                continue;
            };

            let now = Instant::now();
            let dt = now.duration_since(last_loop).as_secs_f64();
            last_loop = now;
            let mut ticked = false;
            if steps > 0 {
                steps -= 1;
                l.world.step();
                l.world.flush();
                ticked = true;
                last_publish = ago(LONG_AGO);
                last_pick = ago(LONG_AGO);
            } else if running {
                match speed {
                    None => {
                        l.world.step();
                        ticked = true;
                    }
                    Some(rate) => {
                        // A short memory, so a stall isn't made up for by a rush afterwards.
                        credit = (credit + rate * dt).min((rate * 0.1).max(1.0));
                        if credit >= 1.0 {
                            credit -= 1.0;
                            l.world.step();
                            ticked = true;
                            if rate < 120.0 {
                                l.world.flush();
                                last_flush = Instant::now();
                                if rate < 10.0 {
                                    last_publish = ago(LONG_AGO);
                                    last_pick = ago(LONG_AGO);
                                }
                            }
                        }
                    }
                }
            }
            if last_flush.elapsed() > Duration::from_millis(16) {
                l.world.flush();
                last_flush = Instant::now();
            }

            // Colours of new cells, while someone's looking at them.
            let slow = speed.is_some_and(|r| r < 10.0) || !running;
            let bake_gap = Duration::from_millis(if slow { 60 } else { 250 });
            if needs_bake && l.world.tick != l.baked_tick && l.last_bake.elapsed() > bake_gap {
                self.gpu
                    .queue
                    .write_buffer(&l.bake_uniform, 0, &l.palette.bytes());
                l.world.bake(&l.bake_uniform);
                l.last_bake = Instant::now();
                l.baked_tick = l.world.tick;
                (self.repaint)();
            }
            if l.last_palette.elapsed() > Duration::from_secs(2) {
                let s = sample_with(&mut l.world, 2000, false);
                l.palette.update(&s.cells);
                l.sample_cells = s.cells;
                cell_range = fit_for(cell_mode, &l.sample_cells);
                l.last_palette = Instant::now();
                l.baked_tick = u64::MAX;
            }
            if last_fields.elapsed() > Duration::from_millis(1500) {
                read_field_ranges(&mut l.world, &mut hi);
                last_fields = Instant::now();
            }
            if last_lock_check.elapsed() > Duration::from_secs(5) {
                let software = self.gpu.name().to_lowercase().contains("llvmpipe");
                sharing = !software && gpu_busy_elsewhere();
                l.world.pacing.duty = if sharing { DUTY_SHARING } else { DUTY };
                last_lock_check = Instant::now();
            }
            if l.last_chart.elapsed() > Duration::from_secs(1) && l.world.tick > l.chart_tick {
                self.chart(l);
            }
            if let Some(w) = &mut watch
                && last_pick.elapsed() > Duration::from_millis(120)
            {
                l.world.flush();
                let tau = f32::from_bits(self.tau.load(Ordering::Relaxed));
                let seen = look(&l.world, w, &l.sample_cells, tau);
                self.status.lock().unwrap().picked = Some(seen);
                last_pick = Instant::now();
                (self.repaint)();
            }
            if last_publish.elapsed() > Duration::from_millis(100) {
                l.world.flush();
                let cells = l.world.alive();
                let elapsed = rate_at.elapsed().as_secs_f64();
                let mut st = self.status.lock().unwrap();
                let ticks = (l.world.tick - rate_tick) as f64;
                // Long enough to hold a tick or two, so slow speeds don't read as stopped.
                if elapsed > 4.0
                    || (elapsed > 1.2 && ticks >= 2.0)
                    || (elapsed > 0.5 && ticks >= 8.0)
                {
                    st.tps = ticks / elapsed.max(1e-6);
                    if ticks > 0.0 {
                        st.gpu_ms_per_tick = (l.world.gpu_seconds - rate_gpu) * 1000.0 / ticks;
                    }
                    st.longest_ms = l.world.longest_submit_ms;
                    l.world.longest_submit_ms = 0.0;
                    rate_tick = l.world.tick;
                    rate_gpu = l.world.gpu_seconds;
                    rate_at = Instant::now();
                }
                st.tick = l.world.tick;
                st.cells = cells;
                st.running = running || steps > 0;
                st.sharing = sharing;
                st.hi = hi;
                st.cell_range = cell_range.map(|r| (cell_mode, r));
                st.sun_k = l.world.sun_k;
                st.season_on = l.world.cfg.f("season_ticks") > 0.0;
                st.season = l.world.current_frame().season;
                st.ground_cap = l.world.cfg.f("ground_cap");
                st.sun_keep = 1.0 - l.world.cfg.f("bloom_share");
                st.drift = if l.world.cfg.b("weather_sun_on") {
                    let x = l.world.tick as f32 * l.world.cfg.f("weather_drift");
                    [x, x * 0.618034]
                } else {
                    [0.0, 0.0]
                };
                drop(st);
                last_publish = Instant::now();
                (self.repaint)();
            }

            if !ticked {
                // Idle, or waiting for the next slow tick. A command wakes this at once.
                let wait = match (running, speed) {
                    (true, Some(rate)) => ((1.0 - credit) / rate).clamp(0.0005, 0.02),
                    _ => 0.02,
                };
                match self.rx.recv_timeout(Duration::from_secs_f64(wait)) {
                    Ok(c) => next = Some(c),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    }

    fn set_busy(&self, what: String) {
        self.progress.store(0, Ordering::Relaxed);
        self.status.lock().unwrap().busy = Some(what);
        (self.repaint)();
    }

    /// Lets go of the world and everything keeping its buffers alive, so its VRAM is free.
    fn release_world(&self, live: &mut Option<Live>, watch: &mut Option<Watch>) {
        *watch = None;
        {
            let mut st = self.status.lock().unwrap();
            st.view = None;
            st.picked = None;
            st.history = Arc::new(vec![]);
            st.running = false;
        }
        *live = None;
        // A frame being drawn may still hold it. That's over within a frame or two.
        std::thread::sleep(Duration::from_millis(60));
        self.gpu.wait();
    }

    fn world_config(&self, spec: &NewWorld) -> Result<Config, String> {
        let mut cfg = Config::new(spec.size, spec.seed, self.shaders.clone())?;
        for (k, v) in &spec.physics {
            cfg.physics.insert(k.clone(), v.clone());
        }
        cfg.hidden_rows = spec.hidden_rows;
        cfg.shard_slots = cfg.max_shard_slots();
        cfg.slots = spec.slots.max(1024);
        Ok(cfg)
    }

    fn make_world(&self, spec: &NewWorld) -> Result<World, String> {
        let cfg = self.world_config(spec)?;
        guarded(&self.gpu, || {
            let mut w = World::with_gpu(
                self.gpu.clone(),
                cfg,
                Pacing {
                    target_ms: TARGET_MS,
                    duty: DUTY,
                },
            )?;
            if !spec.seed_from.is_empty() {
                w.progress = self.progress.clone();
                w.seed_from(&spec.seed_from, spec.seed_keep)?;
            }
            Ok(w)
        })
    }

    fn adopt(&self, result: Result<World, String>, live: &mut Option<Live>, path: Option<String>) {
        let fail = |e: String| {
            let mut st = self.status.lock().unwrap();
            st.busy = None;
            st.message = Some((true, e));
        };
        let mut world = match result {
            Ok(w) => w,
            Err(e) => return fail(e),
        };
        let view = match ViewPipe::new(&world, &self.view_layout) {
            Ok(v) => v,
            Err(e) => return fail(e),
        };
        let bake_uniform = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("bake uniform"),
            size: (12 + 4 + 1) * 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let last = world.counters();
        let mut palette = Palette::default();
        let s = sample_with(&mut world, 2000, false);
        palette.update(&s.cells);
        self.gpu
            .queue
            .write_buffer(&bake_uniform, 0, &palette.bytes());
        world.bake(&bake_uniform);
        {
            let mut st = self.status.lock().unwrap();
            st.world_id += 1;
            st.view = Some(view);
            st.size = world.cfg.size;
            st.seed = world.cfg.seed;
            st.slots = world.cfg.slots;
            st.hidden_rows = world.cfg.hidden_rows;
            st.vram = world.cfg.vram_bytes();
            st.path = path;
            st.tick = world.tick;
            st.cells = world.alive();
            st.physics = world.cfg.physics.clone();
            st.busy = None;
            st.message = None;
            st.adapter = self.gpu.name();
            st.software = self.gpu.name().to_lowercase().contains("llvmpipe");
            st.ground_cap = world.cfg.f("ground_cap");
            st.sun_k = world.sun_k;
            st.tps = 0.0;
        }
        let chart_tick = world.tick;
        *live = Some(Live {
            world,
            palette,
            bake_uniform,
            last,
            last_chart: Instant::now(),
            chart_tick,
            history: vec![],
            last_palette: Instant::now(),
            last_bake: Instant::now(),
            baked_tick: u64::MAX,
            sample_cells: s.cells,
        });
        (self.repaint)();
    }

    fn chart(&self, l: &mut Live) {
        l.world.flush();
        let c = l.world.counters();
        let ticks = (l.world.tick - l.chart_tick).max(1) as f64;
        let rate = |k: usize| c[k].wrapping_sub(l.last[k]) as f64 / ticks;
        let mut diet = [0.0f64; 4];
        for s in &l.sample_cells {
            for (k, d) in diet.iter_mut().enumerate() {
                *d += s.diet[k] as f64;
            }
        }
        let total: f64 = diet.iter().sum::<f64>().max(1e-9);
        diet.iter_mut().for_each(|d| *d /= total);
        let point = Point {
            tick: l.world.tick as f64,
            cells: l.world.alive() as f64,
            diet,
            deaths: [
                rate(counter::STARVED),
                rate(counter::BITTEN),
                rate(counter::BURNED),
                rate(counter::OLD),
            ],
            births: rate(counter::BIRTHS),
        };
        l.last = c;
        l.chart_tick = l.world.tick;
        l.last_chart = Instant::now();
        l.history.push(point);
        // Long runs keep their whole story, at lower detail.
        if l.history.len() > 1200 {
            l.history = l.history.iter().step_by(2).copied().collect();
        }
        self.status.lock().unwrap().history = Arc::new(l.history.clone());
    }
}

fn fresh_rate(live: &Option<Live>) -> (u64, Instant, f64) {
    (
        live.as_ref().map_or(0, |l| l.world.tick),
        Instant::now(),
        live.as_ref().map_or(0.0, |l| l.world.gpu_seconds),
    )
}

fn fit_for(mode: usize, cells: &[Sample]) -> Option<(f32, f32)> {
    let read = CELL_VIEWS.get(mode)?.reading?;
    explore::fit_range(cells.iter().map(read).collect())
}

/// Runs something that makes GPU memory, and turns running out into a message instead of a crash.
fn guarded<T>(gpu: &Gpu, make: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let scope = gpu.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let made = make();
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(format!("The GPU ran out of memory: {e}"));
    }
    made
}

/// Samples each field's squares, so each one's colours fit what's actually there.
fn read_field_ranges(world: &mut World, hi: &mut [f32; explore::HI_COUNT]) {
    world.flush();
    let size = world.cfg.size as u64;
    let squares = size * size;
    const WANT: u64 = 1500;
    let gap = (squares / WANT).max(1);
    let wobble = world.tick.wrapping_mul(2654435761) % gap;
    let spots: Vec<u64> = (0..WANT.min(squares))
        .map(|k| (k * gap + wobble + (k.wrapping_mul(7919) % size)) % squares)
        .collect();
    let of = |bytes: u64| -> Vec<(u64, u64)> { spots.iter().map(|p| (p * bytes, bytes)).collect() };
    let fa = world.read_ranges("fa", &of(16));
    let fb = world.read_ranges("fb", &of(16));
    let fc = world.read_ranges("fc", &of(16));
    let fd = world.read_ranges("fd", &of(16));
    let terrain = world.read_ranges("terrain", &of(4));
    let fa: &[f32] = bytemuck::cast_slice(&fa);
    let fb: &[f32] = bytemuck::cast_slice(&fb);
    let fc: &[f32] = bytemuck::cast_slice(&fc);
    let fd: &[f32] = bytemuck::cast_slice(&fd);
    let t: &[u32] = bytemuck::cast_slice(&terrain);
    let mut update = |k: usize, top: f32| {
        // Quick to rise and slow to fall, so a fading trail doesn't flare up.
        hi[k] = if hi[k] < top {
            top
        } else {
            hi[k] * 0.8 + top * 0.2
        };
    };
    for k in 0..4 {
        let channel = |f: &[f32]| f.iter().skip(k).step_by(4).copied().collect::<Vec<f32>>();
        update(k, explore::fit_top(channel(fa)));
        update(4 + k, explore::fit_top(channel(fb)));
        update(8 + k, explore::fit_top(channel(fc)));
    }
    update(
        explore::HI_TRACKS,
        explore::fit_top(fd.iter().step_by(4).copied().collect()),
    );
    let base: Vec<f32> = t
        .iter()
        .filter(|v| *v & (1 << 24) == 0)
        .map(|v| (v & 0xFFFF) as f32 / 65535.0 * auto_life_gpu::config::SUN_RANGE * world.sun_k)
        .collect();
    let top = explore::fit_top(base);
    update(explore::HI_BASE, top);
    update(explore::HI_SUN, top);
}

fn read_cell(world: &World, slot: u32) -> Vec<u32> {
    let b = world.read_ranges("cell", &[(slot as u64 * CELL_BYTES, CELL_BYTES)]);
    bytemuck::pod_collect_to_vec(&b)
}

fn first_look(world: &World, x: u32, y: u32) -> Watch {
    let p = ((y as u64) << world.cfg.size.trailing_zeros()) | x as u64;
    let occ = world.read_ranges("occ", &[(p * 4, 4)]);
    let o = u32::from_le_bytes(occ[..4].try_into().unwrap());
    let mut w = Watch {
        x,
        y,
        slot: None,
        clade: 0,
        age: 0.0,
        note: String::new(),
    };
    if o != 0 && o != u32::MAX {
        let rec = read_cell(world, o - 1);
        w.slot = Some(o - 1);
        w.clade = rec[cell::CLADE];
        w.age = f32::from_bits(rec[cell::AGE]);
    }
    w
}

/// A fresh look at the picked cell or square.
fn look(world: &World, w: &mut Watch, sample: &[Sample], tau: f32) -> Picked {
    let size = world.cfg.size;
    let log = size.trailing_zeros();
    let mut out = Picked {
        x: w.x,
        y: w.y,
        slot: None,
        note: w.note.clone(),
        cell: vec![0; cell::WORDS],
        stat: vec![0.0; stat::WORDS],
        sketch: [0; 4],
        fa: [0.0; 4],
        fb: [0.0; 4],
        fc: [0.0; 4],
        fd: [0.0; 4],
        rock: false,
        similar_share: None,
    };
    if let Some(s) = w.slot {
        let rec = read_cell(world, s);
        let (e, age) = (f32::from_bits(rec[cell::E]), f32::from_bits(rec[cell::AGE]));
        if rec[cell::CLADE] == w.clade && age >= w.age && rec[cell::POS] != u32::MAX && e > 0.0 {
            w.age = age;
            w.x = rec[cell::POS] & (size - 1);
            w.y = rec[cell::POS] >> log;
            out.x = w.x;
            out.y = w.y;
            out.slot = Some(s);
            out.cell = rec;
            let st = world.read_ranges("stat", &[(s as u64 * STAT_BYTES, STAT_BYTES)]);
            out.stat = bytemuck::pod_collect_to_vec(&st);
            let sk = world.read_ranges("sketch", &[(s as u64 * SKETCH_BYTES, SKETCH_BYTES)]);
            out.sketch.copy_from_slice(bytemuck::cast_slice(&sk));
            let mine: [f32; 16] = std::array::from_fn(|k| {
                ((out.sketch[k / 4] >> ((k % 4) * 8)) as u8 as i8) as f32 / 127.0 * 4.0
            });
            if sample.len() >= 8 {
                let near = sample
                    .iter()
                    .filter(|c| {
                        let d: f32 = (0..16).map(|k| (c.sketch[k] - mine[k]).powi(2)).sum();
                        (d / 16.0).sqrt() < tau
                    })
                    .count();
                out.similar_share = Some(near as f32 / sample.len() as f32);
            }
        } else {
            let cause = if e > 0.0 {
                ""
            } else {
                match (rec[cell::FLAGS] >> cell::CAUSE_SHIFT) & cell::CAUSE_MASK {
                    1 => " It was bitten to death.",
                    2 => " It burned to death.",
                    _ => " It starved or died of old age.",
                }
            };
            w.note = format!("This cell has died.{cause}");
            w.slot = None;
            out.note = w.note.clone();
        }
    }
    let p = ((out.y as u64) << log) | out.x as u64;
    let read4 = |name: &str| -> [f32; 4] {
        let b = world.read_ranges(name, &[(p * 16, 16)]);
        let mut v = [0.0; 4];
        v.copy_from_slice(bytemuck::cast_slice(&b));
        v
    };
    out.fa = read4("fa");
    out.fb = read4("fb");
    out.fc = read4("fc");
    out.fd = read4("fd");
    let t = world.read_ranges("terrain", &[(p * 4, 4)]);
    out.rock = u32::from_le_bytes(t[..4].try_into().unwrap()) & (1 << 24) != 0;
    out
}

fn debug(what: &str) {
    if std::env::var_os("LIFE_DEBUG").is_some() {
        eprintln!("{what}");
    }
}

/// Saved worlds can only seed a world they fit in, all the same size, with the same size of brain.
fn check_seeds(spec: &NewWorld, shaders: &Option<PathBuf>) -> Result<(), String> {
    let mut size = None;
    for path in &spec.seed_from {
        let saved = Config::from_save(path, shaders.clone())?;
        if saved.hidden_rows != spec.hidden_rows {
            return Err(format!(
                "{path} has {} memory rows and this world has {}",
                saved.hidden_rows, spec.hidden_rows
            ));
        }
        if saved.size > spec.size {
            return Err(format!(
                "{path} is {} wide, too big to seed a world {} wide",
                saved.size, spec.size
            ));
        }
        if *size.get_or_insert(saved.size) != saved.size {
            return Err("The saved worlds used as seeds must all be the same size".into());
        }
    }
    Ok(())
}

/// The largest single buffer this device can use.
pub fn buffer_limit(gpu: &Gpu) -> u64 {
    let limits = gpu.device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}
