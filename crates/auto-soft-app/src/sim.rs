//! The thread that owns the world. Everything that waits for the GPU happens here, so the window
//! stays responsive whatever the world is doing. Ticks are paced by `World::flush`, which keeps each
//! submission near its target length.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use auto_core::Gpu;
use auto_life_soft::dump::{self, Palette};
use auto_life_soft::view::ViewPipe;
use auto_life_soft::{Config, Pacing, World, gpu_busy_elsewhere};
use serde_json::{Map, Value};

pub const TARGET_MS: f64 = 8.0;
pub const DUTY: f64 = 0.7;
/// While another GPU job holds the locks, this window takes a smaller share.
const DUTY_SHARING: f64 = 0.35;

pub enum Cmd {
    Run(bool),
    /// Ticks per second. None is as fast as the GPU can go.
    Speed(Option<f64>),
    Step,
    Physics(Map<String, Value>),
    New {
        size: u32,
        seed: u32,
        physics: Map<String, Value>,
    },
    Load(String),
    Save(String),
    Quit,
}

#[derive(Clone, Default)]
pub struct Status {
    pub world_id: u64,
    pub view: Option<ViewPipe>,
    pub size: u32,
    pub seed: u32,
    pub slots: u32,
    pub vram: u64,
    pub tick: u64,
    pub cells: u32,
    pub flip: usize,
    pub running: bool,
    pub tps: f64,
    pub gpu_ms_per_tick: f64,
    pub longest_ms: f64,
    pub overflow: u64,
    pub births: u64,
    pub physics: Map<String, Value>,
    pub physics_applied: u64,
    pub message: Option<(bool, String)>,
    pub busy: Option<String>,
    pub sharing: bool,
    pub adapter: String,
    pub software: bool,
    pub palette: Option<Arc<Palette>>,
    pub tops: [f32; 8],
}

pub struct Sim {
    pub gpu: Arc<Gpu>,
    pub status: Arc<Mutex<Status>>,
    pub view_layout: wgpu::BindGroupLayout,
    pub shaders: Option<PathBuf>,
    pub rx: Receiver<Cmd>,
    pub repaint: Arc<dyn Fn() + Send + Sync>,
}

impl Sim {
    fn set_busy(&self, what: String) {
        self.status.lock().unwrap().busy = Some(what);
        (self.repaint)();
    }

    /// A GPU device can only hand out errors while one is listening, so a bad world is an error
    /// here rather than a crash.
    fn adopt(&self, made: Result<World, String>, live: &mut Option<World>, next_id: &mut u64) {
        let mut st = self.status.lock().unwrap();
        st.busy = None;
        match made {
            Ok(world) => match ViewPipe::new(&world, &self.view_layout) {
                Ok(view) => {
                    *next_id += 1;
                    st.world_id = *next_id;
                    st.view = Some(view);
                    st.size = world.cfg.size;
                    st.seed = world.cfg.seed;
                    st.slots = world.cfg.slots;
                    st.vram = world.cfg.vram_bytes();
                    st.tick = world.tick;
                    st.cells = world.alive();
                    st.flip = world.flip();
                    st.physics = world.cfg.physics.clone();
                    st.adapter = self.gpu.name();
                    st.software = self.gpu.name().to_lowercase().contains("llvmpipe");
                    st.tps = 0.0;
                    st.palette = None;
                    st.tops = [1.0; 8];
                    st.message = None;
                    *live = Some(world);
                }
                Err(e) => st.message = Some((true, e)),
            },
            Err(e) => st.message = Some((true, e)),
        }
        drop(st);
        (self.repaint)();
    }

    pub fn run(self, first: Cmd) {
        let mut live: Option<World> = None;
        let mut next_id = 0u64;
        let mut running = true;
        let mut speed: Option<f64> = None;
        let mut credit = 0.0f64;
        let mut steps = 0u32;
        let start = Instant::now();
        let long_ago = start.checked_sub(Duration::from_secs(3600)).unwrap_or(start);
        let mut last_loop = Instant::now();
        let mut last_flush = Instant::now();
        let mut last_publish = long_ago;
        let mut last_palette = long_ago;
        let mut last_lock_check = long_ago;
        let (mut rate_tick, mut rate_at, mut rate_gpu) = (0u64, Instant::now(), 0.0f64);
        let pacing = || Pacing {
            target_ms: TARGET_MS,
            duty: DUTY,
        };
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
                        last_publish = long_ago;
                    }
                    Cmd::Speed(s) => {
                        speed = s;
                        credit = 0.0;
                    }
                    Cmd::Step => {
                        running = false;
                        steps += 1;
                    }
                    Cmd::Physics(p) => {
                        if let Some(w) = &mut live {
                            let result = w.set_physics(p);
                            let mut st = self.status.lock().unwrap();
                            match result {
                                Ok(()) => {
                                    st.physics = w.cfg.physics.clone();
                                    st.physics_applied += 1;
                                    st.message = None;
                                }
                                Err(e) => st.message = Some((true, format!("Those settings didn't build: {e}"))),
                            }
                        }
                    }
                    Cmd::Save(path) => {
                        if let Some(w) = &mut live {
                            self.set_busy(format!("Saving {path}"));
                            if let Some(dir) = std::path::Path::new(&path).parent() {
                                let _ = std::fs::create_dir_all(dir);
                            }
                            let result = w.save(&path);
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
                        if let Err(e) = Config::from_save(&path, self.shaders.clone()) {
                            self.status.lock().unwrap().message = Some((true, e));
                            continue;
                        }
                        self.release(&mut live);
                        self.set_busy(format!("Loading {path}"));
                        let made = World::open_saved(self.gpu.clone(), &path, self.shaders.clone(), pacing());
                        self.adopt(made, &mut live, &mut next_id);
                        running = live.is_some();
                        last_palette = long_ago;
                        rate_tick = live.as_ref().map_or(0, |w| w.tick);
                        rate_at = Instant::now();
                        rate_gpu = 0.0;
                    }
                    Cmd::New { size, seed, physics } => {
                        let mut cfg = match Config::new(size, seed, self.shaders.clone()) {
                            Ok(c) => c,
                            Err(e) => {
                                self.status.lock().unwrap().message = Some((true, e));
                                continue;
                            }
                        };
                        cfg.physics = physics;
                        let limit = self.gpu.device.limits().max_storage_buffer_binding_size;
                        if let Err(e) = cfg.check(limit) {
                            self.status.lock().unwrap().message = Some((true, e));
                            continue;
                        }
                        self.release(&mut live);
                        self.set_busy(format!("Building a world {size} x {size}"));
                        let made = World::with_gpu(self.gpu.clone(), cfg, pacing());
                        self.adopt(made, &mut live, &mut next_id);
                        running = live.is_some();
                        last_palette = long_ago;
                        rate_tick = 0;
                        rate_at = Instant::now();
                        rate_gpu = 0.0;
                    }
                }
            }

            let Some(w) = &mut live else {
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
            if steps > 0 {
                steps -= 1;
                w.step();
                w.flush();
                last_publish = long_ago;
            } else if running {
                match speed {
                    None => w.step(),
                    Some(rate) => {
                        // A short memory, so a stall isn't made up for by a rush afterwards.
                        credit = (credit + rate * dt).min((rate * 0.1).max(1.0));
                        if credit >= 1.0 {
                            credit -= 1.0;
                            w.step();
                            if rate < 120.0 {
                                w.flush();
                                last_flush = Instant::now();
                                if rate < 10.0 {
                                    last_publish = long_ago;
                                }
                            }
                        }
                    }
                }
            }
            if last_flush.elapsed() > Duration::from_millis(16) {
                w.flush();
                last_flush = Instant::now();
            }
            if last_palette.elapsed() > Duration::from_secs(2) {
                let cells = dump::sample(w, 2000);
                let tops = w.field_tops();
                let palette = (cells.len() >= 8).then(|| Arc::new(Palette::fit(&cells)));
                let mut st = self.status.lock().unwrap();
                st.tops = tops;
                if palette.is_some() {
                    st.palette = palette;
                }
                drop(st);
                last_palette = Instant::now();
            }
            if last_lock_check.elapsed() > Duration::from_secs(5) {
                let software = self.gpu.name().to_lowercase().contains("llvmpipe");
                let sharing = !software && gpu_busy_elsewhere();
                w.pacing.duty = if sharing { DUTY_SHARING } else { DUTY };
                self.status.lock().unwrap().sharing = sharing;
                last_lock_check = Instant::now();
            }
            if last_publish.elapsed() > Duration::from_millis(100) {
                w.flush();
                let cells = w.alive();
                let tally = w.tally();
                let elapsed = rate_at.elapsed().as_secs_f64();
                let mut st = self.status.lock().unwrap();
                let ticks = (w.tick - rate_tick) as f64;
                // Long enough to hold a tick or two, so slow speeds don't read as stopped.
                if elapsed > 4.0 || (elapsed > 1.2 && ticks >= 2.0) || (elapsed > 0.5 && ticks >= 8.0) {
                    st.tps = ticks / elapsed.max(1e-6);
                    if ticks > 0.0 {
                        st.gpu_ms_per_tick = (w.gpu_seconds - rate_gpu) * 1000.0 / ticks;
                    }
                    st.longest_ms = w.longest_submit_ms;
                    w.longest_submit_ms = 0.0;
                    rate_tick = w.tick;
                    rate_at = Instant::now();
                    rate_gpu = w.gpu_seconds;
                }
                st.tick = w.tick;
                st.cells = cells;
                st.flip = w.flip();
                st.running = running;
                st.overflow = tally.overflow;
                st.births = tally.births;
                drop(st);
                last_publish = Instant::now();
                (self.repaint)();
            }
            if !running && steps == 0 {
                // Nothing to do but wait for the next command.
                match self.rx.recv_timeout(Duration::from_millis(30)) {
                    Ok(c) => next = Some(c),
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => return,
                }
            }
        }
    }

    /// Lets go of a world's buffers before making another, so both never need to fit at once.
    fn release(&self, live: &mut Option<World>) {
        if let Some(w) = live {
            w.flush();
        }
        *live = None;
        self.status.lock().unwrap().view = None;
        let _ = self.gpu.device.poll(wgpu::PollType::Poll);
    }
}
