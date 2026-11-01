//! Runs a thermodynamic life world on the GPU and logs what happens.
//!   life --size 1024 --ticks 60000 --out run.jsonl
//!   life --size 4096 --ticks 20000 --video run.mp4 --frame-every 4 --zoom 0.25
//!   life --shaders my-physics/ --set bite_power=45 --size 512 --ticks 20000 --out run.jsonl

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use auto_life_gpu::report::{Tracker, report};
use auto_life_gpu::video::{Layout, Recorder, RecorderOptions};
use auto_life_gpu::{Config, Pacing, World, free_vram, lock_gpu};
use clap::Parser;
use serde_json::json;

/// VRAM always left free for the desktop and everything else.
const VRAM_MARGIN: u64 = 3 << 30;

#[derive(Parser)]
struct Args {
    /// World width and height in squares. A power of two.
    #[arg(long, default_value_t = 256)]
    size: u32,
    #[arg(long, default_value_t = 1)]
    seed: u32,
    #[arg(long, default_value_t = 10000)]
    ticks: u64,
    #[arg(long, default_value_t = 2000)]
    log_every: u64,
    #[arg(long)]
    out: Option<String>,
    /// Directory with defaults.json and the .wgsl files, to run edited physics with no rebuild.
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// Physics setting, as key=value. Repeat for more.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Most cells the world can hold. Defaults to a third of the squares, up to what fits.
    #[arg(long)]
    slots: Option<u32>,
    #[arg(long, default_value_t = 2)]
    hidden_rows: u32,
    /// Aim for GPU submissions this long, so the desktop never waits for more.
    #[arg(long, default_value_t = 8.0)]
    target_ms: f64,
    /// Share of time the GPU may be busy with this job.
    #[arg(long, default_value_t = 0.7)]
    duty: f64,
    /// Stop cleanly after this much running, not counting time spent waiting for the GPU.
    #[arg(long, default_value_t = 15.0)]
    max_minutes: f64,
    /// Let other GPU jobs have a turn after holding the card this long.
    #[arg(long, default_value_t = 3.0)]
    hold_minutes: f64,
    /// Run even if it would nearly fill VRAM or hold the GPU too long. Only if Erik asked.
    #[arg(long)]
    force: bool,
    #[arg(long)]
    video: Option<String>,
    #[arg(long, default_value_t = 4)]
    frame_every: u64,
    #[arg(long, default_value_t = 0)]
    video_from: u64,
    #[arg(long, default_value_t = 1920)]
    width: u32,
    #[arg(long, default_value_t = 1080)]
    height: u32,
    /// genome, diet or split.
    #[arg(long, default_value = "split")]
    view: String,
    /// Cells per pixel when the camera zooms in. 0 keeps the whole world in view.
    #[arg(long, default_value_t = 0.0)]
    zoom: f32,
    #[arg(long, default_value_t = 8.0)]
    seconds_per_shot: f32,
    /// Save the final world here.
    #[arg(long)]
    save: Option<String>,
    /// Start from a saved world instead of an empty one.
    #[arg(long)]
    load: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(why) = run(args) {
        eprintln!("refused: {why}");
        std::process::exit(3);
    }
}

fn run(args: Args) -> Result<(), String> {
    let mut cfg = Config::new(args.size, args.seed, args.shaders.clone())?;
    for s in &args.sets {
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("--set {s} needs key=value"))?;
        cfg.set(k, v)?;
    }
    cfg.hidden_rows = args.hidden_rows;
    cfg.slots = args.slots.unwrap_or_else(|| cfg.default_slots());

    let mut lock = lock_gpu();
    let mut held_since = Instant::now();
    let mut waited = 0.0f64;
    let bytes = cfg.vram_bytes();
    if !args.force
        && let Some(free) = free_vram()
        && bytes + VRAM_MARGIN > free
    {
        return Err(format!(
            "world needs {:.1} GB and only {:.1} GB of VRAM is free, keeping 3 GB spare",
            bytes as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    eprintln!(
        "world {0}x{0} ({1:.1} million squares), up to {2:.2} million cells, {3:.2} GB of VRAM",
        args.size,
        (args.size as f64).powi(2) / 1e6,
        cfg.slots as f64 / 1e6,
        bytes as f64 / 1e9
    );
    let mut world = World::new(
        cfg,
        Pacing {
            target_ms: args.target_ms,
            duty: args.duty,
        },
    )?;
    eprintln!(
        "gpu: {}, sun scale {:.4}, open squares {}",
        world.gpu.name(),
        world.sun_k,
        world.open
    );
    if let Some(path) = &args.load {
        world.load(path)?;
        eprintln!("loaded {path} at tick {}", world.tick);
    }

    let mut log: Box<dyn Write> = match &args.out {
        Some(p) => {
            if let Some(dir) = std::path::Path::new(p).parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            Box::new(std::fs::File::create(p).map_err(|e| e.to_string())?)
        }
        None => Box::new(std::io::stdout()),
    };
    writeln!(log, "{}", json!({"config": world.cfg.physics, "size": args.size, "slots": world.cfg.slots, "seed": args.seed, "hidden_rows": args.hidden_rows, "open": world.open}))
        .map_err(|e| e.to_string())?;

    let layout = match args.view.as_str() {
        "genome" => Layout::Genome,
        "diet" => Layout::Diet,
        "split" => Layout::Split,
        other => return Err(format!("unknown view {other}")),
    };
    let mut recorder = match &args.video {
        Some(path) => Some(Recorder::new(
            &world,
            path,
            RecorderOptions {
                width: args.width,
                height: args.height,
                fps: 30,
                layout,
                zoom: args.zoom,
                seconds_per_shot: args.seconds_per_shot,
                label: format!(
                    "{0} x {0} squares. Left: colour = genome. Right: green ground, red meat and biting, blue gifts",
                    args.size
                ),
                threads: 2,
            },
        )?),
        None => None,
    };

    let mut tracker = Tracker {
        last: world.counters(),
    };
    let started = Instant::now();
    let first = world.tick;
    let (mut last_tick, mut last_gpu, mut last_wall) = (first, 0.0, 0.0);
    for t in 1..=args.ticks {
        world.step();
        if let Some(rec) = &mut recorder
            && world.tick >= args.video_from
            && t % args.frame_every == 0
        {
            rec.frame(&mut world);
        }
        if held_since.elapsed().as_secs_f64() > args.hold_minutes * 60.0 {
            world.flush();
            drop(lock);
            std::thread::sleep(std::time::Duration::from_millis(500));
            let queue = Instant::now();
            lock = lock_gpu();
            waited += queue.elapsed().as_secs_f64() + 0.5;
            held_since = Instant::now();
        }
        let out_of_time = started.elapsed().as_secs_f64() - waited > args.max_minutes * 60.0;
        if t % args.log_every == 0 || t == args.ticks || out_of_time {
            world.flush();
            let mut r = report(&mut world, &mut tracker);
            let wall = started.elapsed().as_secs_f64() - waited;
            let ticks = (world.tick - last_tick).max(1) as f64;
            r["seconds"] = json!(wall);
            r["ticks_per_second"] = json!(ticks / (wall - last_wall).max(1e-6));
            r["gpu_ms_per_tick"] = json!((world.gpu_seconds - last_gpu) * 1000.0 / ticks);
            r["longest_submit_ms"] = json!(world.longest_submit_ms);
            r["seed"] = json!(args.seed);
            writeln!(log, "{r}").map_err(|e| e.to_string())?;
            log.flush().map_err(|e| e.to_string())?;
            eprintln!(
                "tick {:>7}  cells {:>9}  species {:>6}  {:.0} ticks/s  gpu {:.2} ms/tick  longest submit {:.1} ms",
                world.tick,
                r["cells"],
                r.get("effective_species")
                    .and_then(|v| v.as_f64())
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_default(),
                r["ticks_per_second"].as_f64().unwrap_or(0.0),
                r["gpu_ms_per_tick"].as_f64().unwrap_or(0.0),
                world.longest_submit_ms,
            );
            (last_tick, last_gpu, last_wall) = (world.tick, world.gpu_seconds, wall);
            let software = std::env::var("AUTO_ADAPTER").is_ok_and(|a| a.contains("llvmpipe"));
            if !args.force && !software && world.longest_submit_ms > 80.0 {
                return Err(format!(
                    "a submission held the GPU for {:.0} ms, so the desktop would stutter. Use a smaller world.",
                    world.longest_submit_ms
                ));
            }
            world.longest_submit_ms = 0.0;
        }
        if out_of_time {
            eprintln!("stopping at the time limit of {} minutes", args.max_minutes);
            break;
        }
    }
    drop(lock);
    if let Some(rec) = recorder {
        rec.finish();
    }
    if let Some(path) = &args.save {
        let _lock = lock_gpu();
        world.save(path)?;
        eprintln!("saved {path}");
    }
    Ok(())
}
