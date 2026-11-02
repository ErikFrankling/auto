//! Runs a soft life world on the GPU and logs what happens.
//!   soft --size 256 --ticks 20000 --out run.jsonl --png run.png
//!   soft --size 256 --set stick_on=false --seed 3 --save world.bin
//!   soft --size 1024 --dump-every 2000 --dump-dir dumps/ --ticks 40000
//!   soft --shaders my-physics/ --set bite_power=45 --size 512 --ticks 20000

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use auto_life_soft::world::{ROUTES, Tally};
use auto_life_soft::{Config, Pacing, VRAM_MARGIN, World, dump, free_vram, lock_gpu};
use clap::Parser;
use serde_json::{Map, Value, json};

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
    /// Where the JSONL log goes. Standard output if not given.
    #[arg(long)]
    out: Option<String>,
    /// Directory with defaults.json and the .wgsl files, to run edited physics with no rebuild.
    #[arg(long)]
    shaders: Option<PathBuf>,
    /// Physics setting, as key=value. Repeat for more.
    #[arg(long = "set")]
    sets: Vec<String>,
    /// Most cells the world can hold. Defaults to half the squares, up to what fits.
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
    /// Save the final world here.
    #[arg(long)]
    save: Option<String>,
    /// Start from a saved world instead of an empty one.
    #[arg(long)]
    load: Option<String>,
    /// Comma list of saved smaller worlds. Their living cells are copied in as tiles.
    #[arg(long)]
    seed_from: Option<String>,
    /// Share of those cells to keep.
    #[arg(long, default_value_t = 1.0)]
    seed_keep: f64,
    /// Every this many ticks, write every living cell to --dump-dir, for soft-analyse.
    #[arg(long, default_value_t = 0)]
    dump_every: u64,
    #[arg(long)]
    dump_dir: Option<String>,
    /// When the run ends, draw every cell as a disc, coloured by genome, to this PNG.
    #[arg(long)]
    png: Option<String>,
    /// When the run ends, print a hash of everything physical, to compare two runs.
    #[arg(long)]
    fingerprint: bool,
    /// Measure each pass with timestamp queries, and log where the time goes. Slower.
    #[arg(long)]
    profile: bool,
}

fn main() {
    let args = Args::parse();
    if let Err(why) = run(args) {
        eprintln!("refused: {why}");
        std::process::exit(3);
    }
}

fn run(args: Args) -> Result<(), String> {
    let mut cfg = match &args.load {
        Some(path) => Config::from_save(path, args.shaders.clone())?,
        None => {
            let mut cfg = Config::new(args.size, args.seed, args.shaders.clone())?;
            cfg.hidden_rows = args.hidden_rows;
            cfg.shard_slots = cfg.max_shard_slots();
            cfg.slots = args.slots.unwrap_or_else(|| cfg.default_slots());
            cfg
        }
    };
    for s in &args.sets {
        let (k, v) = s
            .split_once('=')
            .ok_or_else(|| format!("--set {s} needs key=value"))?;
        cfg.set(k, v)?;
    }

    let queued = std::env::var("LIFE_LOCK_HELD").is_ok();
    let max_minutes = std::env::var("LIFE_MAX_MINUTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .map_or(args.max_minutes, |cap: f64| args.max_minutes.min(cap));
    let mut lock = lock_gpu();
    let mut held_since = Instant::now();
    let mut waited = 0.0f64;
    let bytes = cfg.vram_bytes();
    let software = std::env::var("AUTO_ADAPTER").is_ok_and(|a| a.contains("llvmpipe"));
    if !args.force
        && !software
        && let Some(free) = free_vram()
        && bytes + VRAM_MARGIN > free
    {
        return Err(format!(
            "world needs {:.1} GB and only {:.1} GB of VRAM is free, keeping 3 GB spare",
            bytes as f64 / 1e9,
            free as f64 / 1e9
        ));
    }
    let size = cfg.size;
    eprintln!(
        "world {size}x{size}, up to {:.2} million cells, {:.2} GB of VRAM",
        cfg.slots as f64 / 1e6,
        bytes as f64 / 1e9
    );
    let duty = std::env::var("LIFE_DUTY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(args.duty);
    let mut world = World::new(
        cfg,
        Pacing {
            target_ms: args.target_ms,
            duty,
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
    if let Some(list) = &args.seed_from {
        let paths: Vec<String> = list.split(',').map(str::to_string).collect();
        let planted = world.seed_from(&paths, args.seed_keep)?;
        eprintln!("planted {planted} cells from {} worlds", paths.len());
    }
    if args.profile && !world.enable_profile() {
        eprintln!("this GPU has no timestamp queries, so no profile");
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
    writeln!(log, "{}", json!({"config": world.cfg.physics, "size": size, "slots": world.cfg.slots, "seed": world.cfg.seed, "hidden_rows": world.cfg.hidden_rows, "open": world.open, "gpu": world.gpu.name()}))
        .map_err(|e| e.to_string())?;

    // A batch keeps many worlds on one map. Their land differs, so note what each one got.
    let tile = world.cfg.f("batch_tile") as usize;
    if tile > 0 {
        let terrain = world.read_terrain();
        let per_side = size as usize / tile;
        let mut tiles = vec![];
        for i in 0..per_side * per_side {
            let (ox, oy) = (i % per_side * tile, i / per_side * tile);
            let (mut open, mut sun) = (0u64, 0.0f64);
            for y in oy..oy + tile {
                for v in &terrain[y * size as usize + ox..y * size as usize + ox + tile] {
                    if v & (1 << 24) == 0 {
                        open += 1;
                        sun += (v & 0xFFFF) as f64 / 65535.0;
                    }
                }
            }
            tiles.push(json!({"tile": i, "open": open, "sun": sun / open.max(1) as f64 * auto_life_soft::config::SUN_RANGE as f64 * world.sun_k as f64}));
        }
        writeln!(log, "{}", json!({"tiles": tiles, "batch_tile": tile})).map_err(|e| e.to_string())?;
    }

    // What was there before any of the flows counted so far, so the ledger can be checked.
    let stock_base = {
        let t = world.tally();
        let s = world.stock();
        s.total() - ((t.flow("sun") + t.flow("rain")) as i128 - t.lost())
    };
    let started = Instant::now();
    let mut last = world.tally();
    let (mut last_tick, mut last_gpu, mut last_wall) = (world.tick, 0.0, 0.0);
    let mut last_submits = world.submits;
    for t in 1..=args.ticks {
        world.step();
        if let Some(dir) = &args.dump_dir
            && args.dump_every > 0
            && t % args.dump_every == 0
        {
            let _ = std::fs::create_dir_all(dir);
            let path = format!("{dir}/t{:09}.bin", world.tick);
            dump::write(&mut world, &path)?;
        }
        if !queued && held_since.elapsed().as_secs_f64() > args.hold_minutes * 60.0 {
            world.flush();
            drop(lock);
            std::thread::sleep(std::time::Duration::from_millis(500));
            let queue = Instant::now();
            lock = lock_gpu();
            waited += queue.elapsed().as_secs_f64() + 0.5;
            held_since = Instant::now();
        }
        let out_of_time = started.elapsed().as_secs_f64() - waited > max_minutes * 60.0;
        if t % args.log_every == 0 || t == args.ticks || out_of_time {
            world.flush();
            let wall = started.elapsed().as_secs_f64() - waited;
            let ticks = (world.tick - last_tick).max(1) as f64;
            let mut r = report(&mut world, &last, stock_base);
            last = world.tally();
            r["seconds"] = json!(wall);
            r["ticks_per_second"] = json!(ticks / (wall - last_wall).max(1e-6));
            r["gpu_ms_per_tick"] = json!((world.gpu_seconds - last_gpu) * 1000.0 / ticks);
            r["longest_submit_ms"] = json!(world.longest_submit_ms);
            r["mean_submit_ms"] = json!((world.gpu_seconds - last_gpu) * 1000.0 / (world.submits - last_submits).max(1) as f64);
            r["seed"] = json!(world.cfg.seed);
            if let Some(p) = world.profile_report() {
                let map: Map<String, Value> = p.into_iter().map(|(n, ms)| (n.to_string(), json!(ms))).collect();
                r["gpu_ms_by_pass"] = Value::Object(map);
            }
            writeln!(log, "{r}").map_err(|e| e.to_string())?;
            log.flush().map_err(|e| e.to_string())?;
            eprintln!(
                "tick {:>7}  cells {:>8}  births {:>6}  deaths {:>6}  overflow {:>5}  {:.0} ticks/s  gpu {:.3} ms/tick  longest submit {:.1} ms",
                world.tick,
                r["cells"],
                r["births"],
                r["deaths"],
                r["bin_overflows"],
                r["ticks_per_second"].as_f64().unwrap_or(0.0),
                r["gpu_ms_per_tick"].as_f64().unwrap_or(0.0),
                world.longest_submit_ms,
            );
            (last_tick, last_gpu, last_wall) = (world.tick, world.gpu_seconds, wall);
            // The clock can't tell this world's work from waiting behind another program on the
            // GPU. A world that's too big is slow nearly every time.
            let submits = (world.submits - last_submits).max(1);
            last_submits = world.submits;
            if !args.force
                && !software
                && world.slow_submits >= 5
                && world.slow_submits as u64 * 4 >= submits
            {
                return Err(format!(
                    "{} of {} submissions held the GPU over 80 ms, the longest {:.0} ms, so the desktop would stutter. Use a smaller world.",
                    world.slow_submits, submits, world.longest_submit_ms
                ));
            }
            world.longest_submit_ms = 0.0;
            world.slow_submits = 0;
        }
        if out_of_time {
            eprintln!("stopping at the time limit of {max_minutes} minutes");
            break;
        }
    }
    if let Some(path) = &args.png {
        let cells = dump::living(&mut world);
        dump::write_png(&cells, world.cfg.size, path)?;
        eprintln!("wrote {path}");
    }
    if args.fingerprint {
        println!("fingerprint {:016x} at tick {}", world.fingerprint(), world.tick);
    }
    if let Some(path) = &args.save {
        world.save(path)?;
        eprintln!("saved {path}");
    }
    drop(lock);
    Ok(())
}

/// A line of the log. Flows are totals since the last line, in energy. With the ledger off they
/// are estimates from one cell in 64. With it on they're exact, and so is the gap.
fn report(world: &mut World, last: &Tally, stock_base: i128) -> Value {
    let now = world.tally();
    let stock = world.stock();
    let per_tick = 1.0 / 65536.0;
    let mut flows = Map::new();
    for (k, name) in ROUTES.iter().enumerate() {
        // Scaled up when only one cell in 64 was counted.
        flows.insert(
            name.to_string(),
            json!((now.flows[k] - last.flows[k]) as f64 * now.sample as f64 * per_tick),
        );
    }
    let mut r = json!({
        "tick": world.tick,
        "cells": stock.alive,
        "births": now.births - last.births,
        "rained": now.rained - last.rained,
        "deaths": (now.starved + now.bitten + now.burned + now.old) - (last.starved + last.bitten + last.burned + last.old),
        "starved": now.starved - last.starved,
        "bitten": now.bitten - last.bitten,
        "burned": now.burned - last.burned,
        "old": now.old - last.old,
        "bin_overflows": now.overflow - last.overflow,
        "no_slot": now.no_slot - last.no_slot,
        "flows": flows,
        "flows_sampled": now.sample != 1,
        "energy_total": stock.total() as f64 * per_tick,
        "stock": {
            "food": stock.food as f64 * per_tick,
            "meat": stock.meat as f64 * per_tick,
            "cells": stock.energy as f64 * per_tick,
            "wombs": stock.womb as f64 * per_tick,
            "bodies": (stock.alive * stock.body_each) as f64 * per_tick,
            "heat": stock.heat as f64 * per_tick,
            "enzyme": stock.enzyme as f64 * per_tick,
        },
    });
    if now.sample == 1 {
        // In and out since the world began.
        r["ledger_in"] = json!((now.flow("sun") + now.flow("rain")) as f64 * per_tick);
        r["ledger_out"] = json!(now.lost() as f64 * per_tick);
        // Energy now, less what the flows say should be here. Zero, to the last unit, or a rule is wrong.
        let gap = stock.total() - (stock_base + (now.flow("sun") + now.flow("rain")) as i128 - now.lost());
        r["ledger_gap_units"] = json!(gap as i64);
        if gap != 0 {
            eprintln!("the ledger is out by {gap} units at tick {}", world.tick);
        }
    }
    r
}
