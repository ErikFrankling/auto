//! Throughput measurement.
//!
//! Grows the batch until one submit takes about 20 ms, then runs batches for
//! `--seconds` and reports the median. Staying near 20 ms per submit keeps the
//! desktop responsive, which shares this GPU.

use std::io::Write;
use std::time::Instant;

use auto_core::{Gpu, GpuOptions, Sim, automata};
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
pub struct Args {
    /// Automaton or kernel id.
    pub id: String,
    /// Sizes as WxH, comma separated.
    #[arg(long, default_value = "1024x1024")]
    pub sizes: String,
    #[arg(long, default_value_t = 3.0)]
    pub seconds: f64,
    #[arg(long, default_value = "results/bench.jsonl")]
    pub out: String,
    /// Tag recorded with the result.
    #[arg(long, default_value = "")]
    pub tag: String,
    /// Steps to run before timing starts, so the state is past its initial burst.
    #[arg(long, default_value_t = 32)]
    pub warmup: u32,
    /// Longest one GPU step may hold the card, since the desktop shares it.
    #[arg(long, default_value_t = 50.0)]
    pub max_step_ms: f64,
    /// Run even if it would stall the desktop or nearly fill VRAM.
    #[arg(long)]
    pub force: bool,
}

#[derive(Serialize)]
pub struct Sample {
    pub id: String,
    pub tag: String,
    pub adapter: String,
    pub checks: String,
    pub width: u32,
    pub height: u32,
    pub cells: u64,
    pub memory_bytes: u64,
    pub steps_per_second: f64,
    pub generations_per_second: f64,
    pub cells_per_second: f64,
    pub batch_steps: u32,
    pub batches: u32,
    pub spread: f64,
}

fn parse_size(s: &str) -> (u32, u32) {
    let (w, h) = s.split_once('x').expect("size looks like WxH");
    (w.parse().expect("width"), h.parse().expect("height"))
}

pub fn run(opts: GpuOptions, args: Args) {
    let gpu = pollster::block_on(Gpu::headless(opts)).expect("no gpu");
    // Always small to big, and stop climbing once a size is refused.
    let mut sizes: Vec<&str> = args.sizes.split(',').collect();
    sizes.sort_by_key(|s| {
        let (w, h) = parse_size(s);
        w as u64 * h as u64
    });
    let limits = crate::guard::Limits {
        max_step_ms: args.max_step_ms,
        force: args.force,
    };
    for size in sizes {
        let (w, h) = parse_size(size);
        if let Err(why) = crate::guard::check(&gpu, &args.id, w, h, &limits) {
            eprintln!("skipped: {why}");
            break;
        }
        let Some(program) = automata::build(&args.id, w, h) else {
            eprintln!("unknown id {} or size {size} unsupported", args.id);
            std::process::exit(2);
        };
        match measure(&gpu, program, &args) {
            Ok(sample) => {
                println!(
                    "{:<26} {:>7}x{:<7} {:>9.1} MB {:>12.1} gen/s {:>14.3} Gcell/s  batch {} spread {:.1}%",
                    sample.id,
                    sample.width,
                    sample.height,
                    sample.memory_bytes as f64 / 1e6,
                    sample.generations_per_second,
                    sample.cells_per_second / 1e9,
                    sample.batch_steps,
                    sample.spread * 100.0
                );
                if let Some(parent) = std::path::Path::new(&args.out).parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&args.out)
                    .expect("results file");
                writeln!(f, "{}", serde_json::to_string(&sample).expect("json")).expect("write");
            }
            Err(e) => eprintln!("{} {size}: {e}", args.id),
        }
    }
}

fn measure(gpu: &Gpu, program: auto_core::Program, args: &Args) -> Result<Sample, String> {
    let (id, w, h, cells, memory) = (
        program.id.clone(),
        program.width,
        program.height,
        program.cells_per_step,
        program.total_bytes(),
    );
    let gens = program.generations_per_step as f64;
    let mut sim = Sim::new(gpu, program, 1)?;
    gpu.wait();
    sim.step(gpu, args.warmup);
    gpu.wait();

    let mut batch = 1u32;
    loop {
        let t = Instant::now();
        sim.step(gpu, batch);
        gpu.wait();
        let dt = t.elapsed().as_secs_f64();
        if dt >= 0.02 || batch >= 1 << 16 {
            break;
        }
        let want = (batch as f64 * 0.025 / dt.max(1e-5)).ceil() as u32;
        batch = want.clamp(batch + 1, batch.saturating_mul(8)).min(1 << 16);
    }

    let mut rates = Vec::new();
    let start = Instant::now();
    while start.elapsed().as_secs_f64() < args.seconds || rates.len() < 3 {
        let t = Instant::now();
        sim.step(gpu, batch);
        gpu.wait();
        rates.push(batch as f64 / t.elapsed().as_secs_f64());
    }
    rates.sort_by(|a, b| a.total_cmp(b));
    let median = rates[rates.len() / 2];
    let spread = (rates[rates.len() * 9 / 10] - rates[rates.len() / 10]) / median;
    Ok(Sample {
        id,
        tag: args.tag.clone(),
        adapter: gpu.name(),
        checks: format!("{:?}", gpu.checks),
        width: w,
        height: h,
        cells,
        memory_bytes: memory,
        steps_per_second: median,
        generations_per_second: median * gens,
        cells_per_second: median * cells as f64 * gens,
        batch_steps: batch,
        batches: rates.len() as u32,
        spread,
    })
}
