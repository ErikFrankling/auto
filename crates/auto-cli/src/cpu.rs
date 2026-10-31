//! CPU measurements, written to the same results file as the GPU ones.

use std::io::Write;
use std::time::Instant;

use auto_cpu::bitgrid::{BitGrid, Kernel};
use auto_cpu::hashlife::{self, HashLife};
use clap::Parser;
use serde::Serialize;

#[derive(Parser)]
pub struct Args {
    #[arg(long, default_value = "1024x1024,4096x4096,16384x16384")]
    pub sizes: String,
    #[arg(long, default_value_t = 2.0)]
    pub seconds: f64,
    #[arg(long, default_value = "results/cpu.jsonl")]
    pub out: String,
    /// Run the Hashlife scale test as well.
    #[arg(long)]
    pub hashlife: bool,
    #[arg(long, default_value_t = 4)]
    pub threads: usize,
}

#[derive(Serialize)]
struct Sample {
    id: String,
    threads: usize,
    width: usize,
    height: usize,
    cells: u64,
    memory_bytes: u64,
    generations_per_second: f64,
    cells_per_second: f64,
}

#[derive(Serialize)]
struct HashSample {
    pattern: String,
    generation: String,
    universe_side_log2: u32,
    population: String,
    seconds: f64,
    nodes: usize,
    memory_bytes: usize,
}

fn append(path: &str, line: String) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("results file");
    writeln!(f, "{line}").expect("write");
}

pub fn run(args: Args) {
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .ok();
    for size in args.sizes.split(',') {
        let (w, h) = size.split_once('x').expect("WxH");
        let (w, h): (usize, usize) = (w.parse().expect("width"), h.parse().expect("height"));
        for (name, kernel) in [
            ("cpu.bytes", Kernel::Bytes),
            ("cpu.swar64", Kernel::Swar),
            ("cpu.avx2", Kernel::Avx2),
            ("cpu.avx512", Kernel::Avx512),
        ] {
            if !kernel.available() || (kernel == Kernel::Bytes && w * h > 1 << 26) {
                continue;
            }
            for threads in [1, args.threads] {
                let threaded = threads > 1;
                let mut g = BitGrid::random(w, h, 7);
                for _ in 0..8 {
                    g.step(kernel, threaded);
                }
                let start = Instant::now();
                let mut gens = 0u64;
                while start.elapsed().as_secs_f64() < args.seconds {
                    for _ in 0..8 {
                        g.step(kernel, threaded);
                    }
                    gens += 8;
                }
                let rate = gens as f64 / start.elapsed().as_secs_f64();
                let cells = (w * h) as u64;
                println!(
                    "{name:<12} t{threads} {w}x{h}  {rate:>12.1} gen/s {:>10.3} Gcell/s",
                    rate * cells as f64 / 1e9
                );
                append(
                    &args.out,
                    serde_json::to_string(&Sample {
                        id: name.into(),
                        threads,
                        width: w,
                        height: h,
                        cells,
                        memory_bytes: cells / 4,
                        generations_per_second: rate,
                        cells_per_second: rate * cells as f64,
                    })
                    .expect("json"),
                );
            }
        }
    }
    if args.hashlife {
        for (name, cells, exps) in [
            (
                "gosper_gun",
                hashlife::gosper_gun(),
                vec![20u32, 30, 40, 50, 60, 70],
            ),
            ("r_pentomino", hashlife::r_pentomino(), vec![20, 40, 60, 70]),
        ] {
            for e in exps {
                let mut h = HashLife::from_cells(&cells);
                let start = Instant::now();
                h.advance(1u128 << e);
                let seconds = start.elapsed().as_secs_f64();
                println!(
                    "hashlife {name} generation 2^{e}: population {}, universe 2^{} per side, {seconds:.3}s, {} nodes",
                    h.population(),
                    h.universe_level(),
                    h.node_count()
                );
                append(
                    &args.out,
                    serde_json::to_string(&HashSample {
                        pattern: name.into(),
                        generation: format!("2^{e}"),
                        universe_side_log2: h.universe_level() as u32,
                        population: h.population().to_string(),
                        seconds,
                        nodes: h.node_count(),
                        memory_bytes: h.memory_bytes(),
                    })
                    .expect("json"),
                );
            }
        }
    }
}
