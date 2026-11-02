//! Measures bodies on the CPU, from a run's dumps or from a saved world.
//!   soft-analyse dumps/ --out summary.json
//!   soft-analyse world.bin
//!   soft-analyse dumps/ --png-last last.png
//! A batch of walled worlds gets a summary for each world.

use auto_life_soft::dump::{self, Dump};
use auto_life_soft::measure::{Frame, Rules, split_batch, summary};
use clap::Parser;
use serde_json::{Value, json};

#[derive(Parser)]
struct Args {
    /// A directory of dumps, one dump, or a saved world.
    path: String,
    /// Where the summary goes. Standard output if not given.
    #[arg(long)]
    out: Option<String>,
    /// Pull below this doesn't count as being stuck.
    #[arg(long, default_value_t = 0.01)]
    min_pull: f32,
    /// Draw the last dump as discs to this PNG.
    #[arg(long)]
    png_last: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(why) = run(args) {
        eprintln!("{why}");
        std::process::exit(1);
    }
}

fn load(path: &str) -> Result<Vec<Dump>, String> {
    if std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?.is_dir() {
        let mut files: Vec<String> = std::fs::read_dir(path)
            .map_err(|e| format!("{path}: {e}"))?
            .flatten()
            .map(|e| e.path().to_string_lossy().to_string())
            .filter(|p| p.ends_with(".bin"))
            .collect();
        files.sort();
        return files.iter().map(|f| dump::read(f)).collect();
    }
    match dump::read(path) {
        Ok(d) => Ok(vec![d]),
        Err(_) => Ok(vec![auto_life_soft::world::read_saved_cells(path)?]),
    }
}

fn run(args: Args) -> Result<(), String> {
    let dumps = load(&args.path)?;
    let Some(first) = dumps.first() else {
        return Err(format!("no dumps in {}", args.path));
    };
    let size = first.size;
    let physics = first.physics.clone();
    if let Some(png) = &args.png_last {
        let last = dumps.last().unwrap();
        dump::write_png(&last.cells, last.size, png)?;
    }
    let mut rules = Rules::from_physics(&physics, size);
    rules.min_pull = args.min_pull;
    let tile = physics["batch_tile"].as_u64().unwrap_or(0) as u32;
    let result = if tile > 0 {
        rules.size = tile as f32;
        let per_side = size / tile;
        let worlds: Vec<Value> = (0..per_side * per_side)
            .map(|i| {
                let frames: Vec<Frame> = dumps
                    .iter()
                    .map(|d| {
                        let cells = split_batch(d, tile).swap_remove(i as usize).1;
                        Frame::new(d.tick, cells, &rules)
                    })
                    .collect();
                let mut s = summary(&frames, &rules);
                s["world"] = json!(i);
                s
            })
            .collect();
        json!({"size": size, "batch_tile": tile, "worlds": worlds})
    } else {
        let frames: Vec<Frame> = dumps
            .iter()
            .map(|d| Frame::new(d.tick, d.cells.clone(), &rules))
            .collect();
        let mut s = summary(&frames, &rules);
        s["size"] = json!(size);
        s
    };
    let text = serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?;
    match &args.out {
        Some(p) => std::fs::write(p, text).map_err(|e| format!("{p}: {e}"))?,
        None => println!("{text}"),
    }
    Ok(())
}
