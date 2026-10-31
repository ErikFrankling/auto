//! Keeps experiments from freezing the desktop, which shares this GPU.
//!
//! Before a big run: check it fits in free VRAM with room to spare, then time the
//! same automaton from small to big and predict how long one GPU step will hold
//! the card. A run that would stall the screen is refused unless forced.

use std::time::Instant;

use auto_core::{Gpu, Program, Sim, automata};

/// VRAM always left free for the desktop and everything else.
const VRAM_MARGIN: u64 = 3 << 30;
/// Grids at or below this size are always safe to run.
const SAFE_CELLS: u64 = 1 << 20;

pub struct Limits {
    /// Longest the GPU may be held by one step, in milliseconds.
    pub max_step_ms: f64,
    pub force: bool,
}

fn read_number(path: std::path::PathBuf) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Free VRAM on the biggest GPU, on Linux with amdgpu. None elsewhere.
pub fn free_vram() -> Option<u64> {
    let mut best: Option<(u64, u64)> = None;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let device = entry.path().join("device");
        let (Some(total), Some(used)) = (
            read_number(device.join("mem_info_vram_total")),
            read_number(device.join("mem_info_vram_used")),
        ) else {
            continue;
        };
        if best.is_none_or(|(t, _)| total > t) {
            best = Some((total, used));
        }
    }
    best.map(|(total, used)| total.saturating_sub(used))
}

/// Time one step of `program`, in milliseconds.
fn time_step(gpu: &Gpu, program: Program) -> Result<f64, String> {
    let mut sim = Sim::new(gpu, program, 1)?;
    gpu.wait();
    sim.step(gpu, 1);
    gpu.wait();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        sim.step(gpu, 1);
        gpu.wait();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    Ok(best)
}

/// Err explains why this run should not start.
pub fn check(gpu: &Gpu, id: &str, w: u32, h: u32, limits: &Limits) -> Result<(), String> {
    if limits.force {
        return Ok(());
    }
    let program = automata::build(id, w, h).ok_or_else(|| format!("unknown id {id}"))?;
    let cells = w as u64 * h as u64;
    let bytes = program.total_bytes();
    if let Some(free) = free_vram()
        && bytes + VRAM_MARGIN > free
    {
        return Err(format!(
            "{id} {w}x{h} needs {:.1} GB and only {:.1} GB of VRAM is free, keeping {:.0} GB spare. Use --force to run anyway.",
            bytes as f64 / 1e9,
            free as f64 / 1e9,
            VRAM_MARGIN as f64 / (1u64 << 30) as f64
        ));
    }
    if cells <= SAFE_CELLS {
        return Ok(());
    }
    // Sharded kernels only take certain widths; the single buffer one predicts them well.
    let probe_id = if id.starts_with("life.sharded") {
        "life"
    } else {
        id
    };
    // Walk up in factors of 4 cells, stopping as soon as the prediction is over the limit.
    let one_dimensional = h <= 2;
    let mut sizes = Vec::new();
    let (mut pw, mut ph) = (w, h);
    while (pw as u64 * ph as u64) > SAFE_CELLS {
        if one_dimensional {
            pw /= 4;
        } else {
            pw /= 2;
            ph /= 2;
        }
        sizes.push((pw.max(64), ph.max(1)));
    }
    sizes.reverse();
    let mut predicted = 0.0;
    let mut last = (0, 0, 0.0);
    for (sw, sh) in sizes {
        let program = automata::build(probe_id, sw, sh)
            .ok_or_else(|| format!("cannot build {probe_id} at {sw}x{sh}"))?;
        let probe_cells = program.width as u64 * program.height as u64;
        let ms = time_step(gpu, program)?;
        // Small grids are dominated by per-dispatch overhead, so only the last,
        // closest probe predicts well. A probe already over the limit stops the climb.
        predicted = ms * cells as f64 / probe_cells as f64;
        last = (sw, sh, ms);
        if ms > limits.max_step_ms {
            break;
        }
    }
    if predicted > limits.max_step_ms * 1.5 {
        let (sw, sh, ms) = last;
        return Err(format!(
            "{id} {w}x{h}: one step took {ms:.2} ms at {sw}x{sh}, so about {predicted:.0} ms at this size, over the {:.0} ms limit. The desktop would stutter. Use --force to run anyway.",
            limits.max_step_ms
        ));
    }
    Ok(())
}
