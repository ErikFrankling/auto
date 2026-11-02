//! Thermodynamic life on the GPU. One kind of cell, one RNN each, energy as the
//! only currency. No types, no scores, and no genome-derived signals: anything a
//! cell shows or smells of is something its own brain chose.

pub mod config;
pub mod explore;
pub mod report;
pub mod video;
pub mod world;

pub use config::Config;
pub use world::{Pacing, World};

/// VRAM always left free for the desktop and everything else.
pub const VRAM_MARGIN: u64 = 3 << 30;

/// Whether another GPU job holds the job locks right now. An open window shouldn't
/// queue behind them or hold them for hours, but it can see them and go gently.
pub fn gpu_busy_elsewhere() -> bool {
    ["/tmp/auto-gpu.lock", "/tmp/cars-gpu.lock"]
        .iter()
        .any(|path| {
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(false)
                .write(true)
                .open(path)
                .is_ok_and(|f| f.try_lock().is_err())
        })
}

/// Waits for the GPU. One job at a time on this machine, and cars jobs share the card.
/// Hold the returned files until the job is done.
pub fn lock_gpu() -> Vec<std::fs::File> {
    let mut held = vec![];
    // The queue runner already holds the locks for its jobs.
    if std::env::var("LIFE_LOCK_HELD").is_ok() {
        return held;
    }
    // A software driver on the CPU shares nothing with the desktop.
    if std::env::var("AUTO_ADAPTER").is_ok_and(|a| a.contains("llvmpipe")) {
        return held;
    }
    for path in ["/tmp/auto-gpu.lock", "/tmp/cars-gpu.lock"] {
        let Ok(f) = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(path)
        else {
            continue;
        };
        if f.try_lock().is_err() {
            eprintln!("waiting for {path}");
            let _ = f.lock();
        }
        held.push(f);
    }
    held
}

/// Free VRAM on the biggest GPU, on Linux with amdgpu.
pub fn free_vram() -> Option<u64> {
    let mut best: Option<(u64, u64)> = None;
    for entry in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let device = entry.path().join("device");
        let read = |name: &str| -> Option<u64> {
            std::fs::read_to_string(device.join(name))
                .ok()?
                .trim()
                .parse()
                .ok()
        };
        let (Some(total), Some(used)) = (read("mem_info_vram_total"), read("mem_info_vram_used"))
        else {
            continue;
        };
        if best.is_none_or(|(t, _)| total > t) {
            best = Some((total, used));
        }
    }
    best.map(|(total, used)| total.saturating_sub(used))
}
