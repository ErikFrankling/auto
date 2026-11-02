//! Reads a saved world and asks why it looks the way it does. Runs on the CPU, with no GPU.
//!   life-analyse NAME world.bin OUT.json
//!
//! Two questions. The economy: what does the world pay for, and does that explain where
//! cells live and how big colonies get? The brains: put every sampled genome through the
//! same set of situations and see what it actually does. Different genomes that act the
//! same are one strategy, whatever their weights look like.

use auto_life_gpu::measure::{self, Rules};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

const NONE: u32 = u32::MAX;
const IN_BLOCKS: usize = auto_life_gpu::config::IN_BLOCKS as usize;
const OUT_ROWS: usize = auto_life_gpu::config::OUT_ROWS as usize;
const CELL_WORDS: usize = auto_life_gpu::config::CELL_WORDS;
const STAT_WORDS: usize = auto_life_gpu::config::STAT_WORDS;
const SUN_RANGE: f32 = 8.0;
const N_ACTIONS: usize = 13;
const ACTIONS: [&str; N_ACTIONS] = [
    "rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme", "armour", "clean", "uptake",
];
/// Which rules are on in this world. Set once, from the save.
static RULES: std::sync::OnceLock<Rules> = std::sync::OnceLock::new();

fn rules() -> Rules {
    RULES.get().copied().unwrap_or_default()
}

/// One tick's budget shares, as the shader splits them. Actions that are off get nothing.
fn split_budget(raw: &[f32; N_ACTIONS]) -> [f32; N_ACTIONS] {
    let on = rules().actions();
    let top = raw
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < 12 || on[2])
        .map(|(_, v)| *v)
        .fold(f32::MIN, f32::max);
    let mut s = [0.0f32; N_ACTIONS];
    let mut z = 0.0;
    for i in 0..N_ACTIONS {
        if i < 10 || on[i - 10] {
            s[i] = (raw[i] - top).clamp(-20.0, 0.0).exp();
            z += s[i];
        }
    }
    s.iter_mut().for_each(|v| *v /= z);
    s
}
const DIRS: [(i32, i32); 8] =
    [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn below(&mut self, n: usize) -> usize {
        ((self.next() >> 32) as usize * n) >> 32
    }
    fn gauss(&mut self) -> f32 {
        let u = self.f().max(1e-7);
        let v = self.f();
        (-2.0 * u.ln()).sqrt() * (6.283_185_3 * v).cos()
    }
}

struct Save {
    file: std::fs::File,
    size: usize,
    slots: usize,
    hr: usize,
    blocks: usize,
    tick: u64,
    sun_k: f32,
    physics: Value,
    genome_at: u64,
    genome1_at: u64,
    shard: u64,
    occ: Vec<u32>,
    terrain: Vec<u32>,
    fd: Vec<[f32; 4]>,
    cell: Vec<[u32; CELL_WORDS]>,
    stat: Vec<[f32; STAT_WORDS]>,
    sketch: Vec<[u32; 4]>,
}

fn read_vec<T: bytemuck::Pod>(f: &mut std::fs::File, at: u64, bytes: u64) -> Vec<T> {
    f.seek(SeekFrom::Start(at)).unwrap();
    let mut v = vec![0u8; bytes as usize];
    f.read_exact(&mut v).unwrap();
    bytemuck::pod_collect_to_vec(&v)
}

fn open(path: &str) -> Save {
    let mut f = std::fs::File::open(path).expect("open world");
    let mut len = [0u8; 8];
    f.read_exact(&mut len).unwrap();
    let n = u64::from_le_bytes(len);
    let mut head = vec![0u8; n as usize];
    f.read_exact(&mut head).unwrap();
    let head: Value = serde_json::from_slice(&head).unwrap();
    if let Err(why) = auto_life_gpu::config::check_layout(&head, path) {
        eprintln!("{why}");
        std::process::exit(2);
    }
    let _ = RULES.set(Rules::from_physics(&head["physics"]));
    let size = head["size"].as_u64().unwrap() as usize;
    let slots = head["slots"].as_u64().unwrap() as usize;
    let hr = head["hidden_rows"].as_u64().unwrap() as usize;
    let blocks = hr * (IN_BLOCKS + hr + OUT_ROWS);
    let max_shard = (1u64 << 31) / (blocks as u64 * 16);
    let shard = head["shard_slots"].as_u64().unwrap_or(max_shard);
    let squares = (size * size) as u64;
    let slots64 = slots as u64;
    let mut ring = 1u64;
    while ring < slots64 {
        ring *= 2;
    }
    let mut at = 8 + n;
    let mut sections: HashMap<&str, (u64, u64)> = HashMap::new();
    let mut names: Vec<(&str, u64)> = vec![
        ("occ", squares * 4),
        ("seen", squares * 4),
        ("seen_next", squares * 4),
        ("terrain", squares * 4),
        ("fa", squares * 16),
        ("fb", squares * 16),
        ("fc", squares * 16),
        ("fd", squares * 16),
        ("cell", slots64 * auto_life_gpu::config::CELL_BYTES),
        ("stat", slots64 * auto_life_gpu::config::STAT_BYTES),
        ("sketch", slots64 * 16),
        ("mem", slots64 * hr as u64 * 16),
        ("genome", slots64.min(shard) * blocks as u64 * 16),
    ];
    if slots64 > shard {
        names.push(("genome1", (slots64 - shard) * blocks as u64 * 16));
    }
    for (name, bytes) in names {
        sections.insert(name, (at, bytes));
        at += bytes;
    }
    let get = |k: &str| sections[k];
    let occ = read_vec(&mut f, get("occ").0, get("occ").1);
    let terrain = read_vec(&mut f, get("terrain").0, get("terrain").1);
    let fd = read_vec(&mut f, get("fd").0, get("fd").1);
    let cell = read_vec(&mut f, get("cell").0, get("cell").1);
    let stat = read_vec(&mut f, get("stat").0, get("stat").1);
    let sketch = read_vec(&mut f, get("sketch").0, get("sketch").1);
    Save {
        size,
        slots,
        hr,
        blocks,
        tick: head["tick"].as_u64().unwrap_or(0),
        sun_k: head["sun_k"].as_f64().unwrap_or(0.0) as f32,
        physics: head["physics"].clone(),
        genome_at: get("genome").0,
        genome1_at: sections.get("genome1").map_or(0, |s| s.0),
        shard,
        occ,
        terrain,
        fd,
        cell,
        stat,
        sketch,
        file: f,
    }
}

impl Save {
    fn genome(&mut self, slot: usize) -> Vec<f32> {
        let g = self.blocks as u64 * 16;
        let at = if (slot as u64) < self.shard {
            self.genome_at + slot as u64 * g
        } else {
            self.genome1_at + (slot as u64 - self.shard) * g
        };
        self.file.seek(SeekFrom::Start(at)).unwrap();
        let mut b = vec![0u8; g as usize];
        self.file.read_exact(&mut b).unwrap();
        b.iter().map(|v| (*v as i8 as f32 / 127.0).max(-1.0)).collect()
    }
    fn sun(&self, p: usize) -> f32 {
        let t = self.terrain[p];
        let base = (t & 0xFFFF) as f32 / 65535.0 * SUN_RANGE;
        let season = if self.physics["season_ticks"].as_f64().unwrap_or(0.0) > 0.0 { 0.55 } else { 1.0 };
        base * self.sun_k * season
    }
    fn rock(&self, p: usize) -> bool {
        self.terrain[p] & (1 << 24) != 0
    }
}

fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p) as usize]
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

fn round5(v: f32) -> f32 {
    (v * 100000.0).round() / 100000.0
}

/// Connected groups of squares that pass `test`, with the squares of each group's size and centre.
fn components(size: usize, test: impl Fn(usize) -> bool) -> Vec<(usize, f32, f32)> {
    let n = size * size;
    let mut seen = vec![false; n];
    let mut out = vec![];
    let mask = size - 1;
    let mut stack = vec![];
    for start in 0..n {
        if seen[start] || !test(start) {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut count, mut sx, mut cx, mut sy, mut cy) = (0usize, 0f32, 0f32, 0f32, 0f32);
        let tau = std::f32::consts::TAU / size as f32;
        while let Some(q) = stack.pop() {
            count += 1;
            let (x, y) = (q & mask, q / size);
            sx += (x as f32 * tau).sin();
            cx += (x as f32 * tau).cos();
            sy += (y as f32 * tau).sin();
            cy += (y as f32 * tau).cos();
            for (dx, dy) in DIRS {
                let nx = (x as i32 + dx) as usize & mask;
                let ny = (y as i32 + dy) as usize & mask;
                let r = ny * size + nx;
                if !seen[r] && test(r) {
                    seen[r] = true;
                    stack.push(r);
                }
            }
        }
        let wrap = |s: f32, c: f32| (s.atan2(c) / tau).rem_euclid(size as f32);
        out.push((count, wrap(sx, cx), wrap(sy, cy)));
    }
    out
}

fn size_summary(sizes: &[usize]) -> Value {
    let mut s: Vec<f32> = sizes.iter().map(|v| *v as f32).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len().max(1) as f32;
    let mean = s.iter().sum::<f32>() / n;
    let var = s.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    // Spread on a log scale: 1 means sizes differ by a factor of 10 for each standard deviation.
    let logs: Vec<f32> = s.iter().map(|v| v.log10()).collect();
    let lm = logs.iter().sum::<f32>() / n;
    let lsd = (logs.iter().map(|v| (v - lm).powi(2)).sum::<f32>() / n).sqrt();
    json!({
        "count": s.len(),
        "p10": percentile(&s, 0.1), "p50": percentile(&s, 0.5), "p90": percentile(&s, 0.9),
        "max": s.last().copied().unwrap_or(0.0), "mean": mean, "cv": var.sqrt() / mean.max(1e-6), "log10_sd": round3(lsd),
    })
}

/// How strongly positions line up on a grid of this period. 1 is a perfect grid, about 1/sqrt(n) is none.
fn grid_strength(points: &[(f32, f32)], period: f32) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let k = std::f32::consts::TAU / period;
    let n = points.len() as f32;
    let axis = |pick: &dyn Fn(&(f32, f32)) -> f32| {
        let (mut s, mut c) = (0.0f32, 0.0f32);
        for p in points {
            s += (pick(p) * k).sin();
            c += (pick(p) * k).cos();
        }
        (s * s + c * c).sqrt() / n
    };
    (axis(&|p| p.0) + axis(&|p| p.1)) / 2.0
}

fn economy(w: &Save) -> Value {
    let size = w.size;
    let n = size * size;
    let mask = size - 1;
    let mut sun_total = 0f64;
    let mut open = 0usize;
    // Where is it worth living? Cells per square, across levels of sun.
    const BINS: usize = 24;
    let lo = 0.003f32.ln();
    let hi = 2.0f32.ln();
    let bin_of = |s: f32| (((s.max(0.003).ln() - lo) / (hi - lo)) * BINS as f32).clamp(0.0, BINS as f32 - 1.0) as usize;
    let mut squares = vec![0u64; BINS];
    let mut occupied = vec![0u64; BINS];
    let mut energy = vec![0f64; BINS];
    let mut other = vec![0f64; BINS];
    let mut moving = vec![0f64; BINS];
    let mut rest = vec![0f64; BINS];
    let mut age = vec![0f64; BINS];
    let (mut flow_ground, mut flow_meat, mut flow_bite) = (0f64, 0f64, 0f64);
    for p in 0..n {
        if w.rock(p) {
            continue;
        }
        open += 1;
        let s = w.sun(p);
        sun_total += s as f64;
        let b = bin_of(s);
        squares[b] += 1;
        let o = w.occ[p];
        if o != 0 && o != NONE {
            occupied[b] += 1;
            let c = &w.cell[o as usize - 1];
            let st = &w.stat[o as usize - 1];
            energy[b] += f32::from_bits(c[0]) as f64;
            age[b] += f32::from_bits(c[2]) as f64;
            let total = (st[0] + st[1] + st[2] + st[3]).max(1e-6);
            other[b] += ((st[1] + st[2]) / total) as f64;
            moving[b] += st[19] as f64;
            rest[b] += st[4] as f64;
            flow_ground += st[0] as f64;
            flow_meat += st[1] as f64;
            flow_bite += st[2] as f64;
        }
    }
    let table: Vec<Value> = (0..BINS)
        .filter(|b| squares[*b] > 30)
        .map(|b| {
            let c = occupied[b].max(1) as f64;
            json!({
                "sun": round5(((b as f32 + 0.5) / BINS as f32 * (hi - lo) + lo).exp()),
                "squares": squares[b],
                "occupied": round3(occupied[b] as f32 / squares[b] as f32),
                "energy": round3((energy[b] / c) as f32), "age": (age[b] / c).round(),
                "from_others": round3((other[b] / c) as f32), "moving": round3((moving[b] / c) as f32), "rest": round3((rest[b] / c) as f32),
            })
        })
        .collect();
    // Sun where half the squares are lived on: the break-even level.
    let mut break_even = 0.0f32;
    for b in 0..BINS {
        if squares[b] > 30 && occupied[b] as f32 / squares[b] as f32 >= 0.5 {
            break_even = ((b as f32 + 0.5) / BINS as f32 * (hi - lo) + lo).exp();
            break;
        }
    }
    let threshold = if break_even > 0.0 { break_even } else { 0.2 };

    // Colonies against the land they sit on.
    let solid = |p: usize| {
        let o = w.occ[p];
        if o == 0 || o == NONE {
            return false;
        }
        // A colony square has most of its neighbours filled, which leaves out the scattered wanderers.
        let (x, y) = (p & mask, p / size);
        let mut around = 0;
        for (dx, dy) in DIRS {
            let q = (((y as i32 + dy) as usize) & mask) * size + (((x as i32 + dx) as usize) & mask);
            let v = w.occ[q];
            if v != 0 && v != NONE {
                around += 1;
            }
        }
        around >= 4
    };
    let colonies: Vec<(usize, f32, f32)> = components(size, solid).into_iter().filter(|c| c.0 >= 30).collect();
    let rich_patches: Vec<(usize, f32, f32)> =
        components(size, |p| !w.rock(p) && w.sun(p) >= threshold).into_iter().filter(|c| c.0 >= 30).collect();
    let (mut both, mut either) = (0u64, 0u64);
    let mut in_rich_sum = 0u64;
    let mut solid_total = 0u64;
    for p in 0..n {
        if w.rock(p) {
            continue;
        }
        let a = solid(p);
        let b = w.sun(p) >= threshold;
        if a && b {
            both += 1;
        }
        if a || b {
            either += 1;
        }
        if a {
            solid_total += 1;
            if b {
                in_rich_sum += 1;
            }
        }
    }
    // Does living by a wall matter? Share of colony squares touching rock, against all open squares.
    let near_rock = |p: usize, reach: i32| {
        let (x, y) = (p & mask, p / size);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let q = (((y as i32 + dy) as usize) & mask) * size + (((x as i32 + dx) as usize) & mask);
                if w.rock(q) {
                    return true;
                }
            }
        }
        false
    };
    let (mut open_near, mut solid_near, mut rich_near, mut rich_all) = (0u64, 0u64, 0u64, 0u64);
    let step = (size / 1024).max(1);
    let mut counted_open = 0u64;
    let mut counted_solid = 0u64;
    for y in (0..size).step_by(step) {
        for x in (0..size).step_by(step) {
            let p = y * size + x;
            if w.rock(p) {
                continue;
            }
            let near = near_rock(p, 3);
            counted_open += 1;
            if near {
                open_near += 1;
            }
            if solid(p) {
                counted_solid += 1;
                if near {
                    solid_near += 1;
                }
            }
            if w.sun(p) >= threshold {
                rich_all += 1;
                if near {
                    rich_near += 1;
                }
            }
        }
    }
    // Is the rich land itself on a grid? The noise is built on a lattice of this size.
    let mut suns: Vec<(f32, usize)> = (0..n).filter(|p| !w.rock(*p)).map(|p| (w.sun(p), p)).collect();
    suns.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let top: Vec<(f32, f32)> =
        suns.iter().take(n / 100).map(|(_, p)| ((p & mask) as f32, (p / size) as f32)).collect();
    let centres: Vec<(f32, f32)> = colonies.iter().map(|c| (c.1, c.2)).collect();
    let rich_centres: Vec<(f32, f32)> = rich_patches.iter().map(|c| (c.1, c.2)).collect();
    let grid = |pts: &[(f32, f32)]| {
        json!({"p16": round3(grid_strength(pts, 16.0)), "p32": round3(grid_strength(pts, 32.0)), "p64": round3(grid_strength(pts, 64.0)), "p48": round3(grid_strength(pts, 48.0)), "chance": round3(1.0 / (pts.len().max(1) as f32).sqrt())})
    };
    // How alike is the sun at a distance? Peaks at lattice steps give the regular look.
    let mean_sun = (sun_total / open.max(1) as f64) as f32;
    let var: f64 = (0..n).step_by(7).filter(|p| !w.rock(*p)).map(|p| ((w.sun(p) - mean_sun) as f64).powi(2)).sum::<f64>() / ((n / 7) as f64);
    let auto: Vec<Value> = [4usize, 8, 16, 24, 32, 40, 48, 64, 96, 128, 256]
        .iter()
        .filter(|lag| **lag < size / 2)
        .map(|lag| {
            let mut sum = 0f64;
            let mut count = 0f64;
            for p in (0..n).step_by(7) {
                let (x, y) = (p & mask, p / size);
                let q = y * size + ((x + lag) & mask);
                if !w.rock(p) && !w.rock(q) {
                    sum += ((w.sun(p) - mean_sun) * (w.sun(q) - mean_sun)) as f64;
                    count += 1.0;
                }
            }
            json!({"lag": lag, "r": round3((sum / count.max(1.0) / var.max(1e-9)) as f32)})
        })
        .collect();
    // Walls: how far you can walk east before hitting one.
    let mut runs = vec![];
    let mut rng = Rng(77);
    for _ in 0..4000 {
        let (x, y) = (rng.below(size), rng.below(size));
        if w.rock(y * size + x) {
            continue;
        }
        let mut d = 1;
        while d < size && !w.rock(y * size + ((x + d) & mask)) {
            d += 1;
        }
        runs.push(d as f32);
    }
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_run = runs.iter().sum::<f32>() / runs.len().max(1) as f32;
    let run_sd = (runs.iter().map(|v| (v - mean_run).powi(2)).sum::<f32>() / runs.len().max(1) as f32).sqrt();
    let col_sizes: Vec<usize> = colonies.iter().map(|c| c.0).collect();
    let rich_sizes: Vec<usize> = rich_patches.iter().map(|c| c.0).collect();
    // Colony sizes, measured the same way as the history, and how much terrain explains them.
    let found = measure::colonies(&|p| w.occ[p] != 0 && w.occ[p] != NONE, size, 30);
    let (mut sum_log_sun, mut count_sq) = (vec![0f64; found.found.len()], vec![0f64; found.found.len()]);
    for p in 0..n {
        let l = found.label[p] as usize;
        if l != 0 {
            sum_log_sun[l - 1] += (w.sun(p).max(1e-4) as f64).log10();
            count_sq[l - 1] += 1.0;
        }
    }
    let xs: Vec<f64> = (0..found.found.len()).map(|i| sum_log_sun[i] / count_sq[i].max(1.0)).collect();
    let ys: Vec<f64> = found.found.iter().map(|c| (c.0 as f64).log10()).collect();
    let colony_sizes = measure::size_stats(&found.found.iter().map(|c| c.0).collect::<Vec<_>>(), 30.0);
    let mineral = mineral_land(w);
    json!({
        // Near 1: colonies are this size because of the land they sit on. Near 0: free of it.
        "colony_size_explained_by_sun": round3(r_squared(&xs, &ys)),
        "colony_sizes": colony_sizes,
        "mineral_land": mineral,
        "mean_sun": round5(mean_sun),
        "sun_per_tick": sun_total,
        "open": open,
        "occupancy_by_sun": table,
        "break_even_sun": round5(break_even),
        // Sampled moving averages of what each cell takes in per tick, summed over cells.
        "flows": {"ground": flow_ground, "meat": flow_meat, "bite": flow_bite},
        "colonies": size_summary(&col_sizes),
        "rich_patches": size_summary(&rich_sizes),
        "colony_cells_on_rich_land": round3(in_rich_sum as f32 / solid_total.max(1) as f32),
        "colony_rich_overlap": round3(both as f32 / either.max(1) as f32),
        "near_rock": {"open": round3(open_near as f32 / counted_open.max(1) as f32), "colony": round3(solid_near as f32 / counted_solid.max(1) as f32), "rich": round3(rich_near as f32 / rich_all.max(1) as f32)},
        "grid": {"sun_top_1pct": grid(&top), "colony_centres": grid(&centres), "rich_patch_centres": grid(&rich_centres)},
        "sun_autocorrelation": auto,
        "walk_to_wall": {"p10": percentile(&runs, 0.1), "p50": percentile(&runs, 0.5), "p90": percentile(&runs, 0.9), "cv": round3(run_sd / mean_run.max(1e-6))},
    })
}

fn r_squared(x: &[f64], y: &[f64]) -> f32 {
    let n = x.len() as f64;
    if x.len() < 3 {
        return 0.0;
    }
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut xx, mut yy, mut xy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        xx += (a - mx) * (a - mx);
        yy += (b - my) * (b - my);
        xy += (a - mx) * (b - my);
    }
    if xx <= 0.0 || yy <= 0.0 {
        return 0.0;
    }
    (xy * xy / xx / yy) as f32
}

/// Land rent, once there are two things land can be good for. Thirds of sun against thirds of
/// mineral richness: who lives where, and how. If mineral matters, rows and columns both change.
fn mineral_land(w: &Save) -> Value {
    if !rules().mineral {
        return Value::Null;
    }
    let n = w.size * w.size;
    let open: Vec<usize> = (0..n).filter(|p| !w.rock(*p)).collect();
    let cuts = |value: &dyn Fn(usize) -> f32| -> (f32, f32) {
        let mut v: Vec<f32> = open.iter().map(|p| value(*p)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (percentile(&v, 1.0 / 3.0), percentile(&v, 2.0 / 3.0))
    };
    let (s1, s2) = cuts(&|p| w.sun(p));
    let (m1, m2) = cuts(&|p| w.fd[p][2]);
    let third = |v: f32, a: f32, b: f32| if v < a { 0 } else if v < b { 1 } else { 2 };
    let mut squares = [[0f64; 3]; 3];
    let mut occupied = [[0f64; 3]; 3];
    let mut ground = [[0f64; 3]; 3];
    let mut store = [[0f64; 3]; 3];
    let mut moving = [[0f64; 3]; 3];
    let mut uptake = [[0f64; 3]; 3];
    let mut other = [[0f64; 3]; 3];
    for p in &open {
        let (s, m) = (third(w.sun(*p), s1, s2), third(w.fd[*p][2], m1, m2));
        squares[s][m] += 1.0;
        ground[s][m] += w.fd[*p][1] as f64;
        let o = w.occ[*p];
        if o != 0 && o != NONE {
            let c = &w.cell[o as usize - 1];
            let st = &w.stat[o as usize - 1];
            occupied[s][m] += 1.0;
            store[s][m] += f32::from_bits(c[20]) as f64;
            moving[s][m] += st[19] as f64;
            uptake[s][m] += st[28] as f64;
            let total = (st[0] + st[1] + st[2] + st[3]).max(1e-6);
            other[s][m] += ((st[1] + st[2]) / total) as f64;
        }
    }
    let mut rows = vec![];
    let names = ["low", "mid", "high"];
    for s in 0..3 {
        for m in 0..3 {
            let c = occupied[s][m].max(1.0);
            rows.push(json!({
                "sun": names[s], "mineral": names[m],
                "occupied": round3((occupied[s][m] / squares[s][m].max(1.0)) as f32),
                "ground_mineral": round3((ground[s][m] / squares[s][m].max(1.0)) as f32),
                "store": round3((store[s][m] / c) as f32),
                "moving": round3((moving[s][m] / c) as f32),
                "uptake": round3((uptake[s][m] / c) as f32),
                "from_others": round3((other[s][m] / c) as f32),
            }));
        }
    }
    let total_ground: f64 = open.iter().map(|p| w.fd[*p][1] as f64).sum();
    let live = || w.occ.iter().filter(|o| **o != 0 && **o != NONE).map(|o| &w.cell[*o as usize - 1]);
    let free: f64 = live().map(|c| f32::from_bits(c[20]) as f64).sum();
    let bound: f64 = live().map(|c| f32::from_bits(c[22]) as f64).sum();
    let start = w.physics["mineral_start"].as_f64().unwrap_or(0.0);
    let birth = w.physics["mineral_birth"].as_f64().unwrap_or(1.0);
    json!({
        "table": rows,
        "on_ground": total_ground, "free_in_cells": free, "bound_in_cells": bound,
        "density_cap": round3((start / birth.max(1e-6)) as f32),
    })
}

// ---- Brains ----------------------------------------------------------------

type Inputs = [[f32; 4]; IN_BLOCKS];

fn squash(x: f32) -> f32 {
    let c = x.clamp(-3.0, 3.0);
    c * (27.0 + c * c) / (27.0 + 9.0 * c * c)
}

fn bt(g: &[f32], block: usize, v: &[f32; 4]) -> [f32; 4] {
    let b = &g[block * 16..block * 16 + 16];
    let mut out = [0.0; 4];
    for w in 0..4 {
        out[w] = 2.0 * (b[w * 4] * v[0] + b[w * 4 + 1] * v[1] + b[w * 4 + 2] * v[2] + b[w * 4 + 3] * v[3]);
    }
    out
}

/// One tick of a brain, as the shader does it.
fn think(g: &[f32], hr: usize, input: &Inputs, mem: &mut [[f32; 4]]) -> [[f32; 4]; OUT_ROWS] {
    let hh = hr * IN_BLOCKS;
    let ho = hh + hr * hr;
    let mut next = vec![[0.0f32; 4]; hr];
    for r in 0..hr {
        let mut acc = [0.0f32; 4];
        for (k, v) in input.iter().enumerate() {
            let t = bt(g, r * IN_BLOCKS + k, v);
            for i in 0..4 {
                acc[i] += t[i];
            }
        }
        for j in 0..hr {
            let t = bt(g, hh + r * hr + j, &mem[j]);
            for i in 0..4 {
                acc[i] += t[i];
            }
        }
        for i in 0..4 {
            next[r][i] = squash(acc[i]);
        }
    }
    mem.copy_from_slice(&next);
    let mut out = [[0.0f32; 4]; OUT_ROWS];
    for (o, row) in out.iter_mut().enumerate() {
        for (j, h) in next.iter().enumerate() {
            let t = bt(g, ho + o * hr + j, h);
            for i in 0..4 {
                row[i] += t[i];
            }
        }
    }
    out
}

fn raw_budget(o: &[[f32; 4]; OUT_ROWS]) -> [f32; N_ACTIONS] {
    [
        o[0][0], o[0][1], o[0][2], o[0][3], o[1][0], o[1][1], o[1][2], o[1][3], o[2][0], o[2][1], o[2][2], o[2][3],
        o[6][0],
    ]
}

/// All the outputs in a situation, from a blank memory after a few ticks.
fn settle(g: &[f32], hr: usize, input: &Inputs) -> [[f32; 4]; OUT_ROWS] {
    let mut mem = vec![[0.0f32; 4]; hr];
    let mut out = [[0.0f32; 4]; OUT_ROWS];
    for _ in 0..3 {
        out = think(g, hr, input, &mut mem);
    }
    out
}

/// Budget shares, heading and willingness in a situation.
fn act(g: &[f32], hr: usize, input: &Inputs) -> ([f32; N_ACTIONS], f32, f32) {
    let out = settle(g, hr, input);
    (split_budget(&raw_budget(&out)), out[3][0], out[3][1])
}

fn sat(v: f32, s: f32) -> f32 {
    v / (s + v)
}

#[derive(Clone, Copy)]
struct Situation {
    energy: f32,
    food: f32,
    food_east: bool,
    /// 0 empty, 1 smaller body, 2 bigger body, 3 rock.
    front: u8,
    hit: bool,
    crowd: f32,
}

fn inputs_for(s: &Situation) -> Inputs {
    let mut i: Inputs = [[0.0; 4]; IN_BLOCKS];
    i[0] = [sat(s.energy, 10.0), 0.3, sat(300.0, 1000.0), if s.hit { sat(1.0, 0.5) } else { 0.0 }];
    i[1] = [0.0, 1.0, s.crowd, 0.0];
    i[2] = [sat(s.food, 2.0), 0.0, 0.0, 0.0];
    if s.food_east {
        i[4][0] = 0.8;
    }
    let me = sat(s.energy, 10.0);
    match s.front {
        1 => {
            let their = me * 0.4;
            i[8][2] = 1.0;
            i[8][3] = (their - me) / (their + me + 0.05);
            i[10][2] = 0.5;
        }
        2 => {
            let their = (me * 1.8).min(0.95);
            i[8][2] = 1.0;
            i[8][3] = (their - me) / (their + me + 0.05);
            i[10][0] = 0.5;
        }
        3 => i[8][2] = -1.0,
        _ => {}
    }
    i
}

/// Inputs and, for those of a rule, the switch that makes them anything but zero.
const INPUT_NAMES: [(&str, usize, usize); 70] = [
    ("front coat 0", 15, 0), ("front coat 1", 15, 1), ("own coat 0", 15, 2), ("own coat 1", 15, 3),
    ("mineral here", 16, 0), ("mineral east", 16, 1), ("mineral south", 16, 2), ("own mineral", 16, 3),
    ("biting east", 17, 0), ("biting south", 17, 1), ("biting nearby", 17, 2), ("biters on front", 17, 3),
    ("own energy", 0, 0), ("womb", 0, 1), ("age", 0, 2), ("felt damage", 0, 3),
    ("noise input", 1, 0), ("bias", 1, 1), ("crowding", 1, 2), ("own hardness", 1, 3),
    ("ground here", 2, 0), ("meat here", 2, 1), ("heat here", 2, 2), ("noise here", 2, 3),
    ("scent 0 here", 3, 0), ("scent 1 here", 3, 1), ("scent 2 here", 3, 2), ("scent 3 here", 3, 3),
    ("ground east", 4, 0), ("meat east", 4, 1), ("heat east", 4, 2), ("noise east", 4, 3),
    ("ground south", 5, 0), ("meat south", 5, 1), ("heat south", 5, 2), ("noise south", 5, 3),
    ("scent 0 east", 6, 0), ("scent 1 east", 6, 1), ("scent 2 east", 6, 2), ("scent 3 east", 6, 3),
    ("scent 0 south", 7, 0), ("scent 1 south", 7, 1), ("scent 2 south", 7, 2), ("scent 3 south", 7, 3),
    ("heard signal 0", 8, 0), ("heard signal 1", 8, 1), ("body in front", 8, 2), ("front is bigger", 8, 3),
    ("front signal 0", 9, 0), ("front signal 1", 9, 1), ("front faces me", 9, 2), ("enzyme here", 9, 3),
    ("bigger east", 10, 0), ("bigger south", 10, 1), ("smaller east", 10, 2), ("smaller south", 10, 3),
    ("signal 0 east", 11, 0), ("signal 0 south", 11, 1), ("signal 1 east", 11, 2), ("signal 1 south", 11, 3),
    ("waste here", 12, 0), ("waste east", 12, 1), ("waste south", 12, 2),
    ("track here", 13, 0), ("track east", 13, 1), ("track south", 13, 2),
    ("practice at grazing", 14, 0), ("practice at bodies", 14, 1), ("felt upkeep", 14, 2), ("felt drag", 14, 3),
];

/// The switch behind a rule's input. An input with its rule off always reads zero, so
/// testing it would only measure weights that do nothing.
fn input_switches(name: &str) -> &'static [&'static str] {
    match name {
        "own hardness" => &["armour_on"],
        "waste here" | "waste east" | "waste south" => &["waste_on"],
        "track here" | "track east" | "track south" => &["roads_on"],
        "practice at grazing" | "practice at bodies" => &["skill_on", "skill_sense"],
        "felt upkeep" | "felt drag" => &["biomes_on", "biome_feel"],
        "front coat 0" | "front coat 1" | "own coat 0" | "own coat 1" => &["coat_on", "coat_sense"],
        "mineral here" | "mineral east" | "mineral south" | "own mineral" => &["mineral_on"],
        "biting east" | "biting south" | "biting nearby" | "biters on front" => &["gang_on", "gang_sense"],
        _ => &[],
    }
}

/// The budget of one situation, leaving out actions that don't exist in this world.
fn print_budget(shares: &[f32; N_ACTIONS], v: &mut Vec<f32>) {
    v.extend_from_slice(&shares[..10]);
    for (k, on) in rules().actions().iter().enumerate() {
        if *on {
            v.push(shares[10 + k]);
        }
    }
}

fn base_actions() -> usize {
    10 + rules().actions().iter().filter(|v| **v).count()
}

/// What a genome does across a fixed set of situations. The base situations come first, each with
/// the actions that exist. Then, for rules that are on, situations of their own and what
/// the brain chooses besides its budget. A rule that's off doesn't change the measure at all.
fn fingerprint(g: &[f32], hr: usize, situations: &[Situation]) -> Vec<f32> {
    let r = rules();
    let mut v = Vec::with_capacity(situations.len() * 12);
    let mut beyond = vec![];
    for s in situations {
        let out = settle(g, hr, &inputs_for(s));
        print_budget(&split_budget(&raw_budget(&out)), &mut v);
        beyond.extend(measure::rule_outputs(&r, &out));
    }
    let make = |energy: f32, food: f32, front: u8| {
        inputs_for(&Situation { energy, food, food_east: false, front, hit: false, crowd: 0.0 })
    };
    let extra = measure::rule_situations(&r, &make);
    let count = situations.len() + extra.len();
    for i in extra {
        let out = settle(g, hr, &i);
        print_budget(&split_budget(&raw_budget(&out)), &mut v);
        beyond.extend(measure::rule_outputs(&r, &out));
    }
    v.extend(measure::outputs_once(&beyond, count));
    v
}

fn distance(a: &[f32], b: &[f32]) -> f32 {
    let sum: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (sum / a.len() as f32).sqrt()
}

fn top_axes(data: &[Vec<f32>], count: usize) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, f32) {
    let dims = data[0].len();
    let n = data.len() as f32;
    let mut mean = vec![0.0f32; dims];
    for d in data {
        for i in 0..dims {
            mean[i] += d[i] / n;
        }
    }
    let total: f32 = data.iter().map(|d| d.iter().zip(&mean).map(|(a, m)| (a - m) * (a - m)).sum::<f32>()).sum::<f32>() / n;
    let mut centred: Vec<Vec<f32>> = data.iter().map(|d| d.iter().zip(&mean).map(|(a, m)| a - m).collect()).collect();
    let mut axes = vec![];
    let mut values = vec![];
    let mut rng = Rng(9);
    for _ in 0..count {
        let mut v: Vec<f32> = (0..dims).map(|_| rng.gauss()).collect();
        let mut lambda = 0.0;
        for _ in 0..40 {
            let mut next = vec![0.0f32; dims];
            for row in &centred {
                let d: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                for i in 0..dims {
                    next[i] += row[i] * d;
                }
            }
            let len = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            next.iter_mut().for_each(|x| *x /= len);
            lambda = len / n;
            v = next;
        }
        for row in centred.iter_mut() {
            let d: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            for i in 0..dims {
                row[i] -= d * v[i];
            }
        }
        values.push(lambda);
        axes.push(v);
    }
    (axes, values, mean, total)
}

fn project(row: &[f32], mean: &[f32], axes: &[Vec<f32>]) -> Vec<f32> {
    axes.iter()
        .map(|a| row.iter().zip(mean).zip(a).map(|((r, m), x)| (r - m) * x).sum())
        .collect()
}

fn kmeans(points: &[Vec<f32>], k: usize, rng: &mut Rng) -> (Vec<usize>, f32) {
    let n = points.len();
    let dist = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum() };
    let mut best: (Vec<usize>, f32) = (vec![0; n], f32::MAX);
    for _ in 0..4 {
        // k-means++ starts.
        let mut centres: Vec<Vec<f32>> = vec![points[rng.below(n)].clone()];
        while centres.len() < k {
            let d: Vec<f32> = points.iter().map(|p| centres.iter().map(|c| dist(p, c)).fold(f32::MAX, f32::min)).collect();
            let total: f32 = d.iter().sum();
            let mut pick = rng.f() * total;
            let mut chosen = n - 1;
            for (i, v) in d.iter().enumerate() {
                pick -= v;
                if pick <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centres.push(points[chosen].clone());
        }
        let mut label = vec![0usize; n];
        for _ in 0..25 {
            for i in 0..n {
                let (mut b, mut bd) = (0, f32::MAX);
                for (c, centre) in centres.iter().enumerate() {
                    let v = dist(&points[i], centre);
                    if v < bd {
                        bd = v;
                        b = c;
                    }
                }
                label[i] = b;
            }
            let dims = points[0].len();
            let mut sum = vec![vec![0.0f32; dims]; k];
            let mut count = vec![0f32; k];
            for i in 0..n {
                count[label[i]] += 1.0;
                for d in 0..dims {
                    sum[label[i]][d] += points[i][d];
                }
            }
            for c in 0..k {
                if count[c] > 0.0 {
                    for d in 0..dims {
                        centres[c][d] = sum[c][d] / count[c];
                    }
                }
            }
        }
        let inertia: f32 = (0..n).map(|i| dist(&points[i], &centres[label[i]])).sum();
        if inertia < best.1 {
            best = (label, inertia);
        }
    }
    best
}

/// Near 1: tight separate clusters. Near 0: no real clusters.
fn silhouette(points: &[Vec<f32>], label: &[usize], k: usize) -> f32 {
    let n = points.len().min(500);
    let step = (points.len() / n).max(1);
    let idx: Vec<usize> = (0..points.len()).step_by(step).take(n).collect();
    let dist = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt() };
    let mut total = 0.0;
    let mut counted = 0.0;
    for &i in &idx {
        let mut sum = vec![0.0f32; k];
        let mut count = vec![0.0f32; k];
        for &j in &idx {
            if i != j {
                sum[label[j]] += dist(&points[i], &points[j]);
                count[label[j]] += 1.0;
            }
        }
        if count[label[i]] == 0.0 {
            continue;
        }
        let a = sum[label[i]] / count[label[i]];
        let b = (0..k).filter(|c| *c != label[i] && count[*c] > 0.0).map(|c| sum[c] / count[c]).fold(f32::MAX, f32::min);
        if b == f32::MAX {
            continue;
        }
        total += (b - a) / a.max(b).max(1e-6);
        counted += 1.0;
    }
    total / f32::max(counted, 1.0)
}

fn heading_east(g: &[f32], hr: usize, input: &Inputs) -> f32 {
    let (_, dx, dy) = act(g, hr, input);
    dy.atan2(dx).cos()
}

fn brains(w: &mut Save) -> Value {
    let hr = w.hr;
    let alive = w.occ.iter().filter(|o| **o != 0 && **o != NONE).count();
    let mut rng = Rng(w.tick ^ 0xABCD);
    // A random sample of cells old enough to have done something.
    let mut pool: Vec<usize> = vec![];
    for (p, o) in w.occ.iter().enumerate() {
        let _ = p;
        if *o != 0 && *o != NONE {
            let s = *o as usize - 1;
            if f32::from_bits(w.cell[s][2]) >= 50.0 {
                pool.push(s);
            }
        }
    }
    let n = pool.len().min(1500);
    for i in 0..n {
        let j = i + rng.below(pool.len() - i);
        pool.swap(i, j);
    }
    pool.truncate(n);
    let genomes: Vec<Vec<f32>> = pool.iter().map(|s| w.genome(*s)).collect();

    let mut situations = vec![];
    for energy in [2.0f32, 10.0, 30.0] {
        for (food, east) in [(0.0f32, false), (2.0, false), (0.1, true)] {
            for front in 0..4u8 {
                for hit in [false, true] {
                    for crowd in [0.0f32, 0.75] {
                        situations.push(Situation { energy, food, food_east: east, front, hit, crowd });
                    }
                }
            }
        }
    }
    let prints: Vec<Vec<f32>> = genomes.iter().map(|g| fingerprint(g, hr, &situations)).collect();

    // How much does a brain change what it does between situations? A constant function scores 0.
    let per = situations.len();
    let actions = base_actions();
    let reactivity = |print: &[f32]| -> f32 {
        let mut mean = [0.0f32; N_ACTIONS];
        for s in 0..per {
            for a in 0..actions {
                mean[a] += print[s * actions + a] / per as f32;
            }
        }
        let mut var = 0.0;
        for s in 0..per {
            for a in 0..actions {
                var += (print[s * actions + a] - mean[a]).powi(2);
            }
        }
        (var / (per * actions) as f32).sqrt()
    };
    let reacts: Vec<f32> = prints.iter().map(|p| reactivity(p)).collect();
    let mean_react = reacts.iter().sum::<f32>() / reacts.len().max(1) as f32;

    // Random genomes, as a yardstick.
    let init = w.physics["gene_init"].as_f64().unwrap_or(0.4) as f32;
    let random_genomes: Vec<Vec<f32>> = (0..300)
        .map(|_| (0..w.blocks * 16).map(|_| (rng.gauss() * init / 2.0).clamp(-1.0, 1.0)).collect())
        .collect();
    let random_prints: Vec<Vec<f32>> = random_genomes.iter().map(|g| fingerprint(g, hr, &situations)).collect();
    let random_react = random_prints.iter().map(|p| reactivity(p)).sum::<f32>() / 300.0;

    // Distinct ways of acting: main axes of the behaviour, then clusters.
    let (axes, values, mean, total) = top_axes(&prints, 12);
    let explained: Vec<f32> = values.iter().map(|v| round3(v / total.max(1e-9))).collect();
    let sum2: f32 = values.iter().map(|v| v * v).sum();
    let sum1: f32 = values.iter().sum();
    let effective_dims = sum1 * sum1 / sum2.max(1e-12);
    let reduced: Vec<Vec<f32>> = prints.iter().map(|p| project(p, &mean, &axes)).collect();
    let mut scores = vec![];
    let mut best = (2usize, f32::MIN, vec![]);
    for k in 2..=10.min(n.saturating_sub(1)) {
        let (label, _) = kmeans(&reduced, k, &mut rng);
        let s = silhouette(&reduced, &label, k);
        scores.push(json!({"k": k, "silhouette": round3(s)}));
        if s > best.1 {
            best = (k, s, label);
        }
    }
    let (k, sil, label) = best;

    // What each strategy does in plain situations, and who the cells using it are.
    let default = Situation { energy: 10.0, food: 2.0, food_east: false, front: 0, hit: false, crowd: 0.0 };
    let named: [(&str, Situation); 8] = [
        ("settled, fed, with food", default),
        ("starving, no food", Situation { energy: 2.0, food: 0.0, ..default }),
        ("fat, food", Situation { energy: 30.0, ..default }),
        ("smaller cell in front", Situation { front: 1, ..default }),
        ("bigger cell in front", Situation { front: 2, ..default }),
        ("wall in front", Situation { front: 3, ..default }),
        ("just got hurt", Situation { hit: true, ..default }),
        ("crowded", Situation { crowd: 0.75, ..default }),
    ];
    let responds = |gs: &[&Vec<f32>]| -> Value {
        let mut out = serde_json::Map::new();
        for (name, s) in &named {
            let i = inputs_for(s);
            let mut m = [0.0f32; N_ACTIONS];
            for g in gs {
                let (shares, _, _) = act(g, hr, &i);
                for a in 0..N_ACTIONS {
                    m[a] += shares[a] / gs.len() as f32;
                }
            }
            let mut b = serde_json::Map::new();
            for a in 0..N_ACTIONS {
                b.insert(ACTIONS[a].into(), json!(round3(m[a])));
            }
            out.insert((*name).to_string(), Value::Object(b));
        }
        Value::Object(out)
    };
    // Direction tests: does a stimulus change where a cell heads? +1 is towards it, -1 is away.
    let mut base = inputs_for(&default);
    base[2][0] = sat(1.0, 2.0);
    let stim = |slot: (usize, usize), v: f32| {
        let mut i = base;
        i[slot.0][slot.1] = v;
        i
    };
    let tests: [(&str, Inputs); 11] = [
        ("food", stim((4, 0), 0.8)),
        ("meat", stim((4, 1), 0.8)),
        ("heat", stim((4, 2), 0.8)),
        ("noise", stim((4, 3), 0.8)),
        ("bigger body", stim((10, 0), 0.5)),
        ("smaller body", stim((10, 2), 0.5)),
        ("scent 0", stim((6, 0), 0.8)),
        ("scent 1", stim((6, 1), 0.8)),
        ("scent 2", stim((6, 2), 0.8)),
        ("scent 3", stim((6, 3), 0.8)),
        ("signal 0", stim((11, 0), 0.5)),
    ];
    let heading_table = |gs: &[&Vec<f32>]| -> Value {
        let mut out = serde_json::Map::new();
        for (name, i) in &tests {
            let (mut sum, mut towards, mut away) = (0.0f32, 0.0f32, 0.0f32);
            for g in gs {
                let d = heading_east(g, hr, i) - heading_east(g, hr, &base);
                sum += d;
                if d > 0.5 {
                    towards += 1.0;
                }
                if d < -0.5 {
                    away += 1.0;
                }
            }
            let c = gs.len().max(1) as f32;
            out.insert((*name).to_string(), json!({"mean": round3(sum / c), "towards": round3(towards / c), "away": round3(away / c)}));
        }
        Value::Object(out)
    };

    let flow_total: [f32; 4] = {
        let mut t = [0f32; 4];
        for o in &w.occ {
            if *o != 0 && *o != NONE {
                let st = &w.stat[*o as usize - 1];
                for d in 0..4 {
                    t[d] += st[d];
                }
            }
        }
        t
    };
    let all: Vec<&Vec<f32>> = genomes.iter().collect();
    let mut strategies = vec![];
    for c in 0..k {
        let members: Vec<usize> = (0..n).filter(|i| label[*i] == c).collect();
        if members.is_empty() {
            continue;
        }
        let gs: Vec<&Vec<f32>> = members.iter().map(|i| &genomes[*i]).collect();
        let (mut diet, mut mv, mut age, mut e, mut nb, mut gen_) = ([0f32; 4], 0f32, 0f32, 0f32, 0f32, 0f32);
        let (mut free, mut bound, mut coat_c, mut coat_s, mut womb) = (0f32, 0f32, 0f32, 0f32, 0f32);
        for &i in &members {
            let st = &w.stat[pool[i]];
            let cell = &w.cell[pool[i]];
            for d in 0..4 {
                diet[d] += st[d];
            }
            mv += st[19];
            nb += st[24] * 8.0;
            age += f32::from_bits(cell[2]);
            e += f32::from_bits(cell[0]);
            gen_ += cell[13] as f32;
            womb += f32::from_bits(cell[1]);
            free += f32::from_bits(cell[20]);
            bound += f32::from_bits(cell[22]);
            let a = f32::from_bits(cell[21]) * std::f32::consts::TAU;
            coat_c += a.cos();
            coat_s += a.sin();
        }
        let m = members.len() as f32;
        let t: f32 = diet.iter().sum::<f32>().max(1e-6);
        // Inside this strategy, how alike are behaviours compared with the whole population?
        strategies.push(json!({
            "share": round3(m / n as f32),
            "lives_on": {"ground": round3(diet[0] / t), "meat": round3(diet[1] / t), "bite": round3(diet[2] / t), "gift": round3(diet[3] / t)},
            // Energy a cell takes in a tick from each source, and this strategy's share of what the
            // whole world takes from it, scaled up from the sample.
            "income": {"ground": round5(diet[0] / m), "meat": round5(diet[1] / m), "bite": round5(diet[2] / m), "gift": round5(diet[3] / m)},
            "share_of_flow": {
                "ground": round3(diet[0] / n as f32 * alive as f32 / flow_total[0].max(1e-6)),
                "meat": round3(diet[1] / n as f32 * alive as f32 / flow_total[1].max(1e-6)),
                "bite": round3(diet[2] / n as f32 * alive as f32 / flow_total[2].max(1e-6)),
                "gift": round3(diet[3] / n as f32 * alive as f32 / flow_total[3].max(1e-6)),
            },
            "mineral": [round3(free / m), round3(bound / m)],
            "coat_spread": round3(1.0 - (coat_c * coat_c + coat_s * coat_s).sqrt() / m),
            "womb": round3(womb / m),
            "moving": round3(mv / m), "neighbours": round3(nb / m), "age": (age / m).round(), "energy": round3(e / m), "generation": (gen_ / m).round(),
            "reactivity": round3(members.iter().map(|i| reacts[*i]).sum::<f32>() / m),
            "responds": responds(&gs),
            "headings": heading_table(&gs),
        }));
    }
    // Who bites whom, in the same strategies, from this one moment: every cell that really bit last
    // tick, and the body in front of it. Classing a cell means putting its brain through the battery,
    // so this takes at most 1500 of them.
    let bite_links = {
        let size = w.size;
        let mask = size - 1;
        let mut pairs: Vec<(usize, usize)> = vec![];
        for (p, o) in w.occ.iter().enumerate() {
            if *o == 0 || *o == NONE {
                continue;
            }
            let flags = w.cell[*o as usize - 1][11];
            if flags & 128 == 0 {
                continue;
            }
            let (dx, dy) = DIRS[(flags & 7) as usize];
            let q = (((p / size) as i32 + dy) as usize & mask) * size + (((p & mask) as i32 + dx) as usize & mask);
            let t = w.occ[q];
            if t != 0 && t != NONE {
                pairs.push((*o as usize - 1, t as usize - 1));
            }
        }
        let total = pairs.len();
        for i in 0..pairs.len().min(1500) {
            let j = i + rng.below(pairs.len() - i);
            pairs.swap(i, j);
        }
        pairs.truncate(1500);
        let classify = |print: &[f32]| -> usize {
            let at = project(print, &mean, &axes);
            let mut best_c = (0usize, f32::MAX);
            for c in 0..k {
                let (mut sum, mut count) = (0f32, 0f32);
                for i in (0..n).step_by((n / 300).max(1)) {
                    if label[i] == c {
                        sum += measure::dist2(&at, &reduced[i]);
                        count += 1.0;
                    }
                }
                if count > 0.0 && sum / count < best_c.1 {
                    best_c = (c, sum / count);
                }
            }
            best_c.0
        };
        let mut matrix = vec![vec![0u32; k]; k];
        for (a, b) in &pairs {
            let (ga, gb) = (w.genome(*a), w.genome(*b));
            let x = classify(&fingerprint(&ga, hr, &situations));
            let y = classify(&fingerprint(&gb, hr, &situations));
            matrix[x][y] += 1;
        }
        let sampled = pairs.len().max(1) as f32;
        let mut links = 0;
        let mut across = 0u32;
        for x in 0..k {
            let by_x: u32 = matrix[x].iter().sum();
            for y in 0..k {
                if x != y {
                    across += matrix[x][y];
                    if by_x >= 10 && matrix[x][y] as f32 >= by_x as f32 * 0.05 {
                        links += 1;
                    }
                }
            }
        }
        json!({
            "biters_now": total,
            "matrix": matrix,
            // Real links: one strategy spends at least 5% of its bites on another.
            "links": links,
            "share_across_strategies": round3(across as f32 / sampled),
        })
    };
    strategies.sort_by(|a, b| b["share"].as_f64().partial_cmp(&a["share"].as_f64()).unwrap());

    // Which senses change behaviour at all?
    let base_shares: Vec<[f32; N_ACTIONS]> = genomes.iter().take(400).map(|g| act(g, hr, &base).0).collect();
    let mut senses = vec![];
    for (name, b, s) in INPUT_NAMES {
        if input_switches(name).iter().any(|k| !w.physics[k].as_bool().unwrap_or(false)) {
            continue;
        }
        let mut sum = 0.0f32;
        let mut i = base;
        i[b][s] = if base[b][s].abs() > 0.5 { 0.0 } else { 0.9 };
        for (g, before) in genomes.iter().take(400).zip(&base_shares) {
            let (after, _, _) = act(g, hr, &i);
            sum += (0..N_ACTIONS).map(|a| (after[a] - before[a]).abs()).sum::<f32>() / 2.0;
        }
        senses.push((name, sum / base_shares.len().max(1) as f32));
    }
    senses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let senses_json: Vec<Value> = senses.iter().map(|(n, v)| json!({"sense": n, "changes_budget_by": round3(*v)})).collect();

    // Does memory do anything? Same situation, with a memory state from a different one.
    let mut memory_effect = 0.0f32;
    for g in genomes.iter().take(400) {
        let mut mem = vec![[0.0f32; 4]; hr];
        for _ in 0..4 {
            think(g, hr, &inputs_for(&Situation { energy: 2.0, food: 0.0, ..default }), &mut mem);
        }
        let out_after = think(g, hr, &base, &mut mem);
        let mut fresh = vec![[0.0f32; 4]; hr];
        let out_fresh = think(g, hr, &base, &mut fresh);
        let (a, b) = (split_budget(&raw_budget(&out_after)), split_budget(&raw_budget(&out_fresh)));
        memory_effect += (0..N_ACTIONS).map(|i| (a[i] - b[i]).abs()).sum::<f32>() / 2.0;
    }
    memory_effect /= genomes.len().min(400).max(1) as f32;

    // Can a strategy be inherited at all? One birth's mutations against the gap between cells.
    let size = w.physics["mut_size"].as_f64().unwrap_or(0.1) as f32;
    let mut pair = 0.0f32;
    let pairs = 800;
    for _ in 0..pairs {
        let (a, b) = (rng.below(n), rng.below(n));
        pair += distance(&prints[a], &prints[b]);
    }
    pair /= pairs as f32;
    let mut mutation = vec![];
    for rate in [0.1f32, 0.03, 0.01, 0.003] {
        let mut sum = 0.0f32;
        let take = 200.min(n);
        for i in 0..take {
            let mut child = genomes[i].clone();
            for w in child.iter_mut() {
                if rng.f() < rate {
                    // Weights here are in the stored -1..1 range, where one unit of weight is 0.5.
                    *w = (*w + rng.gauss() * size / 2.0).clamp(-1.0, 1.0);
                }
            }
            sum += distance(&prints[i], &fingerprint(&child, hr, &situations));
        }
        let change = sum / take.max(1) as f32;
        mutation.push(json!({"rate": rate, "change_per_birth": round5(change), "share_of_gap_between_cells": round3(change / pair.max(1e-6))}));
    }

    json!({
        "sampled": n,
        "reactivity": round5(mean_react),
        "reactivity_random": round5(random_react),
        "behaviour_gap_between_cells": round5(pair),
        "axes_explained": explained,
        "effective_dims": round3(effective_dims),
        "cluster_scores": scores,
        "strategies_k": k,
        "strategies_silhouette": round3(sil),
        "strategies": strategies,
        "bite_links": bite_links,
        "whole_population": {"responds": responds(&all), "headings": heading_table(&all)},
        "senses": senses_json,
        "memory_effect": round3(memory_effect),
        "mutation": mutation,
    })
}

/// Colours for this world's cells from their genome sketches: the three directions in which its
/// genomes differ most become red, green and blue, so kin look alike and strangers don't.
fn genome_colours(w: &Save) -> HashMap<usize, [u8; 3]> {
    let cells: Vec<usize> = w.occ.iter().filter(|o| **o != 0 && **o != NONE).map(|o| *o as usize - 1).collect();
    let mut out = HashMap::new();
    if cells.is_empty() {
        return out;
    }
    let vectors: Vec<[f32; 16]> = cells
        .iter()
        .map(|s| {
            let mut v = [0.0f32; 16];
            for (word, packed) in w.sketch[*s].iter().enumerate() {
                for (byte, value) in packed.to_le_bytes().iter().enumerate() {
                    v[word * 4 + byte] = *value as i8 as f32 / 127.0;
                }
            }
            v
        })
        .collect();
    let n = vectors.len() as f32;
    let mut mean = [0.0f32; 16];
    for v in &vectors {
        for k in 0..16 {
            mean[k] += v[k] / n;
        }
    }
    let mut cov = [[0.0f32; 16]; 16];
    for v in &vectors {
        for a in 0..16 {
            for b in 0..16 {
                cov[a][b] += (v[a] - mean[a]) * (v[b] - mean[b]) / n;
            }
        }
    }
    let mut axes: Vec<[f32; 16]> = vec![];
    for round in 0..3 {
        let mut x = [0.0f32; 16];
        x[round] = 1.0;
        for k in 0..16 {
            x[k] += 0.01 * (k as f32 + 1.0);
        }
        for _ in 0..60 {
            let mut y = [0.0f32; 16];
            for a in 0..16 {
                for b in 0..16 {
                    y[a] += cov[a][b] * x[b];
                }
            }
            for axis in &axes {
                let dot: f32 = (0..16).map(|k| y[k] * axis[k]).sum();
                for k in 0..16 {
                    y[k] -= dot * axis[k];
                }
            }
            let norm = y.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
            for k in 0..16 {
                x[k] = y[k] / norm;
            }
        }
        axes.push(x);
    }
    let along = |v: &[f32; 16], axis: &[f32; 16]| (0..16).map(|k| (v[k] - mean[k]) * axis[k]).sum::<f32>();
    let spread: Vec<f32> = axes
        .iter()
        .map(|axis| (vectors.iter().map(|v| along(v, axis).powi(2)).sum::<f32>() / n).sqrt().max(1e-6))
        .collect();
    for (s, v) in cells.iter().zip(&vectors) {
        let channel = |i: usize| (127.5 + 127.5 * (along(v, &axes[i]) / (2.0 * spread[i])).clamp(-1.0, 1.0)) as u8;
        out.insert(*s, [channel(0), channel(1), channel(2)]);
    }
    out
}

/// A picture of one world to read by eye, as a PPM. Rock is dark in every panel.
/// Top row: cells by genome, colours chosen for this world alone; cells by recent diet (red bites, green ground, blue meat and gifts);
/// cells by how much they move (blue sits, yellow moves).
/// Bottom row: sun; every step taken over the run, on a log scale; tracks on the ground if there
/// are any, otherwise mineral if there's any, otherwise cells by energy.
fn map_picture(w: &Save, path: &str) {
    const GAP: usize = 4;
    let n = w.size;
    let (width, height) = (n * 3 + GAP * 2, n * 2 + GAP);
    let mut px = vec![40u8; width * height * 3];
    let top = |v: &[f32]| v.iter().cloned().fold(1e-6f32, f32::max);
    let sun: Vec<f32> = (0..n * n).map(|p| w.sun(p)).collect();
    let sun_top = top(&sun);
    // Log steps, stretched between the quiet end and the busy end of this world's own traffic,
    // so paths show if there are any.
    let steps: Vec<f32> = w.fd.iter().map(|v| (1.0 + v[3]).ln()).collect();
    let mut ordered: Vec<f32> = (0..n * n).filter(|p| !w.rock(*p)).map(|p| steps[p]).collect();
    ordered.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |share: f64| ordered.get((ordered.len() as f64 * share) as usize).copied().unwrap_or(0.0);
    let (steps_low, steps_high) = (at(0.02), at(0.998).max(at(0.02) + 1e-3));
    let genome = genome_colours(w);
    let tracks: Vec<f32> = w.fd.iter().map(|v| v[0]).collect();
    let mineral: Vec<f32> = w.fd.iter().map(|v| v[1]).collect();
    let (tracks_top, mineral_top) = (top(&tracks), top(&mineral));
    let last = if tracks_top > 1e-3 { 0 } else if mineral_top > 1e-3 { 1 } else { 2 };
    for y in 0..n {
        for x in 0..n {
            let p = y * n + x;
            let rock = w.rock(p);
            let o = w.occ[p];
            let live = o != 0 && o != NONE;
            let empty = if rock { [18, 20, 30] } else { [0, 0, 0] };
            let mut colours = [empty; 6];
            if live {
                let c = &w.cell[o as usize - 1];
                let st = &w.stat[o as usize - 1];
                colours[0] = genome[&(o as usize - 1)];
                let t = (st[0] + st[1] + st[2] + st[3]).max(1e-6);
                let shade = |v: f32| (40.0 + 215.0 * v.clamp(0.0, 1.0)) as u8;
                // Stretched, since a cell with a quarter of its food from biting is a real biter.
                colours[1] = [shade(st[2] / t * 3.0), shade(st[0] / t * (1.0 - (st[2] / t * 3.0).min(0.7))), shade((st[1] + st[3]) / t * 2.0)];
                let m = (st[19] * 2.5).clamp(0.0, 1.0);
                colours[2] = [(255.0 * m) as u8, (230.0 * m) as u8, (255.0 * (1.0 - m)) as u8];
                if last == 2 {
                    let e = (f32::from_bits(c[auto_life_gpu::config::cell::E]) / 30.0).clamp(0.0, 1.0);
                    colours[5] = [(255.0 * e) as u8, (120.0 * e) as u8, 30];
                }
            }
            if !rock {
                let grey = |v: f32| [(255.0 * v.clamp(0.0, 1.0)) as u8; 3];
                colours[3] = {
                    let v = (sun[p] / sun_top).clamp(0.0, 1.0);
                    [(255.0 * v) as u8, (220.0 * v) as u8, (90.0 * v) as u8]
                };
                colours[4] = grey((steps[p] - steps_low) / (steps_high - steps_low));
                if last == 0 {
                    colours[5] = grey((tracks[p] / tracks_top).sqrt());
                } else if last == 1 {
                    colours[5] = grey(mineral[p] / mineral_top);
                }
            }
            for (panel, colour) in colours.iter().enumerate() {
                let (cx, cy) = (panel % 3 * (n + GAP) + x, panel / 3 * (n + GAP) + y);
                let at = (cy * width + cx) * 3;
                px[at..at + 3].copy_from_slice(colour);
            }
        }
    }
    let mut out = format!("P6\n{width} {height}\n255\n").into_bytes();
    out.extend_from_slice(&px);
    std::fs::write(path, out).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: life-analyse NAME world.bin OUT.json [--tile SIZE] [--maps DIR] [--maps-only]");
        std::process::exit(2);
    }
    let maps = args.iter().position(|a| a == "--maps").map(|i| args[i + 1].clone());
    let maps_only = args.iter().any(|a| a == "--maps-only");
    let mut w = open(&args[2]);
    // A batch: every world on the map, measured as if it were alone.
    if let Some(tile) = args.iter().position(|a| a == "--tile").map(|i| args[i + 1].parse::<usize>().expect("--tile SIZE")) {
        let size = w.size;
        let per_side = size / tile;
        let (occ, terrain, fd) = (std::mem::take(&mut w.occ), std::mem::take(&mut w.terrain), std::mem::take(&mut w.fd));
        w.size = tile;
        let mut tiles = vec![];
        for i in 0..per_side * per_side {
            let (ox, oy) = (i % per_side * tile, i / per_side * tile);
            let rows: Vec<usize> = (oy..oy + tile).map(|y| y * size + ox).collect();
            w.occ = rows.iter().flat_map(|r| occ[*r..*r + tile].iter().copied()).collect();
            w.terrain = rows.iter().flat_map(|r| terrain[*r..*r + tile].iter().copied()).collect();
            w.fd = rows.iter().flat_map(|r| fd[*r..*r + tile].iter().copied()).collect();
            let alive = w.occ.iter().filter(|o| **o != 0 && **o != NONE).count();
            if let Some(dir) = &maps {
                map_picture(&w, &format!("{dir}/t{i:03}.ppm"));
            }
            if maps_only {
                continue;
            }
            // Too few cells to sort into strategies says enough by itself.
            if alive < 50 {
                tiles.push(json!({"tile": i, "alive": alive, "dead": true}));
                continue;
            }
            let e = economy(&w);
            let b = brains(&mut w);
            // Where this world's cells have walked over the whole run. Counted on the map, so
            // unlike the rest of the depth measures it can be told apart world by world.
            let mut steps: Vec<f32> = w.fd.iter().map(|v| v[3]).collect();
            let open = w.terrain.iter().filter(|t| *t & (1 << 24) == 0).count().max(1);
            let lines = auto_life_gpu::measure::traffic_lines(&steps, &|p| w.terrain[p] & (1 << 24) == 0, tile, 8);
            let total: f64 = steps.iter().map(|v| *v as f64).sum();
            steps.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
            let busiest: f64 = steps[..(open / 20).clamp(1, steps.len())].iter().map(|v| *v as f64).sum();
            let trails = json!({
                "steps": total,
                "top5_share": (busiest / total.max(1.0) * 1000.0).round() / 1000.0,
                "top5_chance": (auto_life_gpu::measure::chance_top_share(total / open as f64, 0.05) * 1000.0).round() / 1000.0,
                "lines": (lines * 1000.0).round() / 1000.0,
            });
            tiles.push(json!({"tile": i, "tick": w.tick, "size": tile, "alive": alive, "economy": e, "brains": b, "trails": trails}));
            eprintln!("{}: tile {i} of {}, {alive} cells", args[1], per_side * per_side);
        }
        if maps_only {
            return;
        }
        let out = json!({"name": args[1], "tick": w.tick, "batch_tile": tile, "tiles": tiles});
        std::fs::write(&args[3], serde_json::to_string(&out).unwrap()).unwrap();
        return;
    }
    eprintln!("{}: {} squares wide, {} slots, tick {}", args[1], w.size, w.slots, w.tick);
    let alive = w.occ.iter().filter(|o| **o != 0 && **o != NONE).count();
    let e = economy(&w);
    let b = brains(&mut w);
    let out = json!({"name": args[1], "tick": w.tick, "size": w.size, "alive": alive, "economy": e, "brains": b});
    std::fs::write(&args[3], serde_json::to_string(&out).unwrap()).unwrap();
}
