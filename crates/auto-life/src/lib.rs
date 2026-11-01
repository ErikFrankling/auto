//! Thermodynamic life on the CPU. Every cell is the same kind of thing: one RNN
//! with its own weights, one energy store, one square of ground. There are no
//! roles, organs, species or scores. A cell splits its energy across a list of
//! local physics, and whatever keeps copying itself stays.
//!
//! Each square holds at most one cell, so writes to a cell's own square never
//! race and neighbours are a table lookup. That is meant to carry over to WGSL.

use rayon::prelude::*;
use serde_json::{Value, json};

pub const INPUTS: usize = 35;
pub const HIDDEN: usize = 8;
pub const OUTPUTS: usize = 20;
pub const WEIGHTS: usize = INPUTS * HIDDEN + HIDDEN * HIDDEN + OUTPUTS * HIDDEN;
pub const SKETCH: usize = 16;

// The first outputs are scores for the energy budget.
const REST: usize = 0;
const MOVE: usize = 1;
const GROUND: usize = 2;
const MEAT: usize = 3;
const GROW: usize = 4;
const EMIT: usize = 5;
const HEAT: usize = 6;
const BITE: usize = 7;
const GIVE: usize = 8;
const ENZYME: usize = 9;
const BUDGET: usize = 10;
const DIR: usize = 10;
const MIX: usize = 12;
const VOICE: usize = 16;
const MATE: usize = 18;
const SHARE: usize = 19;

pub const ACTION_NAMES: [&str; BUDGET] = [
    "rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme",
];

// Fields. Only ground food, meat, heat and enzyme do anything physically. The
// four pheromones have no meaning except what evolution gives them.
const F_GROUND: usize = 0;
const F_MEAT: usize = 1;
const F_HEAT: usize = 2;
const F_ENZYME: usize = 3;
const F_P: usize = 4;
const CHANNELS: usize = 4;
const FIELDS: usize = F_P + CHANNELS;
/// (decay per tick, diffusion). Pheromones run from long trails to short shouts.
const FLOW: [(f32, f32); FIELDS] = [
    (0.0, 0.0),
    (0.002, 0.02),
    (0.5, 0.5),
    (0.1, 0.2),
    (0.005, 0.05),
    (0.02, 0.2),
    (0.08, 0.5),
    (0.3, 0.9),
];

const SPEND: f32 = 0.25;
const GROUND_CAP: f32 = 4.0;
const WOMB: f32 = 5.0;
const BODY: f32 = 1.0;
const MAX_ENERGY: f32 = 40.0;
const HAZARD: f32 = 1.0 / 4000.0;
const DX: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];
const NONE: u32 = u32::MAX;

const STARVED: u8 = 0;
const BITTEN: u8 = 1;
const BURNED: u8 = 2;
const OLD: u8 = 3;

#[derive(Clone, Debug)]
pub struct Switches {
    pub pheromones: bool,
    pub kin: bool,
    pub voice: bool,
    pub bite: bool,
    pub heat: bool,
    pub give: bool,
    pub enzyme: bool,
    pub sex: bool,
    pub share: bool,
    pub memory: bool,
}

impl Default for Switches {
    fn default() -> Self {
        Switches {
            pheromones: true,
            kin: true,
            voice: true,
            bite: true,
            heat: true,
            give: true,
            enzyme: true,
            sex: true,
            share: true,
            memory: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub size: usize,
    pub seed: u64,
    /// Mean sun per open square per tick.
    pub sun: f32,
    /// Genome distance under which two cells count as kin and can mate.
    pub compat: f32,
    pub mutation_rate: f32,
    pub mutation_size: f32,
    /// One random cell lands every this many ticks, so the world can't die for good.
    pub rain_every: u32,
    pub season_ticks: u32,
    pub rock: bool,
    /// Energy pulled out of a bitten cell per unit spent.
    pub bite_power: f32,
    /// Share of a bite the biter keeps. The same share again falls as meat, and the rest is lost.
    pub bite_yield: f32,
    /// Share of eaten meat that becomes energy. Digesting is never free.
    pub meat_yield: f32,
    /// Spending below about this much barely works, so dabbling in everything is a waste.
    pub commitment: f32,
    /// A bite only reaches a cell's open sides, so a cell inside a clump is protected.
    pub exposure: bool,
    /// Scent every cell leaks per tick. Its mix of channels comes from the genome.
    pub odour: f32,
    /// Share of the sun that arrives as sudden rich blooms, which fade fast.
    pub bloom: f32,
    pub on: Switches,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            size: 128,
            seed: 1,
            sun: 0.05,
            compat: 0.4,
            mutation_rate: 0.1,
            mutation_size: 0.1,
            rain_every: 200,
            season_ticks: 8000,
            rock: true,
            bite_power: 20.0,
            bite_yield: 0.4,
            meat_yield: 0.6,
            commitment: 0.03,
            exposure: true,
            odour: 0.02,
            bloom: 0.0,
            on: Switches::default(),
        }
    }
}

#[derive(Clone, Default)]
pub struct Tally {
    pub births: u64,
    pub births_mixed: u64,
    pub deaths: [u64; 4],
    pub intake: [f64; 4],
    pub bite_kin: f64,
    pub bite_other: f64,
    pub give_kin: f64,
    pub give_other: f64,
    pub front_kin: u64,
    pub front_other: u64,
}

#[derive(Clone)]
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

fn mix(a: u64, b: u64) -> u64 {
    Rng(a.wrapping_mul(0x100000001B3) ^ b).next()
}

/// Smooth noise that wraps around the world edge.
fn noise(seed: u64, x: usize, y: usize, size: usize, cells: usize) -> f32 {
    let (fx, fy) = (x as f32 * cells as f32 / size as f32, y as f32 * cells as f32 / size as f32);
    let (ix, iy) = (fx as usize, fy as usize);
    let (tx, ty) = (fx - ix as f32, fy - iy as f32);
    let (tx, ty) = (tx * tx * (3.0 - 2.0 * tx), ty * ty * (3.0 - 2.0 * ty));
    let corner = |a: usize, b: usize| {
        let h = mix(seed, ((a % cells) as u64) << 32 | (b % cells) as u64);
        (h >> 40) as f32 / (1u64 << 24) as f32
    };
    let top = corner(ix, iy) * (1.0 - tx) + corner(ix + 1, iy) * tx;
    let bottom = corner(ix, iy + 1) * (1.0 - tx) + corner(ix + 1, iy + 1) * tx;
    top * (1.0 - ty) + bottom * ty
}

#[inline]
fn tanh(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // Float sums only vectorise when written as separate lanes.
    let mut lanes = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            lanes[k] += x[k] * y[k];
        }
    }
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    lanes.iter().sum::<f32>() + tail
}

#[inline]
fn sat(v: f32, scale: f32) -> f32 {
    let v = v / scale;
    v / (1.0 + v)
}

/// Contrast between two readings, so a cell can follow a gradient at any strength.
#[inline]
fn contrast(a: f32, b: f32) -> f32 {
    (a - b) / (0.05 + a + b)
}

#[derive(Clone)]
pub struct World {
    pub cfg: Config,
    pub tick: u64,
    rng: Rng,
    pub open: usize,
    rock: Vec<bool>,
    occ: Vec<u32>,
    sun_base: Vec<f32>,
    sun_phase: Vec<f32>,
    sun_now: Vec<f32>,
    fields: Vec<Vec<f32>>,
    scratch: Vec<f32>,
    west: Vec<u32>,
    east: Vec<u32>,
    // Cells, kept dense. A dead cell is swapped with the last one.
    pub n: usize,
    x: Vec<u16>,
    y: Vec<u16>,
    e: Vec<f32>,
    womb: Vec<f32>,
    age: Vec<u32>,
    hit: Vec<f32>,
    dir: Vec<u8>,
    cause: Vec<u8>,
    id: Vec<u64>,
    clade: Vec<u32>,
    voice: Vec<[f32; 2]>,
    genome: Vec<f32>,
    hidden: Vec<f32>,
    out: Vec<f32>,
    sketch: Vec<[f32; SKETCH]>,
    ema_in: Vec<[f32; 4]>,
    ema_sp: Vec<[f32; BUDGET]>,
    ema_bite: Vec<[f32; 2]>,
    sketch_map: Vec<(u8, f32)>,
    next_id: u64,
    pub tally: Tally,
    bloom_bank: f32,
    // Energy ledger, checked by a test.
    pub sun_in: f64,
    pub rain_in: f64,
    pub lost: f64,
    pub start_energy: f64,
}

impl World {
    pub fn new(cfg: Config) -> World {
        let s = cfg.size;
        assert!(s >= 8 && s <= 65_535);
        let cells = s * s;
        let mut rock = vec![false; cells];
        let (mut sun_base, mut sun_phase) = (vec![0.0; cells], vec![0.0; cells]);
        let chunks = (s / 24).max(2);
        for y in 0..s {
            for x in 0..s {
                let i = y * s + x;
                if cfg.rock {
                    // Ridges along the zero line of one noise. Another noise opens gaps,
                    // so the world is many regions joined by corridors.
                    let r = noise(cfg.seed ^ 11, x, y, s, chunks);
                    let gap = noise(cfg.seed ^ 12, x, y, s, chunks * 2);
                    rock[i] = (r - 0.5).abs() < 0.035 && gap > 0.42;
                }
                let rich = noise(cfg.seed ^ 13, x, y, s, chunks);
                let fine = noise(cfg.seed ^ 14, x, y, s, chunks * 4);
                let v = rich * rich * rich * 4.0 + fine * 0.3 + 0.02;
                sun_base[i] = if rock[i] { 0.0 } else { v };
                sun_phase[i] = noise(cfg.seed ^ 15, x, y, s, (chunks / 2).max(2)) * 12.566;
            }
        }
        let open = rock.iter().filter(|r| !**r).count();
        let mut fields = vec![vec![0.0f32; cells]; FIELDS];
        for i in 0..cells {
            if !rock[i] {
                fields[F_GROUND][i] = GROUND_CAP.min(sun_base[i] * 2.0);
            }
        }
        let start_energy: f64 = fields[F_GROUND].iter().map(|v| *v as f64).sum();
        let west: Vec<u32> = (0..s).map(|x| ((x + s - 1) % s) as u32).collect();
        let east: Vec<u32> = (0..s).map(|x| ((x + 1) % s) as u32).collect();
        let mut map = Rng(cfg.seed ^ 77);
        let sketch_map = (0..WEIGHTS)
            .map(|_| (map.below(SKETCH) as u8, if map.f() < 0.5 { -1.0 } else { 1.0 }))
            .collect();
        let rng = Rng(mix(cfg.seed, 1));
        let mut w = World {
            tick: 0,
            rng,
            open,
            rock,
            occ: vec![NONE; cells],
            sun_base,
            sun_phase,
            sun_now: vec![0.0; cells],
            fields,
            scratch: vec![0.0; cells],
            west,
            east,
            n: 0,
            x: vec![],
            y: vec![],
            e: vec![],
            womb: vec![],
            age: vec![],
            hit: vec![],
            dir: vec![],
            cause: vec![],
            id: vec![],
            clade: vec![],
            voice: vec![],
            genome: vec![],
            hidden: vec![],
            out: vec![],
            sketch: vec![],
            ema_in: vec![],
            ema_sp: vec![],
            ema_bite: vec![],
            sketch_map,
            next_id: 0,
            tally: Tally::default(),
            bloom_bank: 0.0,
            sun_in: 0.0,
            rain_in: 0.0,
            lost: 0.0,
            start_energy,
            cfg,
        };
        w.update_sun();
        w
    }

    #[inline]
    fn at(&self, x: usize, y: usize, d: usize) -> usize {
        let s = self.cfg.size as i32;
        let nx = (x as i32 + DX[d]).rem_euclid(s) as usize;
        let ny = (y as i32 + DY[d]).rem_euclid(s) as usize;
        ny * self.cfg.size + nx
    }

    fn update_sun(&mut self) {
        // Regions wax and wane out of step, and the total stays fixed.
        let t = self.tick as f32 / self.cfg.season_ticks.max(1) as f32 * 6.283_185_3;
        let seasons = self.cfg.season_ticks > 0;
        self.sun_base
            .par_iter()
            .zip(&self.sun_phase)
            .zip(self.sun_now.par_iter_mut())
            .for_each(|((b, p), n)| {
                *n = *b * if seasons { 0.55 + 0.45 * (p + t).sin() } else { 1.0 };
            });
        // Summed in order, because a parallel float sum depends on the thread count.
        let total: f32 = self.sun_now.iter().sum();
        let k = self.cfg.sun * (1.0 - self.cfg.bloom) * self.open as f32 / total.max(1e-6);
        self.sun_now.par_iter_mut().for_each(|n| *n *= k);
    }

    fn grow_food(&mut self) {
        for (g, s) in self.fields[F_GROUND].iter_mut().zip(&self.sun_now) {
            let room = (GROUND_CAP - *g).max(0.0);
            let a = s.min(room);
            *g += a;
            self.sun_in += a as f64;
        }
        if self.cfg.bloom > 0.0 {
            self.bloom_tick();
        }
    }

    /// Food too big for one cell, there for a moment. This is what ant trails are for.
    fn bloom_tick(&mut self) {
        const RADIUS: i32 = 12;
        const BANK: f32 = 9000.0;
        const FADE: f32 = 0.02;
        let s = self.cfg.size as i32;
        let mut faded = 0.0f64;
        for g in self.fields[F_GROUND].iter_mut() {
            if *g > GROUND_CAP {
                let gone = (*g - GROUND_CAP) * FADE;
                *g -= gone;
                faded += gone as f64;
            }
        }
        self.lost += faded;
        self.bloom_bank += self.cfg.sun * self.cfg.bloom * self.open as f32;
        if self.bloom_bank < BANK {
            return;
        }
        let (cx, cy) = (self.rng.below(s as usize) as i32, self.rng.below(s as usize) as i32);
        let mut squares = vec![];
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                if dx * dx + dy * dy <= RADIUS * RADIUS {
                    let q = ((cy + dy).rem_euclid(s) * s + (cx + dx).rem_euclid(s)) as usize;
                    if !self.rock[q] {
                        squares.push(q);
                    }
                }
            }
        }
        if !squares.is_empty() {
            let each = self.bloom_bank / squares.len() as f32;
            for q in squares {
                self.fields[F_GROUND][q] += each;
                self.sun_in += each as f64;
            }
            self.bloom_bank = 0.0;
        }
    }

    fn sketch_of(&self, genes: &[f32]) -> [f32; SKETCH] {
        let mut out = [0.0; SKETCH];
        for (g, (slot, sign)) in genes.iter().zip(&self.sketch_map) {
            out[*slot as usize] += g * sign;
        }
        let k = (SKETCH as f32 / WEIGHTS as f32).sqrt();
        out.iter_mut().for_each(|v| *v *= k);
        out
    }

    /// Adds a cell. `genes` holds the weights, `hidden` its starting memory.
    fn spawn(&mut self, pos: usize, energy: f32, genes: &[f32; WEIGHTS], hidden: &[f32; HIDDEN], clade: u32) {
        let s = self.cfg.size;
        debug_assert_eq!(self.occ[pos], NONE);
        self.occ[pos] = self.n as u32;
        self.x.push((pos % s) as u16);
        self.y.push((pos / s) as u16);
        self.e.push(energy);
        self.womb.push(0.0);
        self.age.push(0);
        self.hit.push(0.0);
        let d = self.rng.below(8) as u8;
        self.dir.push(d);
        self.cause.push(STARVED);
        self.next_id += 1;
        self.id.push(self.next_id);
        self.clade.push(if clade == NONE { self.next_id as u32 } else { clade });
        self.voice.push([0.0; 2]);
        self.genome.extend_from_slice(genes);
        self.hidden.extend_from_slice(hidden);
        self.out.extend_from_slice(&[0.0; OUTPUTS]);
        let sk = self.sketch_of(genes);
        self.sketch.push(sk);
        self.ema_in.push([0.0; 4]);
        self.ema_sp.push([0.0; BUDGET]);
        self.ema_bite.push([0.0; 2]);
        self.n += 1;
    }

    fn remove(&mut self, i: usize) {
        let last = self.n - 1;
        let s = self.cfg.size;
        self.occ[self.y[i] as usize * s + self.x[i] as usize] = NONE;
        if i != last {
            self.occ[self.y[last] as usize * s + self.x[last] as usize] = i as u32;
            self.genome.copy_within(last * WEIGHTS..(last + 1) * WEIGHTS, i * WEIGHTS);
            self.hidden.copy_within(last * HIDDEN..(last + 1) * HIDDEN, i * HIDDEN);
            self.out.copy_within(last * OUTPUTS..(last + 1) * OUTPUTS, i * OUTPUTS);
        }
        self.genome.truncate(last * WEIGHTS);
        self.hidden.truncate(last * HIDDEN);
        self.out.truncate(last * OUTPUTS);
        self.x.swap_remove(i);
        self.y.swap_remove(i);
        self.e.swap_remove(i);
        self.womb.swap_remove(i);
        self.age.swap_remove(i);
        self.hit.swap_remove(i);
        self.dir.swap_remove(i);
        self.cause.swap_remove(i);
        self.id.swap_remove(i);
        self.clade.swap_remove(i);
        self.voice.swap_remove(i);
        self.sketch.swap_remove(i);
        self.ema_in.swap_remove(i);
        self.ema_sp.swap_remove(i);
        self.ema_bite.swap_remove(i);
        self.n = last;
    }

    fn random_cell(&mut self, energy: f32) {
        for _ in 0..30 {
            let pos = self.rng.below(self.occ.len());
            if !self.rock[pos] && self.occ[pos] == NONE {
                let mut genes = [0.0; WEIGHTS];
                genes.iter_mut().for_each(|g| *g = self.rng.gauss() * 0.4);
                self.spawn(pos, energy, &genes, &[0.0; HIDDEN], NONE);
                self.rain_in += (energy + BODY) as f64;
                return;
            }
        }
    }

    fn rain(&mut self) {
        // A burst at the start, then a thin drizzle. Newcomers almost always die.
        if self.tick < 3000 {
            let floor = (self.open / 100).max(16);
            for _ in 0..8 {
                if self.n < floor {
                    self.random_cell(6.0);
                }
            }
        } else if self.cfg.rain_every > 0 && self.tick % self.cfg.rain_every as u64 == 0 {
            self.random_cell(6.0);
        }
    }

    fn think(&mut self) {
        let s = self.cfg.size;
        let on = &self.cfg.on;
        let compat = self.cfg.compat;
        let tick = self.tick;
        let (x, y, e, womb, age, hit, dir, id) = (
            &self.x, &self.y, &self.e, &self.womb, &self.age, &self.hit, &self.dir, &self.id,
        );
        let (occ, rock, fields, sketch, voice, genome) =
            (&self.occ, &self.rock, &self.fields, &self.sketch, &self.voice, &self.genome);
        let (w, east) = (&self.west, &self.east);
        self.hidden
            .par_chunks_mut(HIDDEN)
            .zip(self.out.par_chunks_mut(OUTPUTS))
            .enumerate()
            .for_each(|(i, (h, o))| {
                let (cx, cy) = (x[i] as usize, y[i] as usize);
                let p = cy * s + cx;
                let (wx, ex) = (w[cx] as usize, east[cx] as usize);
                let (ny, sy) = ((cy + s - 1) % s, (cy + 1) % s);
                let mut v = [0.0f32; INPUTS];
                let ei = e[i];
                v[0] = sat(ei, 10.0);
                v[1] = womb[i] / WOMB;
                v[2] = sat(age[i] as f32, 1000.0);
                v[3] = sat(hit[i], 0.5);
                for (k, f) in fields.iter().enumerate() {
                    if k >= F_P && !on.pheromones {
                        continue;
                    }
                    v[4 + k] = sat(f[p], 2.0);
                }
                let mut g = 12;
                for (k, f) in fields.iter().enumerate() {
                    if k == F_ENZYME {
                        continue;
                    }
                    if k < F_P || on.pheromones {
                        v[g] = contrast(f[cy * s + ex], f[cy * s + wx]);
                        v[g + 1] = contrast(f[sy * s + cx], f[ny * s + cx]);
                    }
                    g += 2;
                }
                let dist = |j: usize| {
                    let mut d = 0.0;
                    for k in 0..SKETCH {
                        let a = sketch[i][k] - sketch[j][k];
                        d += a * a;
                    }
                    (d / SKETCH as f32).sqrt()
                };
                let (mut count, mut kin, mut vo) = (0.0f32, 0.0f32, [0.0f32; 2]);
                for d in 0..8 {
                    let nx = (cx as i32 + DX[d]).rem_euclid(s as i32) as usize;
                    let nyy = (cy as i32 + DY[d]).rem_euclid(s as i32) as usize;
                    let j = occ[nyy * s + nx];
                    if j != NONE {
                        let j = j as usize;
                        count += 1.0;
                        kin += (1.0 - dist(j) / compat).clamp(-1.0, 1.0);
                        vo[0] += voice[j][0];
                        vo[1] += voice[j][1];
                    }
                }
                v[26] = count / 8.0;
                if on.kin && count > 0.0 {
                    v[27] = kin / count;
                }
                let d = dir[i] as usize;
                let fx = (cx as i32 + DX[d]).rem_euclid(s as i32) as usize;
                let fy = (cy as i32 + DY[d]).rem_euclid(s as i32) as usize;
                let front = fy * s + fx;
                if rock[front] {
                    v[28] = -1.0;
                } else if occ[front] != NONE {
                    let j = occ[front] as usize;
                    v[28] = 1.0;
                    if on.kin {
                        v[29] = (1.0 - dist(j) / compat).clamp(-1.0, 1.0);
                    }
                    v[30] = (e[j] - ei) / (e[j] + ei + 1.0);
                }
                if on.voice && count > 0.0 {
                    v[31] = vo[0] / count;
                    v[32] = vo[1] / count;
                }
                v[33] = (mix(tick, id[i]) >> 40) as f32 / (1u64 << 23) as f32 - 1.0;
                v[34] = 1.0;

                let g = &genome[i * WEIGHTS..(i + 1) * WEIGHTS];
                let (ih, rest) = g.split_at(INPUTS * HIDDEN);
                let (hh, ho) = rest.split_at(HIDDEN * HIDDEN);
                let mut next = [0.0f32; HIDDEN];
                for j in 0..HIDDEN {
                    let a = dot(&ih[j * INPUTS..(j + 1) * INPUTS], &v);
                    let b = if on.memory { dot(&hh[j * HIDDEN..(j + 1) * HIDDEN], h) } else { 0.0 };
                    next[j] = tanh(a + b);
                }
                h.copy_from_slice(&next);
                for (k, out) in o.iter_mut().enumerate() {
                    *out = dot(&ho[k * HIDDEN..(k + 1) * HIDDEN], &next);
                }
            });
    }

    fn distance(&self, a: usize, b: usize) -> f32 {
        let mut d = 0.0;
        for k in 0..SKETCH {
            let v = self.sketch[a][k] - self.sketch[b][k];
            d += v * v;
        }
        (d / SKETCH as f32).sqrt()
    }

    /// A child's weights. Whole hidden units come from one parent or the other,
    /// so a unit's wiring stays together. Mates are close relatives, so the units
    /// still mean the same thing in both.
    fn make_child(&mut self, a: usize, mate: Option<usize>) -> [f32; WEIGHTS] {
        let mut genes = [0.0; WEIGHTS];
        genes.copy_from_slice(&self.genome[a * WEIGHTS..(a + 1) * WEIGHTS]);
        if let Some(b) = mate {
            for j in 0..HIDDEN {
                if self.rng.f() < 0.5 {
                    let g = &self.genome[b * WEIGHTS..(b + 1) * WEIGHTS];
                    genes[j * INPUTS..(j + 1) * INPUTS]
                        .copy_from_slice(&g[j * INPUTS..(j + 1) * INPUTS]);
                    let hh = INPUTS * HIDDEN + j * HIDDEN;
                    genes[hh..hh + HIDDEN].copy_from_slice(&g[hh..hh + HIDDEN]);
                    let ho = INPUTS * HIDDEN + HIDDEN * HIDDEN;
                    for o in 0..OUTPUTS {
                        genes[ho + o * HIDDEN + j] = g[ho + o * HIDDEN + j];
                    }
                }
            }
        }
        // Mutate a few weights. Skipping ahead keeps births cheap.
        let rate = self.cfg.mutation_rate;
        let size = self.cfg.mutation_size;
        let mut i = 0usize;
        let scale = 1.0 / (1.0 - rate).ln();
        loop {
            i += (self.rng.f().max(1e-7).ln() * scale) as usize;
            if i >= WEIGHTS {
                break;
            }
            genes[i] += self.rng.gauss() * size;
            i += 1;
        }
        genes
    }

    fn act(&mut self) {
        let s = self.cfg.size;
        let compat = self.cfg.compat;
        let on = self.cfg.on.clone();
        let n0 = self.n;
        if n0 == 0 {
            return;
        }
        let start = self.rng.below(n0);
        for k in 0..n0 {
            let i = (start + k) % n0;
            if self.e[i] <= 0.0 {
                continue;
            }
            self.age[i] += 1;
            let (cx, cy) = (self.x[i] as usize, self.y[i] as usize);
            let p = cy * s + cx;

            let burn = (self.fields[F_HEAT][p] * 0.5).min(self.e[i]);
            self.e[i] -= burn;
            self.lost += burn as f64;
            self.hit[i] = burn;
            if self.e[i] <= 0.0 {
                self.cause[i] = BURNED;
                continue;
            }

            let o: [f32; OUTPUTS] = self.out[i * OUTPUTS..(i + 1) * OUTPUTS].try_into().unwrap();
            let mut shares = [0.0f32; BUDGET];
            let top = o[..BUDGET].iter().cloned().fold(f32::MIN, f32::max);
            let enabled = |a: usize| match a {
                EMIT => on.pheromones,
                HEAT => on.heat,
                BITE => on.bite,
                GIVE => on.give,
                ENZYME => on.enzyme,
                _ => true,
            };
            let mut z = 0.0;
            for a in 0..BUDGET {
                if enabled(a) {
                    shares[a] = (o[a] - top).clamp(-20.0, 0.0).exp();
                    z += shares[a];
                }
            }
            shares.iter_mut().for_each(|v| *v /= z);
            for a in 0..BUDGET {
                self.ema_sp[i][a] += (shares[a] - self.ema_sp[i][a]) * 0.02;
            }

            // Resting is the only way to spend less, and it nearly stops the clock.
            let rest = shares[REST];
            let upkeep = (0.015 + 0.0004 * self.e[i]) * (1.0 - 0.8 * rest);
            let pay = upkeep.min(self.e[i]);
            self.e[i] -= pay;
            self.lost += pay as f64;
            let want = SPEND * (1.0 - rest);
            let scale = if want > self.e[i] { self.e[i] / want } else { 1.0 };
            let a = |what: usize| SPEND * shares[what] * scale;
            let commit = self.cfg.commitment;
            let work = |what: usize| {
                let v = a(what);
                v * v / (v + commit)
            };

            let boost = 1.0 + self.fields[F_ENZYME][p].min(2.0);
            let g = self.fields[F_GROUND][p] * (work(GROUND) * 6.0 * boost).min(1.0);
            let m = self.fields[F_MEAT][p] * (work(MEAT) * 12.0).min(1.0);
            self.fields[F_GROUND][p] -= g;
            self.fields[F_MEAT][p] -= m;
            let spend = want * scale;
            let grow = if self.womb[i] < WOMB { a(GROW) } else { 0.0 };
            let burned = spend - a(GROW) + (a(GROW) - grow);
            let meat_gain = m * self.cfg.meat_yield;
            self.lost += (m - meat_gain) as f64;
            self.e[i] += g + meat_gain - spend;
            self.e[i] += a(GROW) - grow;
            self.womb[i] += grow;
            self.lost += (burned - (a(GROW) - grow)).max(0.0) as f64;
            self.tally.intake[0] += g as f64;
            self.tally.intake[1] += meat_gain as f64;
            self.ema_in[i][0] += (g - self.ema_in[i][0]) * 0.01;
            self.ema_in[i][1] += (meat_gain - self.ema_in[i][1]) * 0.01;
            self.ema_in[i][2] -= self.ema_in[i][2] * 0.01;
            self.ema_in[i][3] -= self.ema_in[i][3] * 0.01;

            for c in 0..CHANNELS {
                let weight = (o[MIX + c] - o[MIX..MIX + CHANNELS].iter().cloned().fold(f32::MIN, f32::max))
                    .clamp(-20.0, 0.0)
                    .exp();
                self.fields[F_P + c][p] += a(EMIT) * 30.0 * weight;
            }
            if on.pheromones && self.cfg.odour > 0.0 {
                // Everyone smells of their genes. A cue like this can become a signal.
                let sk = self.sketch[i];
                let top = sk[4..4 + CHANNELS].iter().cloned().fold(f32::MIN, f32::max);
                for c in 0..CHANNELS {
                    self.fields[F_P + c][p] += self.cfg.odour * ((sk[4 + c] - top) * 4.0).exp();
                }
            }
            self.fields[F_ENZYME][p] += work(ENZYME) * 6.0;
            if a(HEAT) > 0.0 {
                // Heat lands on the four squares around the emitter, so it hurts
                // whoever is next to it far more than the emitter itself.
                let flame = work(HEAT) * 12.0 / 4.0;
                for d in [0, 2, 4, 6] {
                    let q = self.at(cx, cy, d);
                    if !self.rock[q] {
                        self.fields[F_HEAT][q] += flame;
                    }
                }
            }
            self.voice[i] = [tanh(o[VOICE]), tanh(o[VOICE + 1])];

            let d = ((o[DIR + 1].atan2(o[DIR]) / 0.785_398_2).round() as i32).rem_euclid(8) as usize;
            self.dir[i] = d as u8;
            if self.rng.f() < work(MOVE) * 8.0 {
                let q = self.at(cx, cy, d);
                if !self.rock[q] && self.occ[q] == NONE {
                    self.occ[q] = i as u32;
                    self.occ[p] = NONE;
                    self.x[i] = (q % s) as u16;
                    self.y[i] = (q / s) as u16;
                }
            }

            let (cx, cy) = (self.x[i] as usize, self.y[i] as usize);
            let front = self.at(cx, cy, d);
            let t = self.occ[front];
            if t != NONE && (t as usize) < self.n && self.e[t as usize] > 0.0 {
                let t = t as usize;
                let kin = self.distance(i, t) < compat;
                if kin {
                    self.tally.front_kin += 1;
                } else {
                    self.tally.front_other += 1;
                }
                if a(BITE) > 0.0 {
                    // The stronger cell bites harder, so energy is also power.
                    let force = 2.0 * self.e[i] / (self.e[i] + self.e[t] + 1e-3);
                    let open = if self.cfg.exposure {
                        let (tx, ty) = (self.x[t] as usize, self.y[t] as usize);
                        let sides = (0..8)
                            .filter(|d| {
                                let q = self.at(tx, ty, *d);
                                self.occ[q] == NONE && !self.rock[q]
                            })
                            .count();
                        // The biter is touching it too, so one side is always reachable.
                        (sides as f32 + 1.0) / 8.0
                    } else {
                        1.0
                    };
                    let take = (work(BITE) * self.cfg.bite_power * force * open).min(self.e[t]);
                    let keep = take * self.cfg.bite_yield;
                    let spill = (take - keep).min(keep);
                    self.e[t] -= take;
                    self.hit[t] = self.hit[t].max(take);
                    self.e[i] += keep;
                    self.fields[F_MEAT][front] += spill;
                    self.lost += (take - keep - spill) as f64;
                    self.tally.intake[2] += keep as f64;
                    self.ema_in[i][2] += keep * 0.01;
                    if kin {
                        self.tally.bite_kin += take as f64;
                        self.ema_bite[i][0] += take * 0.01;
                    } else {
                        self.tally.bite_other += take as f64;
                        self.ema_bite[i][1] += take * 0.01;
                    }
                    if self.e[t] <= 0.0 {
                        self.cause[t] = BITTEN;
                    }
                }
                if a(GIVE) > 0.0 && self.e[i] > 0.0 && self.e[t] > 0.0 {
                    let amount = (work(GIVE) * 8.0).min(self.e[i]);
                    self.e[i] -= amount;
                    self.e[t] += amount * 0.9;
                    self.lost += (amount * 0.1) as f64;
                    self.tally.intake[3] += (amount * 0.9) as f64;
                    self.ema_in[t][3] += amount * 0.9 * 0.01;
                    if kin {
                        self.tally.give_kin += amount as f64;
                    } else {
                        self.tally.give_other += amount as f64;
                    }
                }
                if on.share && o[SHARE] > 0.0 && self.out[t * OUTPUTS + SHARE] > 0.0 && self.e[t] > 0.0 {
                    // Two cells that both agree level out their energy, like a gap junction.
                    let flow = (self.e[i] - self.e[t]) * 0.1;
                    self.e[i] -= flow;
                    self.e[t] += flow;
                }
            }
            if self.e[i] <= 0.0 {
                self.cause[i] = STARVED;
                continue;
            }
            if self.e[i] > MAX_ENERGY {
                self.lost += (self.e[i] - MAX_ENERGY) as f64;
                self.e[i] = MAX_ENERGY;
            }

            if self.womb[i] >= WOMB {
                let first = self.rng.below(8);
                let free = (0..8)
                    .map(|k| self.at(cx, cy, (first + k) % 8))
                    .find(|q| !self.rock[*q] && self.occ[*q] == NONE);
                if let Some(q) = free {
                    let mate = if on.sex && o[MATE] > 0.0 {
                        (0..8).find_map(|k| {
                            let j = self.occ[self.at(cx, cy, k)];
                            if j == NONE {
                                return None;
                            }
                            let j = j as usize;
                            (j < n0
                                && self.e[j] > 0.0
                                && self.out[j * OUTPUTS + MATE] > 0.0
                                && self.distance(i, j) < compat)
                                .then_some(j)
                        })
                    } else {
                        None
                    };
                    self.tally.births += 1;
                    if mate.is_some() {
                        self.tally.births_mixed += 1;
                    }
                    let genes = self.make_child(i, mate);
                    let memory: [f32; HIDDEN] =
                        self.hidden[i * HIDDEN..(i + 1) * HIDDEN].try_into().unwrap();
                    let energy = self.womb[i] - BODY;
                    self.womb[i] = 0.0;
                    let clade = self.clade[i];
                    self.spawn(q, energy, &genes, &memory, clade);
                }
            }
        }
        self.reap();
    }

    fn reap(&mut self) {
        let s = self.cfg.size;
        let mut i = self.n;
        while i > 0 {
            i -= 1;
            let old = self.rng.f() < HAZARD;
            if self.e[i] <= 0.0 || old {
                let cause = if self.e[i] > 0.0 { OLD } else { self.cause[i] };
                self.tally.deaths[cause as usize] += 1;
                let p = self.y[i] as usize * s + self.x[i] as usize;
                self.fields[F_MEAT][p] += self.e[i].max(0.0) + self.womb[i] + BODY;
                self.remove(i);
            }
        }
    }

    fn diffuse(&mut self) {
        let s = self.cfg.size;
        // Meat slowly rots back into ground food.
        let mut rot_lost = 0.0f64;
        let (a, b) = self.fields.split_at_mut(F_MEAT);
        for (g, m) in a[F_GROUND].iter_mut().zip(b[0].iter_mut()) {
            let r = *m * FLOW[F_MEAT].0;
            if r > 0.0 {
                *m -= r;
                let room = (GROUND_CAP - *g).max(0.0);
                let kept = r.min(room);
                *g += kept;
                rot_lost += (r - kept) as f64;
            }
        }
        self.lost += rot_lost;
        let rock = &self.rock;
        let (west, east) = (&self.west, &self.east);
        for f in F_MEAT..FIELDS {
            let (decay, d) = FLOW[f];
            let keep = if f == F_MEAT { 1.0 } else { 1.0 - decay };
            let src = &self.fields[f];
            self.scratch.par_chunks_mut(s).enumerate().for_each(|(y, row)| {
                let up = (y + s - 1) % s * s;
                let down = (y + 1) % s * s;
                let me = y * s;
                for x in 0..s {
                    let i = me + x;
                    if rock[i] {
                        row[x] = 0.0;
                        continue;
                    }
                    let c = src[i];
                    let pick = |j: usize| if rock[j] { c } else { src[j] };
                    let around = pick(me + west[x] as usize)
                        + pick(me + east[x] as usize)
                        + pick(up + x)
                        + pick(down + x);
                    row[x] = (c + d * (around * 0.25 - c)) * keep;
                }
            });
            std::mem::swap(&mut self.fields[f], &mut self.scratch);
        }
    }

    pub fn step(&mut self) {
        self.tick += 1;
        if self.tick % 16 == 1 {
            self.update_sun();
        }
        self.grow_food();
        self.rain();
        self.think();
        self.act();
        self.diffuse();
    }

    /// Everything that holds energy. With the ledger it must add up.
    pub fn stored_energy(&self) -> f64 {
        let ground: f64 = self.fields[F_GROUND].iter().map(|v| *v as f64).sum();
        let meat: f64 = self.fields[F_MEAT].iter().map(|v| *v as f64).sum();
        let cells: f64 = (0..self.n).map(|i| (self.e[i] + self.womb[i] + BODY) as f64).sum();
        ground + meat + cells
    }

    /// Mating groups in a sample: two cells are joined if they could mate, and
    /// species are the connected groups.
    fn species(&self, max: usize, compat: f32) -> Vec<Vec<usize>> {
        let n = self.n;
        let step = (n / max.max(1)).max(1);
        let sample: Vec<usize> = (0..n).step_by(step).collect();
        let m = sample.len();
        let mut parent: Vec<usize> = (0..m).collect();
        fn find(p: &mut [usize], mut a: usize) -> usize {
            while p[a] != a {
                p[a] = p[p[a]];
                a = p[a];
            }
            a
        }
        for a in 0..m {
            for b in a + 1..m {
                if self.distance(sample[a], sample[b]) < compat {
                    let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
                    if ra != rb {
                        parent[ra] = rb;
                    }
                }
            }
        }
        let mut groups: std::collections::HashMap<usize, Vec<usize>> = Default::default();
        for a in 0..m {
            let r = find(&mut parent, a);
            groups.entry(r).or_default().push(sample[a]);
        }
        let mut out: Vec<Vec<usize>> = groups.into_values().collect();
        out.sort_by_key(|g| std::cmp::Reverse(g.len()));
        out
    }

    /// Spatial and behavioural measures, plus the tally since the last call.
    pub fn report(&mut self) -> Value {
        let n = self.n.max(1) as f32;
        let s = self.cfg.size;
        let density = self.n as f32 / self.open as f32;
        let (mut neighbours, mut align, mut align_n) = (0.0f32, 0.0f32, 0.0f32);
        let mut chase = [(0.0f32, 0.0f32); FIELDS];
        let mut shares = [0.0f32; BUDGET];
        let mut intake = [0.0f32; 4];
        let ang = |d: u8| d as f32 * 0.785_398_2;
        for i in 0..self.n {
            let (cx, cy) = (self.x[i] as usize, self.y[i] as usize);
            for a in 0..BUDGET {
                shares[a] += self.ema_sp[i][a];
            }
            for a in 0..4 {
                intake[a] += self.ema_in[i][a];
            }
            for d in 0..8 {
                let j = self.occ[self.at(cx, cy, d)];
                if j != NONE {
                    neighbours += 1.0;
                    align += (ang(self.dir[i]) - ang(self.dir[j as usize])).cos();
                    align_n += 1.0;
                }
            }
            let (ca, sa) = (ang(self.dir[i]).cos(), ang(self.dir[i]).sin());
            for f in 0..FIELDS {
                let field = &self.fields[f];
                let gx = field[cy * s + self.east[cx] as usize] - field[cy * s + self.west[cx] as usize];
                let gy = field[(cy + 1) % s * s + cx] - field[(cy + s - 1) % s * s + cx];
                let mag = (gx * gx + gy * gy).sqrt();
                if mag > 1e-3 {
                    chase[f].0 += (ca * gx + sa * gy) / mag;
                    chase[f].1 += 1.0;
                }
            }
        }
        let names = ["ground", "meat", "heat", "enzyme", "p0", "p1", "p2", "p3"];
        let mut follows = serde_json::Map::new();
        for f in 0..FIELDS {
            follows.insert(names[f].into(), json!(chase[f].0 / chase[f].1.max(1.0)));
        }
        let mut budget = serde_json::Map::new();
        for a in 0..BUDGET {
            budget.insert(ACTION_NAMES[a].into(), json!(shares[a] / n));
        }

        let groups = self.species(1500, self.cfg.compat);
        let sampled: usize = groups.iter().map(|g| g.len()).sum();
        let entropy: f32 = groups
            .iter()
            .map(|g| {
                let p = g.len() as f32 / sampled.max(1) as f32;
                -p * p.ln()
            })
            .sum();
        let mut top = vec![];
        for g in groups.iter().take(8) {
            if g.len() < 5 {
                continue;
            }
            top.push(self.describe(g, sampled));
        }

        // Species count at other thresholds, and how far apart genomes really are.
        let mut at_threshold = serde_json::Map::new();
        for tau in [0.05f32, 0.1, 0.15, 0.2, 0.3] {
            let g = self.species(800, tau);
            let m: usize = g.iter().map(|v| v.len()).sum();
            let h: f32 = g.iter().map(|v| { let p = v.len() as f32 / m.max(1) as f32; -p * p.ln() }).sum();
            at_threshold.insert(format!("{tau}"), json!(h.exp()));
        }
        let mut dists = vec![];
        if self.n > 2 {
            for k in 0..4000 {
                let a = (mix(self.tick, k) >> 32) as usize % self.n;
                let b = (mix(self.tick, k + 99999) >> 32) as usize % self.n;
                if a != b {
                    dists.push(self.distance(a, b));
                }
            }
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        let pct = |p: f32| if dists.is_empty() { 0.0 } else { dists[((dists.len() - 1) as f32 * p) as usize] };
        let mut clades: std::collections::HashMap<u32, u32> = Default::default();
        for c in &self.clade {
            *clades.entry(*c).or_default() += 1;
        }
        // How much of each cell's energy comes from other cells, as a histogram.
        let mut diet = [0u32; 10];
        for i in 0..self.n {
            let v = self.ema_in[i];
            let t: f32 = v.iter().sum::<f32>().max(1e-6);
            diet[(((v[1] + v[2]) / t * 10.0) as usize).min(9)] += 1;
        }
        let t = std::mem::take(&mut self.tally);
        json!({
            "species_at": at_threshold,
            "genome_distance": {"p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9), "max": pct(1.0)},
            "clades": clades.values().filter(|v| **v >= 5).count(),
            "diet": diet,
            "tick": self.tick,
            "cells": self.n,
            "density": density,
            "births": t.births,
            "births_mixed": t.births_mixed,
            "deaths": {"starved": t.deaths[0], "bitten": t.deaths[1], "burned": t.deaths[2], "old": t.deaths[3]},
            "intake": {"ground": t.intake[0], "meat": t.intake[1], "bite": t.intake[2], "gift": t.intake[3]},
            "front": {"kin": t.front_kin, "other": t.front_other},
            "bite": {"kin": t.bite_kin, "other": t.bite_other},
            "give": {"kin": t.give_kin, "other": t.give_other},
            "budget": budget,
            "follows": follows,
            "neighbours": neighbours / n,
            "neighbours_if_random": 8.0 * density,
            "alignment": align / align_n.max(1.0),
            "groups": groups.len(),
            "groups_5plus": groups.iter().filter(|g| g.len() >= 5).count(),
            "effective_species": entropy.exp(),
            "species": top,
        })
    }

    fn describe(&self, g: &[usize], sampled: usize) -> Value {
        let n = g.len() as f32;
        let s = self.cfg.size;
        let mut sh = [0.0f32; BUDGET];
        let mut inn = [0.0f32; 4];
        let (mut e, mut age, mut around, mut bk, mut bo) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
        // Circular mean to see how spread out a species is on the wrapped world.
        let (mut cx, mut sx, mut cy, mut sy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for &i in g {
            for a in 0..BUDGET {
                sh[a] += self.ema_sp[i][a] / n;
            }
            for a in 0..4 {
                inn[a] += self.ema_in[i][a];
            }
            e += self.e[i] / n;
            age += self.age[i] as f32 / n;
            bk += self.ema_bite[i][0];
            bo += self.ema_bite[i][1];
            for d in 0..8 {
                if self.occ[self.at(self.x[i] as usize, self.y[i] as usize, d)] != NONE {
                    around += 1.0 / n;
                }
            }
            let ax = self.x[i] as f32 / s as f32 * 6.283_185_3;
            let ay = self.y[i] as f32 / s as f32 * 6.283_185_3;
            cx += ax.cos() / n;
            sx += ax.sin() / n;
            cy += ay.cos() / n;
            sy += ay.sin() / n;
        }
        let total: f32 = inn.iter().sum::<f32>().max(1e-6);
        json!({
            "share": n / sampled.max(1) as f32,
            "energy": e,
            "age": age,
            "neighbours": around,
            "lives_on": {"ground": inn[0] / total, "meat": inn[1] / total, "bite": inn[2] / total, "gift": inn[3] / total},
            "budget": {"rest": sh[REST], "move": sh[MOVE], "ground": sh[GROUND], "meat": sh[MEAT], "grow": sh[GROW], "emit": sh[EMIT], "heat": sh[HEAT], "bite": sh[BITE], "give": sh[GIVE], "enzyme": sh[ENZYME]},
            "bites_kin_share": if bk + bo > 1e-6 { bk / (bk + bo) } else { 0.0 },
            "clustered": ((cx * cx + sx * sx).sqrt() + (cy * cy + sy * sy).sqrt()) / 2.0,
        })
    }

    /// Two views side by side, RGB. Left: colour is roughly the genome. Right:
    /// colour is what a cell lives on (green ground, red meat and biting, blue gifts).
    pub fn render(&self, scale: usize) -> (usize, usize, Vec<u8>) {
        let s = self.cfg.size;
        let (w, h) = (s * scale * 2 + 4, s * scale);
        let mut img = vec![0u8; w * h * 3];
        let put = |img: &mut [u8], ox: usize, x: usize, y: usize, c: [u8; 3], size: usize| {
            for yy in 0..size {
                for xx in 0..size {
                    let at = ((y * scale + yy) * w + ox + x * scale + xx) * 3;
                    img[at..at + 3].copy_from_slice(&c);
                }
            }
        };
        let hue = |sketch: &[f32; SKETCH]| {
            let c = |k: usize| ((tanh(sketch[k] * 2.0) * 0.5 + 0.5) * 255.0) as u8;
            let (r, g, b) = (c(0), c(1), c(2));
            // Push away from grey so lineages stand apart.
            let m = (r as u16 + g as u16 + b as u16) / 3;
            let f = |v: u8| (m as i32 + (v as i32 - m as i32) * 2 + 40).clamp(30, 255) as u8;
            [f(r), f(g), f(b)]
        };
        for y in 0..s {
            for x in 0..s {
                let p = y * s + x;
                let bg = if self.rock[p] {
                    [58, 60, 70]
                } else {
                    let g = (self.fields[F_GROUND][p] / GROUND_CAP).min(1.0);
                    let m = (self.fields[F_MEAT][p] / 3.0).min(1.0);
                    let heat = (self.fields[F_HEAT][p] / 2.0).min(1.0);
                    let pher: f32 = (F_P..FIELDS).map(|f| self.fields[f][p]).sum::<f32>().min(4.0) / 4.0;
                    [
                        (10.0 + m * 70.0 + heat * 200.0) as u8,
                        (12.0 + g * 38.0 + heat * 90.0) as u8,
                        (16.0 + pher * 44.0 + g * 8.0) as u8,
                    ]
                };
                put(&mut img, 0, x, y, bg, scale);
                put(&mut img, s * scale + 4, x, y, bg, scale);
            }
        }
        for i in 0..self.n {
            let (x, y) = (self.x[i] as usize, self.y[i] as usize);
            put(&mut img, 0, x, y, hue(&self.sketch[i]), scale);
            let inn = self.ema_in[i];
            let t = inn.iter().sum::<f32>().max(1e-3);
            let role = [
                (40.0 + 215.0 * (inn[1] + inn[2]) / t) as u8,
                (40.0 + 215.0 * inn[0] / t) as u8,
                (40.0 + 215.0 * inn[3] / t) as u8,
            ];
            put(&mut img, s * scale + 4, x, y, role, scale);
        }
        (w, h, img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(size: usize, seed: u64) -> World {
        World::new(Config { size, seed, ..Config::default() })
    }

    #[test]
    fn energy_adds_up() {
        let mut w = World::new(Config { size: 64, seed: 3, bloom: 0.5, ..Config::default() });
        for _ in 0..1500 {
            w.step();
        }
        assert!(w.n > 0);
        let have = w.stored_energy() + w.lost;
        let put_in = w.start_energy + w.sun_in + w.rain_in;
        let gap = (have - put_in).abs() / put_in;
        assert!(gap < 0.002, "have {have}, put in {put_in}, gap {gap}");
    }

    #[test]
    fn same_seed_same_world_at_any_thread_count() {
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut w = world(48, 9);
                for _ in 0..400 {
                    w.step();
                }
                (w.n, w.stored_energy())
            })
        };
        assert_eq!(run(1), run(3));
    }

    #[test]
    fn grid_and_cells_agree() {
        let mut w = world(48, 5);
        for _ in 0..800 {
            w.step();
        }
        for i in 0..w.n {
            assert_eq!(w.occ[w.y[i] as usize * 48 + w.x[i] as usize], i as u32);
            assert!(!w.rock[w.y[i] as usize * 48 + w.x[i] as usize]);
            assert!(w.e[i] > 0.0);
        }
        let held = w.occ.iter().filter(|o| **o != NONE).count();
        assert_eq!(held, w.n);
        assert_eq!(w.genome.len(), w.n * WEIGHTS);
    }

    #[test]
    fn a_child_mixes_whole_units_from_two_parents() {
        let mut w = world(32, 7);
        w.cfg.mutation_rate = 1e-6;
        let a = [1.0f32; WEIGHTS];
        let b = [2.0f32; WEIGHTS];
        w.spawn(0, 5.0, &a, &[0.0; HIDDEN], NONE);
        w.spawn(1, 5.0, &b, &[0.0; HIDDEN], NONE);
        let child = w.make_child(0, Some(1));
        for j in 0..HIDDEN {
            let v = child[j * INPUTS];
            assert!(child[j * INPUTS..(j + 1) * INPUTS].iter().all(|x| *x == v));
            let ho = INPUTS * HIDDEN + HIDDEN * HIDDEN;
            for o in 0..OUTPUTS {
                assert_eq!(child[ho + o * HIDDEN + j], v);
            }
        }
    }
}
