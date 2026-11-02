//! What's alive and how it lives, measured on a random sample of cells. Nothing
//! here feeds back into the world.

use serde_json::{Value, json};

use crate::config::{CELL_BYTES, STAT_BYTES};
use crate::world::World;

pub const ACTIONS: [&str; 16] = [
    "rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme", "armour",
    "clean", "x0", "x1", "x2", "x3",
];
/// The actions that count as how a cell lives: all but the spare ones.
const LIVING_ACTIONS: usize = 12;
/// With the first spare action, which is mineral uptake and reads 0 unless it's on.
const BEHAVIOUR_DIMS: usize = 22;
/// Where each gauge is in World::gauges(), which starts at counter 32. See common.wgsl.
const G_BITES: usize = 0;
const G_GAIN: usize = 8;
const G_TARGET: usize = 16;
const G_CTRL_N: usize = 24;
const G_CTRL_EXTRA: usize = 25;
const G_CELLS: usize = 26;
const G_MATCH_N: usize = 27;
const G_MATCH: usize = 28;
const G_LIMIT: usize = 29;
const G_FOLLOW: usize = 35;
const G_FOLLOW_N: usize = 39;
const G_CHILD: usize = 40;
const G_CHILD_E: usize = 46;
const SKETCH_RANGE: f32 = 4.0;

#[derive(Clone)]
pub struct Sample {
    pub e: f32,
    pub womb: f32,
    pub age: f32,
    pub pos: u32,
    pub clade: u32,
    pub generation: u32,
    /// Hardness, wound, practice at grazing, practice at eating bodies.
    pub x: [f32; 4],
    /// Free mineral, coat, mineral bound in the body, the key of the last real bite.
    pub y: [f32; 4],
    pub diet: [f32; 4],
    pub budget: [f32; 16],
    pub chase: [f32; 4],
    pub follow: [f32; 4],
    pub misc: [f32; 4],
    /// Rule measures, zero unless roads_stats is on.
    pub measures: [f32; 4],
    pub sketch: [f32; 16],
    pub genome_at: usize,
}

pub struct Samples {
    pub cells: Vec<Sample>,
    pub genomes: Vec<i8>,
    pub blocks_bytes: usize,
}

/// Reads scattered runs of slots, enough to hold about `want` live cells.
pub fn sample(world: &mut World, want: usize) -> Samples {
    world.flush();
    let slots = world.cfg.slots as u64;
    let alive = world.alive().max(1) as u64;
    let genome_bytes = world.cfg.blocks() as u64 * 16;
    let need = (want as u64 * slots / alive).min(slots).min(80_000);
    // Short runs: slots are reused in order of death, so neighbouring slots are cells of the same age.
    const RUN: u64 = 8;
    let runs = need.div_ceil(RUN).max(1);
    let gap = (slots / runs).max(1);
    let starts: Vec<u64> = (0..runs)
        .map(|k| (k * gap).min(slots.saturating_sub(RUN.min(slots))))
        .collect();
    let len = RUN.min(slots);
    let of = |stride: u64| -> Vec<(u64, u64)> {
        starts.iter().map(|s| (s * stride, len * stride)).collect()
    };
    let cell = world.read_ranges("cell", &of(CELL_BYTES));
    let stat = world.read_ranges("stat", &of(STAT_BYTES));
    let sketch = world.read_ranges("sketch", &of(16));
    let run_list: Vec<(u64, u64)> = starts.iter().map(|s| (*s, len)).collect();
    let genome = world.read_genomes(&run_list, genome_bytes);

    let mut cells = vec![];
    let mut genomes = vec![];
    let total = starts.len() * len as usize;
    for i in 0..total {
        let c: &[u8] = &cell[i * CELL_BYTES as usize..(i + 1) * CELL_BYTES as usize];
        let f = |k: usize| f32::from_le_bytes(c[k * 4..k * 4 + 4].try_into().unwrap());
        let u = |k: usize| u32::from_le_bytes(c[k * 4..k * 4 + 4].try_into().unwrap());
        let (e, age) = (f(0), f(2));
        if e <= 0.0 || age < 1.0 {
            continue;
        }
        let st: &[f32] =
            bytemuck::cast_slice(&stat[i * STAT_BYTES as usize..(i + 1) * STAT_BYTES as usize]);
        let mut sk = [0.0f32; 16];
        for (k, v) in sketch[i * 16..(i + 1) * 16].iter().enumerate() {
            sk[k] = (*v as i8) as f32 / 127.0 * SKETCH_RANGE;
        }
        let g = &genome[i * genome_bytes as usize..(i + 1) * genome_bytes as usize];
        let genome_at = genomes.len();
        genomes.extend(g.iter().map(|b| *b as i8));
        let arr4 = |k: usize| -> [f32; 4] { st[k..k + 4].try_into().unwrap() };
        let mut budget = [0.0f32; 16];
        budget[..12].copy_from_slice(&st[4..16]);
        budget[12..].copy_from_slice(&st[28..32]);
        cells.push(Sample {
            e,
            womb: f(1),
            age,
            pos: u(10),
            clade: u(12),
            generation: u(13),
            x: [f(4), f(5), f(6), f(7)],
            y: [f(20), f(21), f(22), f(23)],
            diet: arr4(0),
            budget,
            chase: arr4(16),
            follow: arr4(20),
            misc: arr4(24),
            measures: arr4(32),
            sketch: sk,
            genome_at,
        });
    }
    Samples {
        cells,
        genomes,
        blocks_bytes: genome_bytes as usize,
    }
}

fn sketch_gap(a: &Sample, b: &Sample) -> f32 {
    let mut d = 0.0;
    for k in 0..16 {
        let v = a.sketch[k] - b.sketch[k];
        d += v * v;
    }
    (d / 16.0).sqrt()
}

/// Mating groups in the sample: two cells are joined if their genomes are close,
/// and species are the connected groups.
fn groups(cells: &[Sample], limit: usize, tau: f32) -> Vec<Vec<usize>> {
    let step = (cells.len() / limit.max(1)).max(1);
    let picked: Vec<usize> = (0..cells.len()).step_by(step).collect();
    let m = picked.len();
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
            if sketch_gap(&cells[picked[a]], &cells[picked[b]]) < tau {
                let (x, y) = (find(&mut parent, a), find(&mut parent, b));
                if x != y {
                    parent[x] = y;
                }
            }
        }
    }
    let mut map: std::collections::HashMap<usize, Vec<usize>> = Default::default();
    for a in 0..m {
        let r = find(&mut parent, a);
        map.entry(r).or_default().push(picked[a]);
    }
    let mut out: Vec<Vec<usize>> = map.into_values().collect();
    out.sort_by_key(|g| std::cmp::Reverse(g.len()));
    out
}

fn entropy_species(gs: &[Vec<usize>]) -> f32 {
    let total: usize = gs.iter().map(|g| g.len()).sum();
    let h: f32 = gs
        .iter()
        .map(|g| {
            let p = g.len() as f32 / total.max(1) as f32;
            -p * p.ln()
        })
        .sum();
    h.exp()
}

fn lives_on(diet: [f32; 4]) -> [f32; 4] {
    let t = (diet[0] + diet[1] + diet[2] + diet[3]).max(1e-6);
    [diet[0] / t, diet[1] / t, diet[2] / t, diet[3] / t]
}

fn behaviour(c: &Sample) -> [f32; BEHAVIOUR_DIMS] {
    let l = lives_on(c.diet);
    let mut v = [0.0; BEHAVIOUR_DIMS];
    v[0] = l[0];
    v[1] = l[1];
    v[2] = l[2];
    v[3] = l[3];
    v[4..4 + LIVING_ACTIONS].copy_from_slice(&c.budget[..LIVING_ACTIONS]);
    v[16] = c.chase[0];
    v[17] = c.chase[1];
    v[18] = c.chase[2];
    v[19] = c.chase[3];
    v[20] = c.misc[0];
    v[21] = c.budget[12];
    v
}

fn describe(cells: &[Sample], idx: &[usize], total: usize, size: u32) -> Value {
    let n = idx.len().max(1) as f32;
    let (mut diet, mut budget, mut chase, mut follow) =
        ([0.0f32; 4], [0.0f32; 16], [0.0f32; 4], [0.0f32; 4]);
    let (mut state, mut measures) = ([0.0f32; 4], [0.0f32; 4]);
    let (mut mineral, mut bound, mut coat_c, mut coat_s, mut womb) = (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut e, mut age, mut gen_, mut crowd, mut bk, mut bo) =
        (0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut cx, mut sx, mut cy, mut sy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for &i in idx {
        let c = &cells[i];
        for k in 0..4 {
            diet[k] += c.diet[k];
            chase[k] += c.chase[k] / n;
            follow[k] += c.follow[k] / n;
            state[k] += c.x[k] / n;
            measures[k] += c.measures[k] / n;
        }
        for k in 0..16 {
            budget[k] += c.budget[k] / n;
        }
        e += c.e / n;
        mineral += c.y[0] / n;
        bound += c.y[2] / n;
        coat_c += (c.y[1] * std::f32::consts::TAU).cos() / n;
        coat_s += (c.y[1] * std::f32::consts::TAU).sin() / n;
        womb += c.womb / n;
        age += c.age / n;
        gen_ += c.generation as f32 / n;
        crowd += c.misc[0] * 8.0 / n;
        bk += c.misc[1];
        bo += c.misc[2];
        let ax = (c.pos & (size - 1)) as f32 / size as f32 * std::f32::consts::TAU;
        let ay = (c.pos >> size.trailing_zeros()) as f32 / size as f32 * std::f32::consts::TAU;
        cx += ax.cos() / n;
        sx += ax.sin() / n;
        cy += ay.cos() / n;
        sy += ay.sin() / n;
    }
    let l = lives_on(diet);
    let mut b = serde_json::Map::new();
    for k in 0..16 {
        b.insert(ACTIONS[k].into(), json!(round3(budget[k])));
    }
    json!({
        "share": round3(n / total.max(1) as f32),
        "energy": round3(e),
        "age": age.round(),
        "generation": gen_.round(),
        "neighbours": round3(crowd),
        "lives_on": {"ground": round3(l[0]), "meat": round3(l[1]), "bite": round3(l[2]), "gift": round3(l[3])},
        "budget": b,
        "moving": round3(chase[3]),
        "heads_to_bigger": round3(chase[0]),
        "heads_to_smaller": round3(chase[1]),
        "heads_to_food": round3(chase[2]),
        "follows_scent": [round3(follow[0]), round3(follow[1]), round3(follow[2]), round3(follow[3])],
        "bites_on_similar": if bk + bo > 1e-6 { round3(bk / (bk + bo)) } else { 0.0 },
        "spread": round3(1.0 - ((cx * cx + sx * sx).sqrt() + (cy * cy + sy * sy).sqrt()) / 2.0),
        "hardness": round3(state[0]),
        "wound": round3(state[1]),
        "practice": [round3(state[2]), round3(state[3])],
        "measures": [round3(measures[0]), round3(measures[1]), round3(measures[2]), round3(measures[3])],
        "mineral": [round3(mineral), round3(bound)],
        "coat_spread": round3(1.0 - (coat_c * coat_c + coat_s * coat_s).sqrt()),
        "womb": round3(womb),
    })
}

fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

/// Groups cells by how they live, ignoring genes, using k-means on z-scores.
fn lifestyles(cells: &[Sample], k: usize) -> (Vec<Vec<usize>>, Vec<usize>) {
    let n = cells.len();
    let dims = BEHAVIOUR_DIMS;
    let data: Vec<[f32; BEHAVIOUR_DIMS]> = cells.iter().map(behaviour).collect();
    let mut mean = [0.0f32; BEHAVIOUR_DIMS];
    let mut var = [0.0f32; BEHAVIOUR_DIMS];
    for v in &data {
        for d in 0..dims {
            mean[d] += v[d] / n as f32;
        }
    }
    for v in &data {
        for d in 0..dims {
            var[d] += (v[d] - mean[d]).powi(2) / n as f32;
        }
    }
    let z: Vec<[f32; BEHAVIOUR_DIMS]> = data
        .iter()
        .map(|v| {
            let mut o = [0.0; BEHAVIOUR_DIMS];
            for d in 0..dims {
                o[d] = (v[d] - mean[d]) / var[d].sqrt().max(1e-3);
            }
            o
        })
        .collect();
    let dist = |a: &[f32; BEHAVIOUR_DIMS], b: &[f32; BEHAVIOUR_DIMS]| -> f32 {
        (0..dims).map(|d| (a[d] - b[d]).powi(2)).sum()
    };
    // Farthest-first starts make this the same on every run.
    let mut centres: Vec<[f32; BEHAVIOUR_DIMS]> = vec![z[0]];
    while centres.len() < k {
        let far = (0..n)
            .max_by(|&a, &b| {
                let da = centres
                    .iter()
                    .map(|c| dist(&z[a], c))
                    .fold(f32::MAX, f32::min);
                let db = centres
                    .iter()
                    .map(|c| dist(&z[b], c))
                    .fold(f32::MAX, f32::min);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();
        centres.push(z[far]);
    }
    let mut label = vec![0usize; n];
    for _ in 0..12 {
        for i in 0..n {
            let (mut best, mut bd) = (0, f32::MAX);
            for (c, centre) in centres.iter().enumerate() {
                let d = dist(&z[i], centre);
                if d < bd {
                    bd = d;
                    best = c;
                }
            }
            label[i] = best;
        }
        let mut sum = vec![[0.0f32; BEHAVIOUR_DIMS]; k];
        let mut count = vec![0f32; k];
        for i in 0..n {
            count[label[i]] += 1.0;
            for d in 0..dims {
                sum[label[i]][d] += z[i][d];
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
    let mut out = vec![vec![]; k];
    for i in 0..n {
        out[label[i]].push(i);
    }
    (out, label)
}

/// Share of variation in how cells live that species membership explains, counting
/// only groups of at least 1% of the sample and corrected for the number of groups,
/// since a group of one cell always "explains" itself. Near 0 means species are just
/// genetic labels. High means they really live differently.
fn species_explain_behaviour(cells: &[Sample], gs: &[Vec<usize>], scaled: bool) -> f32 {
    let sampled: usize = gs.iter().map(|g| g.len()).sum();
    let big: Vec<&Vec<usize>> = gs.iter().filter(|g| g.len() * 100 >= sampled).collect();
    let members: Vec<usize> = big.iter().flat_map(|g| g.iter().copied()).collect();
    let k = big.len();
    if k < 2 || members.len() <= k + 1 {
        return 0.0;
    }
    let mut data: Vec<[f32; BEHAVIOUR_DIMS]> = cells.iter().map(behaviour).collect();
    let mut mean = [0.0f32; BEHAVIOUR_DIMS];
    for &i in &members {
        for d in 0..BEHAVIOUR_DIMS {
            mean[d] += data[i][d] / members.len() as f32;
        }
    }
    // Scaled, every measure counts the same, so noisy ones can't drown out a clean split in diet.
    if scaled {
        let mut var = [0.0f32; BEHAVIOUR_DIMS];
        for &i in &members {
            for d in 0..BEHAVIOUR_DIMS {
                var[d] += (data[i][d] - mean[d]).powi(2) / members.len() as f32;
            }
        }
        for row in data.iter_mut() {
            for d in 0..BEHAVIOUR_DIMS {
                row[d] = (row[d] - mean[d]) / var[d].sqrt().max(1e-3);
            }
        }
        mean = [0.0; BEHAVIOUR_DIMS];
    }
    let (mut within, mut total) = (0.0f32, 0.0f32);
    for g in &big {
        let mut gm = [0.0f32; BEHAVIOUR_DIMS];
        for &i in g.iter() {
            for d in 0..BEHAVIOUR_DIMS {
                gm[d] += data[i][d] / g.len() as f32;
            }
        }
        for &i in g.iter() {
            for d in 0..BEHAVIOUR_DIMS {
                within += (data[i][d] - gm[d]).powi(2);
                total += (data[i][d] - mean[d]).powi(2);
            }
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    let between = total - within;
    let per_cell = within / (members.len() - k) as f32;
    ((between - (k - 1) as f32 * per_cell) / (total + per_cell)).max(0.0)
}

pub struct Tracker {
    pub last: [u32; 32],
}

/// One line of the log.
pub fn report(world: &mut World, tracker: &mut Tracker) -> Value {
    let samples = sample(world, 6000);
    let cells = &samples.cells;
    let counters = world.counters();
    let d = |k: usize| counters[k].wrapping_sub(tracker.last[k]);
    let (births, mixed, starved, bitten, burned, old, rained) =
        (d(3), d(4), d(5), d(6), d(7), d(8), d(9));
    let no_slot = d(13);
    let sun_taken = d(14);
    tracker.last = counters;
    let n = world.alive();
    // Fixed here, so logs from all experiments compare. Genomes drift fast, about 0.03 of
    // rms weight per generation, so much below this it only finds recent family.
    let tau = 0.4;
    let size = world.cfg.size;
    let mut out = json!({
        "tick": world.tick,
        "cells": n,
        "density": n as f64 / world.open.max(1) as f64,
        "births": births,
        "births_mixed": mixed,
        "deaths": {"starved": starved, "bitten": bitten, "burned": burned, "old": old},
        "rained": rained,
        "births_no_slot": no_slot,
    });
    if world.cfg.physics.get("plants_meter").and_then(Value::as_bool) == Some(true) {
        // Sampled one tick in 256 at 256 counts per unit: over ticks in a log step, it's sun per tick.
        out["sun_taken"] = json!(sun_taken);
    }
    depth_log(world, cells, &mut out);
    if cells.len() < 10 {
        return out;
    }
    let all: Vec<usize> = (0..cells.len()).collect();
    out["world"] = describe(cells, &all, cells.len(), size);

    let gs = groups(cells, 1500, tau);
    let sampled: usize = gs.iter().map(|g| g.len()).sum();
    out["groups"] = json!(gs.len());
    out["groups_1pct"] = json!(gs.iter().filter(|g| g.len() * 100 >= sampled).count());
    out["effective_species"] = json!(round3(entropy_species(&gs)));
    out["species_explain_behaviour"] = json!(round3(species_explain_behaviour(cells, &gs, false)));
    out["species_explain_scaled"] = json!(round3(species_explain_behaviour(cells, &gs, true)));
    let mut at = serde_json::Map::new();
    for t in [0.05f32, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0] {
        let g = groups(cells, 800, t);
        at.insert(format!("{t}"), json!(round3(entropy_species(&g))));
    }
    out["species_at"] = Value::Object(at);
    out["species"] = Value::Array(
        gs.iter()
            .filter(|g| g.len() * 100 >= sampled)
            .take(10)
            .map(|g| describe(cells, g, sampled, size))
            .collect(),
    );

    // Real weight differences between random pairs, not the projection.
    let m = cells.len();
    let mut gaps = vec![];
    let mut state = world.tick ^ 0x1234_5678;
    let mut pick = |n: usize| {
        state = state
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(0xBF58476D1CE4E5B9);
        ((state >> 33) as usize) % n
    };
    for _ in 0..1500 {
        let (a, b) = (pick(m), pick(m));
        if a == b {
            continue;
        }
        let bytes = samples.blocks_bytes;
        let ga = &samples.genomes[cells[a].genome_at..cells[a].genome_at + bytes];
        let gb = &samples.genomes[cells[b].genome_at..cells[b].genome_at + bytes];
        let sum: f32 = ga
            .iter()
            .zip(gb)
            .map(|(x, y)| (*x as f32 - *y as f32).powi(2))
            .sum();
        gaps.push((sum / bytes as f32).sqrt() / 127.0 * 2.0);
    }
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f32| {
        if gaps.is_empty() {
            0.0
        } else {
            round3(gaps[((gaps.len() - 1) as f32 * p) as usize])
        }
    };
    out["genome_gap"] = json!({"p10": pct(0.1), "p50": pct(0.5), "p90": pct(0.9), "max": pct(1.0)});

    let (life, label) = lifestyles(cells, 6.min(m));
    let mut styles: Vec<&Vec<usize>> = life.iter().filter(|g| !g.is_empty()).collect();
    styles.sort_by_key(|g| std::cmp::Reverse(g.len()));
    out["lifestyles"] = Value::Array(styles.iter().map(|g| describe(cells, g, m, size)).collect());
    let _ = label;

    let mut clades: std::collections::HashMap<u32, u32> = Default::default();
    for c in cells.iter() {
        *clades.entry(c.clade).or_default() += 1;
    }
    out["founder_lines"] = json!(clades.len());
    // The distribution people ask about first: how many cells mostly eat other cells.
    let mut diet_hist = [0u32; 10];
    for c in cells.iter() {
        let l = lives_on(c.diet);
        diet_hist[(((l[1] + l[2]) * 10.0) as usize).min(9)] += 1;
    }
    out["from_other_cells_hist"] = json!(diet_hist);
    out
}

/// The parts of a log line that look for real depth: what's measured in the shaders, since the
/// last line. Nothing here is a verdict. research/life-gpu/PREDICTIONS.md says how to read them.
fn depth_log(world: &mut World, cells: &[Sample], out: &mut Value) {
    if world.cfg.b("depth_stats") {
        let now = world.gauges();
        let d: Vec<u32> = now
            .iter()
            .zip(&world.gauge_last)
            .map(|(a, b)| a.wrapping_sub(*b))
            .collect();
        world.gauge_last = now;

        // Group hunting. "n" is how many others bit the same target last tick.
        let bites: Vec<f32> = (0..8).map(|n| d[G_BITES + n] as f32).collect();
        let total: f32 = bites.iter().sum();
        let per_bite = |base: usize, scale: f32| -> Vec<f32> {
            (0..8)
                .map(|n| {
                    if bites[n] > 0.0 {
                        round3(d[base + n] as f32 / scale / bites[n])
                    } else {
                        0.0
                    }
                })
                .collect()
        };
        let extras: f32 = bites
            .iter()
            .enumerate()
            .map(|(n, b)| n as f32 * b)
            .sum::<f32>()
            / total.max(1.0);
        let gain = per_bite(G_GAIN, 64.0);
        let grouped: f32 = bites[1..].iter().sum();
        let group_gain: f32 = (1..8).map(|n| d[G_GAIN + n] as f32 / 64.0).sum::<f32>() / grouped.max(1.0);
        out["group_hunting"] = json!({
            "bites": bites,
            "gain_per_bite": gain,
            "target_energy": per_bite(G_TARGET, 4.0),
            "extras": round3(extras),
            "extras_control": round3(d[G_CTRL_EXTRA] as f32 / (d[G_CTRL_N] as f32).max(1.0)),
            // If bites fell at random across all bodies.
            "extras_chance": round3(total / (d[G_CELLS] as f32).max(1.0)),
            "share_in_groups": round3(grouped / total.max(1.0)),
            // Above 1 means a bite in company pays better than a bite alone.
            "group_advantage": if bites[0] > 0.0 && gain[0] > 0.0 && grouped > 0.0 { round3(group_gain / gain[0]) } else { 0.0 },
        });

        // What stands between a cell and a child. Cell-ticks, sampled.
        let names = ["energy", "mineral", "both", "room", "waiting", "ready"];
        let lim: Vec<f32> = (0..6).map(|k| d[G_LIMIT + k] as f32).collect();
        let sum: f32 = lim.iter().sum::<f32>().max(1.0);
        let mut limit = serde_json::Map::new();
        for k in 0..6 {
            limit.insert(names[k].into(), json!(round3(lim[k] / sum)));
        }
        out["birth_limit"] = Value::Object(limit);

        let born: f32 = (0..6).map(|k| d[G_CHILD + k] as f32).sum();
        let hist: Vec<u32> = (0..6).map(|k| d[G_CHILD + k]).collect();
        out["child_size_hist"] = json!(hist);
        out["child_energy"] = json!(round3(d[G_CHILD_E] as f32 / 16.0 / born.max(1.0)));

        // Scent ahead against the sides, for cells that hardly emit. 0 is what chance gives.
        let n = d[G_FOLLOW_N] as f32;
        let follow: Vec<f32> = (0..4)
            .map(|k| round3(d[G_FOLLOW + k] as i32 as f32 / 256.0 / n.max(1.0)))
            .collect();
        out["follows_others_scent"] = json!({"mean": follow, "samples": d[G_FOLLOW_N]});

        if world.cfg.b("coat_on") {
            const BINS: usize = 16;
            let (mut coats, mut keys) = ([0.0f32; BINS], [0.0f32; BINS]);
            let mut keyed = 0.0f32;
            for c in cells {
                coats[(c.y[1] * BINS as f32) as usize % BINS] += 1.0;
                if c.y[3] > 0.0 {
                    keys[(c.y[3] * BINS as f32) as usize % BINS] += 1.0;
                    keyed += 1.0;
                }
            }
            let share = |v: &[f32; BINS], n: f32| -> f32 {
                v.iter()
                    .map(|c| c / n.max(1.0))
                    .filter(|p| *p > 0.0)
                    .map(|p| -p * p.ln())
                    .sum::<f32>()
                    .exp()
            };
            let n = cells.len() as f32;
            // The most any one key could match this population, on average.
            let mut best = 0.0f32;
            for k in 0..BINS {
                let mut m = 0.0;
                for (j, c) in coats.iter().enumerate() {
                    let gap = (k as f32 - j as f32) / BINS as f32 * std::f32::consts::TAU;
                    m += c / n.max(1.0) * (0.5 + 0.5 * gap.cos());
                }
                best = best.max(m);
            }
            let matched = d[G_MATCH] as f32 / 256.0 / (d[G_MATCH_N] as f32).max(1.0);
            out["coat"] = json!({
                "effective": round3(share(&coats, n)),
                "effective_keys": round3(share(&keys, keyed)),
                "hist": coats,
                "key_hist": keys,
                "key_match": round3(matched),
                // A key placed at random matches 0.5 of the way, whatever the coats.
                "key_chance": 0.5,
                "key_best": round3(best),
                // 0 is no better than random, 1 is as good as any one key can be.
                "key_tracking": if best > 0.51 && d[G_MATCH_N] > 0 { round3((matched - 0.5) / (best - 0.5)) } else { 0.0 },
            });
        }
    }
    if world.cfg.b("mineral_on") {
        let w = world.cfg.size as u64;
        let rows = 256.min(w);
        let step = (w / rows).max(1);
        let ranges: Vec<(u64, u64)> = (0..rows)
            .map(|k| ((k * step) * w * 16, w * 16))
            .collect();
        let bytes = world.read_ranges("fd", &ranges);
        let (mut ground, mut open) = (0.0f64, 0.0f64);
        for v in bytemuck::cast_slice::<u8, [f32; 4]>(&bytes) {
            if v[2] > 0.0 {
                ground += v[1] as f64;
                open += 1.0;
            }
        }
        let n = cells.len().max(1) as f32;
        let alive = world.alive() as f32 / world.open.max(1) as f32;
        let free: f32 = cells.iter().map(|c| c.y[0]).sum::<f32>() / n * alive;
        let bound: f32 = cells.iter().map(|c| c.y[2]).sum::<f32>() / n * alive;
        // All per open square. Their sum is the world's mineral, estimated from samples here,
        // and the tests check it exactly.
        out["mineral"] = json!({
            "ground": round3((ground / open.max(1.0)) as f32),
            "free": round3(free),
            "bound": round3(bound),
        });
    }
    if world.cfg.b("trail_stats") {
        let fd = world.read_vec4s("fd");
        if world.trail_last.len() != fd.len() {
            world.trail_last = vec![0.0; fd.len()];
        }
        let mut steps: Vec<f32> = fd
            .iter()
            .zip(&world.trail_last)
            .map(|(a, b)| a[3] - b)
            .collect();
        world.trail_last = fd.iter().map(|v| v[3]).collect();
        let total: f64 = steps.iter().map(|v| *v as f64).sum();
        let top = (world.open as usize / 20).clamp(1, steps.len());
        steps.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap());
        let busiest: f64 = steps[..top].iter().map(|v| *v as f64).sum();
        out["trails"] = json!({
            "steps": total,
            "top5_share": round3((busiest / total.max(1.0)) as f32),
            "top5_chance": round3(crate::measure::chance_top_share(total / world.open.max(1) as f64, 0.05) as f32),
        });
    }
}

/// A world's main axes of genome difference, for colouring cells.
pub struct Palette {
    pub m: [[f32; 16]; 3],
    pub mean: [f32; 16],
    pub gain: [f32; 3],
    ready: bool,
}

impl Default for Palette {
    fn default() -> Self {
        Palette {
            m: [[0.0; 16]; 3],
            mean: [0.0; 16],
            gain: [1.0; 3],
            ready: false,
        }
    }
}

impl Palette {
    /// Updates from a sample. Axes turn slowly and keep their sign, so a lineage keeps its colour.
    pub fn update(&mut self, cells: &[Sample]) {
        if cells.len() < 8 {
            return;
        }
        let n = cells.len() as f32;
        let mut mean = [0.0f32; 16];
        for c in cells {
            for k in 0..16 {
                mean[k] += c.sketch[k] / n;
            }
        }
        let mut cov = [[0.0f32; 16]; 16];
        for c in cells {
            for a in 0..16 {
                for b in 0..16 {
                    cov[a][b] += (c.sketch[a] - mean[a]) * (c.sketch[b] - mean[b]) / n;
                }
            }
        }
        let mut axes = [[0.0f32; 16]; 3];
        let mut sd = [1.0f32; 3];
        for (axis, slot) in axes.iter_mut().enumerate() {
            let mut v = if self.ready {
                self.m[axis]
            } else {
                std::array::from_fn(|k| if k == axis { 1.0 } else { 0.1 })
            };
            for _ in 0..30 {
                let mut next = [0.0f32; 16];
                for a in 0..16 {
                    for b in 0..16 {
                        next[a] += cov[a][b] * v[b];
                    }
                }
                let len = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                next.iter_mut().for_each(|x| *x /= len);
                v = next;
            }
            let mut lambda = 0.0;
            let mut cv = [0.0f32; 16];
            for a in 0..16 {
                for b in 0..16 {
                    cv[a] += cov[a][b] * v[b];
                }
                lambda += cv[a] * v[a];
            }
            sd[axis] = lambda.max(1e-6).sqrt();
            for a in 0..16 {
                for b in 0..16 {
                    cov[a][b] -= lambda * v[a] * v[b];
                }
            }
            *slot = v;
        }
        if self.ready {
            for a in 0..3 {
                let along: f32 = (0..16).map(|k| axes[a][k] * self.m[a][k]).sum();
                let flip = if along < 0.0 { -1.0 } else { 1.0 };
                for k in 0..16 {
                    axes[a][k] = self.m[a][k] * 0.8 + axes[a][k] * flip * 0.2;
                }
                let len = axes[a].iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                axes[a].iter_mut().for_each(|x| *x /= len);
            }
            for k in 0..16 {
                self.mean[k] = self.mean[k] * 0.8 + mean[k] * 0.2;
            }
            for a in 0..3 {
                self.gain[a] = self.gain[a] * 0.8 + (1.2 / sd[a]) * 0.2;
            }
        } else {
            self.mean = mean;
            self.gain = [1.2 / sd[0], 1.2 / sd[1], 1.2 / sd[2]];
            self.ready = true;
        }
        self.m = axes;
    }

    /// Bytes for the bake pass's uniform.
    pub fn bytes(&self) -> Vec<u8> {
        let mut v: Vec<f32> = vec![];
        for axis in &self.m {
            v.extend_from_slice(axis);
        }
        // Sketches are stored in +-1 of the snorm range, so the mean and gain are converted.
        for k in 0..16 {
            v.push(self.mean[k] / SKETCH_RANGE);
        }
        v.extend_from_slice(&[
            self.gain[0] * SKETCH_RANGE,
            self.gain[1] * SKETCH_RANGE,
            self.gain[2] * SKETCH_RANGE,
            0.0,
        ]);
        bytemuck::cast_slice(&v).to_vec()
    }
}

/// Writes a sample of living cells and a map of who is where, for studying a run over time.
pub fn dump(world: &mut World, path: &str, want: usize) -> Result<(), String> {
    use std::io::Write;
    world.flush();
    let slots = world.cfg.slots as u64;
    let alive = world.alive().max(1) as u64;
    let genome_bytes = world.cfg.blocks() as u64 * 16;
    let need = (want as u64 * slots / alive).min(slots).min(80_000);
    const RUN: u64 = 8;
    let runs = need.div_ceil(RUN).max(1);
    let gap = (slots / runs).max(1);
    let len = RUN.min(slots);
    let starts: Vec<u64> = (0..runs).map(|k| (k * gap).min(slots.saturating_sub(len))).collect();
    let of = |stride: u64| -> Vec<(u64, u64)> { starts.iter().map(|s| (s * stride, len * stride)).collect() };
    let cell = world.read_ranges("cell", &of(CELL_BYTES));
    let stat = world.read_ranges("stat", &of(STAT_BYTES));
    let run_list: Vec<(u64, u64)> = starts.iter().map(|s| (*s, len)).collect();
    let genome = world.read_genomes(&run_list, genome_bytes);
    let size = world.cfg.size as usize;
    let occ = world.read_u32s("occ");
    let mut map = vec![0u8; size * size / 8 + 1];
    for (p, o) in occ.iter().enumerate() {
        if *o != 0 && *o != u32::MAX {
            map[p / 8] |= 1 << (p % 8);
        }
    }
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| e.to_string())?);
    let mut kept = vec![];
    for i in 0..starts.len() * len as usize {
        let c = &cell[i * CELL_BYTES as usize..(i + 1) * CELL_BYTES as usize];
        let e = f32::from_le_bytes(c[0..4].try_into().unwrap());
        let age = f32::from_le_bytes(c[8..12].try_into().unwrap());
        if e > 0.0 && age >= 50.0 {
            kept.push(i);
        }
    }
    let head = json!({"tick": world.tick, "size": size, "alive": alive, "count": kept.len(), "genome_bytes": genome_bytes, "hidden_rows": world.cfg.hidden_rows, "physics": world.cfg.physics, "layout": crate::config::layout_json()}).to_string();
    let w = |f: &mut std::io::BufWriter<std::fs::File>, b: &[u8]| f.write_all(b).map_err(|e| e.to_string());
    w(&mut f, &(head.len() as u64).to_le_bytes())?;
    w(&mut f, head.as_bytes())?;
    w(&mut f, &map)?;
    for i in kept {
        w(&mut f, &cell[i * CELL_BYTES as usize..(i + 1) * CELL_BYTES as usize])?;
        w(&mut f, &stat[i * STAT_BYTES as usize..(i + 1) * STAT_BYTES as usize])?;
        w(&mut f, &genome[i * genome_bytes as usize..(i + 1) * genome_bytes as usize])?;
    }
    Ok(())
}
