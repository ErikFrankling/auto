//! Finding and following bodies, from dumps of cells, on the CPU. Each measure here has a test on
//! a built case with a known answer, in tests/measures.rs. See "Measures" in DESIGN.md.
//!
//! A body is a connected group under contact with mutual adhesion. Nothing about a body exists in
//! the simulation, so everything here is worked out from where cells are and what they chose.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::dump::{Dump, DumpCell};

/// The physics a body depends on, read from a dump.
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    pub size: f32,
    pub gap: f32,
    pub glue_sharp: f32,
    pub stick_max: f32,
    /// Pull below this doesn't count as being stuck. About a tenth of what a cell at its usual
    /// budget moves itself with, so a passing brush isn't called a bond.
    pub min_pull: f32,
}

impl Rules {
    pub fn from_physics(physics: &Value, size: u32) -> Rules {
        let f = |k: &str, d: f64| physics[k].as_f64().unwrap_or(d) as f32;
        Rules {
            size: size as f32,
            gap: f(k_gap(), 0.2),
            glue_sharp: f("glue_sharp", 4.0),
            stick_max: f("stick_max", 0.4),
            min_pull: 0.01,
        }
    }
}

fn k_gap() -> &'static str {
    "gap"
}

/// The shortest way from one point to another round the torus.
pub fn wrapped(a: f32, b: f32, size: f32) -> f32 {
    let d = b - a;
    d - (d / size).round() * size
}

/// The pull between two cells where they are now. The same rule as contact.wgsl.
pub fn pull(a: &DumpCell, b: &DumpCell, rules: &Rules) -> f32 {
    let dx = wrapped(a.x, b.x, rules.size);
    let dy = wrapped(a.y, b.y, rules.size);
    let d = (dx * dx + dy * dy).sqrt();
    let rr = a.r + b.r;
    let g0 = a.glue[0] - b.glue[0];
    let g1 = a.glue[1] - b.glue[1];
    let m = 1.0 / (1.0 + rules.glue_sharp * (g0 * g0 + g1 * g1));
    a.stick * b.stick * m * rules.stick_max * (1.0 - (d - rr) / rules.gap).clamp(0.0, 1.0)
}

fn touching(a: &DumpCell, b: &DumpCell, rules: &Rules) -> bool {
    let dx = wrapped(a.x, b.x, rules.size);
    let dy = wrapped(a.y, b.y, rules.size);
    let rr = a.r + b.r;
    dx * dx + dy * dy < rr * rr
}

/// What joins two cells into one group.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Link {
    /// Adhesion both ways, strong enough to count.
    Stuck,
    /// Only touching, as a control: what a crowd looks like, with or without any adhesion.
    Touching,
}

/// Which group each cell is in. Numbers run from 0 with no gaps, and a lone cell is its own group.
pub fn groups(cells: &[DumpCell], rules: &Rules, link: Link) -> Vec<u32> {
    let n = cells.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    // A grid two squares wide holds anything within reach, as the bins in the engine do.
    let side = ((rules.size / 2.0).floor() as i64).max(1);
    let cell_of = |c: &DumpCell| {
        let at = |v: f32| ((v / 2.0).floor() as i64).rem_euclid(side);
        (at(c.x), at(c.y))
    };
    let mut grid: HashMap<(i64, i64), Vec<u32>> = HashMap::new();
    for (i, c) in cells.iter().enumerate() {
        grid.entry(cell_of(c)).or_default().push(i as u32);
    }
    for (i, c) in cells.iter().enumerate() {
        let (gx, gy) = cell_of(c);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let key = ((gx + dx).rem_euclid(side), (gy + dy).rem_euclid(side));
                let Some(list) = grid.get(&key) else { continue };
                for &j in list {
                    if (j as usize) <= i {
                        continue;
                    }
                    let other = &cells[j as usize];
                    let joined = match link {
                        Link::Stuck => pull(c, other, rules) >= rules.min_pull,
                        Link::Touching => touching(c, other, rules),
                    };
                    if joined {
                        let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j));
                        parent[a as usize] = b;
                    }
                }
            }
        }
    }
    let mut names: HashMap<u32, u32> = HashMap::new();
    (0..n as u32)
        .map(|i| {
            let root = find(&mut parent, i);
            let next = names.len() as u32;
            *names.entry(root).or_insert(next)
        })
        .collect()
}

/// How many cells each group has.
pub fn group_sizes(labels: &[u32]) -> Vec<u32> {
    let count = labels.iter().max().map_or(0, |m| *m as usize + 1);
    let mut sizes = vec![0u32; count];
    for l in labels {
        sizes[*l as usize] += 1;
    }
    sizes
}

/// How body sizes are spread out.
#[derive(Clone, Debug, Default)]
pub struct SizeStats {
    pub cells: usize,
    pub bodies: usize,
    pub largest: u32,
    /// The share of cells in a body of at least 2, 3 and 10.
    pub in_bodies: f64,
    pub in_3_plus: f64,
    pub in_10_plus: f64,
    /// The size of the body the average cell is in, with a lone cell counting as 1.
    pub mean_per_cell: f64,
    /// How many bodies there are of size 2-3, 4-7, 8-15, and so on, doubling.
    pub histogram: Vec<u32>,
}

pub fn size_stats(labels: &[u32]) -> SizeStats {
    let sizes = group_sizes(labels);
    let cells = labels.len();
    let share = |least: u32| {
        sizes.iter().filter(|s| **s >= least).map(|s| *s as u64).sum::<u64>() as f64 / cells.max(1) as f64
    };
    let mut histogram = vec![];
    for s in sizes.iter().filter(|s| **s >= 2) {
        let bin = (31 - s.leading_zeros()) as usize - 1;
        if histogram.len() <= bin {
            histogram.resize(bin + 1, 0);
        }
        histogram[bin] += 1;
    }
    SizeStats {
        cells,
        bodies: sizes.iter().filter(|s| **s >= 2).count(),
        largest: sizes.iter().copied().max().unwrap_or(0),
        in_bodies: share(2),
        in_3_plus: share(3),
        in_10_plus: share(10),
        mean_per_cell: sizes.iter().map(|s| (*s as u64 * *s as u64) as f64).sum::<f64>() / cells.max(1) as f64,
        histogram,
    }
}

/// A dump of cells, grouped into bodies.
pub struct Frame {
    pub tick: u64,
    pub cells: Vec<DumpCell>,
    pub labels: Vec<u32>,
}

impl Frame {
    pub fn new(tick: u64, cells: Vec<DumpCell>, rules: &Rules) -> Frame {
        let labels = groups(&cells, rules, Link::Stuck);
        Frame { tick, cells, labels }
    }
}

/// A small fixed random number source, so a shuffle comes out the same each time.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

// ---- Following bodies across dumps ----

/// One body's life, from the first dump it's seen in to the last.
#[derive(Clone, Debug)]
pub struct Tracked {
    pub id: usize,
    /// The body it broke off from, if it did.
    pub parent: Option<usize>,
    /// How many times over its line of ancestors has broken off, starting at 0.
    pub generation: u32,
    pub first_tick: u64,
    pub last_tick: u64,
    /// Its size at each dump it's seen in.
    pub sizes: Vec<(u64, u32)>,
    /// Its size at the last dump before it first shed a fragment, if it ever did.
    pub size_at_fragmenting: Option<u32>,
}

/// Follows bodies of 2 or more across dumps, by who's in them. A cell that leaves a body remembers
/// it, and so do its children, so a lone cell that leaves and grows into a new body counts as a
/// fragment of the body it left. When a body splits, the piece with the most of its old members
/// carries on and the others are new bodies.
pub fn track(frames: &[Frame]) -> Vec<Tracked> {
    let mut bodies: Vec<Tracked> = vec![];
    // The body each cell last belonged to, by uid.
    let mut last_body: HashMap<u32, usize> = HashMap::new();
    for frame in frames {
        let sizes = group_sizes(&frame.labels);
        let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, l) in frame.labels.iter().enumerate() {
            if sizes[*l as usize] >= 2 {
                members.entry(*l).or_default().push(i);
            }
        }
        // Who each body in this dump comes from.
        let mut votes: Vec<(u32, HashMap<usize, u32>)> = vec![];
        let mut labels: Vec<u32> = members.keys().copied().collect();
        labels.sort_unstable();
        for l in &labels {
            let mut v: HashMap<usize, u32> = HashMap::new();
            for &i in &members[l] {
                let c = &frame.cells[i];
                let from = last_body.get(&c.uid).or_else(|| last_body.get(&c.parent));
                if let Some(b) = from {
                    *v.entry(*b).or_default() += 1;
                }
            }
            votes.push((*l, v));
        }
        // A body's best predecessor, and for each predecessor the piece that carries it on.
        let best_of = |v: &HashMap<usize, u32>| {
            v.iter().max_by_key(|(id, n)| (**n, std::cmp::Reverse(**id))).map(|(id, n)| (*id, *n))
        };
        let mut heir: HashMap<usize, (u32, u32)> = HashMap::new();
        for (l, v) in &votes {
            if let Some((p, n)) = best_of(v) {
                let e = heir.entry(p).or_insert((*l, n));
                if n > e.1 || (n == e.1 && *l < e.0) {
                    *e = (*l, n);
                }
            }
        }
        let mut names: HashMap<u32, usize> = HashMap::new();
        for (l, v) in &votes {
            let size = members[l].len() as u32;
            let id = match best_of(v) {
                Some((p, _)) if heir[&p].0 == *l => {
                    bodies[p].last_tick = frame.tick;
                    bodies[p].sizes.push((frame.tick, size));
                    p
                }
                other => {
                    // A new body, either from nothing or broken off another.
                    let parent = other.map(|(p, _)| p);
                    let generation = parent.map_or(0, |p| bodies[p].generation + 1);
                    if let Some(p) = parent
                        && bodies[p].size_at_fragmenting.is_none()
                    {
                        // Its size before this dump.
                        let before = bodies[p].sizes.iter().rev().find(|(t, _)| *t < frame.tick);
                        bodies[p].size_at_fragmenting = before.map(|b| b.1);
                    }
                    let id = bodies.len();
                    bodies.push(Tracked {
                        id,
                        parent,
                        generation,
                        first_tick: frame.tick,
                        last_tick: frame.tick,
                        sizes: vec![(frame.tick, size)],
                        size_at_fragmenting: None,
                    });
                    id
                }
            };
            names.insert(*l, id);
        }
        for (l, list) in &members {
            for &i in list {
                last_body.insert(frame.cells[i].uid, names[l]);
            }
        }
    }
    bodies
}

/// Parent to offspring slope of size at fragmenting, and how many pairs it's from.
pub fn size_heritability(bodies: &[Tracked]) -> (Option<f64>, usize) {
    let mut pairs = vec![];
    for b in bodies {
        if let (Some(p), Some(own)) = (b.parent, b.size_at_fragmenting)
            && let Some(theirs) = bodies[p].size_at_fragmenting
        {
            pairs.push((theirs as f64, own as f64));
        }
    }
    if pairs.len() < 3 {
        return (None, pairs.len());
    }
    let n = pairs.len() as f64;
    let mx = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let my = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let sxy: f64 = pairs.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    if sxx <= 0.0 {
        return (None, pairs.len());
    }
    (Some(sxy / sxx), pairs.len())
}

/// Of pairs of cells in one body, the share still in one body after each number of dumps, and the
/// time until half of them have parted. A pair only counts while both cells live, so this is about
/// staying together, not surviving.
#[derive(Clone, Debug, Default)]
pub struct Membership {
    /// Ticks later, share still together, pairs looked at.
    pub curve: Vec<(u64, f64, u64)>,
    pub half_life: Option<f64>,
}

pub fn membership(frames: &[Frame]) -> Membership {
    let mut rng = Rng(0x5eed);
    let mut kept: Vec<(u64, u64, u64)> = vec![];
    for (t, frame) in frames.iter().enumerate() {
        let sizes = group_sizes(&frame.labels);
        let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
        for (i, l) in frame.labels.iter().enumerate() {
            if sizes[*l as usize] >= 2 {
                members.entry(*l).or_default().push(i);
            }
        }
        let mut labels: Vec<u32> = members.keys().copied().collect();
        labels.sort_unstable();
        let mut pairs: Vec<(u32, u32)> = vec![];
        for l in labels {
            let list = &members[&l];
            for _ in 0..16.min(list.len() * (list.len() - 1) / 2) {
                let a = list[rng.below(list.len())];
                let b = list[rng.below(list.len())];
                if a != b {
                    pairs.push((frame.cells[a].uid, frame.cells[b].uid));
                }
            }
        }
        for (lag, later) in frames[t + 1..].iter().enumerate() {
            let label_of: HashMap<u32, u32> = later
                .cells
                .iter()
                .zip(&later.labels)
                .map(|(c, l)| (c.uid, *l))
                .collect();
            let (mut same, mut both) = (0u64, 0u64);
            for (a, b) in &pairs {
                if let (Some(la), Some(lb)) = (label_of.get(a), label_of.get(b)) {
                    both += 1;
                    same += (la == lb) as u64;
                }
            }
            if kept.len() <= lag {
                kept.push((later.tick - frame.tick, 0, 0));
            }
            kept[lag].1 += same;
            kept[lag].2 += both;
        }
    }
    let curve: Vec<(u64, f64, u64)> = kept
        .iter()
        .map(|(ticks, same, both)| (*ticks, *same as f64 / (*both).max(1) as f64, *both))
        .filter(|p| p.2 > 0)
        .collect();
    // Starting from everyone together at no time at all.
    let mut half_life = None;
    let mut before = (0.0f64, 1.0f64);
    for (ticks, share, _) in &curve {
        if *share <= 0.5 {
            let (t0, s0) = before;
            let k = if s0 > *share { (s0 - 0.5) / (s0 - share) } else { 0.0 };
            half_life = Some(t0 + (*ticks as f64 - t0) * k);
            break;
        }
        before = (*ticks as f64, *share);
    }
    Membership { curve, half_life }
}

// ---- Relatedness ----

/// How far apart two genomes are, as root mean square difference across the sketch.
pub fn genome_gap(a: &DumpCell, b: &DumpCell) -> f32 {
    let (x, y) = (a.sketch_f(), b.sketch_f());
    let mut sum = 0.0;
    for k in 0..16 {
        sum += (x[k] - y[k]) * (x[k] - y[k]);
    }
    (sum / 16.0).sqrt()
}

/// Mean genome gap within bodies, between bodies anywhere, and between nearby cells in different
/// bodies. The last is the fair comparison, since relatives also tend to live near each other.
#[derive(Clone, Copy, Debug, Default)]
pub struct Relatedness {
    pub within: Option<f64>,
    pub between: Option<f64>,
    pub between_nearby: Option<f64>,
}

pub fn relatedness(frame: &Frame, rules: &Rules) -> Relatedness {
    let mut rng = Rng(0xabcd);
    let cells = &frame.cells;
    let sizes = group_sizes(&frame.labels);
    let mean = |sum: f64, n: u64| (n > 0).then(|| sum / n as f64);
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, l) in frame.labels.iter().enumerate() {
        if sizes[*l as usize] >= 2 {
            members.entry(*l).or_default().push(i);
        }
    }
    let mut labels: Vec<u32> = members.keys().copied().collect();
    labels.sort_unstable();
    let (mut within, mut wn) = (0.0f64, 0u64);
    for l in &labels {
        let list = &members[l];
        for _ in 0..64 {
            let a = list[rng.below(list.len())];
            let b = list[rng.below(list.len())];
            if a != b {
                within += genome_gap(&cells[a], &cells[b]) as f64;
                wn += 1;
            }
        }
    }
    let (mut far, mut fnn) = (0.0f64, 0u64);
    let (mut near, mut nn) = (0.0f64, 0u64);
    if cells.len() >= 2 && !labels.is_empty() {
        let in_body: Vec<usize> = labels.iter().flat_map(|l| members[l].iter().copied()).collect();
        for _ in 0..20_000 {
            let a = in_body[rng.below(in_body.len())];
            let b = rng.below(cells.len());
            if frame.labels[a] == frame.labels[b] {
                continue;
            }
            let gap = genome_gap(&cells[a], &cells[b]) as f64;
            far += gap;
            fnn += 1;
            let dx = wrapped(cells[a].x, cells[b].x, rules.size);
            let dy = wrapped(cells[a].y, cells[b].y, rules.size);
            if dx * dx + dy * dy < 64.0 {
                near += gap;
                nn += 1;
            }
        }
    }
    Relatedness {
        within: mean(within, wn),
        between: mean(far, fnn),
        between_nearby: mean(near, nn),
    }
}

// ---- Do cells differ by where they are in a body? ----

/// How much of the budget differences among clonemates in a body is explained by how many contacts
/// each has, which is how surface differs from inside. Compared with the same cells' contact
/// counts shuffled within their bodies.
#[derive(Clone, Debug, Default)]
pub struct Position {
    pub bodies: usize,
    pub cells: usize,
    /// Mean variance of a budget share within a body, over cells and actions.
    pub variance_within: f64,
    /// Share of that variance explained by contact count, pooled across actions.
    pub r2: f64,
    /// The same with contacts shuffled, as the mean and the highest of the shuffles.
    pub r2_shuffled: f64,
    pub r2_shuffled_max: f64,
    /// By action, mean share for cells with few contacts, then with many.
    pub surface: [f64; 12],
    pub inside: [f64; 12],
}

/// Pooled R squared of each budget share, within groups, against a value per cell.
fn r2_within(groups_of: &[Vec<(f32, [f32; 12])>]) -> (f64, f64) {
    let (mut sxy2, mut syy) = (0.0f64, 0.0f64);
    for action in 0..12 {
        let (mut sxy, mut sxx) = (0.0f64, 0.0f64);
        for group in groups_of {
            let n = group.len() as f64;
            let mx = group.iter().map(|g| g.0 as f64).sum::<f64>() / n;
            let my = group.iter().map(|g| g.1[action] as f64).sum::<f64>() / n;
            for (x, y) in group {
                let (dx, dy) = (*x as f64 - mx, y[action] as f64 - my);
                sxy += dx * dy;
                sxx += dx * dx;
                syy += dy * dy;
            }
        }
        if sxx > 0.0 {
            sxy2 += sxy * sxy / sxx;
        }
    }
    (sxy2 / syy.max(1e-30), syy)
}

/// `clone_gap` is how close two genomes must be to count as clonemates, and bodies need `least`
/// such cells to be looked at.
pub fn position_effect(frame: &Frame, clone_gap: f32, least: usize) -> Position {
    let cells = &frame.cells;
    let sizes = group_sizes(&frame.labels);
    let mut members: HashMap<u32, Vec<usize>> = HashMap::new();
    for (i, l) in frame.labels.iter().enumerate() {
        if sizes[*l as usize] as usize >= least {
            members.entry(*l).or_default().push(i);
        }
    }
    let mut labels: Vec<u32> = members.keys().copied().collect();
    labels.sort_unstable();
    let mut data: Vec<Vec<(f32, [f32; 12])>> = vec![];
    for l in labels {
        let list = &members[&l];
        // The clone is whoever's near the member most others are near.
        let every = (list.len() / 32).max(1);
        let centre = *list
            .iter()
            .step_by(every)
            .max_by_key(|a| {
                list.iter()
                    .step_by(every)
                    .filter(|b| genome_gap(&cells[**a], &cells[**b]) < clone_gap)
                    .count()
            })
            .unwrap();
        let clone: Vec<(f32, [f32; 12])> = list
            .iter()
            .filter(|i| genome_gap(&cells[**i], &cells[centre]) < clone_gap)
            .map(|i| (cells[*i].contacts as f32, cells[*i].budget))
            .collect();
        if clone.len() >= least {
            data.push(clone);
        }
    }
    let mut out = Position {
        bodies: data.len(),
        cells: data.iter().map(|d| d.len()).sum(),
        ..Default::default()
    };
    if data.is_empty() {
        return out;
    }
    let (r2, syy) = r2_within(&data);
    out.r2 = r2;
    out.variance_within = syy / (out.cells.max(1) as f64 * 12.0);
    // Few and many contacts, split at each body's own mean, since many cells can share a count.
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    for group in &data {
        let mid = group.iter().map(|g| g.0).sum::<f32>() / group.len() as f32;
        for (x, y) in group {
            let row = if *x < mid { &mut out.surface } else { &mut out.inside };
            for k in 0..12 {
                row[k] += y[k] as f64;
            }
            if *x < mid {
                lo += 1.0;
            } else {
                hi += 1.0;
            }
        }
    }
    for k in 0..12 {
        out.surface[k] /= f64::max(lo, 1.0);
        out.inside[k] /= f64::max(hi, 1.0);
    }
    let mut rng = Rng(0xfeed);
    let shuffles = 32;
    let mut total = 0.0;
    for _ in 0..shuffles {
        let mut mixed = data.clone();
        for group in &mut mixed {
            for i in (1..group.len()).rev() {
                let j = rng.below(i + 1);
                let (a, b) = (group[i].0, group[j].0);
                group[i].0 = b;
                group[j].0 = a;
            }
        }
        let (r, _) = r2_within(&mixed);
        total += r;
        out.r2_shuffled_max = out.r2_shuffled_max.max(r);
    }
    out.r2_shuffled = total / shuffles as f64;
    out
}

// ---- A world's summary ----

fn opt(v: Option<f64>) -> Value {
    v.map_or(Value::Null, |x| json!(x))
}

/// Everything measured about one world from its dumps, in time order.
pub fn summary(frames: &[Frame], rules: &Rules) -> Value {
    let Some(last) = frames.last() else {
        return json!({"dumps": 0});
    };
    let stats = size_stats(&last.labels);
    let crowd = size_stats(&groups(&last.cells, rules, Link::Touching));
    let bodies = track(frames);
    let keep = membership(frames);
    let related = relatedness(last, rules);
    let position = position_effect(last, 0.4, 7);
    let (slope, pairs) = size_heritability(&bodies);
    let deepest = bodies.iter().map(|b| b.generation).max().unwrap_or(0);
    let over_time: Vec<Value> = frames
        .iter()
        .map(|f| {
            let s = size_stats(&f.labels);
            json!({"tick": f.tick, "cells": s.cells, "bodies": s.bodies, "largest": s.largest,
                   "in_bodies": s.in_bodies, "in_10_plus": s.in_10_plus, "mean_per_cell": s.mean_per_cell})
        })
        .collect();
    json!({
        "dumps": frames.len(),
        "tick": last.tick,
        "cells": stats.cells,
        "bodies": {
            "count": stats.bodies,
            "largest": stats.largest,
            "share_of_cells_in_bodies": stats.in_bodies,
            "share_in_3_plus": stats.in_3_plus,
            "share_in_10_plus": stats.in_10_plus,
            "mean_size_per_cell": stats.mean_per_cell,
            "histogram_from_2_doubling": stats.histogram,
        },
        "crowds_by_touching_only": {
            "count": crowd.bodies,
            "largest": crowd.largest,
            "share_of_cells": crowd.in_bodies,
            "mean_size_per_cell": crowd.mean_per_cell,
        },
        "over_time": over_time,
        "tracked_bodies": bodies.len(),
        "deepest_body_generation": deepest,
        "membership": {
            "half_life_ticks": opt(keep.half_life),
            "curve": keep.curve.iter().map(|(t, s, n)| json!({"ticks": t, "together": s, "pairs": n})).collect::<Vec<_>>(),
        },
        "relatedness": {
            "gap_within_bodies": opt(related.within),
            "gap_between_bodies": opt(related.between),
            "gap_between_nearby_bodies": opt(related.between_nearby),
        },
        "size_at_fragmenting": {"parent_to_offspring_slope": opt(slope), "pairs": pairs},
        "position": {
            "bodies": position.bodies,
            "cells": position.cells,
            "budget_variance_within_bodies": position.variance_within,
            "r2_by_contact_count": position.r2,
            "r2_shuffled_mean": position.r2_shuffled,
            "r2_shuffled_max": position.r2_shuffled_max,
            "surface_budget": position.surface,
            "inside_budget": position.inside,
        },
    })
}

/// Cuts a dump into the walled worlds of a batch, as (tile number, cells), with positions local.
pub fn split_batch(dump: &Dump, tile: u32) -> Vec<(u32, Vec<DumpCell>)> {
    let per_side = dump.size / tile;
    let mut out: Vec<(u32, Vec<DumpCell>)> = (0..per_side * per_side).map(|i| (i, vec![])).collect();
    for c in &dump.cells {
        let (tx, ty) = (c.x as u32 / tile, c.y as u32 / tile);
        let i = (ty.min(per_side - 1) * per_side + tx.min(per_side - 1)) as usize;
        let mut local = *c;
        local.x = c.x - (tx * tile) as f32;
        local.y = c.y - (ty * tile) as f32;
        out[i].1.push(local);
    }
    out
}
