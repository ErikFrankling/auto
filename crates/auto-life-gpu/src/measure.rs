//! Measures of depth that the log and both analysers share. Plain arithmetic on what a run left
//! behind, so nothing here runs on the GPU and nothing feeds back into a world.

use serde_json::{Value, json};

use crate::config::{IN_BLOCKS, OUT_ROWS};

pub type Inputs = [[f32; 4]; IN_BLOCKS as usize];
pub type Outputs = [[f32; 4]; OUT_ROWS as usize];

/// Which of the rules that change what a brain can do are on in a world, from its settings.
#[derive(Clone, Copy, Default, Debug)]
pub struct Rules {
    pub armour: bool,
    pub clean: bool,
    pub mineral: bool,
    pub coat: bool,
    pub coat_sense: bool,
    pub birth: bool,
    pub gang_sense: bool,
    pub give: bool,
    pub span: f32,
}

impl Rules {
    pub fn from_physics(physics: &Value) -> Rules {
        let on = |k: &str| physics[k].as_bool().unwrap_or(false);
        Rules {
            armour: on("armour_on"),
            clean: on("waste_on") && on("waste_clean"),
            mineral: on("mineral_on"),
            coat: on("coat_on"),
            coat_sense: on("coat_on") && on("coat_sense"),
            birth: on("birth_choice"),
            gang_sense: on("gang_on") && on("gang_sense"),
            give: on("give_on"),
            span: physics["coat_span"].as_f64().unwrap_or(0.5) as f32,
        }
    }

    /// Whether the actions after the first ten exist: armour, clean, mineral uptake.
    pub fn actions(&self) -> [bool; 3] {
        [self.armour, self.clean, self.mineral]
    }
}

/// The situations added to the battery for rules that are on, so that a rule that's off leaves
/// the measure as it was. `make` builds the plain situation for (energy, food, front).
pub fn rule_situations(rules: &Rules, make: &dyn Fn(f32, f32, u8) -> Inputs) -> Vec<Inputs> {
    let mut out = vec![];
    let sat = |v: f32, s: f32| v / (s + v);
    for energy in [2.0f32, 10.0, 30.0] {
        if rules.mineral {
            for (ground, store) in [(0.0f32, 0.0f32), (2.0, 0.0), (0.0, 4.0), (2.0, 4.0)] {
                let mut i = make(energy, 1.0, 0);
                i[16] = [sat(ground, 1.0), 0.0, 0.0, sat(store, 1.0)];
                out.push(i);
            }
            let mut i = make(energy, 1.0, 0);
            i[16] = [0.1, 0.8, 0.0, 0.0];
            out.push(i);
        }
        if rules.coat_sense {
            for k in 0..4 {
                let a = k as f32 * std::f32::consts::FRAC_PI_2;
                let mut i = make(energy, 1.0, 1);
                i[15] = [a.cos(), a.sin(), 1.0, 0.0];
                out.push(i);
            }
        }
        if rules.gang_sense {
            for on_front in [0.0f32, 0.25, 0.75] {
                let mut i = make(energy, 1.0, 2);
                i[17] = [0.0, 0.0, on_front, on_front];
                out.push(i);
            }
        }
    }
    out
}

/// What a brain chooses beyond its budget in a situation, for rules that are on. A coat and a key
/// count as points on a circle, so 0.99 and 0.01 are close. Two cells that agree on everything else
/// and differ here are living differently.
pub fn rule_outputs(rules: &Rules, out: &Outputs) -> Vec<f32> {
    let mut v = vec![];
    if rules.coat {
        for o in [out[5][2], out[5][3]] {
            let a = o * rules.span * std::f32::consts::TAU;
            v.push(a.cos());
            v.push(a.sin());
        }
    }
    if rules.mineral && rules.give {
        v.push(if out[6][1] > 0.0 { 1.0 } else { 0.0 });
    }
    if rules.birth {
        v.push(if out[6][2] > 0.0 { 1.0 } else { 0.0 });
    }
    v
}

/// What a brain chooses besides its budget, once per cell rather than once per situation. Hundreds
/// of copies of a coat angle would outweigh everything it does, and split one way of living
/// into many just because its coats differ.
pub fn outputs_once(all: &[f32], situations: usize) -> Vec<f32> {
    if all.is_empty() || situations == 0 {
        return vec![];
    }
    let width = all.len() / situations;
    (0..width)
        .map(|k| all.iter().skip(k).step_by(width).sum::<f32>() / situations as f32)
        .collect()
}

/// Whether traffic runs along lines. Inside a small block, land and numbers are about the same, so
/// steps that fall evenly there give a variance about equal to their mean, whatever the density.
/// Steps piled onto a few squares of each block give more. Weighted by steps, and blocks with
/// little traffic are left out. Not proof of a trail: cells pacing the few gaps inside a colony
/// also score, so read it against control worlds and look at the map.
pub fn traffic_lines(steps: &[f32], open: &dyn Fn(usize) -> bool, size: usize, block: usize) -> f32 {
    let (mut sum, mut weight) = (0f64, 0f64);
    for by in (0..size).step_by(block) {
        for bx in (0..size).step_by(block) {
            let mut v = vec![];
            for y in by..(by + block).min(size) {
                for x in bx..(bx + block).min(size) {
                    let p = y * size + x;
                    if open(p) {
                        v.push(steps[p] as f64);
                    }
                }
            }
            if v.len() * 2 < block * block {
                continue;
            }
            let total: f64 = v.iter().sum();
            let mean = total / v.len() as f64;
            if mean < 2.0 {
                continue;
            }
            let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64;
            sum += var / mean * total;
            weight += total;
        }
    }
    if weight > 0.0 { (sum / weight) as f32 } else { 0.0 }
}

pub fn round3(v: f32) -> f32 {
    (v * 1000.0).round() / 1000.0
}

/// Spread of sizes on a log scale, and whether the large end looks scale-free. `alpha` is the
/// power law's slope fitted to the sizes (Clauset's estimate for sizes from `x_min` up), and `ks`
/// how far they sit from that law: under about 0.1 is a good fit. A narrow range of sizes can
/// fit anything, so `decades` says how much range the fit had.
pub fn size_stats(sizes: &[usize], x_min: f64) -> Value {
    let mut s: Vec<f64> = sizes
        .iter()
        .map(|v| *v as f64)
        .filter(|v| *v >= x_min)
        .collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if s.is_empty() {
        return json!({"count": 0, "log10_sd": 0.0, "alpha": 0.0, "ks": 0.0, "decades": 0.0});
    }
    let n = s.len() as f64;
    let logs: Vec<f64> = s.iter().map(|v| v.log10()).collect();
    let mean = logs.iter().sum::<f64>() / n;
    let sd = (logs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    let sum: f64 = s.iter().map(|v| (v / (x_min - 0.5)).ln()).sum();
    let alpha = 1.0 + n / sum.max(1e-9);
    let mut ks = 0.0f64;
    for (i, v) in s.iter().enumerate() {
        let model = 1.0 - (v / x_min).powf(1.0 - alpha);
        let below = i as f64 / n;
        let at = (i + 1) as f64 / n;
        ks = ks.max((model - below).abs()).max((model - at).abs());
    }
    json!({
        "count": s.len(),
        "log10_sd": round3(sd as f32),
        "alpha": round3(alpha as f32),
        "ks": round3(ks as f32),
        "decades": round3((s[s.len() - 1] / s[0]).log10() as f32),
        "max": s[s.len() - 1],
        "p50": s[s.len() / 2],
    })
}

/// Are strategies side by side, or only in different places? The chance that two cells from the
/// same tile differ in strategy, over the chance for two cells from anywhere. Counting pairs
/// without replacement means few cells per tile don't bias it. 1 is as mixed as the world,
/// 0 is one strategy per tile.
pub fn side_by_side(cells: &[(u32, u32, usize)], tile: u32, size: u32, k: usize) -> Value {
    let tiles = (size / tile).max(1) as usize;
    let mut counts = vec![vec![0u32; k]; tiles * tiles];
    let mut world = vec![0u32; k];
    for (x, y, s) in cells {
        let at = (*y / tile) as usize * tiles + (*x / tile) as usize;
        counts[at.min(tiles * tiles - 1)][*s] += 1;
        world[*s] += 1;
    }
    let differ = |c: &[u32]| -> Option<(f64, f64)> {
        let n: f64 = c.iter().map(|v| *v as f64).sum();
        if n < 2.0 {
            return None;
        }
        let same: f64 = c.iter().map(|v| *v as f64 * (*v as f64 - 1.0)).sum();
        Some((1.0 - same / (n * (n - 1.0)), n))
    };
    let Some((everywhere, _)) = differ(&world) else {
        return json!({"tile": tile, "overlap": null, "within": 0.0, "world": 0.0});
    };
    let (mut sum, mut weight) = (0.0f64, 0.0f64);
    for c in &counts {
        if let Some((d, n)) = differ(c) {
            sum += d * (n - 1.0);
            weight += n - 1.0;
        }
    }
    let within = sum / weight.max(1.0);
    json!({
        "tile": tile,
        "overlap": if everywhere > 0.0 { json!(round3((within / everywhere) as f32)) } else { Value::Null },
        "within": round3(within as f32),
        "world": round3(everywhere as f32),
    })
}

/// How many of a fine clustering's clusters really differ. Two clusters count as one if, along
/// the line between their centres, the gap is under `MERGE_GAP` of their own spreads. Any lump cut
/// in pieces gives neighbours about 2.7 to 3.5 apart in those terms, however stretched the lump is,
/// and groups with a real gap between them give more. So unlike picking the number of clusters
/// by silhouette, this doesn't lean to 2, and it doesn't lean to the number asked for either.
/// Returns each cluster's group.
pub fn merge_close(centres: &[Vec<f32>], points: &[Vec<f32>], label: &[usize]) -> Vec<usize> {
    let k = centres.len();
    let mut members = vec![vec![]; k];
    for (i, l) in label.iter().enumerate() {
        members[*l].push(i);
    }
    let mut group: Vec<usize> = (0..k).collect();
    fn find(g: &mut [usize], mut a: usize) -> usize {
        while g[a] != a {
            g[a] = g[g[a]];
            a = g[a];
        }
        a
    }
    for a in 0..k {
        for b in a + 1..k {
            if members[a].len() < 2 || members[b].len() < 2 {
                continue;
            }
            let line: Vec<f32> = centres[a].iter().zip(&centres[b]).map(|(x, y)| y - x).collect();
            let length = line.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
            let along = |i: usize| -> f32 { points[i].iter().zip(&line).map(|(p, l)| p * l).sum::<f32>() / length };
            let spread = |m: &[usize]| -> (f32, f32) {
                let n = m.len() as f32;
                let mean = m.iter().map(|i| along(*i)).sum::<f32>() / n;
                (mean, m.iter().map(|i| (along(*i) - mean).powi(2)).sum::<f32>() / n)
            };
            let (ma, va) = spread(&members[a]);
            let (mb, vb) = spread(&members[b]);
            let gap = (ma - mb).abs() / ((va + vb) / 2.0).sqrt().max(1e-9);
            if gap < MERGE_GAP {
                let (x, y) = (find(&mut group, a), find(&mut group, b));
                if x != y {
                    group[x] = y;
                }
            }
        }
    }
    // A lone outlier or an empty centre isn't a strategy, so it counts with its nearest neighbour.
    for a in 0..k {
        if members[a].len() >= 2 {
            continue;
        }
        let nearest = (0..k)
            .filter(|b| *b != a && members[*b].len() >= 2)
            .map(|b| {
                let d: f32 = centres[a].iter().zip(&centres[b]).map(|(x, y)| (x - y) * (x - y)).sum();
                (d, b)
            })
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        if let Some((_, b)) = nearest {
            let (x, y) = (find(&mut group, a), find(&mut group, b));
            if x != y {
                group[x] = y;
            }
        }
    }
    let mut names = std::collections::HashMap::new();
    (0..k)
        .map(|c| {
            let r = find(&mut group, c);
            let next = names.len();
            *names.entry(r).or_insert(next)
        })
        .collect()
}

const MERGE_GAP: f32 = 4.0;

pub fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// The longest run of log points a share stays at or above a level.
pub fn longest_run(shares: &[f32], level: f32) -> usize {
    let (mut best, mut now) = (0, 0);
    for v in shares {
        now = if *v >= level { now + 1 } else { 0 };
        best = best.max(now);
    }
    best
}

/// Colonies on a map of who's where: connected squares with most of their neighbours filled, which
/// leaves out scattered wanderers.
pub struct Colonies {
    pub label: Vec<u32>,
    /// Size, and centre on the torus, of each. Labels are one more than an index here.
    pub found: Vec<(usize, f32, f32)>,
}

const DIRS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];

pub fn colonies(occupied: &dyn Fn(usize) -> bool, size: usize, least: usize) -> Colonies {
    let n = size * size;
    let mask = size - 1;
    let solid = |p: usize| {
        if !occupied(p) {
            return false;
        }
        let (x, y) = (p & mask, p / size);
        let mut around = 0;
        for (dx, dy) in DIRS {
            let q = (((y as i32 + dy) as usize) & mask) * size + (((x as i32 + dx) as usize) & mask);
            if occupied(q) {
                around += 1;
            }
        }
        around >= 4
    };
    let mut label = vec![0u32; n];
    let mut seen = vec![false; n];
    let mut found = vec![];
    let mut stack = vec![];
    let tau = std::f32::consts::TAU / size as f32;
    for start in 0..n {
        if seen[start] || !solid(start) {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let mut members = vec![];
        let (mut sx, mut cx, mut sy, mut cy) = (0f32, 0f32, 0f32, 0f32);
        while let Some(q) = stack.pop() {
            members.push(q);
            let (x, y) = (q & mask, q / size);
            sx += (x as f32 * tau).sin();
            cx += (x as f32 * tau).cos();
            sy += (y as f32 * tau).sin();
            cy += (y as f32 * tau).cos();
            for (dx, dy) in DIRS {
                let r = (((y as i32 + dy) as usize) & mask) * size + (((x as i32 + dx) as usize) & mask);
                if !seen[r] && solid(r) {
                    seen[r] = true;
                    stack.push(r);
                }
            }
        }
        if members.len() >= least {
            found.push((
                members.len(),
                (sx.atan2(cx) / tau).rem_euclid(size as f32),
                (sy.atan2(cy) / tau).rem_euclid(size as f32),
            ));
            for q in members {
                label[q] = found.len() as u32;
            }
        }
    }
    Colonies { label, found }
}

/// Follows each colony from one map to the next by which it overlaps most. `moved` is how far
/// its centre went, as a share of its own radius. A colony that stays put but grows moves little,
/// and one that walks off moves more than 1.
pub struct Travel {
    pub tracked: usize,
    pub lost: usize,
    pub moved: Vec<f32>,
    pub squares: Vec<f32>,
    pub kept: Vec<f32>,
}

pub fn colony_travel(a: &Colonies, b: &Colonies, size: usize) -> Travel {
    let mut t = Travel {
        tracked: 0,
        lost: 0,
        moved: vec![],
        squares: vec![],
        kept: vec![],
    };
    let mut shared: std::collections::HashMap<(u32, u32), u32> = Default::default();
    for (x, y) in a.label.iter().zip(&b.label) {
        if *x != 0 && *y != 0 {
            *shared.entry((*x, *y)).or_default() += 1;
        }
    }
    let mut best = vec![(0u32, 0u32); a.found.len()];
    for ((x, y), n) in &shared {
        let slot = &mut best[*x as usize - 1];
        if *n > slot.0 || (*n == slot.0 && *y < slot.1) {
            *slot = (*n, *y);
        }
    }
    let s = size as f32;
    for (i, (n, x, y)) in a.found.iter().enumerate() {
        let (overlap, to) = best[i];
        if overlap == 0 {
            t.lost += 1;
            continue;
        }
        let (_, u, v) = b.found[to as usize - 1];
        let wrap = |d: f32| {
            let d = d.abs().rem_euclid(s);
            d.min(s - d)
        };
        let d = (wrap(x - u).powi(2) + wrap(y - v).powi(2)).sqrt();
        t.tracked += 1;
        t.squares.push(d);
        t.moved.push(d / (*n as f32 / std::f32::consts::PI).sqrt());
        t.kept.push(overlap as f32 / *n as f32);
    }
    t
}

pub fn percentile(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f32 * p) as usize]
}

/// The share of steps that would land on the busiest `top` share of squares if each step picked
/// a square at random, so that trails are told apart from plain noise in few steps.
pub fn chance_top_share(steps_per_square: f64, top: f64) -> f64 {
    let lambda = steps_per_square;
    if lambda <= 0.0 {
        return 0.0;
    }
    if lambda > 200.0 {
        // A normal curve is close enough, and its top 5% lie 2.06 sd above the mean on average.
        return top * (1.0 + 2.063 * (1.0 / lambda).sqrt());
    }
    let mut p = (-lambda).exp();
    let mut pmfs = vec![p];
    let limit = (lambda + 12.0 * lambda.sqrt() + 30.0) as usize;
    for k in 1..=limit {
        p *= lambda / k as f64;
        pmfs.push(p);
    }
    let (mut left, mut got) = (top, 0.0);
    for (k, q) in pmfs.iter().enumerate().rev() {
        let take = left.min(*q);
        got += take * k as f64;
        left -= take;
        if left <= 0.0 {
            break;
        }
    }
    got / lambda
}
