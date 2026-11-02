//! Follows a run through time from its dumps: which strategies and species exist, when they
//! appear and vanish, and whether anything keeps changing. Runs on the CPU, with no GPU.
//!   life-history NAME DUMP_DIR OUT.json
//!
//! Strategies come from what brains do in a fixed set of situations, so two genomes that act
//! alike count as one. Species come from genomes, so the two can be compared. Labels are
//! plain descriptions of the numbers, for reading. Cells never see them.

use serde_json::{Value, json};
use std::io::Read;

const IN_BLOCKS: usize = 13;
const OUT_ROWS: usize = 6;
const ACTIONS: [&str; 10] = ["rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme"];

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
}

struct Cell {
    diet: [f32; 4],
    budget: [f32; 10],
    moving: f32,
    crowd: f32,
    age: f32,
    energy: f32,
    generation: f32,
    clade: u32,
    genome: Vec<f32>,
    print: Vec<f32>,
}

struct Dump {
    tick: u64,
    alive: u64,
    map: Vec<u8>,
    cells: Vec<Cell>,
}

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

type Inputs = [[f32; 4]; IN_BLOCKS];

fn act(g: &[f32], hr: usize, input: &Inputs) -> ([f32; 10], f32, f32) {
    let hh = hr * IN_BLOCKS;
    let ho = hh + hr * hr;
    let mut mem = vec![[0.0f32; 4]; hr];
    let mut out = [[0.0f32; 4]; OUT_ROWS];
    for _ in 0..3 {
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
        out = [[0.0f32; 4]; OUT_ROWS];
        for (o, row) in out.iter_mut().enumerate() {
            for (j, h) in next.iter().enumerate() {
                let t = bt(g, ho + o * hr + j, h);
                for i in 0..4 {
                    row[i] += t[i];
                }
            }
        }
    }
    let raw = [out[0][0], out[0][1], out[0][2], out[0][3], out[1][0], out[1][1], out[1][2], out[1][3], out[2][0], out[2][1], out[2][2], out[2][3]];
    let top = raw.iter().cloned().fold(f32::MIN, f32::max);
    let mut s = [0.0f32; 10];
    let mut z = 0.0;
    for i in 0..10 {
        s[i] = (raw[i] - top).clamp(-20.0, 0.0).exp();
        z += s[i];
    }
    s.iter_mut().for_each(|v| *v /= z);
    (s, out[3][0], out[3][1])
}

fn sat(v: f32, s: f32) -> f32 {
    v / (s + v)
}

fn situation(energy: f32, food: f32, east: bool, front: u8, hit: bool, crowd: f32) -> Inputs {
    let mut i: Inputs = [[0.0; 4]; IN_BLOCKS];
    i[0] = [sat(energy, 10.0), 0.3, sat(300.0, 1000.0), if hit { sat(1.0, 0.5) } else { 0.0 }];
    i[1] = [0.0, 1.0, crowd, 0.0];
    i[2] = [sat(food, 2.0), 0.0, 0.0, 0.0];
    if east {
        i[4][0] = 0.8;
    }
    let me = sat(energy, 10.0);
    match front {
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

fn battery() -> Vec<Inputs> {
    let mut v = vec![];
    for e in [2.0f32, 10.0, 30.0] {
        for (f, east) in [(0.0f32, false), (2.0, false), (0.1, true)] {
            for front in 0..4u8 {
                for hit in [false, true] {
                    for crowd in [0.0f32, 0.75] {
                        v.push(situation(e, f, east, front, hit, crowd));
                    }
                }
            }
        }
    }
    v
}

fn read_dump(path: &std::path::Path, tests: &[Inputs]) -> Option<(Dump, usize)> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut head).ok()?;
    let head: Value = serde_json::from_slice(&head).ok()?;
    let size = head["size"].as_u64()? as usize;
    let count = head["count"].as_u64()? as usize;
    let gb = head["genome_bytes"].as_u64()? as usize;
    let hr = head["hidden_rows"].as_u64()? as usize;
    let mut map = vec![0u8; size * size / 8 + 1];
    f.read_exact(&mut map).ok()?;
    let mut cells = Vec::with_capacity(count);
    let mut rec = vec![0u8; 64 + 112 + gb];
    for _ in 0..count {
        f.read_exact(&mut rec).ok()?;
        let u: &[u32] = &bytemuck::pod_collect_to_vec::<u8, u32>(&rec[..64]);
        let st: Vec<f32> = bytemuck::pod_collect_to_vec::<u8, f32>(&rec[64..176]);
        let genome: Vec<f32> = rec[176..].iter().map(|b| (*b as i8 as f32 / 127.0).max(-1.0)).collect();
        let mut print = Vec::with_capacity(tests.len() * 10);
        for t in tests {
            print.extend_from_slice(&act(&genome, hr, t).0);
        }
        cells.push(Cell {
            diet: st[0..4].try_into().unwrap(),
            budget: st[4..14].try_into().unwrap(),
            moving: st[19],
            crowd: st[24] * 8.0,
            age: f32::from_bits(u[2]),
            energy: f32::from_bits(u[0]),
            generation: u[13] as f32,
            clade: u[12],
            genome,
            print,
        });
    }
    Some((Dump { tick: head["tick"].as_u64()?, alive: head["alive"].as_u64()?, map, cells }, hr))
}

fn dist2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

fn kmeans(points: &[Vec<f32>], k: usize, rng: &mut Rng) -> (Vec<usize>, Vec<Vec<f32>>) {
    let n = points.len();
    let mut best: (Vec<usize>, Vec<Vec<f32>>, f32) = (vec![0; n], vec![], f32::MAX);
    for _ in 0..3 {
        let mut centres: Vec<Vec<f32>> = vec![points[rng.below(n)].clone()];
        while centres.len() < k {
            let d: Vec<f32> = points.iter().map(|p| centres.iter().map(|c| dist2(p, c)).fold(f32::MAX, f32::min)).collect();
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
        for _ in 0..20 {
            for i in 0..n {
                let (mut b, mut bd) = (0, f32::MAX);
                for (c, centre) in centres.iter().enumerate() {
                    let v = dist2(&points[i], centre);
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
        let inertia: f32 = (0..n).map(|i| dist2(&points[i], &centres[label[i]])).sum();
        if inertia < best.2 {
            best = (label, centres, inertia);
        }
    }
    (best.0, best.1)
}

fn silhouette(points: &[Vec<f32>], label: &[usize], k: usize) -> f32 {
    let step = (points.len() / 400).max(1);
    let idx: Vec<usize> = (0..points.len()).step_by(step).collect();
    let (mut total, mut counted) = (0.0f32, 0.0f32);
    for &i in &idx {
        let mut sum = vec![0.0f32; k];
        let mut count = vec![0.0f32; k];
        for &j in &idx {
            if i != j {
                sum[label[j]] += dist2(&points[i], &points[j]).sqrt();
                count[label[j]] += 1.0;
            }
        }
        if count[label[i]] == 0.0 {
            continue;
        }
        let a = sum[label[i]] / count[label[i]];
        let b = (0..k).filter(|c| *c != label[i] && count[*c] > 0.0).map(|c| sum[c] / count[c]).fold(f32::MAX, f32::min);
        if b < f32::MAX {
            total += (b - a) / a.max(b).max(1e-6);
            counted += 1.0;
        }
    }
    total / counted.max(1.0)
}

fn top_axes(data: &[&[f32]], count: usize) -> (Vec<Vec<f32>>, Vec<f32>) {
    let dims = data[0].len();
    let n = data.len() as f32;
    let mut mean = vec![0.0f32; dims];
    for d in data {
        for i in 0..dims {
            mean[i] += d[i] / n;
        }
    }
    let mut centred: Vec<Vec<f32>> = data.iter().map(|d| d.iter().zip(&mean).map(|(a, m)| a - m).collect()).collect();
    let mut axes = vec![];
    let mut rng = Rng(5);
    for _ in 0..count {
        let mut v: Vec<f32> = (0..dims).map(|_| rng.f() - 0.5).collect();
        for _ in 0..30 {
            let mut next = vec![0.0f32; dims];
            for row in &centred {
                let d: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                for i in 0..dims {
                    next[i] += row[i] * d;
                }
            }
            let len = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            next.iter_mut().for_each(|x| *x /= len);
            v = next;
        }
        for row in centred.iter_mut() {
            let d: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
            for i in 0..dims {
                row[i] -= d * v[i];
            }
        }
        axes.push(v);
    }
    (axes, mean)
}

/// A plain description from the numbers. Only for reading results.
fn label(diet: &[f32; 4], moving: f32, budget: &[f32; 10]) -> String {
    let t = diet.iter().sum::<f32>().max(1e-6);
    let (ground, meat, bite) = (diet[0] / t, diet[1] / t, diet[2] / t);
    let food = if bite >= 0.4 {
        "Hunter"
    } else if meat >= 0.4 {
        "Scavenger"
    } else if bite + meat >= 0.25 {
        "Omnivore"
    } else if ground > 0.5 {
        "Grazer"
    } else {
        "Mixed feeder"
    };
    let way = if budget[0] >= 0.5 && moving < 0.05 {
        "resting"
    } else if moving < 0.03 {
        "sitting"
    } else if moving < 0.15 {
        "slow"
    } else {
        "roaming"
    };
    format!("{way} {}", food.to_lowercase())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: life-history NAME DUMP_DIR OUT.json");
        std::process::exit(2);
    }
    let tests = battery();
    let mut paths: Vec<_> = std::fs::read_dir(&args[2]).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect();
    paths.sort();
    let mut dumps = vec![];
    let mut hr = 2;
    for p in &paths {
        if let Some((d, h)) = read_dump(p, &tests) {
            hr = h;
            dumps.push(d);
        }
    }
    let _ = hr;
    eprintln!("{} dumps, {} cells", dumps.len(), dumps.iter().map(|d| d.cells.len()).sum::<usize>());
    let mut rng = Rng(11);

    // Strategies: one set across the whole run, so a strategy keeps its identity through time.
    let mut pool: Vec<(usize, usize)> = vec![];
    for (i, d) in dumps.iter().enumerate() {
        for j in 0..d.cells.len() {
            pool.push((i, j));
        }
    }
    let mut picked = pool.clone();
    for i in 0..picked.len().min(4000) {
        let j = i + rng.below(picked.len() - i);
        picked.swap(i, j);
    }
    picked.truncate(4000);
    let rows: Vec<&[f32]> = picked.iter().map(|(i, j)| dumps[*i].cells[*j].print.as_slice()).collect();
    let (axes, mean) = top_axes(&rows, 10);
    let reduce = |p: &[f32]| -> Vec<f32> { axes.iter().map(|a| p.iter().zip(&mean).zip(a).map(|((r, m), x)| (r - m) * x).sum()).collect() };
    let reduced: Vec<Vec<f32>> = rows.iter().map(|r| reduce(r)).collect();
    let mut scores = vec![];
    let mut best: (usize, f32, Vec<Vec<f32>>) = (2, f32::MIN, vec![]);
    for k in 2..=14 {
        let (lab, centres) = kmeans(&reduced, k, &mut rng);
        let s = silhouette(&reduced, &lab, k);
        scores.push(json!({"k": k, "silhouette": (s * 1000.0).round() / 1000.0}));
        // Prefer more strategies when the fit is about as good, since rare ones matter here.
        if s > best.1 - 0.02 && (s > best.1 || k > best.0) {
            best = (k, s.max(best.1), centres);
        }
    }
    let (k, sil, centres) = best;
    let classify = |p: &[f32]| -> usize {
        let r = reduce(p);
        let (mut b, mut bd) = (0, f32::MAX);
        for (c, centre) in centres.iter().enumerate() {
            let v = dist2(&r, centre);
            if v < bd {
                bd = v;
                b = c;
            }
        }
        b
    };

    // Each strategy over time, with how it lives.
    let mut share = vec![vec![0.0f32; dumps.len()]; k];
    let mut diet = vec![[0.0f32; 4]; k];
    let mut budget = vec![[0.0f32; 10]; k];
    let mut moving = vec![0.0f32; k];
    let mut crowd = vec![0.0f32; k];
    let mut age = vec![0.0f32; k];
    let mut energy = vec![0.0f32; k];
    let mut count = vec![0.0f32; k];
    for (i, d) in dumps.iter().enumerate() {
        for c in &d.cells {
            let s = classify(&c.print);
            share[s][i] += 1.0 / d.cells.len().max(1) as f32;
            count[s] += 1.0;
            for a in 0..4 {
                diet[s][a] += c.diet[a];
            }
            for a in 0..10 {
                budget[s][a] += c.budget[a];
            }
            moving[s] += c.moving;
            crowd[s] += c.crowd;
            age[s] += c.age;
            energy[s] += c.energy;
        }
    }
    let ticks: Vec<u64> = dumps.iter().map(|d| d.tick).collect();
    let mut strategies = vec![];
    for s in 0..k {
        let m = count[s].max(1.0);
        let b: [f32; 10] = std::array::from_fn(|a| budget[s][a] / m);
        let t: f32 = diet[s].iter().sum::<f32>().max(1e-6);
        let peak = share[s].iter().cloned().fold(0.0f32, f32::max);
        let mut bud = serde_json::Map::new();
        for a in 0..10 {
            bud.insert(ACTIONS[a].into(), json!((b[a] * 1000.0).round() / 1000.0));
        }
        strategies.push(json!({
            "id": s,
            "label": label(&diet[s], moving[s] / m, &b),
            "share_over_time": share[s].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f32>>(),
            "peak": peak,
            "lives_on": {"ground": diet[s][0] / t, "meat": diet[s][1] / t, "bite": diet[s][2] / t, "gift": diet[s][3] / t},
            "moving": moving[s] / m, "neighbours": crowd[s] / m, "age": age[s] / m, "energy": energy[s] / m,
            "budget": bud,
        }));
    }

    // Species: groups of genomes, followed from one dump to the next.
    struct Species {
        id: usize,
        parent: Option<usize>,
        born: usize,
        last: usize,
        centre: Vec<f32>,
        shares: Vec<f32>,
        strategy_counts: Vec<f32>,
        peak: f32,
    }
    let mut species: Vec<Species> = vec![];
    let join_gap = 0.45f32; // rms weight difference, in weight units, for "the same species"
    let mut events: Vec<Value> = vec![];
    for (i, d) in dumps.iter().enumerate() {
        if d.cells.is_empty() {
            continue;
        }
        let step = (d.cells.len() / 800).max(1);
        let idx: Vec<usize> = (0..d.cells.len()).step_by(step).collect();
        // Leader clustering on real weights: a cell joins the nearest group within the gap.
        let mut groups: Vec<(Vec<f32>, Vec<usize>)> = vec![];
        let dims = d.cells[0].genome.len() as f32;
        let gap_of = |a: &[f32], b: &[f32]| (dist2(a, b) / dims).sqrt() * 2.0;
        for &c in &idx {
            let g = &d.cells[c].genome;
            let mut found = None;
            let mut bd = join_gap;
            for (gi, (centre, _)) in groups.iter().enumerate() {
                let v = gap_of(g, centre);
                if v < bd {
                    bd = v;
                    found = Some(gi);
                }
            }
            match found {
                Some(gi) => {
                    let (centre, members) = &mut groups[gi];
                    members.push(c);
                    let n = members.len() as f32;
                    for (x, y) in centre.iter_mut().zip(g) {
                        *x += (y - *x) / n;
                    }
                }
                None => groups.push((g.clone(), vec![c])),
            }
        }
        groups.sort_by_key(|g| std::cmp::Reverse(g.1.len()));
        let live_before: Vec<usize> = species.iter().enumerate().filter(|(_, s)| s.last + 1 == i).map(|(x, _)| x).collect();
        let mut taken: Vec<usize> = vec![];
        for (centre, members) in groups {
            let frac = members.len() as f32 / idx.len() as f32;
            if frac < 0.01 {
                continue;
            }
            // The nearest group from last time is this group's past, or its parent if already claimed.
            let mut near = None;
            let mut nd = f32::MAX;
            for &x in &live_before {
                let v = gap_of(&centre, &species[x].centre);
                if v < nd {
                    nd = v;
                    near = Some(x);
                }
            }
            let mut counts = vec![0.0f32; k];
            for &m in &members {
                counts[classify(&d.cells[m].print)] += 1.0;
            }
            match near {
                Some(x) if nd < join_gap * 1.3 && !taken.contains(&x) => {
                    taken.push(x);
                    let s = &mut species[x];
                    s.centre = centre;
                    s.last = i;
                    s.shares[i] = frac;
                    s.peak = s.peak.max(frac);
                    for (a, b) in s.strategy_counts.iter_mut().zip(&counts) {
                        *a += b;
                    }
                }
                _ => {
                    let id = species.len();
                    let parent = near;
                    let mut shares = vec![0.0f32; dumps.len()];
                    shares[i] = frac;
                    species.push(Species { id, parent, born: i, last: i, centre, shares, strategy_counts: counts, peak: frac });
                }
            }
        }
    }
    let strategy_label = |c: &[f32]| -> String {
        let (mut b, mut bv) = (0, -1.0);
        for (i, v) in c.iter().enumerate() {
            if *v > bv {
                bv = *v;
                b = i;
            }
        }
        strategies[b]["label"].as_str().unwrap_or("").to_string()
    };
    for s in &species {
        if s.peak >= 0.05 && s.born > 0 {
            events.push(json!({"tick": ticks[s.born], "kind": "new species", "species": s.id, "parent": s.parent, "what": strategy_label(&s.strategy_counts)}));
        }
        if s.peak >= 0.05 && s.last + 1 < dumps.len() {
            events.push(json!({"tick": ticks[s.last + 1], "kind": "species gone", "species": s.id, "peak": s.peak, "what": strategy_label(&s.strategy_counts)}));
        }
    }
    for st in &strategies {
        let shares: Vec<f32> = st["share_over_time"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
        let name = st["label"].as_str().unwrap();
        let mut above = false;
        for (i, v) in shares.iter().enumerate() {
            if !above && *v >= 0.1 {
                above = true;
                if i > 0 {
                    events.push(json!({"tick": ticks[i], "kind": "strategy spreads", "what": name, "share": v}));
                }
            } else if above && *v < 0.03 {
                above = false;
                events.push(json!({"tick": ticks[i], "kind": "strategy fades", "what": name, "share": v}));
            }
        }
    }
    for i in 1..dumps.len() {
        let (a, b) = (dumps[i - 1].alive as f32, dumps[i].alive as f32);
        if b < a * 0.7 {
            events.push(json!({"tick": ticks[i], "kind": "population crash", "from": a, "to": b}));
        }
    }
    events.sort_by_key(|e| e["tick"].as_u64().unwrap_or(0));

    // Does the map change? Squares occupied now that were occupied some time ago.
    let overlap = |a: &[u8], b: &[u8]| -> f32 {
        let (mut both, mut either) = (0u64, 0u64);
        for (x, y) in a.iter().zip(b) {
            both += (x & y).count_ones() as u64;
            either += (x | y).count_ones() as u64;
        }
        both as f32 / either.max(1) as f32
    };
    let mut map_change = vec![];
    for lag in [1usize, 2, 5, 10, 20] {
        let mut sum = 0.0f32;
        let mut n = 0.0f32;
        for i in lag..dumps.len() {
            if i * 2 >= dumps.len() {
                sum += overlap(&dumps[i - lag].map, &dumps[i].map);
                n += 1.0;
            }
        }
        if n > 0.0 {
            let dt = if dumps.len() > 1 { ticks[1] - ticks[0] } else { 0 };
            map_change.push(json!({"ticks_apart": lag as u64 * dt, "overlap": (sum / n * 1000.0).round() / 1000.0}));
        }
    }
    // Does behaviour keep moving? Distance of the population's average behaviour from each earlier time.
    let centroid = |d: &Dump| -> Vec<f32> {
        let dims = d.cells[0].print.len();
        let mut m = vec![0.0f32; dims];
        for c in &d.cells {
            for i in 0..dims {
                m[i] += c.print[i] / d.cells.len() as f32;
            }
        }
        m
    };
    let centroids: Vec<Vec<f32>> = dumps.iter().filter(|d| !d.cells.is_empty()).map(centroid).collect();
    let drift: Vec<f32> = (1..centroids.len()).map(|i| ((dist2(&centroids[i], &centroids[i - 1]) / centroids[i].len() as f32).sqrt() * 1000.0).round() / 1000.0).collect();
    let from_end: Vec<f32> = centroids.iter().map(|c| ((dist2(c, centroids.last().unwrap()) / c.len() as f32).sqrt() * 1000.0).round() / 1000.0).collect();
    // New species per dump: Bedau's measure of whether new things keep arriving.
    let mut new_per_dump = vec![0u32; dumps.len()];
    for s in &species {
        if s.peak >= 0.02 {
            new_per_dump[s.born] += 1;
        }
    }
    let alive_species: Vec<u32> = (0..dumps.len()).map(|i| species.iter().filter(|s| s.shares[i] > 0.0).count() as u32).collect();
    let species_json: Vec<Value> = species
        .iter()
        .filter(|s| s.peak >= 0.02)
        .map(|s| {
            json!({
                "id": s.id, "parent": s.parent, "born": ticks[s.born], "last": ticks[s.last], "peak": s.peak,
                "strategy": strategy_label(&s.strategy_counts),
                "shares": s.shares.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f32>>(),
            })
        })
        .collect();
    let founders: Vec<usize> = dumps
        .iter()
        .map(|d| {
            let mut seen = std::collections::HashSet::new();
            for c in &d.cells {
                seen.insert(c.clade);
            }
            seen.len()
        })
        .collect();
    let gens: Vec<f32> = dumps.iter().map(|d| d.cells.iter().map(|c| c.generation).sum::<f32>() / d.cells.len().max(1) as f32).collect();
    let ages: Vec<f32> = dumps.iter().map(|d| d.cells.iter().map(|c| c.age).sum::<f32>() / d.cells.len().max(1) as f32).collect();
    let energies: Vec<f32> = dumps.iter().map(|d| d.cells.iter().map(|c| c.energy).sum::<f32>() / d.cells.len().max(1) as f32).collect();
    let out = json!({
        "name": args[1],
        "ticks": ticks,
        "alive": dumps.iter().map(|d| d.alive).collect::<Vec<u64>>(),
        "strategy_scores": scores,
        "strategies_k": k,
        "strategies_silhouette": sil,
        "strategies": strategies,
        "species": species_json,
        "species_alive": alive_species,
        "new_species": new_per_dump,
        "events": events,
        "map_overlap": map_change,
        "behaviour_drift": drift,
        "behaviour_from_end": from_end,
        "founders": founders,
        "generation": gens,
        "age": ages,
        "energy": energies,
    });
    std::fs::write(&args[3], serde_json::to_string(&out).unwrap()).unwrap();
}
