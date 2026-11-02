//! Follows a run through time from its dumps: which strategies and species exist, when they
//! appear and vanish, and whether anything keeps changing. Runs on the CPU, with no GPU.
//!   life-history NAME DUMP_DIR OUT.json [--tile SIZE]
//!
//! With --tile, the map is a batch of separate worlds, and each gets its own history.
//!
//! Strategies come from what brains do in a fixed set of situations, so two genomes that act
//! alike count as one. Species come from genomes, so the two can be compared. Labels are
//! plain descriptions of the numbers, for reading. Cells never see them.

use auto_life_gpu::measure::{self, Rules};
use serde_json::{Value, json};
use std::io::Read;

const IN_BLOCKS: usize = auto_life_gpu::config::IN_BLOCKS as usize;
const OUT_ROWS: usize = auto_life_gpu::config::OUT_ROWS as usize;
const CELL_BYTES: usize = auto_life_gpu::config::CELL_BYTES as usize;
const STAT_BYTES: usize = auto_life_gpu::config::STAT_BYTES as usize;
const ACTIONS: [&str; 12] = [
    "rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme", "armour", "clean",
];
/// Which rules are on in this run. Set once, from a dump.
static RULES: std::sync::OnceLock<Rules> = std::sync::OnceLock::new();

fn rules() -> Rules {
    RULES.get().copied().unwrap_or_default()
}
/// A fine clustering for counts that don't lean to 2, whatever the silhouette says.
const FINE_K: usize = 12;

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
    budget: [f32; 12],
    moving: f32,
    crowd: f32,
    age: f32,
    energy: f32,
    generation: f32,
    clade: u32,
    x: u32,
    y: u32,
    womb: f32,
    /// Free mineral, coat, bound mineral, last key.
    rule: [f32; 4],
    uptake: f32,
    genome: Vec<f32>,
    print: Vec<f32>,
}

struct Dump {
    tick: u64,
    alive: u64,
    size: usize,
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

fn act(g: &[f32], hr: usize, input: &Inputs) -> ([f32; 13], [[f32; 4]; OUT_ROWS]) {
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
    let raw = [out[0][0], out[0][1], out[0][2], out[0][3], out[1][0], out[1][1], out[1][2], out[1][3], out[2][0], out[2][1], out[2][2], out[2][3], out[6][0]];
    let on = rules().actions();
    let top = raw
        .iter()
        .enumerate()
        .filter(|(i, _)| *i < 12 || on[2])
        .map(|(_, v)| *v)
        .fold(f32::MIN, f32::max);
    let mut s = [0.0f32; 13];
    let mut z = 0.0;
    for i in 0..13 {
        if i < 10 || on[i - 10] {
            s[i] = (raw[i] - top).clamp(-20.0, 0.0).exp();
            z += s[i];
        }
    }
    s.iter_mut().for_each(|v| *v /= z);
    (s, out)
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

/// A dump as it is on disk, so a batch can turn only one world's cells into brains at a time.
struct Raw {
    tick: u64,
    alive: u64,
    size: usize,
    hr: usize,
    map: Vec<u8>,
    rec: usize,
    recs: Vec<u8>,
}

fn read_raw(path: &std::path::Path) -> Option<Raw> {
    let mut f = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut head).ok()?;
    let head: Value = serde_json::from_slice(&head).ok()?;
    if let Err(why) = auto_life_gpu::config::check_layout(&head, &path.display().to_string()) {
        eprintln!("{why}");
        std::process::exit(2);
    }
    let _ = RULES.set(Rules::from_physics(&head["physics"]));
    let size = head["size"].as_u64()? as usize;
    let count = head["count"].as_u64()? as usize;
    let gb = head["genome_bytes"].as_u64()? as usize;
    let hr = head["hidden_rows"].as_u64()? as usize;
    let mut map = vec![0u8; size * size / 8 + 1];
    f.read_exact(&mut map).ok()?;
    let rec = CELL_BYTES + STAT_BYTES + gb;
    let mut recs = vec![0u8; count * rec];
    f.read_exact(&mut recs).ok()?;
    Some(Raw { tick: head["tick"].as_u64()?, alive: head["alive"].as_u64()?, size, hr, map, rec, recs })
}

/// A record as a cell, with its brain run through the standard situations. The rule's own
/// situations and outputs go on the end, and only what exists counts, so a rule that's off
/// doesn't change the measure.
fn cell_of(rec: &[u8], size: usize, hr: usize, tests: &[Inputs]) -> Cell {
    let rule = rules();
    let on_actions = rule.actions();
    let u: &[u32] = &bytemuck::pod_collect_to_vec::<u8, u32>(&rec[..CELL_BYTES]);
    let st: Vec<f32> = bytemuck::pod_collect_to_vec::<u8, f32>(&rec[CELL_BYTES..CELL_BYTES + STAT_BYTES]);
    let genome: Vec<f32> = rec[CELL_BYTES + STAT_BYTES..].iter().map(|b| (*b as i8 as f32 / 127.0).max(-1.0)).collect();
    let mut print = Vec::with_capacity(tests.len() * 13);
    let mut beyond = vec![];
    for t in tests {
        let (shares, out) = act(&genome, hr, t);
        print.extend_from_slice(&shares[..10]);
        for k in 0..3 {
            if on_actions[k] {
                print.push(shares[10 + k]);
            }
        }
        beyond.extend(measure::rule_outputs(&rule, &out));
    }
    print.extend(measure::outputs_once(&beyond, tests.len()));
    Cell {
        diet: st[0..4].try_into().unwrap(),
        budget: st[4..16].try_into().unwrap(),
        moving: st[19],
        crowd: st[24] * 8.0,
        age: f32::from_bits(u[2]),
        energy: f32::from_bits(u[0]),
        generation: u[13] as f32,
        clade: u[12],
        x: u[10] % size as u32,
        y: u[10] / size as u32,
        womb: f32::from_bits(u[1]),
        rule: [f32::from_bits(u[20]), f32::from_bits(u[21]), f32::from_bits(u[22]), f32::from_bits(u[23])],
        uptake: st[28],
        genome,
        print,
    }
}

impl Raw {
    /// The whole map as one world.
    fn whole(&self, tests: &[Inputs]) -> Dump {
        let cells = self.recs.chunks_exact(self.rec).map(|r| cell_of(r, self.size, self.hr, tests)).collect();
        Dump { tick: self.tick, alive: self.alive, size: self.size, map: self.map.clone(), cells }
    }

    /// One world of a batch, with its own map and only its own cells, placed as if it were alone.
    fn tile(&self, tile: usize, index: usize, tests: &[Inputs]) -> Dump {
        let per_side = self.size / tile;
        let (ox, oy) = (index % per_side * tile, index / per_side * tile);
        let mut map = vec![0u8; tile * tile / 8 + 1];
        let mut alive = 0u64;
        for y in 0..tile {
            for x in 0..tile {
                let p = (oy + y) * self.size + ox + x;
                if self.map[p / 8] & (1 << (p % 8)) != 0 {
                    let q = y * tile + x;
                    map[q / 8] |= 1 << (q % 8);
                    alive += 1;
                }
            }
        }
        let mut cells = vec![];
        for r in self.recs.chunks_exact(self.rec) {
            let pos = u32::from_le_bytes(r[40..44].try_into().unwrap()) as usize;
            let (x, y) = (pos % self.size, pos / self.size);
            if x >= ox && x < ox + tile && y >= oy && y < oy + tile {
                let mut c = cell_of(r, self.size, self.hr, tests);
                c.x -= ox as u32;
                c.y -= oy as u32;
                cells.push(c);
            }
        }
        Dump { tick: self.tick, alive, size: tile, map, cells }
    }
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
fn label(diet: &[f32; 4], moving: f32, budget: &[f32; 12]) -> String {
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
        eprintln!("usage: life-history NAME DUMP_DIR OUT.json [--tile SIZE]");
        std::process::exit(2);
    }
    let mut paths: Vec<_> = std::fs::read_dir(&args[2]).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "bin")).collect();
    paths.sort();
    let raws: Vec<Raw> = paths.iter().filter_map(|p| read_raw(p)).collect();
    // The rules are known once a dump has been read, and some bring situations of their own.
    let mut tests = battery();
    let make = |e: f32, food: f32, front: u8| situation(e, food, false, front, false, 0.0);
    tests.extend(measure::rule_situations(&rules(), &make));
    let tile = args.iter().position(|a| a == "--tile").map(|i| args[i + 1].parse::<usize>().expect("--tile SIZE"));
    let out = match (tile, raws.first()) {
        (Some(tile), Some(first)) => {
            let per_side = first.size / tile;
            let count = per_side * per_side;
            let next = std::sync::atomic::AtomicUsize::new(0);
            let done = std::sync::Mutex::new(vec![Value::Null; count]);
            std::thread::scope(|scope| {
                for _ in 0..3 {
                    scope.spawn(|| loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= count {
                            break;
                        }
                        let dumps: Vec<Dump> = raws.iter().map(|r| r.tile(tile, i, &tests)).collect();
                        let mut v = analyse(&format!("{}-t{i:03}", args[1]), &dumps);
                        v["tile"] = json!(i);
                        done.lock().unwrap()[i] = v;
                    });
                }
            });
            json!({"name": args[1], "batch_tile": tile, "tiles": done.into_inner().unwrap()})
        }
        _ => {
            let dumps: Vec<Dump> = raws.iter().map(|r| r.whole(&tests)).collect();
            eprintln!("{} dumps, {} cells", dumps.len(), dumps.iter().map(|d| d.cells.len()).sum::<usize>());
            analyse(&args[1], &dumps)
        }
    };
    std::fs::write(&args[3], serde_json::to_string(&out).unwrap()).unwrap();
}

fn analyse(name: &str, dumps: &[Dump]) -> Value {
    // Nothing to sort into strategies: say what's known, which is a result too.
    if dumps.iter().map(|d| d.cells.len()).sum::<usize>() < 50 {
        return json!({
            "name": name,
            "dead": true,
            "ticks": dumps.iter().map(|d| d.tick).collect::<Vec<u64>>(),
            "alive": dumps.iter().map(|d| d.alive).collect::<Vec<u64>>(),
        });
    }
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

    // A count that doesn't lean to 2. A fine clustering, with centres too close to count as
    // different merged, so one blob is one strategy however it's cut. This is the count to read.
    let fine_k = FINE_K.min(reduced.len().max(1));
    let (fine_label, fine_centres) = kmeans(&reduced, fine_k, &mut rng);
    let fine_group = measure::merge_close(&fine_centres, &reduced, &fine_label);
    let fine_n = fine_group.iter().max().map_or(0, |m| m + 1);
    let classify_fine = |p: &[f32]| -> usize {
        let r = reduce(p);
        let (mut b, mut bd) = (0, f32::MAX);
        for (c, centre) in fine_centres.iter().enumerate() {
            let v = dist2(&r, centre);
            if v < bd {
                bd = v;
                b = c;
            }
        }
        fine_group[b]
    };
    let dump_groups: Vec<Vec<usize>> = dumps.iter().map(|d| d.cells.iter().map(|c| classify_fine(&c.print)).collect()).collect();
    let mut fine_shares = vec![vec![0.0f32; dumps.len()]; fine_n];
    for (i, labels) in dump_groups.iter().enumerate() {
        for l in labels {
            fine_shares[*l][i] += 1.0 / labels.len().max(1) as f32;
        }
    }
    // Lasting means at or above a share across 3 log points in a row, or all of them in a short run.
    let lasting = 3.min(dumps.len().max(1));
    let persistent = |level: f32| fine_shares.iter().filter(|s| measure::longest_run(s, level) >= lasting).count();
    let persistent_1pct = persistent(0.01);
    let persistent_3pct = persistent(0.03);
    let above_now = |level: f32| -> Vec<usize> {
        (0..dumps.len()).map(|i| fine_shares.iter().filter(|s| s[i] >= level).count()).collect()
    };

    // Do strategies live side by side, or only in different places? Against the same fine strategies.
    let coexistence: Vec<Value> = [16u32, 32, 64]
        .iter()
        .map(|tile| {
            let per_dump: Vec<Value> = dumps
                .iter()
                .zip(&dump_groups)
                .map(|(d, labels)| {
                    let rows: Vec<(u32, u32, usize)> = d.cells.iter().zip(labels).map(|(c, l)| (c.x, c.y, *l)).collect();
                    measure::side_by_side(&rows, *tile, d.size as u32, fine_n.max(1))["overlap"].clone()
                })
                .collect();
            let known: Vec<f32> = per_dump.iter().filter_map(|v| v.as_f64()).map(|v| v as f32).collect();
            json!({
                "tile": tile,
                "overlap_over_time": per_dump,
                "mean_overlap": if known.is_empty() { Value::Null } else { json!(measure::round3(known.iter().sum::<f32>() / known.len() as f32)) },
            })
        })
        .collect();

    // Does each fine strategy's centre of mass move? Measured against how spread out it is, since
    // a thinly spread strategy's centre jumps about for no reason.
    let strategy_travel: Vec<Value> = (0..fine_n)
        .map(|g| {
            let mut centres: Vec<Option<(f32, f32, f32)>> = vec![];
            for (d, labels) in dumps.iter().zip(&dump_groups) {
                let tau = std::f32::consts::TAU / d.size as f32;
                let (mut sx, mut cx, mut sy, mut cy, mut n) = (0f32, 0f32, 0f32, 0f32, 0f32);
                for (c, l) in d.cells.iter().zip(labels) {
                    if *l == g {
                        sx += (c.x as f32 * tau).sin();
                        cx += (c.x as f32 * tau).cos();
                        sy += (c.y as f32 * tau).sin();
                        cy += (c.y as f32 * tau).cos();
                        n += 1.0;
                    }
                }
                if n >= 20.0 {
                    let tight = ((sx * sx + cx * cx).sqrt() + (sy * sy + cy * cy).sqrt()) / 2.0 / n;
                    centres.push(Some(((sx.atan2(cx) / tau).rem_euclid(d.size as f32), (sy.atan2(cy) / tau).rem_euclid(d.size as f32), tight)));
                } else {
                    centres.push(None);
                }
            }
            let mut steps = vec![];
            for i in 1..dumps.len() {
                if let (Some(a), Some(b)) = (centres[i - 1], centres[i]) {
                    let s = dumps[i].size as f32;
                    let wrap = |d: f32| {
                        let d = d.abs().rem_euclid(s);
                        d.min(s - d)
                    };
                    let dt = (dumps[i].tick - dumps[i - 1].tick).max(1) as f32;
                    steps.push((wrap(a.0 - b.0).powi(2) + wrap(a.1 - b.1).powi(2)).sqrt() / dt * 10000.0);
                }
            }
            let tight: Vec<f32> = centres.iter().flatten().map(|c| c.2).collect();
            json!({
                "squares_per_10k_ticks": if steps.is_empty() { 0.0 } else { measure::round3(steps.iter().sum::<f32>() / steps.len() as f32) },
                // Near 1 is one tight place, near 0 is everywhere.
                "tightness": if tight.is_empty() { 0.0 } else { measure::round3(tight.iter().sum::<f32>() / tight.len() as f32) },
            })
        })
        .collect();

    // Each strategy over time, with how it lives.
    let mut share = vec![vec![0.0f32; dumps.len()]; k];
    let mut diet = vec![[0.0f32; 4]; k];
    let mut budget = vec![[0.0f32; 12]; k];
    let mut rule_state = vec![[0.0f32; 4]; k];
    let mut coat_xy = vec![[0.0f32; 2]; k];
    let mut uptake = vec![0.0f32; k];
    let mut womb = vec![0.0f32; k];
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
            for a in 0..12 {
                budget[s][a] += c.budget[a];
            }
            moving[s] += c.moving;
            crowd[s] += c.crowd;
            age[s] += c.age;
            energy[s] += c.energy;
            womb[s] += c.womb;
            uptake[s] += c.uptake;
            rule_state[s][0] += c.rule[0];
            rule_state[s][2] += c.rule[2];
            let a = c.rule[1] * std::f32::consts::TAU;
            coat_xy[s][0] += a.cos();
            coat_xy[s][1] += a.sin();
        }
    }
    // Who gets energy from what: each strategy's share of the world's total, from the samples.
    let flow_total: [f32; 4] = std::array::from_fn(|a| diet.iter().map(|d| d[a]).sum::<f32>().max(1e-6));
    let ticks: Vec<u64> = dumps.iter().map(|d| d.tick).collect();
    let mut strategies = vec![];
    for s in 0..k {
        let m = count[s].max(1.0);
        let b: [f32; 12] = std::array::from_fn(|a| budget[s][a] / m);
        let t: f32 = diet[s].iter().sum::<f32>().max(1e-6);
        let peak = share[s].iter().cloned().fold(0.0f32, f32::max);
        let mut bud = serde_json::Map::new();
        for a in 0..12 {
            bud.insert(ACTIONS[a].into(), json!((b[a] * 1000.0).round() / 1000.0));
        }
        strategies.push(json!({
            "id": s,
            "label": label(&diet[s], moving[s] / m, &b),
            "share_over_time": share[s].iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<f32>>(),
            "peak": peak,
            "lives_on": {"ground": diet[s][0] / t, "meat": diet[s][1] / t, "bite": diet[s][2] / t, "gift": diet[s][3] / t},
            "income": {"ground": diet[s][0] / m, "meat": diet[s][1] / m, "bite": diet[s][2] / m, "gift": diet[s][3] / m},
            "share_of_flow": {"ground": diet[s][0] / flow_total[0], "meat": diet[s][1] / flow_total[1], "bite": diet[s][2] / flow_total[2], "gift": diet[s][3] / flow_total[3]},
            "mineral": [rule_state[s][0] / m, rule_state[s][2] / m],
            "coat_spread": 1.0 - (coat_xy[s][0].powi(2) + coat_xy[s][1].powi(2)).sqrt() / m,
            "uptake": uptake[s] / m,
            "womb": womb[s] / m,
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

    // Colonies through time: their sizes, and whether their centres travel.
    let mapped: Vec<measure::Colonies> = dumps
        .iter()
        .map(|d| measure::colonies(&|p| d.map[p / 8] >> (p % 8) & 1 == 1, d.size, 30))
        .collect();
    let colony_sizes: Vec<Value> = mapped
        .iter()
        .map(|c| measure::size_stats(&c.found.iter().map(|f| f.0).collect::<Vec<_>>(), 30.0))
        .collect();
    let (mut tracked, mut lost) = (0usize, 0usize);
    let (mut moved, mut per_10k, mut kept): (Vec<f32>, Vec<f32>, Vec<f32>) = (vec![], vec![], vec![]);
    for i in 1..dumps.len() {
        let t = measure::colony_travel(&mapped[i - 1], &mapped[i], dumps[i].size);
        tracked += t.tracked;
        lost += t.lost;
        let dt = (dumps[i].tick - dumps[i - 1].tick).max(1) as f32;
        moved.extend(&t.moved);
        per_10k.extend(t.squares.iter().map(|d| d / dt * 10000.0));
        kept.extend(&t.kept);
    }
    let sorted = |v: &mut Vec<f32>| v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted(&mut moved);
    sorted(&mut per_10k);
    sorted(&mut kept);
    let colony_travel = json!({
        "tracked": tracked,
        // Colonies with no overlap at all at the next log point: gone, or moved right off themselves.
        "lost": lost,
        "squares_per_10k_ticks": {"p50": measure::percentile(&per_10k, 0.5), "p90": measure::percentile(&per_10k, 0.9)},
        "radii_moved": {"p50": measure::percentile(&moved, 0.5), "p90": measure::percentile(&moved, 0.9)},
        "share_past_own_radius": measure::round3(moved.iter().filter(|m| **m > 1.0).count() as f32 / moved.len().max(1) as f32),
        "squares_kept": {"p50": measure::percentile(&kept, 0.5)},
    });

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
    // What the population lives on, and how much of its behaviour its species explain.
    let mean_of = |d: &Dump, f: &dyn Fn(&Cell) -> f32| d.cells.iter().map(f).sum::<f32>() / d.cells.len().max(1) as f32;
    let others = |c: &Cell| {
        let t: f32 = c.diet.iter().sum::<f32>().max(1e-6);
        (c.diet[1] + c.diet[2]) / t
    };
    let bites = |c: &Cell| {
        let t: f32 = c.diet.iter().sum::<f32>().max(1e-6);
        c.diet[2] / t
    };
    let gifts = |c: &Cell| {
        let t: f32 = c.diet.iter().sum::<f32>().max(1e-6);
        c.diet[3] / t
    };
    let series = |f: &dyn Fn(&Cell) -> f32| -> Vec<f32> { dumps.iter().map(|d| (mean_of(d, f) * 1000.0).round() / 1000.0).collect() };
    let explained: Vec<f32> = dumps.iter().map(|d| species_explain(d, &reduce, join_gap)).collect();
    let out = json!({
        "name": name,
        "from_others": series(&others),
        "bite": series(&bites),
        "gift": series(&gifts),
        "moving": series(&|c: &Cell| c.moving),
        "rest": series(&|c: &Cell| c.budget[0]),
        "species_explain": explained,
        "ticks": ticks,
        "alive": dumps.iter().map(|d| d.alive).collect::<Vec<u64>>(),
        "strategy_scores": scores,
        "strategies_k": k,
        "strategies_silhouette": sil,
        "strategies": strategies,
        "fine_k": fine_k,
        "fine_groups": fine_n,
        "fine_shares_over_time": fine_shares.iter().map(|s| s.iter().map(|v| measure::round3(*v)).collect::<Vec<f32>>()).collect::<Vec<_>>(),
        "persistent_1pct": persistent_1pct,
        "persistent_3pct": persistent_3pct,
        "distinct_3pct_over_time": above_now(0.03),
        "coexistence": coexistence,
        "strategy_travel": strategy_travel,
        "colony_sizes": colony_sizes,
        "colony_travel": colony_travel,
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
    out
}

/// Share of the differences in behaviour that lie between species rather than within them,
/// corrected for the share that groups this small would get by chance.
fn species_explain(d: &Dump, reduce: &dyn Fn(&[f32]) -> Vec<f32>, join_gap: f32) -> f32 {
    if d.cells.len() < 20 {
        return 0.0;
    }
    let step = (d.cells.len() / 800).max(1);
    let idx: Vec<usize> = (0..d.cells.len()).step_by(step).collect();
    let dims = d.cells[0].genome.len() as f32;
    let mut groups: Vec<(Vec<f32>, Vec<usize>)> = vec![];
    for &c in &idx {
        let g = &d.cells[c].genome;
        let mut found = None;
        let mut bd = join_gap;
        for (gi, (centre, _)) in groups.iter().enumerate() {
            let v = (dist2(g, centre) / dims).sqrt() * 2.0;
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
    // Groups under 1% are noise here, as everywhere else in this file.
    let least = (idx.len() as f32 * 0.01).ceil() as usize;
    let big: Vec<&(Vec<f32>, Vec<usize>)> = groups.iter().filter(|g| g.1.len() >= least.max(2)).collect();
    if big.len() < 2 {
        return 0.0;
    }
    let points: Vec<Vec<Vec<f32>>> = big.iter().map(|g| g.1.iter().map(|c| reduce(&d.cells[*c].print)).collect()).collect();
    let all: Vec<&Vec<f32>> = points.iter().flatten().collect();
    let n = all.len() as f32;
    let width = all[0].len();
    let mut mean = vec![0.0f32; width];
    for p in &all {
        for i in 0..width {
            mean[i] += p[i] / n;
        }
    }
    let total: f32 = all.iter().map(|p| dist2(p, &mean)).sum();
    let mut within = 0.0f32;
    for g in &points {
        let m = g.len() as f32;
        let mut centre = vec![0.0f32; width];
        for p in g {
            for i in 0..width {
                centre[i] += p[i] / m;
            }
        }
        within += g.iter().map(|p| dist2(p, &centre)).sum::<f32>();
    }
    let k = points.len() as f32;
    if total <= 0.0 || n <= k {
        return 0.0;
    }
    let between = total - within;
    let within_mean = within / (n - k);
    let omega = (between - (k - 1.0) * within_mean) / (total + within_mean);
    (omega.max(0.0) * 1000.0).round() / 1000.0
}
