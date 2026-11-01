//! Whole-world checks and one-tick checks of each rule against the same rule
//! written as plain Rust. They run on whatever Vulkan device is there, and CI uses
//! a software driver.

use auto_life_gpu::config::{IN_BLOCKS, OUT_ROWS};
use auto_life_gpu::{Config, Pacing, World};

const ROCK: u32 = u32::MAX;
const NONE: u32 = u32::MAX;

fn world_with(size: u32, seed: u32, sets: &[(&str, &str)]) -> World {
    let mut cfg = Config::new(size, seed, None).unwrap();
    for (k, v) in sets {
        cfg.set(k, v).unwrap();
    }
    World::new(
        cfg,
        Pacing {
            target_ms: 50.0,
            duty: 1.0,
        },
    )
    .unwrap()
}

/// A still world: nothing grows, flows, rots or lands, so a tick's effects can be read off directly.
fn still_world(size: u32) -> World {
    still_world_with(size, &[])
}

fn still_world_with(size: u32, more: &[(&str, &str)]) -> World {
    let mut sets: Vec<(&str, &str)> = vec![
        ("sun", "0"),
        ("rock", "false"),
        ("season_ticks", "0"),
        ("rain_p", "0"),
        ("rain_burst_ticks", "0"),
        ("hazard", "0"),
        ("mut_rate", "0"),
        ("noise_on", "false"),
        ("meat_flow", "0"),
        ("meat_rot", "0"),
        ("heat_flow", "0"),
        ("heat_decay", "0"),
        ("enzyme_flow", "0"),
        ("enzyme_decay", "0"),
        ("p0_flow", "0"),
        ("p0_decay", "0"),
        ("p1_flow", "0"),
        ("p1_decay", "0"),
        ("p2_flow", "0"),
        ("p2_decay", "0"),
        ("p3_flow", "0"),
        ("p3_decay", "0"),
    ];
    sets.extend_from_slice(more);
    world_with(size, 7, &sets)
}

fn f32_at(rec: &[u32; 16], k: usize) -> f32 {
    f32::from_bits(rec[k])
}

#[test]
fn energy_adds_up() {
    let mut w = world_with(64, 3, &[("ledger", "true")]);
    let start: f64 = w
        .read_vec4s("fa")
        .iter()
        .map(|v| v[0] as f64 + v[1] as f64)
        .sum();
    for _ in 0..800 {
        w.step();
    }
    w.flush();
    assert!(w.alive() > 0, "the world should have life in it");
    let c = w.counters();
    let scale = 4096.0f64;
    let (lost, sun, rain) = (
        c[10] as f64 / scale,
        c[11] as f64 / scale,
        c[12] as f64 / scale,
    );
    let fields: f64 = w
        .read_vec4s("fa")
        .iter()
        .map(|v| v[0] as f64 + v[1] as f64)
        .sum();
    let body = w.cfg.f("body") as f64;
    let cells: f64 = w
        .read_cells()
        .iter()
        .filter(|r| r[10] != NONE)
        .map(|r| f32_at(r, 0).max(0.0) as f64 + f32_at(r, 1) as f64 + body)
        .sum();
    let have = fields + cells + lost;
    let put_in = start + sun + rain;
    let gap = (have - put_in).abs() / put_in;
    assert!(gap < 0.005, "have {have}, put in {put_in}, gap {gap}");
}

#[test]
fn grid_and_cells_agree() {
    let mut w = world_with(64, 5, &[]);
    for _ in 0..600 {
        w.step();
    }
    w.flush();
    let occ = w.read_u32s("occ");
    let cells = w.read_cells();
    let mut on_grid = 0;
    for (p, o) in occ.iter().enumerate() {
        if *o == 0 || *o == ROCK {
            continue;
        }
        on_grid += 1;
        assert_eq!(
            cells[*o as usize - 1][10] as usize,
            p,
            "cell's own idea of where it is"
        );
    }
    let live = cells.iter().filter(|r| r[10] != NONE).count();
    assert_eq!(on_grid, live);
    assert_eq!(on_grid as u32, w.alive());
    assert!(on_grid > 0);

    // Every slot is either in use or in the free ring, never both.
    let ring = w.read_u32s("ring");
    let c = w.counters();
    let mask = ring.len() as u32 - 1;
    let ctl = w.read_u32s("ctl");
    let mut tail = c[1];
    if (tail.wrapping_sub(ctl[0]) as i32) > 0 {
        tail = ctl[0];
    }
    let mut free = std::collections::HashSet::new();
    let mut t = tail;
    while t != c[0] {
        assert!(free.insert(ring[(t & mask) as usize]), "slot freed twice");
        t = t.wrapping_add(1);
    }
    for o in &occ {
        if *o != 0 && *o != ROCK {
            assert!(!free.contains(&(*o - 1)), "live slot on the free ring");
        }
    }
    assert_eq!(free.len() + on_grid, w.cfg.slots as usize);
}

/// Everything physical, by position, so a different slot for the same cell doesn't matter.
fn fingerprint(w: &World) -> (u32, u64) {
    let occ = w.read_u32s("occ");
    let cells = w.read_cells();
    let sketch = w.read_u32s("sketch");
    let fa = w.read_vec4s("fa");
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |v: u32| {
        h = (h ^ v as u64).wrapping_mul(0x100000001b3);
    };
    for (p, o) in occ.iter().enumerate() {
        mix(fa[p][0].to_bits());
        mix(fa[p][1].to_bits());
        if *o != 0 && *o != ROCK {
            let s = *o as usize - 1;
            mix(p as u32);
            mix(cells[s][0]);
            mix(cells[s][1]);
            mix(sketch[s * 4]);
            mix(sketch[s * 4 + 3]);
        }
    }
    (w.alive(), h)
}

#[test]
fn same_seed_same_world() {
    let run = |ticks: u32| {
        let mut w = world_with(64, 9, &[]);
        for _ in 0..ticks {
            w.step();
        }
        w.flush();
        fingerprint(&w)
    };
    assert_eq!(run(400), run(400));
}

// ---- One-tick rules against plain Rust ----

fn blocks(cfg_hidden: u32) -> usize {
    (cfg_hidden * (IN_BLOCKS + cfg_hidden + OUT_ROWS)) as usize
}

/// A genome whose outputs are fixed: only the bias input drives hidden unit 0, and every
/// output reads only that unit. Returns the genome. `want` lists output values as byte weights.
fn fixed_genome(hr: u32, want: [i8; 24]) -> Vec<u8> {
    let n = blocks(hr);
    let mut g = vec![0u8; n * 16];
    // Input block 1 (i1 = random, bias, crowd, x0), row 0, word 0 (unit 0), byte 1 (bias).
    let block = 1;
    g[block * 16 + 1] = 127u8;
    let ho_base = (hr * IN_BLOCKS + hr * hr) as usize;
    for o in 0..24usize {
        let row = o / 4;
        let word = o % 4;
        // Output row, hidden column 0, word for this output, byte for unit 0.
        let at = (ho_base + row * hr as usize) * 16 + word * 4;
        g[at] = want[o] as u8;
    }
    g
}

fn squash(x: f32) -> f32 {
    let c = x.clamp(-3.0, 3.0);
    c * (27.0 + c * c) / (27.0 + 9.0 * c * c)
}

/// What the fixed genome computes, using the same weights as the shader.
fn outputs(want: [i8; 24]) -> [f32; 24] {
    let w = |b: i8| (b as f32 / 127.0).max(-1.0) * 2.0;
    let h = squash(w(127));
    std::array::from_fn(|k| w(want[k]) * h)
}

struct Plain {
    e: f32,
    womb: f32,
    ate_ground: f32,
    ate_meat: f32,
    work: [f32; 12],
    a: [f32; 12],
}

/// The spending rule as plain arithmetic.
fn spend(w: &World, out: [f32; 24], e0: f32, womb0: f32, ground: f32, meat: f32) -> Plain {
    let c = &w.cfg;
    let top = out[..12].iter().cloned().fold(f32::MIN, f32::max);
    let on: [bool; 12] = [
        true,
        true,
        true,
        true,
        true,
        c.b("pheromones_on"),
        c.b("heat_on"),
        c.b("bite_on"),
        c.b("give_on"),
        c.b("enzyme_on"),
        c.b("x0_on"),
        c.b("x1_on"),
    ];
    let mut s = [0.0f32; 12];
    for k in 0..12 {
        s[k] = if on[k] {
            (out[k] - top).clamp(-20.0, 0.0).exp()
        } else {
            0.0
        };
    }
    let z: f32 = s.iter().sum();
    s.iter_mut().for_each(|v| *v /= z);
    let rest = s[0];
    let mut e = e0;
    let upkeep = (c.f("upkeep") + c.f("upkeep_e") * e) * (1.0 - c.f("rest_saving") * rest);
    e -= upkeep.min(e);
    let want = c.f("spend") * (1.0 - rest);
    let scale = if want > e { e / want } else { 1.0 };
    let a: [f32; 12] = std::array::from_fn(|k| s[k] * c.f("spend") * scale);
    let commit = c.f("commitment");
    let work: [f32; 12] = std::array::from_fn(|k| a[k] * a[k] / (a[k] + commit));
    let ate_ground = ground * (work[2] * c.f("ground_rate")).min(1.0);
    let ate_meat = meat * (work[3] * c.f("meat_rate")).min(1.0);
    let spent = want * scale;
    let grow = if womb0 < c.f("womb") { a[4] } else { 0.0 };
    e += ate_ground + ate_meat * c.f("meat_yield") - spent + (a[4] - grow);
    Plain {
        e,
        womb: womb0 + grow,
        ate_ground,
        ate_meat,
        work,
        a,
    }
}

fn close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() < 2e-4, "{what}: gpu {a}, plain {b}");
}

fn budget(pairs: &[(usize, i8)]) -> [i8; 24] {
    // Everything low by default, with direction, voice and willingness at zero or below.
    let mut v = [0i8; 24];
    for k in 0..12 {
        v[k] = -60;
    }
    v[17] = -60; // mate off
    v[20] = -60;
    v[21] = -60;
    for (k, b) in pairs {
        v[*k] = *b;
    }
    v
}

const REST: usize = 0;
const GROUND: usize = 2;
const MEAT: usize = 3;
const GROW: usize = 4;
const EMIT: usize = 5;
const HEAT: usize = 6;
const BITE: usize = 7;
const GIVE: usize = 8;
const ENZYME: usize = 9;
const DIR_X: usize = 12;
const MATE: usize = 20;
const SHARE: usize = 21;

#[test]
fn eating_matches_plain_rule() {
    let mut w = still_world(32);
    let want = budget(&[(GROUND, 40), (MEAT, 30), (GROW, 20), (DIR_X, 60)]);
    let g = fixed_genome(2, want);
    let (x, y) = (10u32, 10u32);
    w.write_field("fa", x, y, [3.0, 2.0, 0.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &g);
    w.step();
    w.flush();
    let plain = spend(&w, outputs(want), 8.0, 0.0, 3.0, 2.0);
    let cells = w.read_cells();
    close(f32_at(&cells[s as usize], 0), plain.e, "energy");
    close(f32_at(&cells[s as usize], 1), plain.womb, "womb");
    let fa = w.read_vec4s("fa");
    let p = (y * 32 + x) as usize;
    close(fa[p][0], 3.0 - plain.ate_ground, "ground left");
    close(fa[p][1], 2.0 - plain.ate_meat, "meat left");
}

#[test]
fn emissions_match_plain_rule() {
    let mut w = still_world(32);
    let mut want = budget(&[(EMIT, 40), (HEAT, 40), (ENZYME, 40), (DIR_X, 60)]);
    // All scent into channel 2.
    want[16] = -40;
    want[17] = -40;
    want[18] = 40;
    want[19] = -40;
    let g = fixed_genome(2, want);
    let (x, y) = (12u32, 12u32);
    let s = w.plant(x, y, 8.0, 0.0, 0, &g);
    w.step();
    w.flush();
    let out = outputs(want);
    let plain = spend(&w, out, 8.0, 0.0, 0.0, 0.0);
    let cells = w.read_cells();
    close(f32_at(&cells[s as usize], 0), plain.e, "energy");
    let (fa, fb, fc) = (w.read_vec4s("fa"), w.read_vec4s("fb"), w.read_vec4s("fc"));
    let p = (y * 32 + x) as usize;
    let m: [f32; 4] = std::array::from_fn(|k| {
        let top = out[16..20].iter().cloned().fold(f32::MIN, f32::max);
        (out[16 + k] - top).clamp(-20.0, 0.0).exp()
    });
    let total: f32 = m.iter().sum();
    close(
        fb[p][2],
        m[2] / total * plain.a[EMIT] * w.cfg.f("emit_power"),
        "scent",
    );
    close(
        fc[p][0],
        plain.work[ENZYME] * w.cfg.f("enzyme_power"),
        "enzyme",
    );
    let flame = plain.work[HEAT] * w.cfg.f("heat_power") / 4.0;
    close(fa[p + 1][2], flame, "heat east");
    close(fa[p - 1][2], flame, "heat west");
    close(fa[p + 32][2], flame, "heat south");
    close(fa[p - 32][2], flame, "heat north");
    close(fa[p][2], 0.0, "no heat on the emitter");
}

/// One tick for two touching cells facing each other, as plain arithmetic: the first acts,
/// then the second. Returns their energies and the meat spilled on each one's square.
fn two_cells(w: &World, out: [[f32; 24]; 2], start: [f32; 2]) -> ([f32; 2], [f32; 2]) {
    let c = &w.cfg;
    let mut e = start;
    let mut spill = [0.0f32; 2];
    for i in 0..2 {
        let j = 1 - i;
        // Meat that has already spilled on its own square is there to eat.
        let p = spend(w, out[i], e[i], 0.0, 0.0, spill[i]);
        e[i] = p.e;
        spill[i] -= p.ate_meat;
        if c.b("bite_on") {
            // The biter is one of the 8 sides, and can always reach through it.
            let force = 2.0 * e[i] / (e[i] + e[j] + 1e-3);
            let take = (p.work[BITE] * c.f("bite_power") * force).min(e[j]);
            let keep = take * c.f("bite_yield");
            e[j] -= take;
            e[i] += keep;
            spill[j] += (take - keep).min(keep);
        }
        if c.b("give_on") {
            let amount = (p.work[GIVE] * c.f("give_power")).min(e[i]);
            e[i] -= amount;
            e[j] += amount * c.f("give_yield");
        }
        // Agreeing takes both. The first to act hasn't heard from the other yet.
        if c.b("share_on") && i == 1 && out[0][SHARE] > 0.0 && out[1][SHARE] > 0.0 {
            let flow = (e[i] - e[j]) * c.f("share_rate");
            e[i] -= flow;
            e[j] += flow;
        }
    }
    (e, spill)
}

/// Checks a pair against both orders it could have run in, since the phases are shuffled.
fn check_pair(
    w: &World,
    slots: [u32; 2],
    squares: [usize; 2],
    out: [[f32; 24]; 2],
    start: [f32; 2],
) {
    let cells = w.read_cells();
    let got = [
        f32_at(&cells[slots[0] as usize], 0),
        f32_at(&cells[slots[1] as usize], 0),
    ];
    let fa = w.read_vec4s("fa");
    let meat = [fa[squares[0]][1], fa[squares[1]][1]];
    let (ab, spill_ab) = two_cells(w, out, start);
    let (ba_rev, spill_ba_rev) = two_cells(w, [out[1], out[0]], [start[1], start[0]]);
    let (ba, spill_ba) = ([ba_rev[1], ba_rev[0]], [spill_ba_rev[1], spill_ba_rev[0]]);
    let near = |x: [f32; 2], y: [f32; 2]| (x[0] - y[0]).abs() < 2e-4 && (x[1] - y[1]).abs() < 2e-4;
    assert!(
        (near(got, ab) && near(meat, spill_ab)) || (near(got, ba) && near(meat, spill_ba)),
        "gpu energy {got:?} meat {meat:?}, plain {ab:?} {spill_ab:?} or {ba:?} {spill_ba:?}"
    );
}

#[test]
fn biting_matches_plain_rule() {
    let mut w = still_world_with(
        32,
        &[
            ("give_on", "false"),
            ("heat_on", "false"),
            ("share_on", "false"),
        ],
    );
    let biter = budget(&[(BITE, 60), (DIR_X, 60)]);
    let resting = budget(&[(REST, 60), (DIR_X, -60)]);
    let (x, y) = (16u32, 16u32);
    let a = w.plant(x, y, 12.0, 0.0, 0, &fixed_genome(2, biter));
    let b = w.plant(x + 1, y, 6.0, 0.0, 4, &fixed_genome(2, resting));
    w.step();
    w.flush();
    let p = (y * 32 + x) as usize;
    check_pair(
        &w,
        [a, b],
        [p, p + 1],
        [outputs(biter), outputs(resting)],
        [12.0, 6.0],
    );
}

#[test]
fn giving_and_sharing_match_plain_rule() {
    let mut w = still_world_with(32, &[("bite_on", "false"), ("heat_on", "false")]);
    let giver = budget(&[(GIVE, 60), (DIR_X, 60), (SHARE, 40)]);
    let taker = budget(&[(REST, 60), (DIR_X, -60), (SHARE, 40)]);
    let (x, y) = (16u32, 8u32);
    let a = w.plant(x, y, 20.0, 0.0, 0, &fixed_genome(2, giver));
    let b = w.plant(x + 1, y, 4.0, 0.0, 4, &fixed_genome(2, taker));
    w.step();
    w.flush();
    let p = (y * 32 + x) as usize;
    check_pair(
        &w,
        [a, b],
        [p, p + 1],
        [outputs(giver), outputs(taker)],
        [20.0, 4.0],
    );
}

#[test]
fn birth_copies_the_parent() {
    let mut w = still_world(32);
    let want = budget(&[(REST, 60), (DIR_X, 60)]);
    let g = fixed_genome(2, want);
    let (x, y) = (20u32, 20u32);
    let womb = w.cfg.f("womb");
    let s = w.plant(x, y, 8.0, womb + 0.5, 0, &g);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let live: Vec<usize> = (0..cells.len()).filter(|k| cells[*k][10] != NONE).collect();
    assert_eq!(live.len(), 2, "one parent, one child");
    let kid = *live.iter().find(|k| **k != s as usize).unwrap();
    close(
        f32_at(&cells[kid], 0),
        womb + 0.5 - w.cfg.f("body"),
        "child's energy",
    );
    close(f32_at(&cells[s as usize], 1), 0.0, "parent's womb");
    assert_eq!(
        cells[kid][13],
        cells[s as usize][13] + 1,
        "one generation on"
    );
    assert_eq!(
        w.read_genome(kid as u32),
        g,
        "with no mutation a child is a copy"
    );
    let (m0, m1) = (w.read_mem(s), w.read_mem(kid as u32));
    assert_eq!(m0, m1, "a child starts with its parent's memory");
    // It takes up a neighbouring square.
    let p = cells[kid][10];
    let (kx, ky) = (p % 32, p / 32);
    assert!(kx.abs_diff(x) <= 1 && ky.abs_diff(y) <= 1);
}

#[test]
fn a_child_mixes_whole_units_from_two_parents() {
    let mut w = still_world(32);
    let hr = 2u32;
    let womb = w.cfg.f("womb");
    let mut want = budget(&[(REST, 60), (DIR_X, 60)]);
    want[MATE] = 40;
    let make = |mark: u8| {
        let mut g = fixed_genome(hr, want);
        // A parent's mark in input block 5, which these fixed genomes don't use for anything visible:
        // the inputs there are field gradients, and a still empty world has none.
        for row in 0..hr as usize {
            for word in 0..4usize {
                g[(row * IN_BLOCKS as usize + 5) * 16 + word * 4] = mark;
                g[(hr as usize * IN_BLOCKS as usize + row * hr as usize) * 16 + word * 4 + 3] =
                    mark;
            }
        }
        g
    };
    let (ga, gb) = (make(5), make(9));
    let (x, y) = (8u32, 24u32);
    let a = w.plant(x, y, 8.0, 0.0, 0, &ga);
    let b = w.plant(x + 1, y, 8.0, 0.0, 0, &gb);
    // Both say yes on the first tick. Fill a womb on the second, by hand.
    w.step();
    w.flush();
    let bytes = womb.to_bits().to_le_bytes();
    w.gpu
        .queue
        .write_buffer(&w.buffers["cell"], a as u64 * 64 + 4, &bytes);
    w.gpu.queue.submit([]);
    w.gpu.wait();
    w.step();
    w.flush();
    let counters = w.counters();
    assert_eq!(counters[4], 1, "one mixed birth");
    let cells = w.read_cells();
    let kid = (0..cells.len())
        .find(|k| cells[*k][10] != NONE && *k != a as usize && *k != b as usize)
        .unwrap() as u32;
    let g = w.read_genome(kid);
    let mut from = [0u8; 8];
    for row in 0..hr as usize {
        for word in 0..4usize {
            from[row * 4 + word] = g[(row * IN_BLOCKS as usize + 5) * 16 + word * 4];
            let again =
                g[(hr as usize * IN_BLOCKS as usize + row * hr as usize) * 16 + word * 4 + 3];
            assert_eq!(
                from[row * 4 + word],
                again,
                "a unit's input and memory wiring come from one parent"
            );
            assert!(from[row * 4 + word] == 5 || from[row * 4 + word] == 9);
        }
    }
    // Output wiring of each unit comes from the same parent. Input block 1's bias is unit 0 in both.
    let ho = (hr * IN_BLOCKS + hr * hr) as usize;
    let (ga_ho, gb_ho) = (&ga[ho * 16..], &gb[ho * 16..]);
    for (unit, parent) in from.iter().enumerate() {
        let (row, byte) = (unit / 4, unit % 4);
        let chosen = &g[ho * 16..];
        let mine = if *parent == 5 { ga_ho } else { gb_ho };
        for o in 0..OUT_ROWS as usize {
            for word in 0..4usize {
                let at = (o * hr as usize + row) * 16 + word * 4 + byte;
                assert_eq!(chosen[at], mine[at]);
            }
        }
    }
}
