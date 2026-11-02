//! What the test files share: worlds to test in, genomes with fixed outputs, the spending
//! rule as plain arithmetic, and checks that apply to any set of rules.
#![allow(dead_code)]

use auto_life_gpu::config::{CELL_BYTES, IN_BLOCKS, OUT_ROWS};
use auto_life_gpu::{Config, Pacing, World};

/// Every rule on at once, with settings gentle enough for life to get going.
pub const ALL_RULES: &[(&str, &str)] = &[
    ("mass_on", "true"),
    ("armour_on", "true"),
    ("pack_on", "true"),
    ("waste_on", "true"),
    ("waste_clean", "true"),
    ("waste_holds", "0.2"),
    ("plants_on", "true"),
    ("plants_peak", "2"),
    ("weather_sun_on", "true"),
    ("weather_fire_on", "true"),
    ("weather_fire_rate", "0.000001"),
    ("mate_pay", "0.5"),
    ("compat", "0.6"),
    ("skill_on", "true"),
    ("skill_sense", "true"),
    ("skill_naive", "0.8"),
    ("biomes_on", "true"),
    ("walls_on", "true"),
    ("wall_pass", "8"),
    ("wall_thick", "2"),
    ("bloom_share", "0.3"),
    ("roads_on", "true"),
];

pub const ROCK: u32 = u32::MAX;
pub const NONE: u32 = u32::MAX;

pub fn world_with(size: u32, seed: u32, sets: &[(&str, &str)]) -> World {
    // LIFE_TEST_SHADERS runs these tests on edited shaders, to check that a broken rule is caught.
    let shaders = std::env::var("LIFE_TEST_SHADERS").ok().map(Into::into);
    let mut cfg = Config::new(size, seed, shaders).unwrap();
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
pub fn still_world(size: u32) -> World {
    still_world_with(size, &[])
}

pub fn still_world_with(size: u32, more: &[(&str, &str)]) -> World {
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

pub fn f32_at(rec: &[u32; 16], k: usize) -> f32 {
    f32::from_bits(rec[k])
}

pub fn blocks(cfg_hidden: u32) -> usize {
    (cfg_hidden * (IN_BLOCKS + cfg_hidden + OUT_ROWS)) as usize
}

/// A genome whose outputs are fixed: only the bias input drives hidden unit 0, and every
/// output reads only that unit. Returns the genome. `want` lists output values as byte weights.
pub fn fixed_genome(hr: u32, want: [i8; 24]) -> Vec<u8> {
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

pub fn squash(x: f32) -> f32 {
    let c = x.clamp(-3.0, 3.0);
    c * (27.0 + c * c) / (27.0 + 9.0 * c * c)
}

/// What the fixed genome computes, using the same weights as the shader.
pub fn outputs(want: [i8; 24]) -> [f32; 24] {
    let w = |b: i8| (b as f32 / 127.0).max(-1.0) * 2.0;
    let h = squash(w(127));
    std::array::from_fn(|k| w(want[k]) * h)
}

pub struct Plain {
    pub e: f32,
    pub womb: f32,
    pub ate_ground: f32,
    pub ate_meat: f32,
    pub work: [f32; 12],
    pub a: [f32; 12],
    pub upkeep: f32,
}

/// The spending rule as plain arithmetic.
pub fn spend(w: &World, out: [f32; 24], e0: f32, womb0: f32, ground: f32, meat: f32) -> Plain {
    spend_with(w, out, e0, womb0, ground, meat, 0.0, 1.0)
}

/// The same, with a rule's extra upkeep and the tempo of the place.
#[allow(clippy::too_many_arguments)]
pub fn spend_with(
    w: &World,
    out: [f32; 24],
    e0: f32,
    womb0: f32,
    ground: f32,
    meat: f32,
    extra_upkeep: f32,
    tempo: f32,
) -> Plain {
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
        c.b("armour_on"),
        c.b("waste_on") && c.b("waste_clean"),
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
    let upkeep = ((c.f("upkeep") + c.f("upkeep_e") * e) * (1.0 - c.f("rest_saving") * rest)
        + extra_upkeep)
        * tempo;
    e -= upkeep.min(e);
    let spend_here = c.f("spend") * tempo;
    let want = spend_here * (1.0 - rest);
    let scale = if want > e { e / want } else { 1.0 };
    let a: [f32; 12] = std::array::from_fn(|k| s[k] * spend_here * scale);
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
        upkeep,
    }
}

pub fn close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() < 2e-4, "{what}: gpu {a}, plain {b}");
}

pub fn budget(pairs: &[(usize, i8)]) -> [i8; 24] {
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

pub const REST: usize = 0;
pub const GROUND: usize = 2;
pub const MEAT: usize = 3;
pub const GROW: usize = 4;
pub const EMIT: usize = 5;
pub const HEAT: usize = 6;
pub const BITE: usize = 7;
pub const GIVE: usize = 8;
pub const ENZYME: usize = 9;
pub const DIR_X: usize = 12;
pub const MATE: usize = 20;
pub const SHARE: usize = 21;

/// Everything physical, by position, so a different slot for the same cell doesn't matter.
pub fn fingerprint(w: &World) -> (u32, u64) {
    let occ = w.read_u32s("occ");
    let cells = w.read_cells();
    let sketch = w.read_u32s("sketch");
    let fa = w.read_vec4s("fa");
    let fc = w.read_vec4s("fc");
    let fd = w.read_vec4s("fd");
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |v: u32| {
        h = (h ^ v as u64).wrapping_mul(0x100000001b3);
    };
    for (p, o) in occ.iter().enumerate() {
        mix(fa[p][0].to_bits());
        mix(fa[p][1].to_bits());
        mix(fc[p][1].to_bits());
        mix(fc[p][2].to_bits());
        mix(fd[p][0].to_bits());
        if *o != 0 && *o != ROCK {
            let s = *o as usize - 1;
            mix(p as u32);
            mix(cells[s][0]);
            mix(cells[s][1]);
            for word in 4..8 {
                mix(cells[s][word]);
            }
            mix(sketch[s * 4]);
            mix(sketch[s * 4 + 3]);
        }
    }
    (w.alive(), h)
}

/// Writes one word of a cell's record, such as its hardness.
pub fn set_cell_word(w: &World, slot: u32, word: u64, v: f32) {
    w.gpu.queue.write_buffer(
        &w.buffers["cell"],
        slot as u64 * CELL_BYTES + word * 4,
        &v.to_bits().to_le_bytes(),
    );
    w.gpu.queue.submit([]);
    w.gpu.wait();
}

/// Everything put into the world and everything in it or lost must match.
pub fn assert_energy_adds_up(rules: &[(&str, &str)]) {
    energy_gap_under(rules, 800, 0.005);
}

pub fn energy_gap_under(rules: &[(&str, &str)], ticks: u32, limit: f64) {
    let mut sets = vec![("ledger", "true")];
    sets.extend_from_slice(rules);
    let mut w = world_with(64, 3, &sets);
    let start: f64 = w
        .read_vec4s("fa")
        .iter()
        .map(|v| v[0] as f64 + v[1] as f64)
        .sum();
    for _ in 0..ticks {
        w.step();
    }
    w.flush();
    assert!(w.alive() > 0, "the world should have life in it");
    let c = w.counters();
    let scale = w.cfg.f("ledger_scale") as f64;
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
    // A cell whose energy isn't a number would keep spending what it doesn't have, and max()
    // below would hide it.
    for r in w.read_cells().iter().filter(|r| r[10] != NONE) {
        assert!(
            f32_at(r, 0).is_finite() && f32_at(r, 1).is_finite(),
            "a cell with energy {} and womb {}",
            f32_at(r, 0),
            f32_at(r, 1)
        );
    }
    let cells: f64 = w
        .read_cells()
        .iter()
        .filter(|r| r[10] != NONE)
        .map(|r| f32_at(r, 0).max(0.0) as f64 + f32_at(r, 1) as f64 + body)
        .sum();
    let have = fields + cells + lost;
    let put_in = start + sun + rain;
    let gap = (have - put_in).abs() / put_in;
    eprintln!(
        "ledger {rules:?}: alive {}, have {have:.1}, put in {put_in:.1}, lost {lost:.1}, gap {:.2} ({gap:.5})",
        w.alive(),
        have - put_in
    );
    assert!(
        gap < limit,
        "have {have} (fields {fields}, cells {cells}, lost {lost}), put in {put_in} (start {start}, sun {sun}, rain {rain}), gap {gap}"
    );
}

/// The grid, the cells' own idea of where they are, and the free ring agree.
pub fn assert_grid_and_cells_agree(rules: &[(&str, &str)]) {
    let mut w = world_with(64, 5, rules);
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
