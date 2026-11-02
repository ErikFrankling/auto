//! What the test files share: worlds to test in, genomes with fixed outputs, and the spending
//! rule as plain arithmetic.
#![allow(dead_code)]

use auto_life_soft::config::{E_ONE, IN_BLOCKS, OUT_ROWS};
use auto_life_soft::world::{Cell, F_ALIVE};
use auto_life_soft::{Config, Pacing, World};

pub fn world_with(size: u32, seed: u32, sets: &[(&str, &str)]) -> World {
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

/// A still world: nothing grows, flows, decays or lands, and no cell can spend, so a tick's
/// effects can be read off directly. Brains run every tick. The ledger is exact.
pub fn still_world(size: u32, more: &[(&str, &str)]) -> World {
    let mut sets: Vec<(&str, &str)> = vec![
        ("sun", "0"),
        ("rock", "false"),
        ("food_start_ticks", "0"),
        ("rain_p", "0"),
        ("rain_burst_ticks", "0"),
        ("hazard", "0"),
        ("mut_rate", "0"),
        ("mut_heritable", "false"),
        ("brain_every", "1"),
        ("ledger", "true"),
        ("upkeep", "0"),
        ("upkeep_e", "0"),
        ("spend", "0"),
        ("spend_share", "0"),
        ("meat_flow", "0"),
        ("meat_rot", "0"),
        ("heat_flow", "0"),
        ("heat_decay", "0"),
        ("heat_rate", "0"),
        ("heat_harm", "0"),
        ("enzyme_flow", "0"),
        ("enzyme_decay", "0"),
        ("s0_flow", "0"),
        ("s0_decay", "0"),
        ("s1_flow", "0"),
        ("s1_decay", "0"),
        ("s2_flow", "0"),
        ("s2_decay", "0"),
        ("s3_flow", "0"),
        ("s3_decay", "0"),
        ("locked", "0"),
    ];
    sets.extend_from_slice(more);
    world_with(size, 7, &sets)
}

pub const REST: usize = 0;
pub const THRUST: usize = 1;
pub const EAT: usize = 2;
pub const MEAT: usize = 3;
pub const GROW: usize = 4;
pub const EMIT: usize = 5;
pub const HEAT: usize = 6;
pub const BITE: usize = 7;
pub const GIVE: usize = 8;
pub const ENZYME: usize = 9;
pub const STICK: usize = 10;
pub const PUSH: usize = 11;
pub const TURN_X: usize = 12;
pub const TURN_Y: usize = 13;
pub const VOICE0: usize = 14;
pub const VOICE1: usize = 15;
pub const MIX0: usize = 16;
pub const GLUE0: usize = 20;
pub const GLUE1: usize = 21;
pub const SPLIT: usize = 22;
pub const SHARE: usize = 23;
pub const ANGLE_X: usize = 24;
pub const ANGLE_Y: usize = 25;
pub const OUTPUTS: usize = (OUT_ROWS * 4) as usize;

pub fn blocks(hr: u32) -> usize {
    (hr * (IN_BLOCKS + hr + OUT_ROWS)) as usize
}

/// A genome whose outputs are fixed: only the bias input drives hidden unit 0, and every output
/// reads only that unit. `want` lists output values as byte weights.
pub fn fixed_genome(hr: u32, want: [i8; OUTPUTS]) -> Vec<u8> {
    let n = blocks(hr);
    let mut g = vec![0u8; n * 16];
    // Input block 1 (noise, bias, size, pressure), row 0, word 0 (unit 0), byte 1 (bias).
    g[16 + 1] = 127u8;
    let ho_base = (hr * IN_BLOCKS + hr * hr) as usize;
    for (o, w) in want.iter().enumerate() {
        let row = o / 4;
        let word = o % 4;
        // Output row, hidden column 0, word for this output, byte for unit 0.
        let at = (ho_base + row * hr as usize) * 16 + word * 4;
        g[at] = *w as u8;
    }
    g
}

pub fn squash(x: f32) -> f32 {
    let c = x.clamp(-3.0, 3.0);
    c * (27.0 + c * c) / (27.0 + 9.0 * c * c)
}

/// What the fixed genome computes, using the same weights as the shader.
pub fn outputs(want: [i8; OUTPUTS]) -> [f32; OUTPUTS] {
    let w = |b: i8| (b as f32 / 127.0).max(-1.0) * 2.0;
    let h = squash(w(127));
    std::array::from_fn(|k| w(want[k]) * h)
}

/// Outputs with every action low, and everything else at zero.
pub fn budget(pairs: &[(usize, i8)]) -> [i8; OUTPUTS] {
    let mut v = [0i8; OUTPUTS];
    for slot in v.iter_mut().take(12) {
        *slot = -120;
    }
    for (k, b) in pairs {
        v[*k] = *b;
    }
    v
}

pub fn genome(pairs: &[(usize, i8)]) -> Vec<u8> {
    fixed_genome(2, budget(pairs))
}

/// Which actions a world's switches leave on, in budget order.
pub fn enabled(w: &World) -> [bool; 12] {
    let c = &w.cfg;
    [
        true,
        c.b("thrust_on"),
        true,
        c.b("meat_on"),
        true,
        c.b("scent_on"),
        c.b("heat_on"),
        c.b("bite_on"),
        c.b("give_on"),
        c.b("enzyme_on"),
        c.b("stick_on"),
        c.b("push_on"),
    ]
}

/// The shares of the budget a brain with these outputs asks for.
pub fn shares(w: &World, want: [i8; OUTPUTS]) -> [f32; 12] {
    let out = outputs(want);
    let on = enabled(w);
    let top = out[..12].iter().cloned().fold(f32::MIN, f32::max);
    let mut s = [0.0f32; 12];
    for k in 0..12 {
        s[k] = if on[k] { (out[k] - top).clamp(-20.0, 0.0).exp() } else { 0.0 };
    }
    let z: f32 = s.iter().sum();
    s.iter_mut().for_each(|v| *v /= z);
    s
}

/// One tick's spending, as plain arithmetic, in energy.
pub struct Plain {
    pub e: f32,
    pub womb: f32,
    pub unit: f32,
    pub a: [f32; 12],
    pub work: [f32; 12],
}

pub fn spend(w: &World, want: [i8; OUTPUTS], e0: f32, womb0: f32) -> Plain {
    let c = &w.cfg;
    let s = shares(w, want);
    let rest = s[0];
    let upkeep = (c.f("upkeep") + c.f("upkeep_e") * e0) * (1.0 - c.f("rest_saving") * rest);
    let mut e = e0 - upkeep.min(e0);
    let budget = c.f("spend").min(c.f("spend_share") * e0);
    let wanted = budget * (1.0 - rest);
    let spent = wanted.min(e);
    let unit = if wanted > 0.0 { budget * spent / wanted } else { budget };
    let a: [f32; 12] = std::array::from_fn(|k| s[k] * unit);
    let commit = c.f("commitment");
    let work: [f32; 12] = std::array::from_fn(|k| if a[k] > 0.0 { a[k] * a[k] / (a[k] + commit) } else { 0.0 });
    e -= spent;
    let mut womb = womb0;
    if womb0 + a[GROW] > c.f("womb_cap") {
        e += a[GROW];
    } else {
        womb += a[GROW];
    }
    Plain { e, womb, unit, a, work }
}

pub fn close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() < 4e-4, "{what}: gpu {a}, plain {b}");
}

pub fn e_of(c: &Cell) -> f32 {
    c.e as f32 / E_ONE
}

pub fn womb_of(c: &Cell) -> f32 {
    c.womb as f32 / E_ONE
}

pub fn alive(cells: &[Cell]) -> Vec<&Cell> {
    cells.iter().filter(|c| c.flags & F_ALIVE != 0).collect()
}

/// Where a cell is, in squares.
pub fn at(w: &World, c: &Cell) -> (f32, f32) {
    let k = 1.0 / (1u64 << (32 - w.cfg.size.trailing_zeros())) as f64;
    ((c.x as f64 * k) as f32, (c.y as f64 * k) as f32)
}

/// Checks that everything that came in, less everything that left, is what's there now, to the
/// last unit. Needs the ledger on, and `start` from before the first tick.
pub fn assert_ledger(w: &mut World, start: i128, what: &str) {
    w.flush();
    let t = w.tally();
    assert_eq!(t.sample, 1, "the ledger must be on");
    let stock = w.stock();
    let income = (t.flow("sun") + t.flow("rain")) as i128;
    let gap = stock.total() - (start + income - t.lost());
    assert_eq!(
        gap, 0,
        "{what}: {gap} units missing at tick {}. stock {:?}, {:?}",
        w.tick, stock, t
    );
}

/// Everything physical, as a hash, so two runs can be compared.
pub fn fingerprint(w: &mut World) -> u64 {
    w.fingerprint()
}

/// Settings that switch off every action except these, and give cells a budget. A fixed genome
/// can't make a share exactly zero, so this is how a test sees one rule on its own.
pub fn only(on: &[&'static str]) -> Vec<(&'static str, &'static str)> {
    let mut sets = vec![("spend", "0.2"), ("spend_share", "0.01")];
    for k in [
        "thrust_on", "meat_on", "scent_on", "heat_on", "bite_on", "give_on", "enzyme_on", "stick_on",
        "push_on",
    ] {
        if !on.contains(&k) {
            sets.push((k, "false"));
        }
    }
    sets
}

/// A still world with only these actions on, and these other settings.
pub fn quiet_world(size: u32, on: &[&'static str], more: &[(&'static str, &'static str)]) -> World {
    let mut sets = only(on);
    sets.extend_from_slice(more);
    still_world(size, &sets)
}
