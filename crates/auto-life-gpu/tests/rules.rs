//! Every rule in research/life-gpu/RULES.md, one tick at a time against the same rule written
//! as plain Rust, and then together. They run the real shaders on whatever Vulkan device is
//! there, and CI uses a software driver.

mod common;

use auto_life_gpu::World;
use auto_life_gpu::config::IN_BLOCKS;
use common::*;

const MOVE: usize = 1;
const ARMOUR: usize = 10;
const CLEAN: usize = 11;

const W32: usize = 32;

fn at(x: u32, y: u32) -> usize {
    (y as usize) * W32 + x as usize
}

/// A rich world where each rule moves a lot of energy, so a rule that loses count of even a
/// small part shows up. Rounding tiny amounts alone gives a gap of up to about 0.3% here, and
/// taking any one rule's lose() out of its shader gives 1% or more.
fn strict_ledger(rule: &[(&str, &str)]) {
    // A low bite yield, because at 0.5 and above all of a bite's waste spills as meat.
    let mut sets = vec![
        ("sun", "0.2"),
        ("hazard", "0.002"),
        ("bite_power", "60"),
        ("bite_yield", "0.3"),
    ];
    sets.extend_from_slice(rule);
    energy_gap_under(&sets, 3000, 0.005);
}

// ---- Bites, as every rule that changes them sees them ----

#[derive(Clone, Copy)]
struct Body {
    out: [f32; 24],
    e: f32,
    hard: f32,
    wound: f32,
    waste: f32,
}

/// One tick for two touching cells facing each other, as plain arithmetic: `first` acts, then
/// the other. Even a resting cell spends a little on everything, so both can bite. Reads
/// which rules are on from the world's settings. Assumes nobody can eat, and no waste at the start.
fn bite_tick(w: &World, start: [Body; 2], first: usize) -> [Body; 2] {
    let c = &w.cfg;
    let (armour, pack) = (c.b("armour_on"), c.b("pack_on"));
    let holds = if c.b("waste_on") {
        c.f("waste_holds")
    } else {
        0.0
    };
    let skill = if c.b("skill_on") {
        c.f("skill_naive")
    } else {
        1.0
    };
    let mut b = start;
    for i in [first, 1 - first] {
        let j = 1 - i;
        if armour {
            b[i].hard *= 1.0 - c.f("armour_fade");
        }
        if pack {
            b[i].wound *= c.f("pack_wound_keep");
        }
        let extra = if armour {
            c.f("armour_upkeep") * b[i].hard
        } else {
            0.0
        };
        let p = spend_with(w, b[i].out, b[i].e, 0.0, 0.0, 0.0, extra, 1.0);
        b[i].e = p.e;
        if armour {
            b[i].hard = (b[i].hard + p.work[ARMOUR] * c.f("armour_build")).min(1.0);
        }
        // The biter is one of the 8 sides, and can always reach through it.
        let force = 2.0 * b[i].e / (b[i].e + b[j].e + 1e-3);
        let factor = if pack {
            c.f("pack_intact") + (1.0 - c.f("pack_intact")) * b[j].wound
        } else {
            1.0
        };
        if pack {
            b[j].wound = (b[j].wound + p.work[BITE] * c.f("pack_wound_rate")).min(1.0);
        }
        let resist = if armour {
            1.0 + c.f("armour_resist") * b[j].hard
        } else {
            1.0
        };
        let take = (p.work[BITE] * c.f("bite_power") * force * factor / resist).min(b[j].e);
        let keep = take * c.f("bite_yield") * skill;
        b[j].e -= take;
        b[i].e += keep * (1.0 - holds);
        if c.b("waste_on") {
            b[i].waste += keep * c.f("waste_deposit");
        }
    }
    b
}

/// Runs one tick of a biter on a resting target and checks it against both orders it could have run in.
fn check_bite(sets: &[(&str, &str)], biter_hard: f32, target_hard: f32, target_wound: f32) {
    let mut all = vec![
        ("give_on", "false"),
        ("heat_on", "false"),
        ("share_on", "false"),
        ("meat_rate", "0"),
    ];
    all.extend_from_slice(sets);
    let mut w = still_world_with(32, &all);
    let biter = budget(&[(BITE, 60), (DIR_X, 60)]);
    let resting = budget(&[(REST, 60), (DIR_X, -60)]);
    let (x, y) = (16u32, 16u32);
    let a = w.plant(x, y, 12.0, 0.0, 0, &fixed_genome(2, biter));
    let b = w.plant(x + 1, y, 6.0, 0.0, 4, &fixed_genome(2, resting));
    set_cell_word(&w, a, 4, biter_hard);
    set_cell_word(&w, b, 4, target_hard);
    set_cell_word(&w, b, 5, target_wound);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let fc = w.read_vec4s("fc");
    let read = |slot: u32, p: usize| {
        let r = &cells[slot as usize];
        [f32_at(r, 0), f32_at(r, 4), f32_at(r, 5), fc[p][1]]
    };
    let got = [read(a, at(x, y)), read(b, at(x + 1, y))];
    let mk = |out, e, hard, wound| Body {
        out,
        e,
        hard,
        wound,
        waste: 0.0,
    };
    let start = [
        mk(outputs(biter), 12.0, biter_hard, 0.0),
        mk(outputs(resting), 6.0, target_hard, target_wound),
    ];
    let near = |p: &[Body; 2]| {
        (0..2).all(|k| {
            let want = [p[k].e, p[k].hard, p[k].wound, p[k].waste];
            (0..4).all(|i| (got[k][i] - want[i]).abs() < 2e-4)
        })
    };
    let (ab, ba) = (bite_tick(&w, start, 0), bite_tick(&w, start, 1));
    let show = |p: &[Body; 2]| p.map(|b| [b.e, b.hard, b.wound, b.waste]);
    assert!(
        near(&ab) || near(&ba),
        "gpu {got:?}, plain {:?} or {:?}",
        show(&ab),
        show(&ba)
    );
}

/// Whether each input reaches the brain, by making grazing depend on it alone. Returns the
/// ground eaten by a cell with that one extra wire, given a setup to run first.
fn eaten_with_wire(
    sets: &[(&str, &str)],
    wire: Option<(usize, usize)>,
    setup: impl Fn(&mut World, u32),
) -> f32 {
    let mut w = still_world_with(32, sets);
    let want = budget(&[(GROUND, 30), (REST, 30), (DIR_X, 60)]);
    let mut g = fixed_genome(2, want);
    if let Some((block, input)) = wire {
        let hr = 2usize;
        // Input to hidden unit 1 in row 0, and hidden unit 1 to the grazing output.
        g[block * 16 + 4 + input] = 127;
        let ho = hr * IN_BLOCKS as usize + hr * hr;
        g[ho * 16 + 2 * 4 + 1] = 127;
    }
    let (x, y) = (10u32, 10u32);
    w.write_field("fa", x, y, [3.0, 0.0, 0.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &g);
    setup(&mut w, s);
    w.step();
    w.flush();
    3.0 - w.read_vec4s("fa")[at(x, y)][0]
}

/// An input wired up must change nothing while its rule is off, and must change grazing when on.
fn check_input(
    on: &[(&str, &str)],
    off: &[(&str, &str)],
    wire: (usize, usize),
    setup: impl Fn(&mut World, u32) + Copy,
) {
    let bare = eaten_with_wire(off, None, setup);
    let wired_off = eaten_with_wire(off, Some(wire), setup);
    assert_eq!(
        wired_off, bare,
        "rule off, but its input {wire:?} reached the brain"
    );
    let wired_on = eaten_with_wire(on, Some(wire), setup);
    let bare_on = eaten_with_wire(on, None, setup);
    assert!(
        (wired_on - bare_on).abs() > 1e-3,
        "rule on, but input {wire:?} changed nothing: {wired_on} and {bare_on}"
    );
}

// ---- mass ----

/// What a successful step costs, or None if it's blocked.
fn mass_step(
    energy: f32,
    target: Option<f32>,
    on: bool,
    scale: f32,
    step: f32,
    push: f32,
) -> Option<f32> {
    let mass = energy.max(0.0) / scale.max(1e-6);
    let mut cost = if on { step * (1.0 + mass) } else { 0.0 };
    if let Some(other) = target {
        if !on || other <= 0.0 || other >= energy {
            return None;
        }
        cost += push * (1.0 + mass + other / scale.max(1e-6));
    }
    (energy > cost).then_some(cost)
}

#[test]
fn mass_push_matches_plain_rule() {
    for (on, target_energy, push_cost) in [
        ("false", 6.0, "0.08"),
        ("true", 6.0, "0.08"),
        ("true", 20.0, "0.08"),
        ("true", 6.0, "100"),
    ] {
        let mut w = still_world_with(
            32,
            &[
                ("mass_on", on),
                ("mass_push_cost", push_cost),
                ("mass_drag", "0"),
                ("move_rate", "10"),
                ("bite_on", "false"),
                ("give_on", "false"),
                ("heat_on", "false"),
                ("share_on", "false"),
            ],
        );
        let walk = budget(&[(MOVE, 60), (DIR_X, 60)]);
        // Barely any effort goes on moving, or the heavier target would shove the mover.
        let rest = budget(&[(REST, 60), (MOVE, -127), (DIR_X, -60)]);
        let a = w.plant(16, 16, 12.0, 0.0, 0, &fixed_genome(2, walk));
        let b = w.plant(17, 16, target_energy, 0.0, 4, &fixed_genome(2, rest));
        let pa = spend(&w, outputs(walk), 12.0, 0.0, 0.0, 0.0);
        let pb = spend(&w, outputs(rest), target_energy, 0.0, 0.0, 0.0);
        w.step();
        w.flush();
        let cells = w.read_cells();
        let swapped = cells[a as usize][10] as usize == at(17, 16);
        let (enabled, push) = (on == "true", push_cost.parse::<f32>().unwrap());
        let before = mass_step(pa.e, Some(target_energy), enabled, 6.0, 0.04, push);
        let after = mass_step(pa.e, Some(pb.e), enabled, 6.0, 0.04, push);
        assert_eq!(
            swapped,
            before.is_some(),
            "on {on}, target {target_energy}, push {push}"
        );
        let got = f32_at(&cells[a as usize], 0);
        let near = |cost: Option<f32>| (got - (pa.e - cost.unwrap_or(0.0))).abs() < 2e-4;
        assert!(
            near(before) || near(after),
            "mover has {got}, plain {} less {before:?} or {after:?}",
            pa.e
        );
        let target_got = f32_at(&cells[b as usize], 0);
        // A shoved cell that hadn't acted waits a tick, and that's the only way it keeps it all.
        assert!(
            (target_got - pb.e).abs() < 2e-4
                || (swapped && (target_got - target_energy).abs() < 2e-4)
        );
        let occ = w.read_u32s("occ");
        for slot in [a, b] {
            assert_eq!(
                occ[cells[slot as usize][10] as usize],
                slot + 1,
                "grid and cell agree"
            );
        }
        if enabled && swapped {
            assert_eq!(
                cells[b as usize][10] as usize,
                at(16, 16),
                "swapped into the old square"
            );
        }
        let seen = w.read_u32s("seen");
        for p in [at(16, 16), at(17, 16)] {
            assert_ne!(seen[p] & 0x80, 0, "both squares show a body");
        }
    }
}

#[test]
fn mass_slows_heavy_bodies_and_steps_cost_energy() {
    let e0 = 18.0;
    let run = |drag: &str| {
        let mut w = still_world_with(
            64,
            &[
                ("mass_on", "true"),
                ("mass_drag", drag),
                ("move_rate", "20"),
                ("bite_on", "false"),
                ("give_on", "false"),
                ("heat_on", "false"),
                ("share_on", "false"),
            ],
        );
        let walk = budget(&[(MOVE, 60), (DIR_X, 60)]);
        let mut slots = vec![];
        for y in (0..64).step_by(8) {
            for x in (0..64).step_by(8) {
                slots.push(w.plant(x, y, e0, 0.0, 0, &fixed_genome(2, walk)));
            }
        }
        let plain = spend(&w, outputs(walk), e0, 0.0, 0.0, 0.0);
        w.step();
        w.flush();
        let cells = w.read_cells();
        let mass = plain.e / 6.0;
        let chance = plain.work[MOVE] * w.cfg.f("move_rate") / (1.0 + w.cfg.f("mass_drag") * mass);
        let cost = w.cfg.f("mass_step_cost") * (1.0 + mass);
        let mut moved = 0;
        for (k, s) in slots.iter().enumerate() {
            let rec = &cells[*s as usize];
            let (y, x) = (k / 8 * 8, k % 8 * 8);
            let went = rec[10] as usize != y * 64 + x;
            moved += went as u32;
            let want = if went { plain.e - cost } else { plain.e };
            close(f32_at(rec, 0), want, "energy after a step");
        }
        (moved, chance.min(1.0), slots.len() as f32)
    };
    let (free, c0, n) = run("0");
    assert_eq!(c0, 1.0, "with no drag this budget always moves");
    assert_eq!(free as f32, n);
    let (slowed, c1, n) = run("1");
    let sd = (n * c1 * (1.0 - c1)).sqrt();
    assert!(
        (slowed as f32 - n * c1).abs() < 4.0 * sd + 1.0,
        "{slowed} of {n} moved, expected {}",
        n * c1
    );
    assert!(c1 < 0.9, "this test needs drag to matter, chance {c1}");
}

#[test]
fn energy_adds_up_with_mass() {
    strict_ledger(&[
        ("mass_on", "true"),
        ("mass_step_cost", "0.3"),
        ("mass_push_cost", "0.5"),
    ]);
}

// ---- armour ----

#[test]
fn armour_builds_fades_and_costs_upkeep() {
    let mut w = still_world_with(32, &[("armour_on", "true")]);
    let want = budget(&[(ARMOUR, 50), (GROUND, 20), (DIR_X, 60)]);
    let (x, y) = (10u32, 10u32);
    w.write_field("fa", x, y, [0.0, 0.0, 4.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &fixed_genome(2, want));
    set_cell_word(&w, s, 4, 0.5);
    w.step();
    w.flush();
    let c = &w.cfg;
    let hard = 0.5 * (1.0 - c.f("armour_fade"));
    let burn = 4.0 * c.f("heat_burn") / (1.0 + c.f("armour_resist") * hard);
    let plain = spend_with(
        &w,
        outputs(want),
        8.0 - burn,
        0.0,
        0.0,
        0.0,
        c.f("armour_upkeep") * hard,
        1.0,
    );
    let rec = &w.read_cells()[s as usize];
    close(f32_at(rec, 0), plain.e, "energy");
    close(
        f32_at(rec, 4),
        hard + plain.work[ARMOUR] * c.f("armour_build"),
        "hardness",
    );
    close(f32_at(rec, 3), burn, "the heat it felt");
}

#[test]
fn armour_divides_bites() {
    check_bite(&[("armour_on", "true")], 0.0, 0.8, 0.0);
}

#[test]
fn armour_slows_steps_and_is_not_inherited() {
    // So much drag that a hard cell can't step, where a soft one always does.
    let walk = budget(&[(MOVE, 60), (DIR_X, 60)]);
    let sets = [
        ("armour_on", "true"),
        ("armour_build", "0"),
        ("armour_drag", "1000000"),
        ("move_rate", "1000"),
        ("bite_on", "false"),
    ];
    let mut w = still_world_with(32, &sets);
    let (hard, soft) = (
        w.plant(8, 8, 8.0, 0.0, 0, &fixed_genome(2, walk)),
        w.plant(20, 20, 8.0, 0.0, 0, &fixed_genome(2, walk)),
    );
    set_cell_word(&w, hard, 4, 1.0);
    w.step();
    w.flush();
    let cells = w.read_cells();
    assert_eq!(
        cells[hard as usize][10] as usize,
        at(8, 8),
        "too hard to move"
    );
    assert_eq!(
        cells[soft as usize][10] as usize,
        at(21, 20),
        "no armour, so it steps"
    );

    let mut w = still_world_with(32, &[("armour_on", "true")]);
    let rest = budget(&[(REST, 60), (DIR_X, 60)]);
    let womb = w.cfg.f("womb");
    let s = w.plant(20, 20, 8.0, womb + 0.5, 0, &fixed_genome(2, rest));
    set_cell_word(&w, s, 4, 0.9);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let kid = (0..cells.len())
        .find(|k| cells[*k][10] != NONE && *k != s as usize)
        .unwrap();
    assert_eq!(f32_at(&cells[kid], 4), 0.0, "a child starts soft");
    assert!(
        f32_at(&cells[s as usize], 4) > 0.8,
        "the parent keeps its own"
    );
}

#[test]
fn armour_action_and_input_do_nothing_when_off() {
    // Most of the budget asks for the armour and clean actions, which don't exist yet.
    let mut w = still_world(32);
    let want = budget(&[(ARMOUR, 90), (CLEAN, 90), (GROUND, 40), (DIR_X, 60)]);
    let (x, y) = (10u32, 10u32);
    w.write_field("fa", x, y, [3.0, 0.0, 0.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &fixed_genome(2, want));
    w.step();
    w.flush();
    let plain = spend(&w, outputs(want), 8.0, 0.0, 3.0, 0.0);
    let rec = &w.read_cells()[s as usize];
    close(f32_at(rec, 0), plain.e, "energy");
    assert_eq!(f32_at(rec, 4), 0.0, "no hardness");
    check_input(&[("armour_on", "true")], &[], (1, 3), |w, s| {
        set_cell_word(w, s, 4, 0.9)
    });
}

#[test]
fn energy_adds_up_with_armour() {
    strict_ledger(&[
        ("armour_on", "true"),
        ("armour_build", "6"),
        ("armour_upkeep", "0.2"),
    ]);
}

// ---- pack ----

#[test]
fn wounds_make_later_bites_worse() {
    check_bite(&[("pack_on", "true")], 0.0, 0.0, 0.0);
    check_bite(&[("pack_on", "true")], 0.0, 0.0, 0.5);
    check_bite(&[("pack_on", "true"), ("armour_on", "true")], 0.0, 0.5, 0.5);
}

#[test]
fn energy_adds_up_with_wounds() {
    strict_ledger(&[("pack_on", "true"), ("pack_intact", "0.5")]);
}

// ---- waste ----

const WASTE_STILL: [(&str, &str); 5] = [
    ("waste_on", "true"),
    ("waste_clean", "true"),
    ("waste_holds", "0.2"),
    ("waste_flow", "0"),
    ("waste_decay", "0"),
];

#[test]
fn waste_matches_plain_rule() {
    let mut w = still_world_with(32, &WASTE_STILL);
    let want = budget(&[
        (GROUND, 40),
        (MEAT, 30),
        (GROW, 20),
        (CLEAN, 50),
        (DIR_X, 60),
    ]);
    let (x, y) = (10u32, 10u32);
    let waste = 2.0f32;
    w.write_field("fa", x, y, [3.0, 2.0, 0.0, 0.0]);
    w.write_field("fc", x, y, [0.0, waste, 0.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &fixed_genome(2, want));
    w.step();
    w.flush();
    let c = &w.cfg;
    let (deposit, holds) = (c.f("waste_deposit"), c.f("waste_holds"));
    let plain = spend(
        &w,
        outputs(want),
        8.0 - waste * c.f("waste_harm"),
        0.0,
        3.0,
        2.0,
    );
    let meal = plain.ate_ground + plain.ate_meat * c.f("meat_yield");
    let cleaned = waste * (plain.work[CLEAN] * c.f("waste_rate")).min(1.0);
    let e = plain.e - meal * holds + cleaned * holds / deposit * c.f("waste_yield");
    close(f32_at(&w.read_cells()[s as usize], 0), e, "energy");
    close(
        w.read_vec4s("fc")[at(x, y)][1],
        waste + meal * deposit - cleaned,
        "waste left",
    );
}

#[test]
fn waste_from_a_bite_stays_with_the_biter() {
    check_bite(&WASTE_STILL, 0.0, 0.0, 0.0);
}

#[test]
fn waste_can_kill_and_counts_as_burned() {
    let mut w = still_world_with(
        32,
        &[
            ("waste_on", "true"),
            ("waste_flow", "0"),
            ("waste_decay", "0"),
        ],
    );
    let want = budget(&[(REST, 60), (DIR_X, 60)]);
    w.write_field("fc", 9, 9, [0.0, 1000.0, 0.0, 0.0]);
    let s = w.plant(9, 9, 2.0, 0.0, 0, &fixed_genome(2, want));
    w.step();
    w.step();
    w.flush();
    assert_eq!(w.read_cells()[s as usize][10], NONE, "dead");
    assert_eq!(w.counters()[7], 1, "counted as burned");
}

#[test]
fn waste_input_is_zero_when_off() {
    let set = |w: &mut World, _: u32| w.write_field("fc", 10, 10, [0.0, 3.0, 0.0, 0.0]);
    check_input(&WASTE_STILL, &[], (12, 0), set);
}

#[test]
fn energy_adds_up_with_waste() {
    strict_ledger(&[
        ("waste_on", "true"),
        ("waste_harm", "0.1"),
        ("waste_deposit", "0.2"),
    ]);
}

#[test]
fn energy_adds_up_with_waste_that_holds_energy() {
    strict_ledger(&[
        ("waste_on", "true"),
        ("waste_clean", "true"),
        ("waste_holds", "0.5"),
        ("waste_harm", "0.05"),
    ]);
}

// ---- plants ----

/// A world where only the sun acts, so the growth rule shows directly.
fn sunny_world(more: &[(&str, &str)]) -> World {
    let mut sets = vec![("sun", "0.05")];
    sets.extend_from_slice(more);
    still_world_with(32, &sets)
}

fn sun_of(w: &World, t: u32) -> f32 {
    (t & 0xFFFF) as f32 / 65535.0 * 8.0 * w.sun_k
}

#[test]
fn plants_grow_like_plain_rule() {
    let mut w = sunny_world(&[
        ("plants_on", "true"),
        ("plants_peak", "1.5"),
        ("plants_flow", "0"),
    ]);
    let cap = w.cfg.f("ground_cap");
    let (peak, trickle) = (w.cfg.f("plants_peak"), w.cfg.f("plants_trickle"));
    let squares: [(u32, u32, f32); 5] = [
        (3, 3, 0.0),
        (5, 3, 0.2),
        (7, 3, 2.0),
        (9, 3, 3.9),
        (11, 3, 4.0),
    ];
    for (x, y, v) in squares {
        w.write_field("fa", x, y, [v, 0.0, 0.0, 0.0]);
    }
    let terrain = w.read_u32s("terrain");
    w.step();
    w.flush();
    let fa = w.read_vec4s("fa");
    for (x, y, v) in squares {
        let p = at(x, y);
        let d = (v / cap).clamp(0.0, 1.0);
        let light = sun_of(&w, terrain[p]) * (peak * 4.0 * d + trickle) * (1.0 - d);
        close(fa[p][0], v + light.min((cap - v).max(0.0)), "ground");
    }
    assert_eq!(fa[at(11, 3)][0], cap, "nothing grows on full ground");
}

/// One tick of pair trading over the whole grid, as the sweep does it.
fn pair_trade(v: &mut [f32], n: usize, flow: f32, off: usize) {
    for j in 0..n {
        for i in (off..n).step_by(2) {
            let (a, b) = (j * n + i, j * n + (i + 1) % n);
            let f = flow * (v[a] - v[b]);
            v[a] -= f;
            v[b] += f;
        }
    }
    for j in (off..n).step_by(2) {
        for i in 0..n {
            let (a, b) = (j * n + i, ((j + 1) % n) * n + i);
            let f = flow * (v[a] - v[b]);
            v[a] -= f;
            v[b] += f;
        }
    }
}

#[test]
fn plants_spread_by_pair_trading_and_keep_the_total() {
    let flow = 0.05f32;
    let mut w = still_world_with(32, &[("plants_on", "true"), ("plants_flow", "0.05")]);
    let mut plain = vec![0.0f32; W32 * W32];
    for (x, y, v) in [(4u32, 4u32, 3.0f32), (5, 9, 1.0), (20, 20, 2.0)] {
        w.write_field("fa", x, y, [v, 0.0, 0.0, 0.0]);
        plain[at(x, y)] = v;
    }
    for tick in 1..=6usize {
        w.step();
        pair_trade(&mut plain, W32, flow, tick & 1);
    }
    w.flush();
    let fa = w.read_vec4s("fa");
    for p in 0..W32 * W32 {
        close(fa[p][0], plain[p], "ground");
    }
    let total: f32 = fa.iter().map(|v| v[0]).sum();
    close(total, 6.0, "total food");
}

#[test]
fn energy_adds_up_with_plants() {
    strict_ledger(&[
        ("plants_on", "true"),
        ("plants_peak", "3"),
        ("plants_flow", "0.1"),
    ]);
}

// ---- weather ----

#[test]
fn drifting_sun_matches_plain_rule() {
    let mut w = sunny_world(&[
        ("weather_sun_on", "true"),
        ("weather_drift", "1.25"),
        ("ground_cap", "1000"),
    ]);
    let terrain = w.read_u32s("terrain");
    let before = w.read_vec4s("fa");
    w.step();
    w.flush();
    let after = w.read_vec4s("fa");
    // The first tick is tick 1, so the map has slid by the drift once.
    let (sx, sy) = (1.25f32, 1.25f32 * 0.618034);
    let (dx, fx, dy, fy) = (sx as usize, sx.fract(), sy as usize, sy.fract());
    let (mut gained, mut total) = (0.0f32, 0.0f32);
    for y in 0..W32 {
        for x in 0..W32 {
            let sun = |a: usize, b: usize| {
                sun_of(&w, terrain[(y + dy + b) % W32 * W32 + (x + dx + a) % W32])
            };
            let near = sun(0, 0) * (1.0 - fx) + sun(1, 0) * fx;
            let far = sun(0, 1) * (1.0 - fx) + sun(1, 1) * fx;
            let p = y * W32 + x;
            close(
                after[p][0] - before[p][0],
                near * (1.0 - fy) + far * fy,
                "sun landed",
            );
            gained += after[p][0] - before[p][0];
            total += sun_of(&w, terrain[p]);
        }
    }
    assert!(
        (gained - total).abs() < total * 1e-3,
        "sliding keeps the total: {gained} and {total}"
    );
}

#[derive(Clone)]
struct FireGrid {
    ground: Vec<f32>,
    flame: Vec<f32>,
    heat: Vec<f32>,
    rock: Vec<bool>,
    burned: f32,
}

/// The fire rule on the whole grid, with spreading made certain, one sweep at a time.
fn fire_sweep(
    g: &mut FireGrid,
    fuel: f32,
    burn: f32,
    fade: f32,
    heat: f32,
    rest: f32,
    pairs: Vec<(usize, usize)>,
) {
    let next = |own: f32, other: f32, food: f32| -> f32 {
        if own > 0.0 {
            let f = own * fade;
            return if f >= 0.05 { f } else { -1.0 };
        }
        if own < 0.0 {
            return (own + 0.5 / rest).min(0.0);
        }
        if food < fuel {
            return 0.0;
        }
        if other > 0.0 { 1.0 } else { 0.0 }
    };
    for (a, b) in pairs {
        if g.rock[a] && g.rock[b] {
            continue;
        }
        let (fa, fb) = (g.flame[a], g.flame[b]);
        if !g.rock[a] {
            g.flame[a] = next(fa, if g.rock[b] { 0.0 } else { fb }, g.ground[a]);
        }
        if !g.rock[b] {
            g.flame[b] = next(fb, if g.rock[a] { 0.0 } else { fa }, g.ground[b]);
        }
        for k in [a, b] {
            if g.flame[k] > 0.0 {
                let c = g.ground[k] * burn;
                g.ground[k] -= c;
                g.burned += c;
                g.heat[k] += g.flame[k] * heat;
            }
        }
    }
}

#[test]
fn fire_matches_plain_rule_and_counts_what_it_burns() {
    let n = W32;
    let mut w = still_world_with(
        32,
        &[
            ("ledger", "true"),
            ("weather_fire_on", "true"),
            ("weather_fire_rate", "0"),
            ("weather_fire_spread", "2"),
            ("weather_fire_rest", "6"),
        ],
    );
    let c = &w.cfg;
    let (fuel, burn, fade, heat, rest) = (
        c.f("weather_fire_fuel"),
        c.f("weather_fire_burn"),
        c.f("weather_fire_fade"),
        c.f("weather_fire_heat"),
        c.f("weather_fire_rest"),
    );
    // Thick ground everywhere, with a band of thin ground and a band of rock across it.
    let mut g = FireGrid {
        ground: vec![4.0; n * n],
        flame: vec![0.0; n * n],
        heat: vec![0.0; n * n],
        rock: vec![false; n * n],
        burned: 0.0,
    };
    let mut terrain = w.read_u32s("terrain");
    for x in 0..n {
        g.ground[8 * n + x] = fuel - 0.5;
        g.ground[9 * n + x] = fuel - 0.5;
        for y in [24, 25] {
            g.rock[y * n + x] = true;
            terrain[y * n + x] |= 1 << 24;
            g.ground[y * n + x] = 0.0;
        }
    }
    w.gpu
        .queue
        .write_buffer(&w.buffers["terrain"], 0, bytemuck::cast_slice(&terrain));
    for p in 0..n * n {
        let v = [g.ground[p], 0.0, 0.0, 0.0];
        w.write_field("fa", (p % n) as u32, (p / n) as u32, v);
    }
    g.flame[16 * n + 16] = 1.0;
    w.write_field("fc", 16, 16, [0.0, 0.0, 1.0, 0.0]);
    for tick in 1..=40usize {
        w.step();
        let off = tick & 1;
        let across: Vec<(usize, usize)> = (0..n)
            .flat_map(|j| {
                (off..n)
                    .step_by(2)
                    .map(move |i| (j * n + i, j * n + (i + 1) % n))
            })
            .collect();
        let down: Vec<(usize, usize)> = (off..n)
            .step_by(2)
            .flat_map(|j| (0..n).map(move |i| (j * n + i, ((j + 1) % n) * n + i)))
            .collect();
        fire_sweep(&mut g, fuel, burn, fade, heat, rest, across);
        fire_sweep(&mut g, fuel, burn, fade, heat, rest, down);
    }
    w.flush();
    let (fa, fc) = (w.read_vec4s("fa"), w.read_vec4s("fc"));
    for p in 0..n * n {
        if g.rock[p] {
            continue;
        }
        close(fc[p][2], g.flame[p], "flame");
        close(fa[p][0], g.ground[p], "ground");
        close(fa[p][2], g.heat[p], "heat");
    }
    assert!(g.burned > 100.0, "a real fire, burned {}", g.burned);
    assert!(g.ground[20 * n + 3] < 0.5, "it crossed the thick ground");
    assert_eq!(g.ground[8 * n + 3], fuel - 0.5, "thin ground doesn't burn");
    assert_eq!(
        g.ground[3 * n + 3],
        4.0,
        "and stops it, as rock does the other way"
    );
    let lost = w.counters()[10] as f32 / 4096.0;
    assert!(
        (lost - g.burned).abs() < g.burned * 0.01,
        "ledger {lost}, burned {}",
        g.burned
    );
}

#[test]
fn a_strange_rest_setting_still_compiles() {
    let w = still_world_with(
        32,
        &[("weather_fire_on", "true"), ("weather_fire_rest", "0")],
    );
    drop(w);
}

#[test]
fn energy_adds_up_with_weather() {
    strict_ledger(&[
        ("weather_sun_on", "true"),
        ("weather_fire_on", "true"),
        ("weather_fire_rate", "0.00001"),
        ("weather_fire_fuel", "1"),
        ("weather_fire_rest", "200"),
    ]);
}

// ---- mating ----

#[test]
fn a_mate_pays_for_an_early_birth_and_energy_only_moves() {
    let mut w = still_world_with(
        32,
        &[
            ("mate_pay", "0.6"),
            ("bite_on", "false"),
            ("give_on", "false"),
            ("heat_on", "false"),
            ("share_on", "false"),
            ("move_rate", "0"),
        ],
    );
    let want = budget(&[(REST, 60), (DIR_X, 60), (MATE, 40)]);
    let womb = w.cfg.f("womb");
    let price = 0.6 * womb;
    let g = fixed_genome(2, want);
    let (a, b) = (
        w.plant(8, 24, 8.0, 0.0, 0, &g),
        w.plant(9, 24, 8.0, 0.0, 0, &g),
    );
    // Both say yes on the first tick. Then one can't afford to be a mate, and can start a birth.
    w.step();
    w.flush();
    let before = w.read_cells();
    let (wa, wb) = (womb * 0.5, price + 0.4);
    assert!(wa >= womb * 0.4 && wa < price);
    set_cell_word(&w, a, 1, wa);
    set_cell_word(&w, b, 1, wb);
    w.step();
    w.flush();
    let c = w.counters();
    assert_eq!(c[3], 1, "one birth");
    assert_eq!(c[4], 1, "and it's mixed");
    let cells = w.read_cells();
    let (ea, eb) = (
        f32_at(&before[a as usize], 0),
        f32_at(&before[b as usize], 0),
    );
    let (pa, pb) = (
        spend(&w, outputs(want), ea, wa, 0.0, 0.0),
        spend(&w, outputs(want), eb, wb, 0.0, 0.0),
    );
    let kid = (0..cells.len())
        .find(|k| cells[*k][10] != NONE && *k != a as usize && *k != b as usize)
        .unwrap();
    close(
        f32_at(&cells[kid], 0),
        pa.womb + price - w.cfg.f("body"),
        "the child gets both shares",
    );
    close(
        f32_at(&cells[a as usize], 1),
        0.0,
        "the first womb is empty",
    );
    close(
        f32_at(&cells[b as usize], 1),
        pb.womb - price,
        "the mate paid its share",
    );
    close(f32_at(&cells[a as usize], 0), pa.e, "no other energy moved");
    close(f32_at(&cells[b as usize], 0), pb.e, "no other energy moved");
}

#[test]
fn nobody_gives_birth_early_alone_or_with_a_mate_that_cannot_pay() {
    let mut w = still_world_with(32, &[("mate_pay", "0.6"), ("move_rate", "0")]);
    let want = budget(&[(REST, 60), (DIR_X, 60), (MATE, 40)]);
    let womb = w.cfg.f("womb");
    let g = fixed_genome(2, want);
    let alone = w.plant(4, 4, 8.0, womb * 0.5, 0, &g);
    let (a, b) = (
        w.plant(20, 20, 8.0, 0.0, 0, &g),
        w.plant(21, 20, 8.0, 0.0, 0, &g),
    );
    w.step();
    w.flush();
    set_cell_word(&w, alone, 1, womb * 0.5);
    set_cell_word(&w, a, 1, womb * 0.5);
    set_cell_word(&w, b, 1, womb * 0.5);
    w.step();
    w.flush();
    assert_eq!(w.counters()[3], 0, "no births");
}

#[test]
fn energy_adds_up_with_paying_mates() {
    strict_ledger(&[("mate_pay", "0.5")]);
}

// ---- skills ----

#[test]
fn practice_matches_plain_rule() {
    let (naive, learn, fade) = (0.5f32, 0.1f32, 0.01f32);
    let mut w = still_world_with(
        32,
        &[
            ("skill_on", "true"),
            ("skill_naive", "0.5"),
            ("skill_learn", "0.1"),
            ("skill_fade", "0.01"),
        ],
    );
    let want = budget(&[(GROUND, 40), (MEAT, 30), (GROW, 20), (DIR_X, 60)]);
    let (x, y) = (10u32, 10u32);
    w.write_field("fa", x, y, [3.0, 2.0, 0.0, 0.0]);
    let s = w.plant(x, y, 8.0, 0.0, 0, &fixed_genome(2, want));
    w.step();
    w.flush();
    let plain = spend(&w, outputs(want), 8.0, 0.0, 3.0, 2.0);
    let cells = w.read_cells();
    let rec = &cells[s as usize];
    // A new cell has no practice, so it keeps only the naive share of what it eats.
    let wasted =
        plain.ate_ground * (1.0 - naive) + plain.ate_meat * w.cfg.f("meat_yield") * (1.0 - naive);
    close(f32_at(rec, 0), plain.e - wasted, "energy");
    // The food still leaves the ground in full.
    let fa = w.read_vec4s("fa");
    close(fa[at(x, y)][0], 3.0 - plain.ate_ground, "ground left");
    close(fa[at(x, y)][1], 2.0 - plain.ate_meat, "meat left");
    // Practice moves toward the mix of what the cell did, and then fades.
    let (did_ground, did_flesh) = (
        plain.a[GROUND] / w.cfg.f("spend"),
        plain.a[MEAT] / w.cfg.f("spend"),
    );
    let did = did_ground + did_flesh;
    let rate = learn * did.min(1.0);
    close(
        f32_at(rec, 6),
        did_ground / did * rate * (1.0 - fade),
        "ground practice",
    );
    close(
        f32_at(rec, 7),
        did_flesh / did * rate * (1.0 - fade),
        "flesh practice",
    );
}

#[test]
fn a_bite_needs_practice_too() {
    check_bite(
        &[
            ("skill_on", "true"),
            ("skill_naive", "0.5"),
            ("skill_learn", "0"),
        ],
        0.0,
        0.0,
        0.0,
    );
}

#[test]
fn practice_is_inherited_only_as_far_as_the_setting_says() {
    for (inherit, share) in [("0", 0.0f32), ("1", 1.0)] {
        let mut w = still_world_with(
            32,
            &[
                ("skill_on", "true"),
                ("skill_inherit", inherit),
                ("skill_fade", "0"),
            ],
        );
        let rest = budget(&[(REST, 60), (DIR_X, 60)]);
        let womb = w.cfg.f("womb");
        let s = w.plant(20, 20, 8.0, womb + 0.5, 0, &fixed_genome(2, rest));
        set_cell_word(&w, s, 6, 0.7);
        set_cell_word(&w, s, 7, 0.2);
        w.step();
        w.flush();
        let cells = w.read_cells();
        let kid = (0..cells.len())
            .find(|k| cells[*k][10] != NONE && *k != s as usize)
            .unwrap();
        close(
            f32_at(&cells[kid], 6),
            0.7 * share,
            "child's ground practice",
        );
        close(
            f32_at(&cells[kid], 7),
            0.2 * share,
            "child's flesh practice",
        );
    }
}

#[test]
fn practice_input_is_zero_when_off() {
    let set = |w: &mut World, s: u32| {
        set_cell_word(w, s, 6, 0.9);
        set_cell_word(w, s, 7, 0.9);
    };
    let on = [("skill_on", "true"), ("skill_sense", "true")];
    check_input(&on, &[("skill_on", "true")], (14, 0), set);
    check_input(&on, &[("skill_on", "true")], (14, 1), set);
}

#[test]
fn energy_adds_up_with_practice() {
    strict_ledger(&[("skill_on", "true"), ("skill_naive", "0.8")]);
}

// ---- biomes ----

fn place_of(terrain: u32) -> usize {
    let room = terrain >> 28;
    (((room & 3) + (room >> 2)) & 3) as usize
}

const KINDS: [&str; 4] = ["cold", "hot", "swamp", "barren"];

#[test]
fn each_kind_of_place_sets_tempo_like_plain_rule() {
    let mut w = still_world_with(
        32,
        &[
            ("biomes_on", "true"),
            ("bite_on", "false"),
            ("heat_on", "false"),
        ],
    );
    let want = budget(&[(GROUND, 40), (GROW, 30), (DIR_X, -60)]);
    let terrain = w.read_u32s("terrain");
    let mut planted = vec![];
    for (y, x) in [
        (4u32, 4u32),
        (4, 12),
        (4, 20),
        (4, 28),
        (12, 4),
        (20, 4),
        (28, 4),
    ] {
        let p = at(x, y);
        planted.push((
            w.plant(x, y, 8.0, 0.0, 0, &fixed_genome(2, want)),
            place_of(terrain[p]),
        ));
    }
    let kinds: std::collections::HashSet<usize> = planted.iter().map(|k| k.1).collect();
    assert!(
        kinds.len() >= 3,
        "this test should cover several kinds of place"
    );
    w.step();
    w.flush();
    let cells = w.read_cells_full();
    for (s, place) in planted {
        let tempo = w.cfg.f(&format!("tempo_{}", KINDS[place]));
        let drag = w.cfg.f(&format!("drag_{}", KINDS[place]));
        let plain = spend_with(&w, outputs(want), 8.0, 0.0, 0.0, 0.0, 0.0, tempo);
        let rec = &cells[s as usize];
        close(f32::from_bits(rec[0]), plain.e, "energy");
        close(f32::from_bits(rec[1]), plain.womb, "womb");
        let upkeep = plain.upkeep;
        close(
            f32::from_bits(rec[16]),
            upkeep / (w.cfg.f("upkeep") + upkeep),
            "felt upkeep",
        );
        close(
            f32::from_bits(rec[17]),
            (drag - 1.0) / (1.0 + drag - 1.0),
            "felt drag",
        );
    }
}

#[test]
fn drag_blocks_steps_and_rot_follows_the_place() {
    // So much drag that nobody can step, where a cell always steps with none.
    let walk = budget(&[(MOVE, 60), (DIR_X, 60)]);
    let drags = [
        ("drag_cold", "1000000"),
        ("drag_hot", "1000000"),
        ("drag_swamp", "1000000"),
        ("drag_barren", "1000000"),
    ];
    let mut sets = vec![("biomes_on", "true"), ("bite_on", "false")];
    sets.extend_from_slice(&drags);
    let mut w = still_world_with(32, &sets);
    let s = w.plant(8, 8, 8.0, 0.0, 0, &fixed_genome(2, walk));
    w.step();
    w.flush();
    assert_eq!(w.read_cells()[s as usize][10] as usize, at(8, 8));

    let mut w = still_world_with(32, &[("biomes_on", "true"), ("meat_rot", "0.01")]);
    let terrain = w.read_u32s("terrain");
    let squares = [(4u32, 4u32), (4, 12), (12, 4), (12, 12), (20, 20)];
    for (x, y) in squares {
        w.write_field("fa", x, y, [0.0, 2.0, 0.0, 0.0]);
    }
    w.step();
    w.flush();
    let fa = w.read_vec4s("fa");
    for (x, y) in squares {
        let p = at(x, y);
        let rotx = w.cfg.f(&format!("rotx_{}", KINDS[place_of(terrain[p])]));
        let rot = 2.0 * 0.01 * rotx;
        close(fa[p][1], 2.0 - rot, "meat");
        close(fa[p][0], rot, "ground from rot");
    }
}

/// How many regions of open squares a world falls into, biggest first.
fn regions(occ: &[u32], n: usize) -> Vec<usize> {
    let mut seen = vec![false; occ.len()];
    let mut out = vec![];
    for start in 0..occ.len() {
        if seen[start] || occ[start] == ROCK {
            continue;
        }
        let (mut stack, mut size) = (vec![start], 0);
        seen[start] = true;
        while let Some(p) = stack.pop() {
            size += 1;
            let (x, y) = (p % n, p / n);
            for q in [
                y * n + (x + 1) % n,
                y * n + (x + n - 1) % n,
                (y + 1) % n * n + x,
                (y + n - 1) % n * n + x,
            ] {
                if !seen[q] && occ[q] != ROCK {
                    seen[q] = true;
                    stack.push(q);
                }
            }
        }
        out.push(size);
    }
    out.sort_by(|a, b| b.cmp(a));
    out
}

#[test]
fn walls_cut_rooms_and_passes_join_them() {
    let world = |pass: &str| {
        let w = world_with(
            128,
            11,
            &[
                ("rock", "false"),
                ("walls_on", "true"),
                ("wall_thick", "4"),
                ("wall_pass", pass),
            ],
        );
        regions(&w.read_u32s("occ"), 128)
    };
    let sealed = world("0");
    let big = sealed.iter().filter(|s| **s > 200).count();
    assert_eq!(big, 16, "sixteen sealed rooms: {sealed:?}");
    let open = world("12");
    assert_eq!(
        open.iter().filter(|s| **s > 200).count(),
        1,
        "rooms joined by passes: {open:?}"
    );
}

#[test]
fn sun_differs_by_kind_of_place() {
    let w = world_with(128, 11, &[("rock", "false"), ("biomes_on", "true")]);
    let mut sum = [0.0f64; 4];
    let mut count = [0.0f64; 4];
    for t in w.read_u32s("terrain") {
        sum[place_of(t)] += (t & 0xFFFF) as f64;
        count[place_of(t)] += 1.0;
    }
    let mean = |k: usize| sum[k] / count[k].max(1.0);
    assert!(
        mean(1) > mean(0) * 1.5,
        "hot gets more sun than cold: {sum:?} {count:?}"
    );
    assert!(
        mean(0) > mean(3) * 1.5,
        "cold gets more than barren: {sum:?} {count:?}"
    );
}

#[test]
fn felt_inputs_are_zero_when_off() {
    let set = |w: &mut World, s: u32| {
        set_cell_word(w, s, 16, 0.9);
        set_cell_word(w, s, 17, 0.9);
    };
    let off = [("biomes_on", "true"), ("biome_feel", "false")];
    check_input(&[("biomes_on", "true")], &off, (14, 2), set);
    check_input(&[("biomes_on", "true")], &off, (14, 3), set);
}

#[test]
fn energy_adds_up_with_biomes_and_walls() {
    strict_ledger(&[
        ("biomes_on", "true"),
        ("walls_on", "true"),
        ("wall_pass", "8"),
        ("wall_thick", "2"),
    ]);
}

// ---- blooms and roads ----

#[test]
fn deep_piles_spoil_and_cannot_be_swallowed_whole() {
    let mut w = still_world_with(
        32,
        &[
            ("ledger", "true"),
            ("bloom_share", "0.3"),
            ("bloom_fade", "0.1"),
        ],
    );
    let cap = w.cfg.f("ground_cap");
    let want = budget(&[(GROUND, 60), (DIR_X, -60)]);
    w.write_field("fa", 4, 4, [cap + 10.0, 0.0, 0.0, 0.0]);
    w.write_field("fa", 20, 20, [cap + 10.0, 0.0, 0.0, 0.0]);
    let s = w.plant(20, 20, 8.0, 0.0, 0, &fixed_genome(2, want));
    w.step();
    w.flush();
    let fa = w.read_vec4s("fa");
    close(fa[at(4, 4)][0], cap + 9.0, "a tenth of the excess spoiled");
    let plain = spend(&w, outputs(want), 8.0, 0.0, cap, 0.0);
    assert!(plain.ate_ground > 0.0);
    close(
        f32_at(&w.read_cells()[s as usize], 0),
        plain.e,
        "ate from the cap, not the pile",
    );
    let left = cap + 10.0 - plain.ate_ground;
    close(fa[at(20, 20)][0], left - (left - cap) * 0.1, "ground left");
}

#[test]
fn blooms_land_the_share_of_sun_they_took() {
    let (every, ticks) = (50usize, 400usize);
    let mut w = world_with(
        64,
        6,
        &[
            ("rock", "false"),
            ("season_ticks", "0"),
            ("rain_p", "0"),
            ("rain_burst_ticks", "0"),
            ("ground_cap", "100000"),
            ("meat_rot", "0"),
            ("bloom_share", "0.5"),
            ("bloom_every", "50"),
            ("bloom_fade", "0"),
            ("ledger", "true"),
        ],
    );
    let start: f64 = w.read_vec4s("fa").iter().map(|v| v[0] as f64).sum();
    for _ in 0..ticks {
        w.step();
    }
    w.flush();
    let end: f64 = w.read_vec4s("fa").iter().map(|v| v[0] as f64).sum();
    let gained = end - start;
    let want = 0.05 * 64.0 * 64.0 * ticks as f64;
    assert!(
        (gained - want).abs() < want * 0.05,
        "gained {gained}, baseline sun {want}, every {every}"
    );
    let counted = w.counters()[11] as f64 / 4096.0;
    assert!(
        (counted - gained).abs() < gained * 0.005,
        "ledger {counted}, gained {gained}"
    );
}

#[test]
fn tracks_wear_in_and_make_steps_easier() {
    // Steps happen with this chance on bare ground. Worn ground multiplies it.
    let sets = [
        ("roads_on", "true"),
        ("track_flow", "0"),
        ("track_decay", "0"),
        ("move_rate", "3"),
        ("bite_on", "false"),
    ];
    let mut w = still_world_with(64, &sets);
    let walk = budget(&[(MOVE, 60), (DIR_X, 60)]);
    let road = 3.0f32;
    let (mut worn, mut bare) = (vec![], vec![]);
    for y in (0..64u32).step_by(8) {
        for x in (0..64u32).step_by(8) {
            let g = fixed_genome(2, walk);
            let s = w.plant(x, y, 8.0, 0.0, 0, &g);
            if y < 32 {
                w.write_field("fd", x + 1, y, [road, 0.0, 0.0, 0.0]);
                worn.push(s);
            } else {
                bare.push(s);
            }
        }
    }
    let plain = spend(&w, outputs(walk), 8.0, 0.0, 0.0, 0.0);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let fd = w.read_vec4s("fd");
    let n = 64usize;
    let moved = |list: &Vec<u32>| -> f32 {
        let mut count = 0.0;
        for s in list {
            let p = cells[*s as usize][10] as usize;
            let went = fd[p][0] > 0.0 && (p % n) % 8 == 1;
            if went {
                count += 1.0;
                let was = if (p / n) < 32 { road } else { 0.0 };
                close(
                    fd[p][0],
                    was + w.cfg.f("track_lay"),
                    "a step wears the square it lands on",
                );
            }
        }
        count
    };
    let c = &w.cfg;
    let base = plain.work[MOVE] * c.f("move_rate");
    let ease = 1.0
        + c.f("road_ease") * road * road / (road * road + c.f("track_half") * c.f("track_half"));
    assert!(
        base < 0.5 && base * ease > 0.5,
        "this test needs the road to matter: {base}, {ease}"
    );
    let (mw, mb) = (moved(&worn), moved(&bare));
    let expect =
        |chance: f32, k: f32| (k * chance, 4.0 * (k * chance * (1.0 - chance)).sqrt() + 1.0);
    let (ew, sw) = expect((base * ease).min(1.0), worn.len() as f32);
    let (eb, sb) = expect(base, bare.len() as f32);
    assert!(
        (mw - ew).abs() < sw,
        "{mw} moved on worn ground, expected {ew}"
    );
    assert!(
        (mb - eb).abs() < sb,
        "{mb} moved on bare ground, expected {eb}"
    );
}

#[test]
fn track_input_is_zero_when_off() {
    let set = |w: &mut World, _: u32| w.write_field("fd", 10, 10, [3.0, 0.0, 0.0, 0.0]);
    check_input(&[("roads_on", "true")], &[], (13, 0), set);
}

#[test]
fn energy_adds_up_with_blooms_and_roads() {
    strict_ledger(&[
        ("bloom_share", "0.5"),
        ("bloom_fade", "0.05"),
        ("roads_on", "true"),
    ]);
}

// ---- the spare parts ----

#[test]
fn spare_actions_stay_off_until_switched_on_and_then_cost_energy() {
    let run = |sets: &[(&str, &str)]| {
        let mut w = still_world_with(32, sets);
        let want = budget(&[(GROUND, 20), (DIR_X, 60)]);
        let mut g = fixed_genome(2, want);
        // Output row 6, its x0 action, read from hidden unit 0.
        let ho = 2 * IN_BLOCKS as usize + 2 * 2;
        g[(ho + 6 * 2) * 16] = 100;
        let s = w.plant(10, 10, 8.0, 0.0, 0, &g);
        w.step();
        w.flush();
        (
            f32_at(&w.read_cells()[s as usize], 0),
            spend(&w, outputs(want), 8.0, 0.0, 0.0, 0.0).e,
        )
    };
    let (off, plain) = run(&[]);
    close(off, plain, "off, an action that nobody can spend on");
    let (on, _) = run(&[("x0_on", "true")]);
    assert!(
        on < off - 0.01 || on > off + 0.01,
        "on, it changes the budget: {on} and {off}"
    );
}

// ---- together ----

#[test]
fn energy_adds_up_with_every_rule_on() {
    energy_gap_under(ALL_RULES, 3000, 0.002);
}

#[test]
fn energy_adds_up_with_every_rule_on_and_rich_ground() {
    let mut sets = ALL_RULES.to_vec();
    sets.extend_from_slice(&[
        ("sun", "0.3"),
        ("hazard", "0.002"),
        ("bite_yield", "0.3"),
        ("waste_harm", "0.01"),
        ("skill_naive", "0.95"),
        // Amounts under half a ledger unit round to nothing, and at the usual scale that's
        // 0.4% of this world. This scale runs out at 131,072 units, and this world uses a quarter.
        ("ledger_scale", "32768"),
    ]);
    energy_gap_under(&sets, 3000, 0.0005);
}

#[test]
fn grid_and_cells_agree_with_every_rule_on() {
    assert_grid_and_cells_agree(ALL_RULES);
}

#[test]
fn every_rule_on_is_the_same_every_time() {
    let run = || {
        let mut w = world_with(64, 9, ALL_RULES);
        for _ in 0..400 {
            w.step();
        }
        w.flush();
        fingerprint(&w)
    };
    assert_eq!(run(), run());
}

#[test]
fn every_rule_on_saves_and_continues_the_same() {
    let dir = std::env::temp_dir().join(format!("life-rules-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("world.bin").to_string_lossy().to_string();
    let mut a = world_with(64, 21, ALL_RULES);
    for _ in 0..200 {
        a.step();
    }
    a.save(&path).unwrap();
    for _ in 0..100 {
        a.step();
    }
    a.flush();
    let want = fingerprint(&a);
    drop(a);
    let mut b = world_with(64, 21, ALL_RULES);
    b.load(&path).unwrap();
    for _ in 0..100 {
        b.step();
    }
    b.flush();
    assert_eq!(fingerprint(&b), want);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_old_saved_world_is_refused_with_a_clear_message() {
    let dir = std::env::temp_dir().join(format!("life-old-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("old.bin").to_string_lossy().to_string();
    let head = br#"{"size":32,"slots":1024,"hidden_rows":2,"shard_slots":2000000,"tick":5}"#;
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(head);
    std::fs::write(&path, bytes).unwrap();
    let mut w = still_world(32);
    let why = w.load(&path).unwrap_err();
    assert!(why.contains("older layout"), "{why}");
    assert!(why.contains("Rerun"), "{why}");
    let why = w
        .seed_from(std::slice::from_ref(&path), 1.0)
        .map(|_| ())
        .unwrap_err();
    assert!(why.contains("older layout"), "{why}");
    std::fs::remove_dir_all(&dir).unwrap();
}
