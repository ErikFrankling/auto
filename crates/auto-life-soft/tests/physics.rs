//! One-tick checks of each rule against the same rule written as plain Rust, and whole-world
//! checks of the ledger and of repeating exactly. They run on whatever Vulkan device is there.

mod common;

use auto_life_soft::config::E_ONE;
use auto_life_soft::world::F_ALIVE;
use common::*;

fn radius(w: &auto_life_soft::World, e: f32) -> f32 {
    (w.cfg.f("r0") * (1.0 + e / w.cfg.f("e_size")).sqrt()).min(w.cfg.f("r_max"))
}

#[test]
fn crowding_matches_plain_rule() {
    let mut w = still_world(32, &[]);
    let g = genome(&[(REST, 127)]);
    let r = radius(&w, 4.0);
    let d = 2.0 * r - 0.3;
    let a = w.plant(10.0, 10.0, 4.0, 0.0, 0.0, &g);
    let b = w.plant(10.0 + d, 10.0, 4.0, 0.0, 0.0, &g);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let step = w.cfg.f("k_push") * 0.3 / w.cfg.f("drag");
    close(at(&w, &cells[a as usize]).0, 10.0 - step, "left disc");
    close(at(&w, &cells[b as usize]).0, 10.0 + d + step, "right disc");
    close(at(&w, &cells[a as usize]).1, 10.0, "stays on its row");
}

#[test]
fn crowding_can_be_off() {
    let mut w = still_world(32, &[("crowd_on", "false")]);
    let g = genome(&[(REST, 127)]);
    let a = w.plant(10.0, 10.0, 4.0, 0.0, 0.0, &g);
    w.plant(10.2, 10.0, 4.0, 0.0, 0.0, &g);
    w.step();
    w.flush();
    close(at(&w, &w.read_cells()[a as usize]).0, 10.0, "no push");
}

#[test]
fn the_step_is_capped() {
    let mut w = still_world(32, &[("k_push", "40")]);
    let g = genome(&[(REST, 127)]);
    let a = w.plant(10.0, 10.0, 4.0, 0.0, 0.0, &g);
    w.plant(10.1, 10.0, 4.0, 0.0, 0.0, &g);
    w.step();
    w.flush();
    close(at(&w, &w.read_cells()[a as usize]).0, 10.0 - w.cfg.f("step_max"), "capped");
}

/// Stick spending as plain arithmetic: what a cell with this much budget can pull with.
fn stick_of(w: &auto_life_soft::World, want: [i8; OUTPUTS], e: f32) -> f32 {
    let p = spend(w, want, e, 0.0);
    p.work[STICK] / (p.work[STICK] + w.cfg.f("stick_half"))
}

fn sticky() -> Vec<(&'static str, &'static str)> {
    let mut s = only(&["stick_on"]);
    s.push(("spend_share", "0.004"));
    s.push(("commitment", "0.002"));
    s
}

#[test]
fn adhesion_matches_plain_rule() {
    let mut w = still_world(32, &sticky());
    let pairs = [(REST, 0), (STICK, 100)];
    let want = budget(&pairs);
    let g = fixed_genome(2, want);
    let e = 10.0;
    let plain = spend(&w, want, e, 0.0);
    let r = radius(&w, plain.e);
    let s = stick_of(&w, want, e);
    let gap = 0.1;
    let d = 2.0 * r + gap;
    let a = w.plant(10.0, 10.0, e, 0.0, 0.0, &g);
    let b = w.plant(10.0 + d, 10.0, e, 0.0, 0.0, &g);
    w.step();
    w.flush();
    let cells = w.read_cells();
    // Identical glue, so match is 1. The pull fades to nothing at gap beyond contact.
    let pull = s * s * w.cfg.f("stick_max") * (1.0 - gap / w.cfg.f("gap"));
    let step = pull / w.cfg.f("drag");
    close(at(&w, &cells[a as usize]).0, 10.0 + step, "left pulled in");
    close(at(&w, &cells[b as usize]).0, 10.0 + d - step, "right pulled in");
    assert_ledger(&mut w, 2 * ((e * E_ONE) as i128 + 65536), "adhesion");
}

#[test]
fn a_pair_pulled_past_the_gap_stops_sticking() {
    let mut w = still_world(32, &sticky());
    let g = genome(&[(REST, 0), (STICK, 100)]);
    let e = 10.0;
    let plain = spend(&w, budget(&[(REST, 0), (STICK, 100)]), e, 0.0);
    let d = 2.0 * radius(&w, plain.e) + w.cfg.f("gap") + 0.02;
    let a = w.plant(10.0, 10.0, e, 0.0, 0.0, &g);
    w.plant(10.0 + d, 10.0, e, 0.0, 0.0, &g);
    w.step();
    w.flush();
    close(at(&w, &w.read_cells()[a as usize]).0, 10.0, "no pull");
}

#[test]
fn glue_that_differs_sticks_less() {
    let mut w = still_world(32, &sticky());
    let e = 10.0;
    let (wa, wb) = (
        budget(&[(REST, 0), (STICK, 100), (GLUE0, 100)]),
        budget(&[(REST, 0), (STICK, 100), (GLUE0, -100)]),
    );
    let plain = spend(&w, wa, e, 0.0);
    let d = 2.0 * radius(&w, plain.e) + 0.1;
    let a = w.plant(10.0, 10.0, e, 0.0, 0.0, &fixed_genome(2, wa));
    w.plant(10.0 + d, 10.0, e, 0.0, 0.0, &fixed_genome(2, wb));
    w.step();
    w.flush();
    let cells = w.read_cells();
    let g0 = squash(outputs(wa)[GLUE0]);
    // Bytes are written and read as 8-bit snorm, as the shader's packing does.
    let g = (g0 * 127.0).round() / 127.0;
    let m = 1.0 / (1.0 + w.cfg.f("glue_sharp") * (2.0 * g) * (2.0 * g));
    let s = stick_of(&w, wa, e);
    let pull = s * s * m * w.cfg.f("stick_max") * (1.0 - 0.1 / w.cfg.f("gap"));
    close(at(&w, &cells[a as usize]).0, 10.0 + pull / w.cfg.f("drag"), "weak pull");
}

#[test]
fn stick_off_means_no_pull_and_no_budget() {
    let mut w = quiet_world(32, &[], &[]);
    let want = budget(&[(REST, 0), (STICK, 100)]);
    let g = fixed_genome(2, want);
    let d = 2.0 * radius(&w, 10.0) + 0.1;
    let a = w.plant(10.0, 10.0, 10.0, 0.0, 0.0, &g);
    w.plant(10.0 + d, 10.0, 10.0, 0.0, 0.0, &g);
    w.step();
    w.flush();
    let plans = w.read_plans();
    assert_eq!(plans[a as usize].s2[2], 0.0, "no share of the budget for it");
    assert_eq!(w.read_cells()[a as usize].stick, 0.0);
    close(at(&w, &w.read_cells()[a as usize]).0, 10.0, "no pull");
}

#[test]
fn forces_cancel_to_the_last_bit() {
    // Drag of 2 scales by a power of two, so a step is exactly the force in position units, and
    // the sum of positions can't change if every pair's forces cancel exactly.
    let mut w = still_world(
        64,
        &[("spend", "0.2"), ("spend_share", "0.01"), ("k_push", "0.3"), ("thrust_on", "false"), ("push_power", "1"), ("step_max", "4")],
    );
    let mut r = 12345u32;
    let mut next = move || {
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        (r >> 8) as f32 / 16777216.0
    };
    let g = genome(&[(REST, 0), (STICK, 100), (PUSH, 90), (GLUE0, 30), (TURN_X, 50), (TURN_Y, 70)]);
    for _ in 0..30 {
        let (x, y, h) = (20.0 + next() * 6.0, 20.0 + next() * 6.0, next() * 6.28);
        w.plant(x, y, 8.0 + next() * 10.0, 0.0, h, &g);
    }
    let sums = |w: &auto_life_soft::World| {
        let cells = w.read_cells();
        let mut sx = 0u32;
        let mut sy = 0u32;
        for c in alive(&cells) {
            sx = sx.wrapping_add(c.x);
            sy = sy.wrapping_add(c.y);
        }
        (sx, sy)
    };
    let before = sums(&w);
    let mut moved = false;
    let first: Vec<u32> = w.read_cells().iter().map(|c| c.x).collect();
    for _ in 0..20 {
        w.step();
        w.flush();
        assert_eq!(sums(&w), before, "at tick {}", w.tick);
        moved |= w.read_cells().iter().zip(&first).any(|(c, x)| c.x != *x);
    }
    assert!(moved, "nothing moved, so this proved nothing");
}

#[test]
fn spending_matches_plain_rule() {
    let mut w = still_world(
        32,
        &[("spend", "0.2"), ("spend_share", "0.01"), ("upkeep", "0.01"), ("upkeep_e", "0.001"), ("heat_rate", "1")],
    );
    let want = budget(&[(REST, 40), (GROW, 60), (EMIT, 20), (HEAT, 30), (ENZYME, 50), (EAT, 10)]);
    let g = fixed_genome(2, want);
    let s = w.plant(10.5, 10.5, 12.0, 0.5, 0.0, &g);
    let plain = spend(&w, want, 12.0, 0.5);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let c = &cells[s as usize];
    close(e_of(c), plain.e, "energy");
    close(womb_of(c), plain.womb, "womb");
    close(c.unit, plain.unit, "budget");
    close(c.r, radius(&w, plain.e), "radius");
    // No food here, so nothing else changes it. The ledger still adds up.
    assert_ledger(&mut w, (12.5f32 * E_ONE) as i128 + 65536, "spending");
    let fa = w.read_fa();
    let p = (10 * 32 + 10) as usize;
    let shed = 12.5 - plain.e - plain.womb;
    let heat = shed * w.cfg.f("heat_rate") + plain.work[HEAT] * w.cfg.f("heat_power");
    close(fa[p][2] as f32 / E_ONE, heat, "heat here");
    close(fa[p][3] as f32 / E_ONE, plain.work[ENZYME] * w.cfg.f("enzyme_power"), "enzyme here");
    let scents = w.read_fb();
    let total: f32 = scents[p].iter().map(|v| *v as f32 / E_ONE).sum();
    close(total, plain.a[EMIT] * w.cfg.f("emit_power"), "scent here");
}

#[test]
fn harvest_is_supply_and_demand() {
    let mut w = quiet_world(32, &[], &[("k_push", "0")]);
    let want = budget(&[(REST, 60), (EAT, 100)]);
    let g = fixed_genome(2, want);
    // Plenty on one square, and little on another with three eaters.
    w.write_fa(5, 5, [3.0, 0.0, 0.0, 0.0]);
    let tiny = 0.05;
    w.write_fa(20, 20, [tiny, 0.0, 0.0, 0.0]);
    let rich = w.plant(5.5, 5.5, 10.0, 0.0, 0.0, &g);
    let poor: Vec<u32> = [20.3, 20.5, 20.7]
        .iter()
        .map(|x| w.plant(*x, 20.5, 10.0, 0.0, 0.0, &g))
        .collect();
    let start = (0..1).map(|_| w.stock().total()).next().unwrap();
    let plain = spend(&w, want, 10.0, 0.0);
    let wish = (plain.work[EAT] * w.cfg.f("eat_rate")).min(w.cfg.f("eat_max"));
    w.step();
    w.flush();
    let cells = w.read_cells();
    close(e_of(&cells[rich as usize]), plain.e + wish, "full wish");
    // With three after the same little food, each gets a third of it.
    assert!(wish * 3.0 > tiny, "this test needs demand above supply");
    for s in &poor {
        close(e_of(&cells[*s as usize]), plain.e + tiny / 3.0, "a third of the supply");
    }
    let fa = w.read_fa();
    close(fa[(5 * 32 + 5) as usize][0] as f32 / E_ONE, 3.0 - wish, "rich square");
    assert!(fa[(20 * 32 + 20) as usize][0].abs() <= 1, "poor square is bare: {}", fa[(20 * 32 + 20) as usize][0]);
    assert_ledger(&mut w, start, "harvest");
}

#[test]
fn locked_food_reach_matches_plain_rule() {
    let mut w = quiet_world(32, &[], &[("locked", "0.9"), ("eat_max", "10"), ("eat_rate", "20")]);
    let want = budget(&[(REST, 60), (EAT, 100)]);
    let g = fixed_genome(2, want);
    let z = 2.0;
    w.write_fa(5, 5, [30.0, 0.0, 0.0, z]);
    w.write_fa(20, 20, [3.0, 0.0, 0.0, 0.0]);
    let helped = w.plant(5.5, 5.5, 10.0, 0.0, 0.0, &g);
    let alone = w.plant(20.5, 20.5, 10.0, 0.0, 0.0, &g);
    let plain = spend(&w, want, 10.0, 0.0);
    let wish = plain.work[EAT] * w.cfg.f("eat_rate");
    w.step();
    w.flush();
    let cells = w.read_cells();
    let locked = 0.9;
    let reach = (1.0 - locked) + locked * z / (z + w.cfg.f("enzyme_half"));
    close(e_of(&cells[helped as usize]), plain.e + wish * reach, "with enzyme");
    close(e_of(&cells[alone as usize]), plain.e + wish * (1.0 - locked), "without");
}

#[test]
fn heat_harm_matches_plain_rule() {
    let mut w = still_world(32, &[("heat_harm", "0.1"), ("heat_scale", "8")]);
    let g = genome(&[(REST, 127)]);
    w.write_fa(5, 5, [0.0, 0.0, 2.0, 0.0]);
    w.write_fa(20, 20, [0.0, 0.0, 2.0, 0.0]);
    let thin = w.plant(5.5, 5.5, 2.0, 0.0, 0.0, &g);
    let rich = w.plant(20.5, 20.5, 30.0, 0.0, 0.0, &g);
    w.step();
    w.flush();
    let cells = w.read_cells();
    close(e_of(&cells[thin as usize]), 2.0 - 0.1 * 2.0 / (1.0 + 2.0 / 8.0), "thin");
    close(e_of(&cells[rich as usize]), 30.0 - 0.1 * 2.0 / (1.0 + 30.0 / 8.0), "rich");
    let harm = w.tally().flow("harm") as f32 / E_ONE;
    close(harm, 0.16 + 0.2 / (1.0 + 30.0 / 8.0), "counted as lost");
}

#[test]
fn many_biters_on_one_thin_cell_cant_overdraw_it() {
    let mut w = still_world(
        32,
        &[only(&["bite_on"]), vec![("bite_power", "30"), ("bite_yield", "0.5"), ("k_push", "0")]].concat(),
    );
    let want = budget(&[(REST, 0), (BITE, 120)]);
    let biter = fixed_genome(2, want);
    let idle = genome(&[(REST, 127)]);
    let thin_e = 0.05;
    let (cx, cy) = (16.0f32, 16.0f32);
    let victim = w.plant(cx, cy, thin_e, 0.0, 0.0, &idle);
    let n = 12;
    let mut biters = vec![];
    for k in 0..n {
        // In a ring, each facing the middle.
        let a = k as f32 / n as f32 * std::f32::consts::TAU;
        let d = 0.8;
        let (x, y) = (cx + a.cos() * d, cy + a.sin() * d);
        biters.push(w.plant(x, y, 10.0, 0.0, a + std::f32::consts::PI, &biter));
    }
    let start = w.stock().total();
    let plain = spend(&w, want, 10.0, 0.0);
    let wish = plain.work[BITE] * w.cfg.f("bite_power");
    assert!(wish * n as f32 > thin_e, "this test needs demand above what it has");
    w.step();
    w.flush();
    let cells = w.read_cells();
    // It held 0.05 and lost all of it. Unsigned energy would show a huge number if it overdrew.
    let v = &cells[victim as usize];
    assert_eq!(v.flags & F_ALIVE, 0, "drained to nothing, so dead");
    let t = w.tally();
    let paid = t.flow("bite_paid") as f32 / E_ONE;
    let got = t.flow("bite_got") as f32 / E_ONE;
    close(paid, thin_e, "paid no more than it had");
    assert!(got <= paid * 0.5 + 1e-3, "biters got {got}, from {paid}");
    let mut total_gain = 0.0;
    for s in &biters {
        let e = e_of(&cells[*s as usize]);
        // Nobody below zero, and each gets its equal share.
        assert!(e > 0.0 && e < 11.0, "biter at {e}");
        total_gain += e - plain.e;
    }
    close(total_gain, thin_e * 0.5, "the biters' gain is half of what was paid");
    assert_eq!(t.bitten, 1);
    assert_ledger(&mut w, start, "bites");
}

#[test]
fn a_bite_only_reaches_a_contact_in_front() {
    let mut w = quiet_world(32, &["bite_on"], &[("k_push", "0")]);
    let biter = genome(&[(REST, 0), (BITE, 120)]);
    let idle = genome(&[(REST, 127)]);
    // Facing away from it.
    let a = w.plant(10.0, 10.0, 10.0, 0.0, std::f32::consts::PI, &biter);
    let b = w.plant(10.8, 10.0, 5.0, 0.0, 0.0, &idle);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let idle_e = spend(&w, budget(&[(REST, 127)]), 5.0, 0.0).e;
    close(e_of(&cells[b as usize]), idle_e, "untouched");
    assert!(e_of(&cells[a as usize]) < 10.0);
}

#[test]
fn bite_off_means_no_bites() {
    let mut w = quiet_world(32, &[], &[("k_push", "0")]);
    let biter = genome(&[(REST, 0), (BITE, 120)]);
    let idle = genome(&[(REST, 127)]);
    w.plant(10.0, 10.0, 10.0, 0.0, 0.0, &biter);
    let b = w.plant(10.8, 10.0, 5.0, 0.0, 0.0, &idle);
    w.step();
    w.flush();
    let idle_e = spend(&w, budget(&[(REST, 127)]), 5.0, 0.0).e;
    close(e_of(&w.read_cells()[b as usize]), idle_e, "untouched");
}

#[test]
fn gifts_go_by_match_with_a_loss() {
    let mut w = quiet_world(32, &["give_on"], &[("k_push", "0"), ("give_yield", "0.9")]);
    let want = budget(&[(REST, 0), (GIVE, 120), (GLUE0, 100)]);
    let giver = fixed_genome(2, want);
    let kin = genome(&[(REST, 127), (GLUE0, 100)]);
    let other = genome(&[(REST, 127), (GLUE0, -100)]);
    let a = w.plant(10.0, 10.0, 10.0, 0.0, 0.0, &giver);
    let b = w.plant(10.8, 10.0, 5.0, 0.0, 0.0, &kin);
    let c = w.plant(9.2, 10.0, 5.0, 0.0, 0.0, &other);
    let start = w.stock().total();
    let plain = spend(&w, want, 10.0, 0.0);
    let amount = plain.work[GIVE] * w.cfg.f("give_power");
    w.step();
    w.flush();
    let cells = w.read_cells();
    let g = (squash(outputs(want)[GLUE0]) * 127.0).round() / 127.0;
    let m_other = 1.0 / (1.0 + w.cfg.f("glue_sharp") * (2.0 * g) * (2.0 * g));
    let sum = 1.0 + m_other;
    // Idle cells have a tiny share for giving too, and each touches only the giver.
    let idle = spend(&w, budget(&[(REST, 127)]), 5.0, 0.0);
    let back = idle.work[GIVE] * w.cfg.f("give_power");
    close(e_of(&cells[a as usize]), plain.e - amount + 2.0 * back * 0.9, "giver");
    let idle_e = spend(&w, budget(&[(REST, 127)]), 5.0, 0.0).e;
    close(e_of(&cells[b as usize]), idle_e - back + amount / sum * 0.9, "the one alike");
    close(e_of(&cells[c as usize]), idle_e - back + amount * m_other / sum * 0.9, "the other");
    assert_ledger(&mut w, start, "gifts");
}

#[test]
fn a_shove_pushes_what_is_in_front() {
    let mut w = quiet_world(32, &["push_on"], &[("k_push", "0"), ("push_power", "5")]);
    let want = budget(&[(REST, 0), (PUSH, 120)]);
    let g = fixed_genome(2, want);
    let idle = genome(&[(REST, 127)]);
    let a = w.plant(10.0, 10.0, 10.0, 0.0, 0.0, &g);
    let b = w.plant(10.8, 10.0, 10.0, 0.0, 0.0, &idle);
    let plain = spend(&w, want, 10.0, 0.0);
    let f = plain.work[PUSH] * w.cfg.f("push_power");
    w.step();
    w.flush();
    let cells = w.read_cells();
    let step = f / w.cfg.f("drag");
    close(at(&w, &cells[b as usize]).0, 10.8 + step, "pushed away");
    close(at(&w, &cells[a as usize]).0, 10.0 - step, "and back");
}

#[test]
fn thrust_moves_along_the_heading() {
    let mut w = quiet_world(32, &["thrust_on"], &[("thrust_rate", "40")]);
    let want = budget(&[(REST, 0), (THRUST, 120)]);
    let g = fixed_genome(2, want);
    let h = 0.7f32;
    let a = w.plant(10.0, 10.0, 10.0, 0.0, h, &g);
    let plain = spend(&w, want, 10.0, 0.0);
    let step = (plain.work[THRUST] * 40.0 / w.cfg.f("drag")).min(w.cfg.f("step_max"));
    w.step();
    w.flush();
    let cells = w.read_cells();
    let (x, y) = at(&w, &cells[a as usize]);
    close(x, 10.0 + step * h.cos(), "x");
    close(y, 10.0 + step * h.sin(), "y");
}

#[test]
fn splitting_makes_a_child_at_contact_distance() {
    let mut w = still_world(32, &[("split_share_max", "0.5"), ("mut_rate", "0")]);
    // No turn, split at 90 degrees to the left, share at most half.
    let want = budget(&[(REST, 127), (SPLIT, 50), (SHARE, 127), (ANGLE_Y, 100), (GLUE0, 60), (VOICE0, 40)]);
    let g = fixed_genome(2, want);
    let parent = w.plant(10.0, 10.0, 8.0, 2.5, 0.0, &g);
    let start = w.stock().total();
    w.step();
    w.flush();
    let cells = w.read_cells();
    let live = alive(&cells);
    assert_eq!(live.len(), 2, "one parent, one child");
    let p = &cells[parent as usize];
    let c = live.iter().find(|c| c.uid != p.uid).unwrap();
    // The share: full output of the share neuron saturates at 1, times the maximum.
    let out = outputs(want);
    let share = (0.5 + 0.5 * out[SHARE]).clamp(0.0, 1.0) * 0.5;
    close(e_of(p), 8.0 * (1.0 - share), "parent's energy");
    close(womb_of(p), 0.0, "womb emptied");
    close(e_of(c), 2.5 - 1.0 + 8.0 * share, "child's energy");
    assert_eq!(c.parent, p.uid);
    assert_eq!(c.generation, p.generation + 1);
    assert_eq!(c.clade, p.clade);
    // Directly to the left, in contact.
    let (px, py) = at(&w, p);
    let (cx, cy) = at(&w, c);
    let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
    close(d, p.r + radius(&w, e_of(c)), "contact distance");
    assert!(cy > py && (cx - px).abs() < 0.01, "to the left of a heading of +x: {cx},{cy} from {px},{py}");
    // Same outputs, so it sticks to its parent with no recognition rule.
    assert_eq!(c.gv, p.gv);
    // And the same genome, since this world has no mutation.
    let slot = cells.iter().position(|x| x.uid == c.uid).unwrap() as u32;
    assert_eq!(w.read_genome(slot), g);
    assert_ledger(&mut w, start, "splitting");
}

#[test]
fn a_child_keeps_its_parents_memory() {
    let mut w = still_world(32, &[]);
    let mut want = budget(&[(REST, 127), (SPLIT, 50)]);
    want[SHARE] = -127;
    // Hidden unit 0 is driven by the bias, so memory holds the same thing after a thought.
    let g = fixed_genome(2, want);
    let parent = w.plant(10.0, 10.0, 8.0, 2.5, 0.0, &g);
    w.step();
    w.flush();
    let mem = w.read_mem();
    let cells = w.read_cells();
    let child = cells
        .iter()
        .position(|c| c.flags & F_ALIVE != 0 && c.uid != cells[parent as usize].uid)
        .unwrap();
    let hr = w.cfg.hidden_rows as usize;
    for r in 0..hr {
        assert_eq!(mem[parent as usize * hr + r], mem[child * hr + r], "row {r}");
    }
    assert!(mem[child * hr][0] != 0.0, "memory should hold something");
}

#[test]
fn a_mutation_rate_is_handed_down_with_a_nudge() {
    let mut w = still_world(32, &[("mut_rate", "0.1"), ("mut_heritable", "true"), ("mut_rate_step", "0.2")]);
    let g = genome(&[(REST, 127), (SPLIT, 50)]);
    let parent = w.plant(10.0, 10.0, 8.0, 2.5, 0.0, &g);
    w.step();
    w.flush();
    let cells = w.read_cells();
    let child = alive(&cells).into_iter().find(|c| c.parent == cells[parent as usize].uid).unwrap();
    assert!(child.mu != cells[parent as usize].mu, "nudged");
    assert!((child.mu / cells[parent as usize].mu).ln().abs() < 1.5);
    let mut off = still_world(32, &[("mut_rate", "0.1")]);
    let p2 = off.plant(10.0, 10.0, 8.0, 2.5, 0.0, &g);
    off.step();
    off.flush();
    let cells = off.read_cells();
    let child = alive(&cells).into_iter().find(|c| c.parent == cells[p2 as usize].uid).unwrap();
    assert_eq!(child.mu, cells[p2 as usize].mu, "off means no nudge");
}

#[test]
fn death_leaves_body_and_stores_as_meat_and_frees_the_slot() {
    let mut w = still_world(32, &[("upkeep", "0.5")]);
    let g = genome(&[(REST, -127), (THRUST, -127)]);
    let s = w.plant(10.5, 10.5, 0.2, 1.5, 0.0, &g);
    let start = w.stock().total();
    w.step();
    w.flush();
    let cells = w.read_cells();
    assert_eq!(cells[s as usize].flags & F_ALIVE, 0);
    let fa = w.read_fa();
    close(fa[(10 * 32 + 10) as usize][1] as f32 / E_ONE, 1.5 + 1.0, "womb and body as meat");
    assert_eq!(w.tally().starved, 1);
    assert_eq!(w.alive(), 0);
    assert_ledger(&mut w, start, "death");
}

#[test]
fn eating_meat_matches_plain_rule() {
    let mut w = quiet_world(32, &["meat_on"], &[("meat_yield", "0.7")]);
    let want = budget(&[(REST, 60), (MEAT, 100)]);
    let g = fixed_genome(2, want);
    w.write_fa(5, 5, [0.0, 3.0, 0.0, 0.0]);
    let s = w.plant(5.5, 5.5, 10.0, 0.0, 0.0, &g);
    let start = w.stock().total();
    let plain = spend(&w, want, 10.0, 0.0);
    let wish = (plain.work[MEAT] * w.cfg.f("meat_rate")).min(w.cfg.f("meat_max"));
    w.step();
    w.flush();
    close(e_of(&w.read_cells()[s as usize]), plain.e + wish * 0.7, "energy");
    let fa = w.read_fa();
    close(fa[(5 * 32 + 5) as usize][1] as f32 / E_ONE, 3.0 - wish, "left");
    assert_ledger(&mut w, start, "meat");
}

#[test]
fn a_full_bin_drops_the_extra_for_a_tick_and_counts_it() {
    let mut w = still_world(32, &[]);
    let g = genome(&[(REST, 127)]);
    // Twelve cells in one bin, far enough apart that they're all near but none need move much.
    let mut slots = vec![];
    for k in 0..12 {
        slots.push(w.plant(10.1 + k as f32 * 0.15, 10.5, 4.0, 0.0, 0.0, &g));
    }
    w.step();
    w.flush();
    assert_eq!(w.tally().overflow, 4, "room for 8, so 4 are left out");
    assert_eq!(w.alive(), 12);
    // The bin keeps its highest slots, so the lowest four are the ones that felt nothing.
    let cells = w.read_cells();
    for s in &slots[..4] {
        assert_eq!(cells[*s as usize].contacts, 0, "slot {s} was dropped");
    }
    for s in &slots[8..] {
        assert!(cells[*s as usize].contacts > 0, "slot {s} was kept");
    }
}

#[test]
fn a_full_world_refuses_births_and_counts_them() {
    let mut w = world_with_slots(4);
    let g = genome(&[(REST, 127), (SPLIT, 50)]);
    for k in 0..4 {
        w.plant(4.0 + k as f32 * 4.0, 10.0, 8.0, 2.5, 0.0, &g);
    }
    w.step();
    w.flush();
    assert_eq!(w.alive(), 4);
    assert_eq!(w.tally().no_slot, 4);
    assert_eq!(w.tally().births, 0);
}

fn world_with_slots(n: u32) -> auto_life_soft::World {
    let mut cfg = auto_life_soft::Config::new(32, 7, None).unwrap();
    for (k, v) in [
        ("sun", "0"),
        ("rock", "false"),
        ("rain_p", "0"),
        ("rain_burst_ticks", "0"),
        ("hazard", "0"),
        ("brain_every", "1"),
        ("upkeep", "0"),
        ("upkeep_e", "0"),
        ("spend", "0"),
        ("heat_rate", "0"),
        ("food_start_ticks", "0"),
    ] {
        cfg.set(k, v).unwrap();
    }
    cfg.slots = n;
    auto_life_soft::World::new(cfg, auto_life_soft::Pacing { target_ms: 50.0, duty: 1.0 }).unwrap()
}

#[test]
fn slots_are_reused() {
    let mut w = world_with_slots(4);
    let dies = genome(&[(REST, 127)]);
    for k in 0..4 {
        let e = if k == 0 { 0.0 } else { 8.0 };
        w.plant(4.0 + k as f32 * 4.0, 10.0, e, 0.0, 0.0, &dies);
    }
    w.step();
    w.flush();
    assert_eq!(w.alive(), 3, "one had no energy");
    assert_eq!(w.read_cells()[0].flags & F_ALIVE, 0);
    // A split now finds the freed slot.
    let g = genome(&[(REST, 127), (SPLIT, 50)]);
    w.write_genomes(1, &g);
    let mut c = w.read_cells()[1];
    c.womb = (2.5 * E_ONE) as u32;
    w.gpu.queue.write_buffer(&w.buffers["cell"], 96, bytemuck::bytes_of(&c));
    w.step();
    w.flush();
    assert_eq!(w.alive(), 4);
    let cells = w.read_cells();
    assert_eq!(cells[0].flags & F_ALIVE, F_ALIVE);
    assert_eq!(cells[0].parent, cells[1].uid);
}

/// The same rule as fields.wgsl, in plain Rust, for one layer of four values.
fn plain_fields(
    now: &[[i32; 4]],
    size: usize,
    flow: [f32; 4],
    decay: [f32; 4],
    rock: &[bool],
) -> Vec<[i32; 4]> {
    let mut out = now.to_vec();
    for y in 0..size {
        for x in 0..size {
            let p = y * size + x;
            if rock[p] {
                continue;
            }
            let v = now[p];
            let mut next = v;
            for (nx, ny) in [
                ((x + 1) % size, y),
                ((x + size - 1) % size, y),
                (x, (y + 1) % size),
                (x, (y + size - 1) % size),
            ] {
                let q = ny * size + nx;
                if rock[q] {
                    continue;
                }
                for k in 0..4 {
                    next[k] += ((now[q][k] - v[k]) as f32 * flow[k]) as i32;
                }
            }
            for k in 0..4 {
                next[k] -= (next[k] as f32 * decay[k]) as i32;
            }
            out[p] = next;
        }
    }
    out
}

#[test]
fn fields_match_plain_rule_exactly() {
    let sets = [
        ("meat_flow", "0.05"),
        ("heat_flow", "0.15"),
        ("heat_decay", "0.05"),
        ("enzyme_flow", "0.1"),
        ("enzyme_decay", "0.02"),
        ("s0_flow", "0.02"),
        ("s0_decay", "0.005"),
        ("s1_flow", "0.08"),
        ("s1_decay", "0.02"),
        ("s2_flow", "0.15"),
        ("s2_decay", "0.08"),
        ("s3_flow", "0.24"),
        ("s3_decay", "0.3"),
        ("rock", "true"),
        ("food_cap", "4"),
        ("batch_tile", "32"),
    ];
    let mut w = still_world(64, &sets);
    let size = 64;
    let terrain = w.read_terrain();
    let rock: Vec<bool> = terrain.iter().map(|t| t & (1 << 24) != 0).collect();
    assert!(rock.iter().any(|r| *r), "needs some rock in the way");
    let mut r = 99u32;
    let mut next = move || {
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        r >> 13
    };
    let mut fa: Vec<[i32; 4]> = (0..size * size)
        .map(|p| {
            if rock[p] { [0; 4] } else { [0, next() as i32, next() as i32, next() as i32] }
        })
        .collect();
    let mut fb: Vec<[i32; 4]> = (0..size * size)
        .map(|p| {
            if rock[p] { [0; 4] } else { [next() as i32, next() as i32, next() as i32, next() as i32] }
        })
        .collect();
    w.write_fa_all(&fa);
    w.gpu.queue.write_buffer(&w.buffers["fb_a"], 0, bytemuck::cast_slice(&fb));
    let flow_a = [0.0, 0.05, 0.15, 0.1];
    let decay_a = [0.0, 0.0, 0.05, 0.02];
    let flow_b = [0.02, 0.08, 0.15, 0.24];
    let decay_b = [0.005, 0.02, 0.08, 0.3];
    let start_meat: i64 = fa.iter().map(|v| v[1] as i64).sum();
    for _ in 0..12 {
        w.step();
        w.flush();
        fa = plain_fields(&fa, size, flow_a, decay_a, &rock);
        fb = plain_fields(&fb, size, flow_b, decay_b, &rock);
        assert_eq!(w.read_fa(), fa, "food, meat, heat, enzyme at tick {}", w.tick);
        assert_eq!(w.read_fb(), fb, "scents at tick {}", w.tick);
    }
    let meat: i64 = w.read_fa().iter().map(|v| v[1] as i64).sum();
    assert_eq!(meat, start_meat, "exchange keeps every unit");
}

#[test]
fn walled_worlds_share_nothing() {
    // Two tiles by two, each with its own land. Fill one with heat and scent and a crowd.
    let mut w = still_world(
        128,
        &[
            ("batch_tile", "64"),
            ("rock", "false"),
            ("heat_flow", "0.2"),
            ("s3_flow", "0.2"),
            ("k_push", "1"),
            ("spend", "0.2"),
            ("spend_share", "0.01"),
            ("thrust_rate", "400"),
        ],
    );
    let terrain = w.read_terrain();
    let size = 128usize;
    for y in 0..size {
        for x in 0..size {
            let wall = x % 64 < 5 || y % 64 < 5;
            assert_eq!(terrain[y * size + x] & (1 << 24) != 0, wall, "at {x},{y}");
        }
    }
    let g = genome(&[(REST, 0), (THRUST, 127)]);
    let mut layer = vec![[0i32; 4]; size * size];
    for y in 10..50 {
        for x in 10..50 {
            layer[y * size + x][2] = 1 << 20;
        }
    }
    w.write_fa_all(&layer);
    // Cells heading into every wall of the first tile.
    let mut k = 0;
    for side in 0..4 {
        let h = side as f32 * std::f32::consts::FRAC_PI_2;
        for n in 0..8 {
            let along = 12.0 + n as f32 * 5.0;
            let (x, y) = match side {
                0 => (60.0, along),
                1 => (along, 60.0),
                2 => (6.0, along),
                _ => (along, 6.0),
            };
            w.plant(x, y, 30.0, 0.0, h, &g);
            k += 1;
        }
    }
    assert_eq!(k, 32);
    for _ in 0..120 {
        w.step();
    }
    w.flush();
    let cells = w.read_cells();
    for c in alive(&cells) {
        let (x, y) = at(&w, c);
        let (ix, iy) = (x as usize, y as usize);
        assert_eq!(terrain[iy * size + ix] & (1 << 24), 0, "a cell in rock at {x},{y}");
        assert!((5..64).contains(&ix) && (5..64).contains(&iy), "a cell left its tile: {x},{y}");
    }
    let fa = w.read_fa();
    for y in 0..size {
        for x in 0..size {
            if x >= 64 || y >= 64 {
                assert_eq!(fa[y * size + x][2], 0, "heat crossed a wall to {x},{y}");
            }
        }
    }
}

#[test]
fn energy_adds_up_exactly_in_a_free_world() {
    let mut w = world_with(64, 3, &[("ledger", "true"), ("sun", "0.05"), ("meat_flow", "0.02"), ("food_flow", "0.01")]);
    let start = w.stock().total();
    for k in 0..600 {
        w.step();
        if k % 100 == 99 {
            assert_ledger(&mut w, start, "free world");
        }
    }
    let t = w.tally();
    assert!(t.flow("sun") > 0 && t.flow("upkeep") > 0 && t.flow("spend") > 0, "{t:?}");
    assert!(t.rained > 0);
    // Every cell's energy is sane, which an underflow would break.
    let cells = w.read_cells();
    for c in alive(&cells) {
        assert!(e_of(c) <= w.cfg.f("max_energy"), "energy {}", e_of(c));
    }
}

#[test]
fn energy_adds_up_with_every_trade_forced_on() {
    // Brains biased to do everything, so every route carries energy in one world.
    let mut w = still_world(
        64,
        &[
            ("sun", "0.1"),
            ("spend", "0.2"),
            ("spend_share", "0.02"),
            ("upkeep", "0.01"),
            ("heat_rate", "1"),
            ("heat_harm", "0.02"),
            ("heat_flow", "0.1"),
            ("heat_decay", "0.1"),
            ("meat_flow", "0.02"),
            ("meat_rot", "0.01"),
            ("food_cap", "2"),
            ("max_energy", "9"),
            ("hazard", "0.002"),
            ("locked", "0.5"),
            ("enzyme_flow", "0.1"),
            ("enzyme_decay", "0.05"),
        ],
    );
    let want = budget(&[
        (REST, 0),
        (THRUST, 10),
        (EAT, 70),
        (MEAT, 40),
        (GROW, 50),
        (EMIT, 10),
        (HEAT, 10),
        (BITE, 50),
        (GIVE, 40),
        (ENZYME, 30),
        (STICK, 40),
        (PUSH, 10),
        (SPLIT, 50),
        (TURN_X, 60),
        (TURN_Y, 30),
    ]);
    let g = fixed_genome(2, want);
    let mut r = 7u32;
    let mut next = move || {
        r = r.wrapping_mul(1664525).wrapping_add(1013904223);
        (r >> 8) as f32 / 16777216.0
    };
    for _ in 0..200 {
        let (x, y, h) = (next() * 64.0, next() * 64.0, next() * 6.28);
        w.plant(x, y, 2.0 + next() * 6.0, next() * 2.0, h, &g);
    }
    let start = w.stock().total();
    for k in 0..400 {
        w.step();
        if k % 50 == 49 {
            assert_ledger(&mut w, start, "every trade");
        }
    }
    let t = w.tally();
    for name in [
        "sun", "upkeep", "spend", "stick", "harm", "eaten", "eat_got", "meat_eaten", "meat_got",
        "bite_paid", "bite_got", "give_out", "give_in", "death", "grow",
    ] {
        assert!(t.flow(name) > 0, "nothing went by {name}: {t:?}");
    }
    assert!(t.births > 0 && t.starved + t.bitten + t.burned + t.old > 0, "{t:?}");
    let cells = w.read_cells();
    for c in alive(&cells) {
        assert!(e_of(c) <= 9.0 && c.e < u32::MAX / 2, "energy {}", e_of(c));
    }
}

#[test]
fn same_seed_same_world() {
    let run = |seed: u32, extra: &[(&str, &str)]| {
        let mut w = world_with(64, seed, extra);
        for _ in 0..500 {
            w.step();
        }
        fingerprint(&mut w)
    };
    assert_eq!(run(9, &[]), run(9, &[]), "same seed");
    assert_ne!(run(9, &[]), run(10, &[]), "different seed");
}

#[test]
fn same_seed_same_world_with_bins_overflowing_and_slots_short() {
    // Cells rain in fast onto very little room, so bins overflow and births run out of slots,
    // which are the two places that a timing-dependent design couldn't repeat.
    let run = || {
        let mut cfg = auto_life_soft::Config::new(32, 11, None).unwrap();
        for (k, v) in [("sun", "0.2"), ("rain_p", "40"), ("rain_burst_ticks", "2000"), ("rock", "false"), ("r0", "0.2"), ("k_push", "0.02"), ("hazard", "0")] {
            cfg.set(k, v).unwrap();
        }
        cfg.slots = 1800;
        let mut w = auto_life_soft::World::new(cfg, auto_life_soft::Pacing { target_ms: 50.0, duty: 1.0 }).unwrap();
        for _ in 0..400 {
            w.step();
        }
        w.flush();
        let t = w.tally();
        (fingerprint(&mut w), t.no_slot, t.overflow)
    };
    let a = run();
    let b = run();
    assert_eq!(a, b);
    assert!(a.1 > 0, "this run never ran out of slots, so it proved nothing about that");
    assert!(a.2 > 0, "this run never overflowed a bin, so it proved nothing about that");
}

#[test]
fn saving_and_loading_loses_nothing() {
    let path = std::env::temp_dir().join(format!("soft-save-test-{}.bin", std::process::id()));
    let path = path.to_str().unwrap().to_string();
    let mut a = world_with(64, 5, &[]);
    for _ in 0..301 {
        a.step();
    }
    a.save(&path).unwrap();
    for _ in 0..200 {
        a.step();
    }
    let want = fingerprint(&mut a);
    drop(a);
    let gpu = std::sync::Arc::new(
        pollster::block_on(auto_core::Gpu::headless(auto_core::GpuOptions {
            adapter_name: std::env::var("AUTO_ADAPTER").ok(),
            ..Default::default()
        }))
        .unwrap(),
    );
    let mut b = auto_life_soft::World::open_saved(
        gpu,
        &path,
        None,
        auto_life_soft::Pacing { target_ms: 50.0, duty: 1.0 },
    )
    .unwrap();
    assert_eq!(b.tick, 301);
    for _ in 0..200 {
        b.step();
    }
    assert_eq!(fingerprint(&mut b), want);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn a_saved_world_can_seed_a_bigger_one() {
    let path = std::env::temp_dir().join(format!("soft-seed-test-{}.bin", std::process::id()));
    let path = path.to_str().unwrap().to_string();
    let mut a = world_with(32, 5, &[("rock", "false"), ("rain_burst_ticks", "100")]);
    for _ in 0..120 {
        a.step();
    }
    a.save(&path).unwrap();
    let n = a.flush();
    let _ = n;
    let alive_there = a.alive() as u64;
    assert!(alive_there > 0);
    let mut b = world_with(64, 6, &[("rock", "false"), ("rain_p", "0"), ("rain_burst_ticks", "0")]);
    let planted = b.seed_from(std::slice::from_ref(&path), 1.0).unwrap();
    assert_eq!(planted, alive_there * 4, "one world's cells in each of four tiles");
    assert_eq!(b.alive() as u64, planted);
    b.step();
    b.flush();
    assert!(b.alive() > 0);
    let _ = std::fs::remove_file(&path);
}
