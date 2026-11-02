//! Whole-world checks and one-tick checks of each rule against the same rule
//! written as plain Rust. They run on whatever Vulkan device is there, and CI uses
//! a software driver.

mod common;

use auto_life_gpu::config::{CELL_BYTES, IN_BLOCKS, OUT_ROWS, STAT_BYTES};
use auto_life_gpu::{Config, Pacing, World};
use common::*;

#[test]
fn energy_adds_up() {
    assert_energy_adds_up(&[]);
}

#[test]
fn grid_and_cells_agree() {
    assert_grid_and_cells_agree(&[]);
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
        .write_buffer(&w.buffers["cell"], a as u64 * CELL_BYTES + 4, &bytes);
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

#[test]
fn a_saved_world_continues_the_same() {
    let dir = std::env::temp_dir().join(format!("life-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("world.bin").to_string_lossy().to_string();
    let mut a = world_with(64, 21, &[]);
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
    let mut b = world_with(64, 21, &[]);
    b.load(&path).unwrap();
    for _ in 0..100 {
        b.step();
    }
    b.flush();
    assert_eq!(fingerprint(&b), want);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn seeding_a_bigger_world_keeps_it_consistent() {
    let dir = std::env::temp_dir().join(format!("life-seed-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("small.bin").to_string_lossy().to_string();
    let mut small = world_with(64, 4, &[]);
    for _ in 0..1500 {
        small.step();
    }
    small.save(&path).unwrap();
    let small_alive = small.alive();
    assert!(small_alive > 0);
    drop(small);
    let mut big = world_with(128, 8, &[]);
    let planted = big.seed_from(&[path], 1.0).unwrap();
    // Four tiles, less whatever landed on rock.
    assert!(
        planted > small_alive as u64 * 3,
        "planted {planted} from {small_alive}"
    );
    assert!(planted <= small_alive as u64 * 4);
    assert_eq!(big.alive() as u64, planted);
    for _ in 0..100 {
        big.step();
    }
    big.flush();
    let occ = big.read_u32s("occ");
    let cells = big.read_cells();
    let mut on_grid = 0;
    for (p, o) in occ.iter().enumerate() {
        if *o != 0 && *o != ROCK {
            on_grid += 1;
            assert_eq!(cells[*o as usize - 1][10] as usize, p);
        }
    }
    assert_eq!(on_grid as u32, big.alive());
    assert!(on_grid > 0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn two_genome_buffers_change_nothing() {
    let run = |shard: Option<u32>| {
        let mut cfg = Config::new(64, 31, None).unwrap();
        if let Some(s) = shard {
            cfg.shard_slots = s;
        }
        let mut w = World::new(
            cfg,
            Pacing {
                target_ms: 50.0,
                duty: 1.0,
            },
        )
        .unwrap();
        for _ in 0..500 {
            w.step();
        }
        w.flush();
        assert!(w.alive() > 0);
        fingerprint(&w)
    };
    // The split lands inside the slots in use, so cells live on both sides of it.
    assert_eq!(run(None), run(Some(60)));
}

#[test]
fn a_new_cell_starts_with_blank_measurements() {
    let mut w = still_world(32);
    let want = budget(&[(REST, 60), (DIR_X, 60)]);
    let g = fixed_genome(2, want);
    let womb = w.cfg.f("womb");
    // Slot 1 will go to the child. Dirty it as a dead cell would have left it.
    let dirty = [9.0f32; STAT_BYTES as usize / 4];
    w.gpu
        .queue
        .write_buffer(&w.buffers["stat"], STAT_BYTES, bytemuck::cast_slice(&dirty));
    w.gpu.queue.submit([]);
    w.gpu.wait();
    w.plant(20, 20, 8.0, womb + 0.5, 0, &g);
    w.step();
    w.flush();
    let kid = w.read_ranges("stat", &[(STAT_BYTES, STAT_BYTES)]);
    assert!(
        bytemuck::cast_slice::<u8, f32>(&kid)
            .iter()
            .all(|v| v.abs() < 1.0),
        "child carries the old owner's numbers"
    );
}
