//! Each measure on a built case with a known answer. No GPU.

use auto_life_soft::dump::{self, DumpCell};
use auto_life_soft::measure::*;

const RULES: Rules = Rules {
    size: 128.0,
    gap: 0.2,
    glue_sharp: 4.0,
    stick_max: 0.4,
    min_pull: 0.01,
};

fn cell(uid: u32, x: f32, y: f32, stick: f32) -> DumpCell {
    DumpCell {
        uid,
        parent: u32::MAX,
        x,
        y,
        r: 0.5,
        stick,
        ..Default::default()
    }
}

/// A row of touching sticky cells.
fn row(first_uid: u32, n: u32, x: f32, y: f32) -> Vec<DumpCell> {
    (0..n).map(|k| cell(first_uid + k, x + k as f32 * 0.9, y, 1.0)).collect()
}

fn labels(cells: &[DumpCell]) -> Vec<u32> {
    groups(cells, &RULES, Link::Stuck)
}

#[test]
fn a_chain_is_one_body_and_a_gap_breaks_it() {
    let mut cells = row(0, 5, 10.0, 10.0);
    cells.extend(row(100, 3, 20.0, 10.0));
    cells.push(cell(200, 40.0, 40.0, 1.0));
    let l = labels(&cells);
    let sizes = group_sizes(&l);
    let mut sorted = sizes.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, [1, 3, 5]);
    assert_eq!(l[0], l[4]);
    assert_ne!(l[0], l[5]);
}

#[test]
fn touching_with_no_adhesion_is_no_body_but_is_a_crowd() {
    let mut cells = row(0, 4, 10.0, 10.0);
    for c in &mut cells {
        c.stick = 0.0;
    }
    assert_eq!(size_stats(&labels(&cells)).bodies, 0);
    let crowd = groups(&cells, &RULES, Link::Touching);
    assert_eq!(size_stats(&crowd).largest, 4);
}

#[test]
fn one_sided_adhesion_is_no_body() {
    let mut cells = row(0, 2, 10.0, 10.0);
    cells[1].stick = 0.0;
    assert_eq!(size_stats(&labels(&cells)).bodies, 0, "both have to stick");
}

#[test]
fn glue_that_differs_is_no_body() {
    let mut cells = row(0, 2, 10.0, 10.0);
    cells[0].stick = 0.4;
    cells[1].stick = 0.4;
    assert_eq!(size_stats(&labels(&cells)).bodies, 1, "same glue");
    cells[0].glue = [1.0, 1.0];
    cells[1].glue = [-1.0, -1.0];
    assert_eq!(size_stats(&labels(&cells)).bodies, 0, "opposite glue");
}

#[test]
fn bodies_wrap_round_the_world() {
    let mut cells = vec![cell(0, 0.2, 10.0, 1.0), cell(1, 127.7, 10.0, 1.0)];
    assert_eq!(size_stats(&labels(&cells)).bodies, 1);
    cells[1].x = 100.0;
    assert_eq!(size_stats(&labels(&cells)).bodies, 0);
}

#[test]
fn a_pair_out_of_reach_is_no_body() {
    // Contact is 1.0 apart here, and the pull ends at gap beyond that.
    let cells = vec![cell(0, 10.0, 10.0, 1.0), cell(1, 11.21, 10.0, 1.0)];
    assert_eq!(size_stats(&labels(&cells)).bodies, 0);
}

#[test]
fn size_stats_add_up() {
    let mut cells = row(0, 10, 10.0, 10.0);
    cells.extend(row(100, 3, 30.0, 10.0));
    cells.extend(row(200, 2, 50.0, 10.0));
    cells.push(cell(300, 70.0, 70.0, 1.0));
    let s = size_stats(&labels(&cells));
    assert_eq!(s.cells, 16);
    assert_eq!(s.bodies, 3);
    assert_eq!(s.largest, 10);
    assert!((s.in_bodies - 15.0 / 16.0).abs() < 1e-9);
    assert!((s.in_3_plus - 13.0 / 16.0).abs() < 1e-9);
    assert!((s.in_10_plus - 10.0 / 16.0).abs() < 1e-9);
    assert!((s.mean_per_cell - (100.0 + 9.0 + 4.0 + 1.0) / 16.0).abs() < 1e-9);
    // 2-3, 4-7, 8-15.
    assert_eq!(s.histogram, [2, 0, 1]);
}

/// A dump in which each body in `sets` is a row of those uids, laid out apart.
fn frame(tick: u64, sets: &[Vec<u32>]) -> Frame {
    let mut cells = vec![];
    for (i, set) in sets.iter().enumerate() {
        for (k, uid) in set.iter().enumerate() {
            cells.push(cell(*uid, 5.0 + k as f32 * 0.9, 5.0 + i as f32 * 5.0, 1.0));
        }
    }
    Frame::new(tick, cells, &RULES)
}

fn range(a: u32, b: u32) -> Vec<u32> {
    (a..b).collect()
}

#[test]
fn a_body_keeps_its_id_and_a_piece_breaking_off_is_its_offspring() {
    let frames = vec![
        frame(100, &[range(0, 20)]),
        frame(200, &[range(0, 20)]),
        // 6 break off.
        frame(300, &[range(0, 14), range(14, 20)]),
        frame(400, &[range(0, 14), range(14, 20)]),
    ];
    let bodies = track(&frames);
    assert_eq!(bodies.len(), 2);
    let (a, b) = (&bodies[0], &bodies[1]);
    assert_eq!(a.parent, None);
    assert_eq!(a.generation, 0);
    assert_eq!(a.sizes, [(100, 20), (200, 20), (300, 14), (400, 14)]);
    assert_eq!(a.size_at_fragmenting, Some(20));
    assert_eq!(b.parent, Some(0));
    assert_eq!(b.generation, 1);
    assert_eq!(b.first_tick, 300);
}

#[test]
fn children_count_as_members_of_their_parents_body() {
    let mut f2 = frame(200, &[range(0, 6)]);
    // Six new cells, each the child of an old member, in a body of their own.
    let kids: Vec<DumpCell> = (0..6)
        .map(|k| DumpCell {
            parent: k,
            ..cell(100 + k, 5.0 + k as f32 * 0.9, 40.0, 1.0)
        })
        .collect();
    f2.cells.extend(kids);
    f2.labels = groups(&f2.cells, &RULES, Link::Stuck);
    let bodies = track(&[frame(100, &[range(0, 6)]), f2]);
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[1].parent, Some(0));
}

#[test]
fn a_lone_cell_that_leaves_and_grows_is_a_fragment() {
    let mut f3 = frame(300, &[range(0, 5)]);
    let grown: Vec<DumpCell> = (0..4)
        .map(|k| DumpCell {
            parent: 5,
            ..cell(50 + k, 5.0 + k as f32 * 0.9, 40.0, 1.0)
        })
        .collect();
    f3.cells.extend(grown);
    f3.labels = groups(&f3.cells, &RULES, Link::Stuck);
    let mut f2 = frame(200, &[range(0, 5)]);
    // Cell 5 has left and is alone, far away.
    f2.cells.push(cell(5, 60.0, 60.0, 1.0));
    f2.labels = groups(&f2.cells, &RULES, Link::Stuck);
    let bodies = track(&[frame(100, &[range(0, 6)]), f2, f3]);
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[1].parent, Some(0));
    assert_eq!(bodies[1].generation, 1);
}

#[test]
fn generations_go_down_a_line_of_splits() {
    let frames = vec![
        frame(100, &[range(0, 32)]),
        frame(200, &[range(0, 16), range(16, 32)]),
        frame(300, &[range(0, 16), range(16, 24), range(24, 32)]),
        frame(400, &[range(0, 16), range(16, 24), range(24, 28), range(28, 32)]),
    ];
    let bodies = track(&frames);
    let deepest = bodies.iter().map(|b| b.generation).max().unwrap();
    assert_eq!(deepest, 3);
    assert_eq!(bodies.len(), 4);
}

#[test]
fn size_at_fragmenting_slope_is_found() {
    // Parents and offspring by hand: offspring fragment at 2x + 1 of their parent's size.
    let mut bodies = vec![];
    let mut add = |parent: Option<usize>, size: u32| {
        let id = bodies.len();
        bodies.push(Tracked {
            id,
            parent,
            generation: parent.map_or(0, |_| 1),
            first_tick: 0,
            last_tick: 0,
            sizes: vec![],
            size_at_fragmenting: Some(size),
        });
        id
    };
    for size in [4, 6, 9, 12, 20] {
        let p = add(None, size);
        add(Some(p), 2 * size + 1);
    }
    let (slope, pairs) = size_heritability(&bodies);
    assert_eq!(pairs, 5);
    assert!((slope.unwrap() - 2.0).abs() < 1e-9);

    // No relation: every offspring the same.
    let mut flat = vec![];
    for (i, size) in [4u32, 6, 9, 12].iter().enumerate() {
        flat.push(Tracked {
            id: i * 2,
            parent: None,
            generation: 0,
            first_tick: 0,
            last_tick: 0,
            sizes: vec![],
            size_at_fragmenting: Some(*size),
        });
        flat.push(Tracked {
            id: i * 2 + 1,
            parent: Some(i * 2),
            generation: 1,
            first_tick: 0,
            last_tick: 0,
            sizes: vec![],
            size_at_fragmenting: Some(7),
        });
    }
    assert!(size_heritability(&flat).0.unwrap().abs() < 1e-9);
}

#[test]
fn membership_half_life_of_a_group_that_stays_is_unbounded() {
    let frames: Vec<Frame> = (1..=5).map(|k| frame(k * 100, &[range(0, 30)])).collect();
    let m = membership(&frames);
    assert_eq!(m.half_life, None);
    assert!(m.curve.iter().all(|p| p.1 == 1.0));
}

#[test]
fn membership_half_life_of_a_group_that_scatters_is_known() {
    // Every pair is together at tick 100, and by tick 200 no two are.
    let spread: Vec<Vec<u32>> = (0..12).map(|k| vec![k]).collect();
    let frames = vec![frame(100, &[range(0, 12)]), frame(200, &spread)];
    let m = membership(&frames);
    // Together from 1 to 0 over 100 ticks, so half are gone at 50.
    assert!((m.half_life.unwrap() - 50.0).abs() < 1e-6, "{:?}", m);
}

#[test]
fn membership_counts_only_pairs_that_both_live() {
    // Half the cells die, and the rest stay together: no one has left.
    let frames = vec![frame(100, &[range(0, 20)]), frame(200, &[range(0, 10)])];
    let m = membership(&frames);
    assert_eq!(m.half_life, None);
    assert_eq!(m.curve[0].1, 1.0);
}

fn with_genome(mut c: DumpCell, byte: i8) -> DumpCell {
    let word = u32::from_le_bytes([byte as u8; 4]);
    c.sketch = [word; 4];
    c
}

#[test]
fn relatedness_within_bodies_against_between() {
    // Two bodies, each a clone of its own genome, a known distance apart.
    let mut cells: Vec<DumpCell> = row(0, 6, 10.0, 10.0).into_iter().map(|c| with_genome(c, 0)).collect();
    cells.extend(row(100, 6, 10.0, 14.0).into_iter().map(|c| with_genome(c, 127)));
    let f = Frame::new(0, cells, &RULES);
    let r = relatedness(&f, &RULES);
    assert_eq!(r.within.unwrap(), 0.0);
    // Byte 127 is +4 in the sketch's scale, so the root mean square gap between the bodies is 4.
    assert!((r.between.unwrap() - 4.0).abs() < 1e-6, "{r:?}");
    assert!((r.between_nearby.unwrap() - 4.0).abs() < 1e-6, "{r:?}");
}

#[test]
fn relatedness_of_a_mixed_body_shows_no_difference() {
    let mut cells: Vec<DumpCell> = row(0, 40, 10.0, 10.0);
    for (k, c) in cells.iter_mut().enumerate() {
        *c = with_genome(*c, if k % 2 == 0 { 0 } else { 127 });
    }
    let mut other = row(100, 40, 10.0, 14.0);
    for (k, c) in other.iter_mut().enumerate() {
        *c = with_genome(*c, if k % 2 == 0 { 0 } else { 127 });
    }
    cells.extend(other);
    let f = Frame::new(0, cells, &RULES);
    let r = relatedness(&f, &RULES);
    let (w, b) = (r.within.unwrap(), r.between.unwrap());
    assert!((w - b).abs() < 0.5, "within {w}, between {b}");
}

/// A grid of clonemates, whose contact count falls towards the rim.
fn body_with_contacts(budget_of: impl Fn(u32) -> [f32; 12]) -> Frame {
    let mut cells = vec![];
    for y in 0..6 {
        for x in 0..6 {
            let edge = x == 0 || y == 0 || x == 5 || y == 5;
            let contacts = if edge { 3 } else { 6 };
            let mut c = cell(y * 6 + x, 10.0 + x as f32 * 0.9, 10.0 + y as f32 * 0.9, 1.0);
            c.contacts = contacts;
            c.budget = budget_of(contacts);
            cells.push(c);
        }
    }
    Frame::new(0, cells, &RULES)
}

#[test]
fn position_effect_finds_budgets_set_by_contacts() {
    let f = body_with_contacts(|contacts| {
        let mut b = [0.05f32; 12];
        b[2] = if contacts >= 6 { 0.5 } else { 0.1 };
        b
    });
    let p = position_effect(&f, 0.4, 7);
    assert_eq!(p.bodies, 1);
    assert_eq!(p.cells, 36);
    assert!(p.r2 > 0.999, "{p:?}");
    assert!(p.r2_shuffled < 0.3, "{p:?}");
    assert!((p.inside[2] - 0.5).abs() < 1e-5, "{p:?}");
    assert!((p.surface[2] - 0.1).abs() < 1e-5, "{p:?}");
}

#[test]
fn position_effect_finds_nothing_when_budgets_ignore_position() {
    // Budgets vary, but with no relation to contacts.
    let mut f = body_with_contacts(|_| [0.05; 12]);
    for (k, c) in f.cells.iter_mut().enumerate() {
        c.budget[2] = ((k as u32 * 2654435761u32) >> 24) as f32 / 256.0;
    }
    let p = position_effect(&f, 0.4, 7);
    assert!(p.r2 < 0.2, "{p:?}");
    assert!(p.r2 <= p.r2_shuffled_max * 3.0 + 0.05, "{p:?}");
}

#[test]
fn position_effect_ignores_cells_that_arent_clonemates() {
    // Half a body is another genome, whose budgets differ. Only the clone is measured.
    let mut f = body_with_contacts(|_| [0.05; 12]);
    for (k, c) in f.cells.iter_mut().enumerate() {
        if k % 2 == 1 {
            *c = with_genome(*c, 127);
            c.budget[2] = 0.9;
        }
    }
    let p = position_effect(&f, 0.4, 7);
    assert_eq!(p.cells, 18);
    assert!(p.variance_within < 1e-6, "{p:?}");
}

#[test]
fn a_summary_has_every_measure() {
    let frames = vec![
        frame(100, &[range(0, 20)]),
        frame(200, &[range(0, 12), range(12, 20)]),
        frame(300, &[range(0, 12), range(12, 20)]),
    ];
    let s = summary(&frames, &RULES);
    assert_eq!(s["dumps"], 3);
    assert_eq!(s["cells"], 20);
    assert_eq!(s["bodies"]["count"], 2);
    assert_eq!(s["tracked_bodies"], 2);
    assert_eq!(s["deepest_body_generation"], 1);
    assert!(s["membership"]["curve"].is_array());
    assert!(s["relatedness"]["gap_within_bodies"].is_number());
    assert!(s["position"]["r2_by_contact_count"].is_number());
}

#[test]
fn a_batch_splits_into_its_worlds() {
    let d = dump::Dump {
        tick: 0,
        size: 128,
        physics: serde_json::json!({"batch_tile": 64}),
        cells: vec![cell(0, 10.0, 10.0, 1.0), cell(1, 70.0, 10.0, 1.0), cell(2, 70.0, 70.0, 1.0)],
    };
    let parts = split_batch(&d, 64);
    assert_eq!(parts.len(), 4);
    assert_eq!(parts[0].1.len(), 1);
    assert_eq!(parts[1].1.len(), 1);
    assert_eq!(parts[2].1.len(), 0);
    assert_eq!(parts[3].1.len(), 1);
    assert!((parts[3].1[0].x - 6.0).abs() < 1e-5);
}

#[test]
fn a_dump_comes_back_as_it_went_in() {
    use std::io::Write;
    let path = std::env::temp_dir().join(format!("soft-dump-test-{}.bin", std::process::id()));
    let cells = row(5, 3, 10.0, 10.0);
    let head = serde_json::json!({
        "tick": 42, "size": 128, "physics": {"gap": 0.2},
        "record_bytes": std::mem::size_of::<DumpCell>(), "cells": 3,
    })
    .to_string();
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&(head.len() as u64).to_le_bytes()).unwrap();
    f.write_all(head.as_bytes()).unwrap();
    f.write_all(bytemuck::cast_slice(&cells)).unwrap();
    drop(f);
    let d = dump::read(path.to_str().unwrap()).unwrap();
    assert_eq!(d.tick, 42);
    assert_eq!(d.size, 128);
    assert_eq!(d.cells.len(), 3);
    assert_eq!(d.cells[1].uid, 6);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn colours_tell_two_genomes_apart() {
    let mut cells: Vec<DumpCell> = (0..30).map(|k| with_genome(cell(k, 10.0, 10.0, 0.0), -80)).collect();
    cells.extend((30..60).map(|k| with_genome(cell(k, 20.0, 10.0, 0.0), 80)));
    let p = dump::Palette::fit(&cells);
    let (a, b) = (p.colour(&cells[0]), p.colour(&cells[59]));
    let gap: i32 = (0..3).map(|k| (a[k] as i32 - b[k] as i32).abs()).sum();
    assert!(gap > 100, "{a:?} against {b:?}");
    assert_eq!(p.colour(&cells[0]), p.colour(&cells[1]), "same genome, same colour");
}

#[test]
fn discs_are_drawn_round_and_wrap() {
    let mut cells = vec![cell(0, 0.0, 16.0, 0.0)];
    cells[0].r = 1.0;
    let (px, rgb) = dump::draw_discs(&cells, 32, 4);
    assert_eq!(px, 128);
    let lit = |x: usize, y: usize| rgb[(y * 128 + x) * 3..(y * 128 + x) * 3 + 3].iter().any(|v| *v > 0);
    // Radius 4 pixels, centred on the world's left edge, so it wraps to the right edge.
    assert!(lit(1, 64));
    assert!(lit(126, 64));
    assert!(!lit(10, 64));
    assert!(!lit(1, 80));
}
