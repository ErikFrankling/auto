//! Each automaton takes its GPU-seeded state, runs a few steps, and has to match
//! a plain CPU version of the same rule.

use auto_core::{Gpu, GpuOptions, Sim, automata};

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless(GpuOptions::default())).expect("needs a GPU or lavapipe")
}

fn read_u32(sim: &Sim, gpu: &Gpu, field: &str) -> Vec<u32> {
    bytemuck::cast_slice(&sim.read_field(gpu, field)).to_vec()
}

fn setup(id: &str, w: u32, h: u32, seed: u32) -> (Gpu, Sim) {
    let gpu = gpu();
    let program = automata::build(id, w, h).expect("known id");
    let sim = Sim::new(&gpu, program, seed).expect("sim");
    gpu.wait();
    (gpu, sim)
}

fn neighbours(w: usize, h: usize, x: usize, y: usize) -> [usize; 8] {
    let at = |dx: usize, dy: usize| (y + dy) % h * w + (x + dx) % w;
    let (l, r) = (w - 1, 1);
    let (u, d) = (h - 1, 1);
    [
        at(l, u),
        at(0, u),
        at(r, u),
        at(l, 0),
        at(r, 0),
        at(l, d),
        at(0, d),
        at(r, d),
    ]
}

/// Compare the `cells` field with a CPU rule over `steps` ticks.
fn check_stencil(id: &str, steps: u32, rule: impl Fn(u32, [u32; 8]) -> u32) {
    let (w, h) = (97usize, 61usize);
    let (gpu, mut sim) = setup(id, w as u32, h as u32, 5);
    let mut want = read_u32(&sim, &gpu, "cells");
    sim.step(&gpu, steps);
    gpu.wait();
    for _ in 0..steps {
        let mut next = vec![0; w * h];
        for y in 0..h {
            for x in 0..w {
                let n = neighbours(w, h, x, y).map(|i| want[i]);
                next[y * w + x] = rule(want[y * w + x], n);
            }
        }
        want = next;
    }
    assert_eq!(read_u32(&sim, &gpu, "cells"), want, "{id}");
}

#[test]
fn cyclic() {
    check_stencil("cyclic", 20, |c, n| {
        let succ = (c + 1) % 16;
        if n.contains(&succ) { succ } else { c }
    });
}

#[test]
fn hodgepodge() {
    check_stencil("hodgepodge", 15, |c, n| {
        const N: u32 = 200;
        let infected = n.iter().filter(|v| **v > 0 && **v < N).count() as u32;
        let ill = n.iter().filter(|v| **v == N).count() as u32;
        let sum: u32 = c + n.iter().sum::<u32>();
        if c == N {
            0
        } else if c == 0 {
            (infected / 2 + ill / 3).min(N)
        } else {
            (sum / (infected + ill + 1) + 28).min(N)
        }
    });
}

#[test]
fn ants() {
    check_stencil("ants", 40, |c, n| {
        let leaving = |v: u32| {
            let m = (v >> 8) & 15;
            if (0xCu32 >> (v & 255)) & 1 == 1 {
                ((m << 1) | (m >> 3)) & 15
            } else {
                ((m >> 1) | (m << 3)) & 15
            }
        };
        let arrive =
            (leaving(n[6]) & 1) | (leaving(n[3]) & 2) | (leaving(n[1]) & 4) | (leaving(n[4]) & 8);
        let here = ((c >> 8) & 15).count_ones();
        (((c & 255) + here) % 4) | (arrive << 8)
    });
}

#[test]
fn sandpile_conserves_grains() {
    let (gpu, mut sim) = setup("sandpile", 200, 200, 1);
    sim.step(&gpu, 5000);
    gpu.wait();
    let total: u64 = read_u32(&sim, &gpu, "cells")
        .iter()
        .map(|c| *c as u64)
        .sum();
    assert_eq!(total, 1 << 20);
}

fn bit(words: &[u32], w: usize, x: usize, y: usize) -> u32 {
    (words[y * (w / 32) + x / 32] >> (x % 32)) & 1
}

fn count(words: &[u32], w: usize, h: usize, x: usize, y: usize) -> u32 {
    let mut n = 0;
    for dy in [h - 1, 0, 1] {
        for dx in [w - 1, 0, 1] {
            if dx != 0 || dy != 0 {
                n += bit(words, w, (x + dx) % w, (y + dy) % h);
            }
        }
    }
    n
}

fn set(words: &mut [u32], w: usize, x: usize, y: usize, v: u32) {
    words[y * (w / 32) + x / 32] |= v << (x % 32);
}

#[test]
fn brain() {
    let (w, h) = (128usize, 90usize);
    let (gpu, mut sim) = setup("brain", w as u32, h as u32, 3);
    let mut on = read_u32(&sim, &gpu, "on");
    let mut dying = read_u32(&sim, &gpu, "dying");
    sim.step(&gpu, 12);
    gpu.wait();
    for _ in 0..12 {
        let mut next_on = vec![0; on.len()];
        for y in 0..h {
            for x in 0..w {
                let idle = bit(&on, w, x, y) == 0 && bit(&dying, w, x, y) == 0;
                set(
                    &mut next_on,
                    w,
                    x,
                    y,
                    u32::from(idle && count(&on, w, h, x, y) == 2),
                );
            }
        }
        dying = on;
        on = next_on;
    }
    assert_eq!(read_u32(&sim, &gpu, "on"), on);
    assert_eq!(read_u32(&sim, &gpu, "dying"), dying);
}

#[test]
fn wireworld() {
    let (w, h) = (160usize, 128usize);
    let (gpu, mut sim) = setup("wireworld", w as u32, h as u32, 2);
    let wire = read_u32(&sim, &gpu, "wire");
    let mut head = read_u32(&sim, &gpu, "head");
    let mut tail = read_u32(&sim, &gpu, "tail");
    assert!(
        head.iter().any(|v| *v != 0),
        "the board should start with electrons"
    );
    sim.step(&gpu, 50);
    gpu.wait();
    for _ in 0..50 {
        let mut next = vec![0; head.len()];
        for y in 0..h {
            for x in 0..w {
                let n = count(&head, w, h, x, y);
                let fire = bit(&wire, w, x, y) == 1
                    && bit(&head, w, x, y) == 0
                    && bit(&tail, w, x, y) == 0
                    && (n == 1 || n == 2);
                set(&mut next, w, x, y, u32::from(fire));
            }
        }
        tail = head;
        head = next;
    }
    assert_eq!(read_u32(&sim, &gpu, "head"), head);
    assert_eq!(read_u32(&sim, &gpu, "tail"), tail);
}

#[test]
fn elementary_rules() {
    for (id, rule) in [("rule30", 30u8), ("rule110", 110u8)] {
        let (w, h) = (96usize, 40usize);
        let (gpu, mut sim) = setup(id, w as u32, h as u32, 9);
        let start = read_u32(&sim, &gpu, "rows");
        let mut row: Vec<u32> = (0..w).map(|x| bit(&start, w, x, 0)).collect();
        sim.step(&gpu, 25);
        gpu.wait();
        for _ in 0..25 {
            row = (0..w)
                .map(|x| {
                    let k = (row[(x + w - 1) % w] << 2) | (row[x] << 1) | row[(x + 1) % w];
                    ((rule >> k) & 1) as u32
                })
                .collect();
        }
        let got = read_u32(&sim, &gpu, "rows");
        let got: Vec<u32> = (0..w).map(|x| bit(&got, w, x, 25)).collect();
        assert_eq!(got, row, "{id}");
    }
}

/// The benchmark's two-row history must evolve the same as a long one.
#[test]
fn elementary_with_two_rows_of_history() {
    let w = 96usize;
    let (gpu, mut sim) = setup("rule110", w as u32, 2, 9);
    let start = read_u32(&sim, &gpu, "rows");
    let mut row: Vec<u32> = (0..w).map(|x| bit(&start, w, x, 0)).collect();
    sim.step(&gpu, 25);
    gpu.wait();
    for _ in 0..25 {
        row = (0..w)
            .map(|x| {
                let k = (row[(x + w - 1) % w] << 2) | (row[x] << 1) | row[(x + 1) % w];
                ((110u8 >> k) & 1) as u32
            })
            .collect();
    }
    let got = read_u32(&sim, &gpu, "rows");
    // After 25 ticks from row 0, the newest row is row 1.
    let got: Vec<u32> = (0..w).map(|x| bit(&got, w, x, 1)).collect();
    assert_eq!(got, row);
}

#[test]
fn sand_keeps_rock_in_place() {
    let (gpu, mut sim) = setup("sand", 128, 96, 4);
    let rock = |cells: &[u32]| {
        cells
            .iter()
            .map(|c| u32::from(c & 255 == 1))
            .collect::<Vec<_>>()
    };
    let before = rock(&read_u32(&sim, &gpu, "cells"));
    sim.step(&gpu, 2000);
    gpu.wait();
    let after_cells = read_u32(&sim, &gpu, "cells");
    assert_eq!(before, rock(&after_cells));
    assert!(
        after_cells.iter().any(|c| c & 255 == 2 || c & 255 == 3),
        "sources should have poured"
    );
}

#[test]
fn gray_scott_matches_cpu() {
    let (w, h) = (64usize, 48usize);
    let (gpu, mut sim) = setup("grayscott", w as u32, h as u32, 6);
    let bytes = sim.read_field(&gpu, "cells");
    let mut want: Vec<[f32; 2]> = bytemuck::cast_slice(&bytes).to_vec();
    sim.step(&gpu, 30);
    gpu.wait();
    for _ in 0..30 {
        let mut next = want.clone();
        for y in 0..h {
            for x in 0..w {
                let n = neighbours(w, h, x, y).map(|i| want[i]);
                let c = want[y * w + x];
                let lap = |k: usize| {
                    (n[3][k] + n[4][k] + n[1][k] + n[6][k]) * 0.2
                        + (n[0][k] + n[2][k] + n[5][k] + n[7][k]) * 0.05
                        - c[k]
                };
                let feed = 0.022 + 0.040 * (y as f32 / h as f32);
                let kill = 0.056 + 0.0095 * (x as f32 / w as f32);
                let re = c[0] * c[1] * c[1];
                next[y * w + x] = [
                    c[0] + lap(0) - re + feed * (1.0 - c[0]),
                    c[1] + 0.5 * lap(1) + re - (feed + kill) * c[1],
                ];
            }
        }
        want = next;
    }
    let got = sim.read_field(&gpu, "cells");
    let got: &[[f32; 2]] = bytemuck::cast_slice(&got);
    for (a, b) in got.iter().zip(&want) {
        assert!(
            (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3,
            "{a:?} vs {b:?}"
        );
    }
}

/// The FFT path has to give the same answer as adding up the neighbourhood directly.
#[test]
fn lenia_fft_matches_direct_convolution() {
    let (w, h) = (128usize, 64usize);
    // A small world has few creature slots, so find a seed that places one.
    let (gpu, mut sim, bytes) = (1..64)
        .find_map(|seed| {
            let (gpu, sim) = setup("lenia", w as u32, h as u32, seed);
            let bytes = sim.read_field(&gpu, "state");
            let found = bytemuck::cast_slice::<u8, f32>(&bytes)
                .iter()
                .any(|v| *v > 0.0);
            found.then_some((gpu, sim, bytes))
        })
        .expect("some seed places an Orbium");
    let start: &[f32] = bytemuck::cast_slice(&bytes);

    let radius = 13.0f64;
    let core = |r: f64| {
        let q = r / radius;
        if q <= 0.0 || q >= 1.0 {
            0.0
        } else {
            (4.0 - 1.0 / (q * (1.0 - q))).exp()
        }
    };
    let mut sum = 0.0;
    for dy in -14i32..=14 {
        for dx in -14i32..=14 {
            sum += core(((dx * dx + dy * dy) as f64).sqrt());
        }
    }
    let mut want = vec![0.0f32; w * h];
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut u = 0.0;
            for dy in -14i32..=14 {
                for dx in -14i32..=14 {
                    let k = core(((dx * dx + dy * dy) as f64).sqrt());
                    if k > 0.0 {
                        let i = (y + dy).rem_euclid(h as i32) as usize * w
                            + (x + dx).rem_euclid(w as i32) as usize;
                        u += k * start[i] as f64;
                    }
                }
            }
            let d = (u / sum - 0.15) / 0.015;
            let g = 2.0 * (-0.5 * d * d).exp() - 1.0;
            want[y as usize * w + x as usize] =
                (start[y as usize * w + x as usize] as f64 + 0.1 * g).clamp(0.0, 1.0) as f32;
        }
    }
    sim.step(&gpu, 1);
    gpu.wait();
    let got = sim.read_field(&gpu, "state");
    let got: &[f32] = bytemuck::cast_slice(&got);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 0.002, "worst cell differs by {worst}");
}

#[test]
fn physarum_agents_stay_in_the_world() {
    let (gpu, mut sim) = setup("physarum", 128, 96, 1);
    sim.step(&gpu, 100);
    gpu.wait();
    let bytes = sim.read_field(&gpu, "agents");
    let agents: &[[f32; 4]] = bytemuck::cast_slice(&bytes);
    assert!(
        agents
            .iter()
            .all(|a| a[0] >= 0.0 && a[0] < 128.0 && a[1] >= 0.0 && a[1] < 96.0)
    );
    let trail = sim.read_field(&gpu, "trail");
    let trail: &[f32] = bytemuck::cast_slice(&trail);
    assert!(trail.iter().sum::<f32>() > 0.0);
}

/// The blocked brain kernel must compute the same RNN as a plain matrix multiply.
#[test]
fn blocked_brains_match_cpu() {
    let (agents, hidden) = (64u32, 8u32);
    let (gpu, mut sim) = setup(&format!("brainsb.h{hidden}.b32"), agents, 1, 3);
    let rows = (hidden / 4) as usize;
    let bytes = sim.read_field(&gpu, "genome");
    let genome: &[[[f32; 4]; 4]] = bytemuck::cast_slice(&bytes);
    let mut state = vec![[0.0f32; 4]; agents as usize * rows];
    assert!(genome.iter().flatten().flatten().all(|w| w.abs() <= 0.5));

    let pcg = |v: u32| {
        let s = v.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let rand3 = |a: u32, b: u32, c: u32| pcg(pcg(pcg(a) ^ b) ^ c);
    for tick in 0..5u32 {
        let mut next = state.clone();
        for a in 0..agents as usize {
            let r = rand3(a as u32, tick, 3);
            let bit = |k: u32| ((r >> k) & 1) as f32;
            let inputs = [
                [bit(0), bit(1), bit(2), bit(3)],
                [bit(4), bit(5), bit(6), bit(7)],
            ];
            for row in 0..rows {
                let base = (a * rows + row) * (rows + 2);
                let mut acc = [0.0f32; 4];
                let mut add = |m: &[[f32; 4]; 4], v: &[f32; 4]| {
                    // WGSL matrices are column major: result[r] = sum over c of m[c][r] * v[c].
                    for (r_i, out) in acc.iter_mut().enumerate() {
                        *out += (0..4).map(|c| m[c][r_i] * v[c]).sum::<f32>();
                    }
                };
                for j in 0..rows {
                    add(&genome[base + j], &state[a * rows + j]);
                }
                add(&genome[base + rows], &inputs[0]);
                add(&genome[base + rows + 1], &inputs[1]);
                next[a * rows + row] = acc.map(|v| v / (1.0 + v.abs()));
            }
        }
        state = next;
    }
    sim.step(&gpu, 5);
    gpu.wait();
    let got = sim.read_field(&gpu, "state");
    let got: &[[f32; 4]] = bytemuck::cast_slice(&got);
    for (a, b) in got.iter().flatten().zip(state.iter().flatten()) {
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
    }
}
