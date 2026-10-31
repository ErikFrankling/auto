//! Every Life kernel must match a plain CPU implementation bit for bit.

use auto_core::{Gpu, GpuOptions, Program, Sim, automata::life};

fn gpu() -> Gpu {
    pollster::block_on(Gpu::headless(GpuOptions::default())).expect("needs a GPU or lavapipe")
}

fn reference_step(c: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut n = 0;
            for dy in [h - 1, 0, 1] {
                for dx in [w - 1, 0, 1] {
                    if dx != 0 || dy != 0 {
                        n += c[(y + dy) % h * w + (x + dx) % w];
                    }
                }
            }
            out[y * w + x] = u8::from(n == 3 || (n == 2 && c[y * w + x] == 1));
        }
    }
    out
}

fn soup(w: usize, h: usize) -> Vec<u8> {
    let mut s = 0x1234_5678_9abc_def0u64;
    (0..w * h)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            u8::from(s % 100 < 37)
        })
        .collect()
}

fn pack(c: &[u8]) -> Vec<u8> {
    let mut words = vec![0u32; c.len() / 32];
    for (i, cell) in c.iter().enumerate() {
        words[i / 32] |= (*cell as u32) << (i % 32);
    }
    bytemuck::cast_slice(&words).to_vec()
}

fn check_bits(gpu: &Gpu, program: Program, field: &str, steps: u32) {
    let (w, h) = (program.width as usize, program.height as usize);
    let gens = program.generations_per_step;
    let id = program.id.clone();
    let start = soup(w, h);
    let mut sim = Sim::new(gpu, program, 1).expect("sim");
    gpu.wait();
    sim.write_field(gpu, field, &pack(&start));
    sim.step(gpu, steps);
    gpu.wait();
    let got = sim.read_field(gpu, field);
    let mut want = start;
    for _ in 0..steps * gens {
        want = reference_step(&want, w, h);
    }
    assert_eq!(
        got,
        pack(&want),
        "{id} {w}x{h} differs after {} generations",
        steps * gens
    );
}

#[test]
fn all_kernels_match_reference() {
    let gpu = gpu();
    for (w, h) in [(64, 64), (320, 300), (1024, 512), (96, 2000)] {
        check_bits(&gpu, life::bits32(w, h, [64, 1]), "bits", 7);
        check_bits(&gpu, life::bits32(w, h, [8, 8]), "bits", 3);
        for (t, tile_w, rows) in [
            (1, 30, 64),
            (2, 30, 64),
            (4, 30, 256),
            (8, 14, 128),
            (16, 30, 256),
            (32, 30, 256),
        ] {
            check_bits(&gpu, life::ghost(w, h, t, tile_w, rows), "bits", 3);
        }
    }
    for (w, h) in [(64, 40), (321, 123)] {
        let program = life::naive(w, h);
        let start = soup(w as usize, h as usize);
        let bytes: Vec<u32> = start.iter().map(|c| *c as u32).collect();
        let mut sim = Sim::new(&gpu, program, 1).expect("sim");
        gpu.wait();
        sim.write_field(&gpu, "cells", bytemuck::cast_slice(&bytes));
        sim.step(&gpu, 9);
        gpu.wait();
        let got = sim.read_field(&gpu, "cells");
        let got: &[u32] = bytemuck::cast_slice(&got);
        let mut want = start;
        for _ in 0..9 {
            want = reference_step(&want, w as usize, h as usize);
        }
        let want: Vec<u32> = want.iter().map(|c| *c as u32).collect();
        assert_eq!(got, &want[..], "naive {w}x{h}");
    }
}

fn pack_columns(c: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut words = vec![0u32; w / 32 * h];
    for y in 0..h {
        for x in 0..w {
            words[x / 32 * h + y] |= (c[y * w + x] as u32) << (x % 32);
        }
    }
    bytemuck::cast_slice(&words).to_vec()
}

#[test]
fn column_kernel_matches_reference() {
    let gpu = gpu();
    for (w, h) in [(64, 64), (320, 300), (1024, 512), (96, 2000)] {
        for (t, threads, k) in [
            (1, 256, 3),
            (2, 64, 3),
            (8, 128, 3),
            (16, 256, 3),
            (16, 32, 16),
            (4, 16, 8),
        ] {
            let (wu, hu) = (w as usize, h as usize);
            let start = soup(wu, hu);
            let mut sim = Sim::new(&gpu, life::column(w, h, t, threads, k), 1).expect("sim");
            gpu.wait();
            sim.write_field(&gpu, "s0", &pack_columns(&start, wu, hu));
            sim.step(&gpu, 3);
            gpu.wait();
            let mut want = start;
            for _ in 0..3 * t {
                want = reference_step(&want, wu, hu);
            }
            assert_eq!(
                sim.read_field(&gpu, "s0"),
                pack_columns(&want, wu, hu),
                "t{t} g{threads} k{k} {w}x{h}"
            );
        }
    }
}

#[test]
fn sharded_kernel_matches_reference() {
    let gpu = gpu();
    for (w, h, shards) in [(64, 40, 2), (192, 300, 3), (1024, 512, 4), (512, 700, 8)] {
        let (wu, hu) = (w as usize, h as usize);
        let cols = wu / 32 / shards as usize;
        let start = soup(wu, hu);
        let mut sim = Sim::new(&gpu, life::sharded(w, h, 16, 256, 3, shards), 1).expect("sim");
        gpu.wait();
        let names = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];
        let split = |cells: &[u8], i: usize| -> Vec<u8> {
            pack_columns(cells, wu, hu)[i * cols * hu * 4..(i + 1) * cols * hu * 4].to_vec()
        };
        for i in 0..shards as usize {
            sim.write_field(&gpu, names[i], &split(&start, i));
        }
        sim.step(&gpu, 3);
        gpu.wait();
        let mut want = start;
        for _ in 0..48 {
            want = reference_step(&want, wu, hu);
        }
        for i in 0..shards as usize {
            assert_eq!(
                sim.read_field(&gpu, names[i]),
                split(&want, i),
                "shard {i} of {shards}, {w}x{h}"
            );
        }
    }
}

#[test]
fn bits64_matches_reference() {
    let gpu = gpu();
    if !gpu.has_int64() {
        return;
    }
    let (w, h) = (256usize, 128usize);
    let start = soup(w, h);
    let mut sim = Sim::new(&gpu, life::bits64(w as u32, h as u32), 1).expect("sim");
    gpu.wait();
    sim.write_field(&gpu, "bits", &pack(&start));
    sim.step(&gpu, 11);
    gpu.wait();
    let mut want = start;
    for _ in 0..11 {
        want = reference_step(&want, w, h);
    }
    assert_eq!(sim.read_field(&gpu, "bits"), pack(&want));
}
