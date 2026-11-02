//! Cells on disk, for analysis on the CPU: dumps during a run, and pictures from them.

use bytemuck::{Pod, Zeroable};
use serde_json::{Value, json};

use crate::config::{CELL_BYTES, E_ONE, PLAN_BYTES, SKETCH_BYTES, layout_json};
use crate::world::{Cell, F_ALIVE, Plan, World};

/// One living cell in a dump.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub struct DumpCell {
    pub uid: u32,
    pub parent: u32,
    pub gen_: u32,
    pub clade: u32,
    pub x: f32,
    pub y: f32,
    pub r: f32,
    pub e: f32,
    pub womb: f32,
    pub age: u32,
    pub contacts: u32,
    pub stick: f32,
    pub push: f32,
    pub glue: [f32; 2],
    pub voice: [f32; 2],
    pub hx: f32,
    pub hy: f32,
    pub mu: f32,
    /// rest thrust eat meat, grow emit heat bite, give enzyme stick push.
    pub budget: [f32; 12],
    pub sketch: [u32; 4],
}

impl DumpCell {
    pub fn sketch_f(&self) -> [f32; 16] {
        let mut out = [0.0; 16];
        for (w, word) in self.sketch.iter().enumerate() {
            for k in 0..4 {
                let v = (word >> (k * 8)) as u8 as i8;
                out[w * 4 + k] = (v as f32 / 127.0).max(-1.0) * 4.0;
            }
        }
        out
    }
}

/// A dump's cells and what's known about the world they came from.
pub struct Dump {
    pub tick: u64,
    pub size: u32,
    pub physics: Value,
    pub cells: Vec<DumpCell>,
}

pub fn from_records(cell: &Cell, plan: &Plan, sketch: [u32; 4], to_sq: f32) -> DumpCell {
    let bytes = cell.gv.to_le_bytes();
    let snorm = |k: usize| (bytes[k] as i8 as f32 / 127.0).max(-1.0);
    let mut budget = [0.0; 12];
    budget[..4].copy_from_slice(&plan.s0);
    budget[4..8].copy_from_slice(&plan.s1);
    budget[8..12].copy_from_slice(&plan.s2);
    DumpCell {
        uid: cell.uid,
        parent: cell.parent,
        gen_: cell.generation,
        clade: cell.clade,
        x: cell.x as f32 * to_sq,
        y: cell.y as f32 * to_sq,
        r: cell.r,
        e: cell.e as f32 / E_ONE,
        womb: cell.womb as f32 / E_ONE,
        age: cell.age,
        contacts: cell.contacts,
        stick: cell.stick,
        push: cell.push,
        glue: [snorm(0), snorm(1)],
        voice: [snorm(2), snorm(3)],
        hx: cell.hx,
        hy: cell.hy,
        mu: cell.mu,
        budget,
        sketch,
    }
}

/// Reads every living cell in a world, in short pieces.
pub fn living(world: &mut World) -> Vec<DumpCell> {
    world.flush();
    let to_sq = 1.0 / (1u64 << (32 - world.cfg.size.trailing_zeros())) as f32;
    let slots = world.cfg.slots as u64;
    let chunk = 65536u64;
    let mut out = Vec::new();
    let mut s = 0;
    while s < slots {
        let n = chunk.min(slots - s);
        let cells: Vec<Cell> = bytemuck::pod_collect_to_vec(
            &world.read_ranges("cell", &[(s * CELL_BYTES, n * CELL_BYTES)]),
        );
        let plans: Vec<Plan> = bytemuck::pod_collect_to_vec(
            &world.read_ranges("plan", &[(s * PLAN_BYTES, n * PLAN_BYTES)]),
        );
        let sketches: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(
            &world.read_ranges("sketch", &[(s * SKETCH_BYTES, n * SKETCH_BYTES)]),
        );
        for k in 0..n as usize {
            if cells[k].flags & F_ALIVE != 0 {
                out.push(from_records(&cells[k], &plans[k], sketches[k], to_sq));
            }
        }
        s += n;
    }
    out
}

/// Writes a world's living cells to a file.
pub fn write(world: &mut World, path: &str) -> Result<usize, String> {
    use std::io::Write;
    let cells = living(world);
    let head = json!({
        "tick": world.tick, "size": world.cfg.size, "seed": world.cfg.seed, "physics": world.cfg.physics,
        "layout": layout_json(), "record_bytes": std::mem::size_of::<DumpCell>(), "cells": cells.len(),
    })
    .to_string();
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(|e| format!("{path}: {e}"))?);
    f.write_all(&(head.len() as u64).to_le_bytes()).map_err(|e| e.to_string())?;
    f.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    f.write_all(bytemuck::cast_slice(&cells)).map_err(|e| e.to_string())?;
    Ok(cells.len())
}

pub fn read(path: &str) -> Result<Dump, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if bytes.len() < 8 {
        return Err(format!("{path} isn't a dump"));
    }
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let head: Value = bytes
        .get(8..8 + len)
        .and_then(|h| serde_json::from_slice(h).ok())
        .ok_or_else(|| format!("{path} isn't a dump"))?;
    if head["record_bytes"].as_u64() != Some(std::mem::size_of::<DumpCell>() as u64) {
        return Err(format!("{path} was written by a different version"));
    }
    let cells: Vec<DumpCell> = bytemuck::pod_collect_to_vec(&bytes[8 + len..]);
    Ok(Dump {
        tick: head["tick"].as_u64().unwrap_or(0),
        size: head["size"].as_u64().unwrap_or(0) as u32,
        physics: head["physics"].clone(),
        cells,
    })
}

/// The population's three main axes of genome difference, as a colour for each cell. Similar
/// weights get similar colours. The same idea as the lattice engine's palette.
pub struct Palette {
    pub mean: [f32; 16],
    pub axes: [[f32; 16]; 3],
    pub gain: [f32; 3],
}

impl Palette {
    pub fn fit(cells: &[DumpCell]) -> Palette {
        let every = (cells.len() / 4000).max(1);
        let sample: Vec<[f32; 16]> = cells.iter().step_by(every).map(DumpCell::sketch_f).collect();
        let n = sample.len().max(1) as f32;
        let mut mean = [0.0f32; 16];
        for s in &sample {
            for k in 0..16 {
                mean[k] += s[k] / n;
            }
        }
        let mut cov = [[0.0f32; 16]; 16];
        for s in &sample {
            for a in 0..16 {
                for b in 0..16 {
                    cov[a][b] += (s[a] - mean[a]) * (s[b] - mean[b]) / n;
                }
            }
        }
        let mut axes = [[0.0f32; 16]; 3];
        let mut gain = [1.0f32; 3];
        for axis in 0..3 {
            let mut v: [f32; 16] = std::array::from_fn(|k| if k == axis { 1.0 } else { 0.1 });
            for _ in 0..30 {
                let mut next = [0.0f32; 16];
                for a in 0..16 {
                    for b in 0..16 {
                        next[a] += cov[a][b] * v[b];
                    }
                }
                let len = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
                next.iter_mut().for_each(|x| *x /= len);
                v = next;
            }
            let mut lambda = 0.0;
            let mut cv = [0.0f32; 16];
            for a in 0..16 {
                for b in 0..16 {
                    cv[a] += cov[a][b] * v[b];
                }
                lambda += cv[a] * v[a];
            }
            for a in 0..16 {
                for b in 0..16 {
                    cov[a][b] -= lambda * v[a] * v[b];
                }
            }
            gain[axis] = 1.2 / lambda.max(1e-6).sqrt();
            axes[axis] = v;
        }
        Palette { mean, axes, gain }
    }

    pub fn colour(&self, c: &DumpCell) -> [u8; 3] {
        let s = c.sketch_f();
        std::array::from_fn(|axis| {
            let mut pc = 0.0;
            for k in 0..16 {
                pc += self.axes[axis][k] * (s[k] - self.mean[k]);
            }
            let t = pc * self.gain[axis];
            let t = t / (1.0 + t.abs());
            // Never far from bright, since the ground is black.
            ((0.62 + 0.38 * t) * 255.0) as u8
        })
    }
}

/// Draws cells as discs on black, `scale` pixels to a square, with the wrap. Returns RGB bytes.
pub fn draw_discs(cells: &[DumpCell], size: u32, scale: u32) -> (u32, Vec<u8>) {
    let palette = Palette::fit(cells);
    let px = size * scale;
    let mut rgb = vec![0u8; px as usize * px as usize * 3];
    let wrap = px as i64;
    for c in cells {
        let colour = palette.colour(c);
        let (cx, cy) = (c.x * scale as f32, c.y * scale as f32);
        let r = c.r * scale as f32;
        let reach = r.ceil() as i64 + 1;
        let (x0, y0) = (cx.floor() as i64, cy.floor() as i64);
        for dy in -reach..=reach {
            for dx in -reach..=reach {
                let (fx, fy) = ((x0 + dx) as f32 + 0.5 - cx, (y0 + dy) as f32 + 0.5 - cy);
                let d = (fx * fx + fy * fy).sqrt();
                if d > r {
                    continue;
                }
                // A little darker towards the rim, so touching discs can be told apart.
                let shade = if r >= 2.0 { (1.25 - 0.5 * d / r).min(1.0) } else { 1.0 };
                let (ix, iy) = ((x0 + dx).rem_euclid(wrap), (y0 + dy).rem_euclid(wrap));
                let at = (iy as usize * px as usize + ix as usize) * 3;
                for k in 0..3 {
                    rgb[at + k] = (colour[k] as f32 * shade) as u8;
                }
            }
        }
    }
    (px, rgb)
}

/// Writes a picture of the cells with ffmpeg from PATH. Big enough to see discs, up to about 4096
/// pixels across.
pub fn write_png(cells: &[DumpCell], size: u32, path: &str) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let scale = (4096 / size).clamp(1, 8);
    let (px, rgb) = draw_discs(cells, size, scale);
    if let Some(dir) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s"])
        .arg(format!("{px}x{px}"))
        .args(["-i", "-", "-frames:v", "1", "-update", "1"])
        .arg(path)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    child
        .stdin
        .take()
        .ok_or("ffmpeg has no stdin")?
        .write_all(&rgb)
        .map_err(|e| format!("ffmpeg: {e}"))?;
    let status = child.wait().map_err(|e| format!("ffmpeg: {e}"))?;
    if !status.success() {
        return Err(format!("ffmpeg failed writing {path}"));
    }
    Ok(())
}

/// Reads short runs of slots spread across the world, for about this many living cells. Far
/// less to read than every cell, and enough for a palette.
pub fn sample(world: &mut World, want: usize) -> Vec<DumpCell> {
    world.flush();
    let to_sq = 1.0 / (1u64 << (32 - world.cfg.size.trailing_zeros())) as f32;
    let slots = world.cfg.slots as u64;
    let run = 64u64.min(slots);
    let runs = (want as u64 * 2 / run).clamp(1, slots / run);
    let gap = slots / runs;
    let mut out = Vec::new();
    for k in 0..runs {
        let s = k * gap;
        let of = |stride: u64| [(s * stride, run * stride)];
        let cells: Vec<Cell> = bytemuck::pod_collect_to_vec(&world.read_ranges("cell", &of(CELL_BYTES)));
        let plans: Vec<Plan> = bytemuck::pod_collect_to_vec(&world.read_ranges("plan", &of(PLAN_BYTES)));
        let sketches: Vec<[u32; 4]> = bytemuck::pod_collect_to_vec(&world.read_ranges("sketch", &of(SKETCH_BYTES)));
        for i in 0..run as usize {
            if cells[i].flags & F_ALIVE != 0 {
                out.push(from_records(&cells[i], &plans[i], sketches[i], to_sq));
            }
        }
    }
    out
}
