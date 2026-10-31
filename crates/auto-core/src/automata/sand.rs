//! Falling sand on a Margolus neighbourhood: cells are grouped into 2x2 blocks,
//! the block grid shifts every tick, and each block rearranges itself in place.
//! Blocks never overlap, so one buffer is enough and matter is conserved.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

pub const ENTRY: Entry = Entry {
    id: "sand",
    name: "Falling sand",
    blurb: "A block automaton, no double buffer. Sand piles, water levels out, through rock.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 2,
    build,
};

pub fn build(w: u32, h: u32) -> Program {
    assert!(w % 2 == 0 && h % 2 == 0);
    let step = r#"
const AIR: u32 = 0u; const ROCK: u32 = 1u; const SAND: u32 = 2u; const WATER: u32 = 3u;
fn kind(c: u32) -> u32 { return c & 255u; }
fn weight(c: u32) -> u32 {
    switch (kind(c)) {
        case 1u: { return 100u; }
        case 2u: { return 3u; }
        case 3u: { return 2u; }
        default: { return 1u; }
    }
}
fn moves(c: u32) -> bool { return kind(c) != ROCK; }
@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let phase = frame.tick & 3u;
    let ox = phase & 1u;
    let oy = (phase >> 1u) & 1u;
    let x = id.x * 2u + ox;
    let y = id.y * 2u + oy;
    if (x + 1u >= P.w || y + 1u >= P.h) { return; }
    let i_tl = y * P.w + x;
    let i_tr = i_tl + 1u;
    let i_bl = i_tl + P.w;
    let i_br = i_bl + 1u;
    var tl = cells[i_tl]; var tr = cells[i_tr]; var bl = cells[i_bl]; var br = cells[i_br];
    let r = rand3(x + y * 65521u, frame.tick, frame.seed);

    // Sources along the top, switching on and off over time.
    if (y < 4u) {
        let band = x / 180u;
        let on = (pcg(band + (frame.tick >> 10u) * 131u) & 3u);
        if (on == 1u && (r & 7u) == 0u && kind(tl) == AIR) { tl = SAND | (r & 0xFFFF00u); }
        if (on == 2u && (r & 3u) == 0u && kind(tl) == AIR) { tl = WATER | (r & 0xFFFF00u); }
    }

    // Gravity: straight down first.
    if (weight(tl) > weight(bl) && moves(tl) && moves(bl)) { let t = tl; tl = bl; bl = t; }
    if (weight(tr) > weight(br) && moves(tr) && moves(br)) { let t = tr; tr = br; br = t; }
    // Then sliding off a pile.
    if (weight(tl) > weight(br) && weight(tl) > weight(tr) && moves(tl) && moves(br) && (r & 16u) != 0u) { let t = tl; tl = br; br = t; }
    else if (weight(tr) > weight(bl) && weight(tr) > weight(tl) && moves(tr) && moves(bl) && (r & 32u) != 0u) { let t = tr; tr = bl; bl = t; }
    // Water spreads sideways.
    if (kind(tl) == WATER && kind(tr) == AIR || kind(tr) == WATER && kind(tl) == AIR) { if ((r & 64u) != 0u) { let t = tl; tl = tr; tr = t; } }
    if (kind(bl) == WATER && kind(br) == AIR || kind(br) == WATER && kind(bl) == AIR) { if ((r & 128u) != 0u) { let t = bl; bl = br; br = t; } }
    // The floor drains slowly so the world never fills up.
    if (y + 2u >= P.h && ((r >> 8u) & 255u) == 0u) {
        if (moves(bl)) { bl = AIR; }
        if (moves(br)) { br = AIR; }
    }

    cells[i_tl] = tl; cells[i_tr] = tr; cells[i_bl] = bl; cells[i_br] = br;
}
"#;
    let init = r#"
fn noise(x: f32, y: f32, s: u32) -> f32 {
    let ix = u32(floor(x)); let iy = u32(floor(y));
    let fx = fract(x); let fy = fract(y);
    let ux = fx * fx * (3.0 - 2.0 * fx); let uy = fy * fy * (3.0 - 2.0 * fy);
    let a = unit_f(rand3(ix, iy, s)); let b = unit_f(rand3(ix + 1u, iy, s));
    let c = unit_f(rand3(ix, iy + 1u, s)); let d = unit_f(rand3(ix + 1u, iy + 1u, s));
    return mix(mix(a, b, ux), mix(c, d, ux), uy);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.w || id.y >= P.h) { return; }
    let fx = f32(id.x); let fy = f32(id.y); let fh = f32(P.h);
    let n = noise(fx / 120.0, fy / 80.0, frame.seed) * 0.65 + noise(fx / 30.0, fy / 25.0, frame.seed + 9u) * 0.35;
    // More rock lower down, none near the top.
    let depth = clamp((fy / fh - 0.18) / 0.82, 0.0, 1.0);
    let rock = n < 0.14 + depth * 0.42;
    cells[id.y * P.w + id.x] = select(0u, 1u | (rand3(id.x, id.y, 3u) & 0xFFFF00u), rock && depth > 0.0);
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let c = view_cells[y * P.w + x];
    let v = f32((c >> 8u) & 255u) / 255.0;
    switch (c & 255u) {
        case 1u: { return mix3(vec3f(0.17, 0.15, 0.17), vec3f(0.26, 0.23, 0.25), v); }
        case 2u: { return mix3(vec3f(0.8, 0.6, 0.28), vec3f(0.98, 0.82, 0.5), v); }
        case 3u: { return mix3(vec3f(0.08, 0.33, 0.75), vec3f(0.2, 0.55, 0.95), v); }
        default: { return mix3(vec3f(0.02, 0.03, 0.07), vec3f(0.06, 0.07, 0.15), f32(y) / f32(P.h)); }
    }
}
"#;
    Program {
        id: "sand".into(),
        width: w,
        height: h,
        fields: vec![Field::single("cells", "u32", 4, w as u64 * h as u64)],
        params_wgsl: "w: u32, h: u32, _a: u32, _b: u32,".into(),
        params: bytemuck::cast_slice(&[w, h, 0u32, 0u32]).to_vec(),
        init: vec![Pass::new(
            "init",
            init,
            vec![Bind::ReadWrite("cells", "cells")],
            [groups(w, 64), h, 1],
        )],
        step: vec![Pass::new(
            "sand",
            step,
            vec![Bind::ReadWrite("cells", "cells")],
            [groups(w / 2, 32), h / 2, 1],
        )],
        view: view.into(),
        view_binds: vec![Bind::Read("cells", "view_cells")],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: 1,
    }
}
