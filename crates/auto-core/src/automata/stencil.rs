//! Integer automata where a cell reads its 8 neighbours: one u32 per cell.
//! A new rule of this kind is one `fn rule` plus a `fn color`.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

const PARAMS: &str = "w: u32, h: u32, _a: u32, _b: u32,";

fn params(w: u32, h: u32) -> Vec<u8> {
    bytemuck::cast_slice(&[w, h, 0u32, 0u32]).to_vec()
}

/// `rule` defines `fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32`,
/// `seed` defines `fn seed(x: u32, y: u32) -> u32` and `view` defines
/// `fn shade(c: u32, x: u32, y: u32) -> vec3f`.
pub fn program(id: &str, w: u32, h: u32, rule: &str, seed: &str, view: &str) -> Program {
    let cells = w as u64 * h as u64;
    let step = format!(
        r#"{rule}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let w = P.w; let h = P.h; let x = id.x; let y = id.y;
    if (x >= w || y >= h) {{ return; }}
    let xl = select(x - 1u, w - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == w);
    let up = select(y - 1u, h - 1u, y == 0u) * w;
    let me = y * w;
    let dn = select(y + 1u, 0u, y + 1u == h) * w;
    let n = array<u32, 8>(
        cur[up + xl], cur[up + x], cur[up + xr],
        cur[me + xl],              cur[me + xr],
        cur[dn + xl], cur[dn + x], cur[dn + xr],
    );
    next[me + x] = rule(cur[me + x], n, x, y);
}}
"#
    );
    let init = format!(
        r#"{seed}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x >= P.w || id.y >= P.h) {{ return; }}
    cur[id.y * P.w + id.x] = seed(id.x, id.y);
}}
"#
    );
    let view = format!(
        r#"{view}
fn color(x: u32, y: u32) -> vec3f {{ return shade(view_cells[y * P.w + x], x, y); }}
"#
    );
    Program {
        id: id.into(),
        width: w,
        height: h,
        fields: vec![Field::double("cells", "u32", 4, cells)],
        params_wgsl: PARAMS.into(),
        params: params(w, h),
        init: vec![Pass::new(
            "init",
            init,
            vec![Bind::ReadWrite("cells", "cur")],
            [groups(w, 64), h, 1],
        )],
        step: vec![
            Pass::new(
                "step",
                step,
                vec![Bind::Read("cells", "cur"), Bind::Write("cells", "next")],
                [groups(w, 64), h, 1],
            )
            .swap(&["cells"]),
        ],
        view,
        view_binds: vec![Bind::Read("cells", "view_cells")],
        cells_per_step: cells,
        generations_per_step: 1,
    }
}

pub const CYCLIC: Entry = Entry {
    id: "cyclic",
    name: "Cyclic cellular automaton",
    blurb: "N colours in a ring. A cell advances if a neighbour is one colour ahead. Noise becomes spirals.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: |w, h| {
        program(
            "cyclic",
            w,
            h,
            r#"
const N: u32 = 16u;
fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32 {
    let succ = (c + 1u) % N;
    var hits = 0u;
    for (var i = 0u; i < 8u; i++) { hits += u32(n[i] == succ); }
    return select(c, succ, hits >= 1u);
}
"#,
            "fn seed(x: u32, y: u32) -> u32 { return rand3(x, y, frame.seed) % 16u; }",
            r#"
fn shade(c: u32, x: u32, y: u32) -> vec3f {
    let t = f32(c) / 16.0 * 6.2831853;
    return vec3f(0.5) + vec3f(0.5, 0.45, 0.5) * cos(t + vec3f(0.0, 2.1, 4.2));
}
"#,
        )
    },
};

pub const HODGEPODGE: Entry = Entry {
    id: "hodgepodge",
    name: "Hodgepodge machine",
    blurb: "Gerhardt and Schuster's model of the Belousov-Zhabotinsky chemical wave reaction.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: |w, h| {
        program(
            "hodgepodge",
            w,
            h,
            r#"
const N: u32 = 200u;
const K1: u32 = 2u;
const K2: u32 = 3u;
const G: u32 = 28u;
fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32 {
    var infected = 0u; var ill = 0u; var sum = c;
    for (var i = 0u; i < 8u; i++) {
        let v = n[i];
        infected += u32(v > 0u && v < N);
        ill += u32(v == N);
        sum += v;
    }
    if (c == N) { return 0u; }
    if (c == 0u) { return min(infected / K1 + ill / K2, N); }
    return min(sum / (infected + ill + 1u) + G, N);
}
"#,
            "fn seed(x: u32, y: u32) -> u32 { return rand3(x, y, frame.seed) % 201u; }",
            r#"
fn shade(c: u32, x: u32, y: u32) -> vec3f {
    return ramp(f32(c) / 200.0, vec3f(0.02, 0.01, 0.08), vec3f(0.33, 0.03, 0.45), vec3f(0.85, 0.2, 0.3), vec3f(0.99, 0.65, 0.2), vec3f(1.0, 0.98, 0.8));
}
"#,
        )
    },
};

pub const FOREST: Entry = Entry {
    id: "forest",
    name: "Forest fire",
    blurb: "Trees grow, lightning strikes, fire spreads. A stochastic rule that drives itself to criticality.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: |w, h| {
        program(
            "forest",
            w,
            h,
            r#"
fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32 {
    if (c >= 2u) { return select(c + 1u, 0u, c >= 9u); }
    let r = unit_f(rand3(x + y * 65521u, frame.tick, frame.seed));
    if (c == 0u) { return u32(r < 0.002); }
    var fire = false;
    for (var i = 0u; i < 8u; i++) { fire = fire || (n[i] >= 2u && n[i] <= 3u); }
    return select(1u, 2u, fire || r < 0.000002);
}
"#,
            "fn seed(x: u32, y: u32) -> u32 { return u32(unit_f(rand3(x, y, frame.seed)) < 0.4); }",
            r#"
fn shade(c: u32, x: u32, y: u32) -> vec3f {
    if (c == 0u) { return vec3f(0.03, 0.025, 0.02); }
    if (c == 1u) {
        let v = f32(pcg(x * 3u + y * 65537u) & 255u) / 255.0;
        return mix3(vec3f(0.05, 0.3, 0.12), vec3f(0.1, 0.5, 0.2), v);
    }
    return ramp(f32(c - 2u) / 7.0, vec3f(1.0, 0.96, 0.6), vec3f(1.0, 0.72, 0.15), vec3f(0.95, 0.3, 0.05), vec3f(0.45, 0.07, 0.03), vec3f(0.1, 0.04, 0.03));
}
"#,
        )
    },
};

pub const SANDPILE: Entry = Entry {
    id: "sandpile",
    name: "Abelian sandpile",
    blurb: "A million grains on one cell. Any cell holding 4 topples to its neighbours, and a fractal appears.",
    default_size: (1024, 1024),
    needs_pot: false,
    width_step: 1,
    build: |w, h| {
        program(
            "sandpile",
            w,
            h,
            r#"
fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32 {
    let gain = (n[1] >> 2u) + (n[3] >> 2u) + (n[4] >> 2u) + (n[6] >> 2u);
    return (c & 3u) + gain;
}
"#,
            r#"
fn seed(x: u32, y: u32) -> u32 {
    // One big pile in the middle, then let it collapse.
    return select(0u, 1u << 20u, x == P.w / 2u && y == P.h / 2u);
}
"#,
            r#"
fn shade(c: u32, x: u32, y: u32) -> vec3f {
    switch (min(c, 4u)) {
        case 0u: { return vec3f(0.03, 0.03, 0.1); }
        case 1u: { return vec3f(0.1, 0.32, 0.65); }
        case 2u: { return vec3f(0.95, 0.75, 0.25); }
        case 3u: { return vec3f(0.8, 0.14, 0.3); }
        default: { return vec3f(1.0); }
    }
}
"#,
        )
    },
};

pub const ANTS: Entry = Entry {
    id: "ants",
    name: "Langton's ants",
    blurb: "Thousands of LLRR turmites as a true CA: each cell keeps a 4-bit mask of ants, so nothing collides.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: |w, h| {
        program(
            "ants",
            w,
            h,
            r#"
// Low byte: colour. Bits 8..12: one flag per heading (N, E, S, W) of ants here.
const COLORS: u32 = 4u;
const RULE: u32 = 0xCu; // Bit c set means turn right on colour c, so LLRR.
fn leaving(c: u32) -> u32 {
    let m = (c >> 8u) & 15u;
    let right = (RULE >> (c & 255u)) & 1u;
    if (right == 1u) { return ((m << 1u) | (m >> 3u)) & 15u; }
    return ((m >> 1u) | (m << 3u)) & 15u;
}
fn rule(c: u32, n: array<u32, 8>, x: u32, y: u32) -> u32 {
    // An ant heading north arrives from the cell below, and so on round.
    let north = leaving(n[6]) & 1u;
    let east = leaving(n[3]) & 2u;
    let south = leaving(n[1]) & 4u;
    let west = leaving(n[4]) & 8u;
    let here = countOneBits((c >> 8u) & 15u);
    return (((c & 255u) + here) % COLORS) | ((north | east | south | west) << 8u);
}
"#,
            r#"
fn seed(x: u32, y: u32) -> u32 {
    let r = rand3(x, y, frame.seed);
    if (r % 8192u != 0u) { return 0u; }
    return 1u << (8u + ((r >> 16u) & 3u));
}
"#,
            r#"
fn shade(c: u32, x: u32, y: u32) -> vec3f {
    if ((c >> 8u) != 0u) { return vec3f(1.0); }
    switch (c & 255u) {
        case 0u: { return vec3f(0.04, 0.05, 0.09); }
        case 1u: { return vec3f(0.1, 0.55, 0.6); }
        case 2u: { return vec3f(0.95, 0.65, 0.2); }
        default: { return vec3f(0.75, 0.15, 0.4); }
    }
}
"#,
        )
    },
};
