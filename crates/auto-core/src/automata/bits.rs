//! Multi-state automata stored as bit planes, 32 cells per word, updated with
//! bitwise logic. Same speed trick as Life, for rules with more than two states.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

const WG: u32 = 256;
const PARAMS: &str = "w: u32, h: u32, wr: u32, _a: u32,";

/// Counts set neighbours: `one`, `two` are the low bits, `many` means 4 or more.
const COUNT: &str = r#"
struct Count { ones: u32, twos: u32, many: u32 }
fn count8(p: ptr<function, array<u32, 9>>) -> Count {
    let tl = (*p)[0]; let tc = (*p)[1]; let tr = (*p)[2];
    let ml = (*p)[3]; let mc = (*p)[4]; let mr = (*p)[5];
    let bl = (*p)[6]; let bc = (*p)[7]; let br = (*p)[8];
    let n0 = (tc << 1u) | (tl >> 31u);
    let n1 = tc;
    let n2 = (tc >> 1u) | (tr << 31u);
    let n3 = (mc << 1u) | (ml >> 31u);
    let n4 = (mc >> 1u) | (mr << 31u);
    let n5 = (bc << 1u) | (bl >> 31u);
    let n6 = bc;
    let n7 = (bc >> 1u) | (br << 31u);
    let x01 = n0 ^ n1;
    let s0 = x01 ^ n2;
    let c0 = (n0 & n1) | (x01 & n2);
    let x34 = n3 ^ n4;
    let s1 = x34 ^ n5;
    let c1 = (n3 & n4) | (x34 & n5);
    let s2 = n6 ^ n7;
    let c2 = n6 & n7;
    let x = s0 ^ s1;
    let c3 = (s0 & s1) | (x & s2);
    return Count(x ^ s2, c0 ^ c1 ^ c2 ^ c3, (c0 & c1) | (c2 & c3) | ((c0 | c1) & (c2 | c3)));
}
"#;

const LOAD: &str = r#"
    let wr = P.wr; let h = P.h; let x = id.x; let y = id.y;
    if (x >= wr || y >= h) { return; }
    let up = select(y - 1u, h - 1u, y == 0u) * wr;
    let me = y * wr;
    let dn = select(y + 1u, 0u, y + 1u == h) * wr;
    let xl = select(x - 1u, wr - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == wr);
"#;

fn words(w: u32, h: u32) -> u64 {
    (w / 32) as u64 * h as u64
}

fn params(w: u32, h: u32) -> Vec<u8> {
    bytemuck::cast_slice(&[w, h, w / 32, 0u32]).to_vec()
}

fn word_dispatch(w: u32, h: u32) -> [u32; 3] {
    [groups(w / 32, WG), h, 1]
}

fn init_dispatch(w: u32, h: u32) -> [u32; 3] {
    [16384, words(w, h).div_ceil(16384 * 64) as u32, 1]
}

/// Brian Silverman's Brian's Brain: on goes to dying goes to off, off fires on
/// exactly two neighbours.
pub fn brain(w: u32, h: u32) -> Program {
    assert!(w % 32 == 0);
    let step = format!(
        r#"{COUNT}
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) id: vec3u) {{
    {LOAD}
    var a = array<u32, 9>(
        on_cur[up + xl], on_cur[up + x], on_cur[up + xr],
        on_cur[me + xl], on_cur[me + x], on_cur[me + xr],
        on_cur[dn + xl], on_cur[dn + x], on_cur[dn + xr],
    );
    let n = count8(&a);
    let on = a[4];
    let exactly_two = ~n.ones & n.twos & ~n.many;
    on_next[me + x] = ~on & ~dying_cur[me + x] & exactly_two;
    dying_next[me + x] = on;
}}
"#
    );
    let init = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 16384u * 64u;
    if (i >= P.wr * P.h) { return; }
    on_w[i] = rand3(i, 1u, frame.seed) & rand3(i, 2u, frame.seed);
    dying_w[i] = 0u;
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let i = y * P.wr + x / 32u;
    let b = x & 31u;
    if (((view_on[i] >> b) & 1u) == 1u) { return vec3f(0.85, 1.0, 1.0); }
    if (((view_dying[i] >> b) & 1u) == 1u) { return vec3f(0.15, 0.45, 0.95); }
    return vec3f(0.015, 0.02, 0.05);
}
"#;
    Program {
        id: "brain".into(),
        width: w,
        height: h,
        fields: vec![
            Field::double("on", "u32", 4, words(w, h)),
            Field::double("dying", "u32", 4, words(w, h)),
        ],
        params_wgsl: PARAMS.into(),
        params: params(w, h),
        init: vec![Pass::new(
            "init",
            init,
            vec![
                Bind::ReadWrite("on", "on_w"),
                Bind::ReadWrite("dying", "dying_w"),
            ],
            init_dispatch(w, h),
        )],
        step: vec![
            Pass::new(
                "brain",
                step,
                vec![
                    Bind::Read("on", "on_cur"),
                    Bind::Read("dying", "dying_cur"),
                    Bind::Write("on", "on_next"),
                    Bind::Write("dying", "dying_next"),
                ],
                word_dispatch(w, h),
            )
            .swap(&["on", "dying"]),
        ],
        view: view.into(),
        view_binds: vec![
            Bind::Read("on", "view_on"),
            Bind::Read("dying", "view_dying"),
        ],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: 1,
    }
}

/// Brian Silverman's Wireworld. A static copper plane and two moving planes.
pub fn wireworld(w: u32, h: u32) -> Program {
    assert!(w % 32 == 0);
    let step = format!(
        r#"{COUNT}
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) id: vec3u) {{
    {LOAD}
    var a = array<u32, 9>(
        head_cur[up + xl], head_cur[up + x], head_cur[up + xr],
        head_cur[me + xl], head_cur[me + x], head_cur[me + xr],
        head_cur[dn + xl], head_cur[dn + x], head_cur[dn + xr],
    );
    let n = count8(&a);
    let head = a[4];
    let one_or_two = (n.ones ^ n.twos) & ~n.many;
    head_next[me + x] = wire[me + x] & ~head & ~tail_cur[me + x] & one_or_two;
    tail_next[me + x] = head;
}}
"#
    );
    // A random circuit board. Each 32 by 32 tile joins its left or upper neighbour,
    // which makes a forest with no loops, so pulses fan out and end instead of filling
    // the wires. Some tiles hold a ring with one electron in it: a clock.
    let init = r#"
const TILE: u32 = 32u;
fn joins_up(tx: u32, ty: u32) -> bool { return (rand3(tx, ty, frame.seed) & 1u) == 1u; }
fn is_ring(tx: u32, ty: u32) -> bool { return (rand3(tx, ty, frame.seed + 77u) & 15u) == 0u; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 16384u * 64u;
    if (i >= P.wr * P.h) { return; }
    let y = i / P.wr;
    let tx = i % P.wr;
    let ty = y / TILE;
    let ry = y % TILE;
    let tiles_h = P.h / TILE;
    let tr = select(tx + 1u, 0u, tx + 1u == P.wr);
    let td = select(ty + 1u, 0u, ty + 1u >= tiles_h);
    let ring = is_ring(tx, ty);
    var copper = 0u;
    var heads = 0u;
    var tails = 0u;
    if (ry == 16u) {
        if (!joins_up(tx, ty)) { copper |= 0x0001FFFFu; }
        if (!joins_up(tr, ty)) { copper |= 0xFFFF0000u; }
        // Inside a ring, the wire stops at the ring's edge.
        if (ring) { copper &= ~0x003FF800u; }
    }
    if (ry < 16u && joins_up(tx, ty) && !(ring && ry > 10u)) { copper |= 1u << 16u; }
    if (ry >= 16u && joins_up(tx, td) && !(ring && ry < 22u)) { copper |= 1u << 16u; }
    if (ring) {
        let box_hi = 22u; let box_lo = 10u;
        let top = 0x007FFC00u;
        if (ry == box_lo || ry == box_hi) { copper |= top; }
        if (ry > box_lo && ry < box_hi) { copper |= (1u << box_lo) | (1u << box_hi); }
        if (ry == box_lo) {
            let at = 12u + (rand3(tx, ty, frame.seed + 5u) & 7u);
            heads |= 1u << at;
            tails |= 1u << (at - 1u);
        }
    }
    wire_w[i] = copper;
    head_w[i] = heads;
    tail_w[i] = tails;
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let i = y * P.wr + x / 32u;
    let b = x & 31u;
    if (((view_head[i] >> b) & 1u) == 1u) { return vec3f(0.95, 1.0, 1.0); }
    if (((view_tail[i] >> b) & 1u) == 1u) { return vec3f(0.25, 0.6, 1.0); }
    if (((view_wire[i] >> b) & 1u) == 1u) { return vec3f(0.62, 0.36, 0.14); }
    return vec3f(0.02, 0.025, 0.03);
}
"#;
    Program {
        id: "wireworld".into(),
        width: w,
        height: h,
        fields: vec![
            Field::single("wire", "u32", 4, words(w, h)),
            Field::double("head", "u32", 4, words(w, h)),
            Field::double("tail", "u32", 4, words(w, h)),
        ],
        params_wgsl: PARAMS.into(),
        params: params(w, h),
        init: vec![Pass::new(
            "init",
            init,
            vec![
                Bind::ReadWrite("wire", "wire_w"),
                Bind::ReadWrite("head", "head_w"),
                Bind::ReadWrite("tail", "tail_w"),
            ],
            init_dispatch(w, h),
        )],
        step: vec![
            Pass::new(
                "wireworld",
                step,
                vec![
                    Bind::Read("wire", "wire"),
                    Bind::Read("head", "head_cur"),
                    Bind::Read("tail", "tail_cur"),
                    Bind::Write("head", "head_next"),
                    Bind::Write("tail", "tail_next"),
                ],
                word_dispatch(w, h),
            )
            .swap(&["head", "tail"]),
        ],
        view: view.into(),
        view_binds: vec![
            Bind::Read("wire", "view_wire"),
            Bind::Read("head", "view_head"),
            Bind::Read("tail", "view_tail"),
        ],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: 1,
    }
}

/// Wolfram's elementary rules. One row per tick written over a circular history,
/// so only the current row runs. The rule is compiled to a boolean expression.
pub fn elementary(id: &str, rule: u8, w: u32, h: u32, single_seed: bool) -> Program {
    assert!(w % 32 == 0);
    assert!(
        h >= 2,
        "with one row the kernel would read the row it is writing"
    );
    let mut terms = Vec::new();
    for k in 0..8 {
        if rule >> k & 1 == 1 {
            let bit = |set: bool, name: &str| {
                if set {
                    name.to_string()
                } else {
                    format!("~{name}")
                }
            };
            terms.push(format!(
                "({} & {} & {})",
                bit(k & 4 != 0, "l"),
                bit(k & 2 != 0, "c"),
                bit(k & 1 != 0, "r")
            ));
        }
    }
    let expr = if terms.is_empty() {
        "0u".to_string()
    } else {
        terms.join(" | ")
    };
    let step = format!(
        r#"
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let wr = P.wr; let x = id.x;
    if (x >= wr) {{ return; }}
    let src = frame.cursor * wr;
    let dst = (frame.cursor + 1u) % P.h * wr;
    let xl = select(x - 1u, wr - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == wr);
    let c = rows[src + x];
    // Bit i is x offset i, so the left neighbour comes from the lower bit.
    let l = (c << 1u) | (rows[src + xl] >> 31u);
    let r = (c >> 1u) | (rows[src + xr] << 31u);
    rows[dst + x] = {expr};
}}
"#
    );
    let seed_expr = if single_seed {
        "select(0u, 1u << 16u, i == P.wr / 2u)"
    } else {
        "rand3(i, 11u, frame.seed)"
    };
    let init = format!(
        r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x + id.y * 16384u * 64u;
    if (i >= P.wr * P.h) {{ return; }}
    rows[i] = select(0u, {seed_expr}, i < P.wr);
}}
"#
    );
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    // The newest row is at the bottom and history scrolls upwards.
    let row = (frame.cursor + 1u + y) % P.h;
    let alive = (view_rows[row * P.wr + x / 32u] >> (x & 31u)) & 1u;
    let t = f32(y) / f32(P.h);
    return select(vec3f(0.97, 0.96, 0.92), mix3(vec3f(0.1, 0.15, 0.45), vec3f(0.8, 0.1, 0.25), t), alive == 1u);
}
"#;
    Program {
        id: id.into(),
        width: w,
        height: h,
        fields: vec![Field::single("rows", "u32", 4, words(w, h))],
        params_wgsl: PARAMS.into(),
        params: params(w, h),
        init: vec![Pass::new(
            "init",
            init,
            vec![Bind::ReadWrite("rows", "rows")],
            init_dispatch(w, h),
        )],
        step: vec![Pass::new(
            "elementary",
            step,
            vec![Bind::ReadWrite("rows", "rows")],
            [groups(w / 32, WG), 1, 1],
        )],
        view: view.into(),
        view_binds: vec![Bind::Read("rows", "view_rows")],
        cells_per_step: w as u64,
        generations_per_step: 1,
    }
}

pub const BRAIN: Entry = Entry {
    id: "brain",
    name: "Brian's Brain",
    blurb: "Three states, in two bit planes. Fires on exactly 2 neighbours, then a refractory step. Nearly all motion.",
    default_size: (3840, 2160),
    needs_pot: false,
    width_step: 32,
    build: brain,
};

pub const WIREWORLD: Entry = Entry {
    id: "wireworld",
    name: "Wireworld",
    blurb: "Electrons on copper. A random circuit board with ring clocks, in three bit planes.",
    default_size: (1920, 1088),
    needs_pot: false,
    width_step: 32,
    build: wireworld,
};

pub const RULE30: Entry = Entry {
    id: "rule30",
    name: "Rule 30",
    blurb: "Wolfram's chaotic 1D rule from a single cell. One row per tick over a circular history.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 32,
    build: |w, h| elementary("rule30", 30, w, h, true),
};

pub const RULE110: Entry = Entry {
    id: "rule110",
    name: "Rule 110",
    blurb: "The 1D rule Matthew Cook proved Turing complete, from a random row.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 32,
    build: |w, h| elementary("rule110", 110, w, h, false),
};
