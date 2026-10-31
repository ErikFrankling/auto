//! Conway's Game of Life, and every other life-like rule, in several kernels.
//! The kernels exist to find out what is fastest; `ENTRY` uses the winner.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

pub const ENTRY: Entry = Entry {
    id: "life",
    name: "Conway's Game of Life",
    blurb: "B3/S23. 32 cells per word, 16 generations per dispatch in GPU shared memory.",
    default_size: (3840, 2160),
    needs_pot: false,
    width_step: 32,
    build: |w, h| column(w, h, 16, 256, 3),
};

/// Workgroup shape of the 32-bit kernel, in words.
pub const BITS_WG: [u32; 2] = [64, 1];

const PARAMS: &str = "w: u32, h: u32, wr: u32, density: f32,";

fn params(w: u32, h: u32, wr: u32, density: f32) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(bytemuck::cast_slice(&[w, h, wr]));
    v.extend_from_slice(bytemuck::bytes_of(&density));
    v
}

/// Adds up the eight neighbour words and applies Life's rule in parallel on
/// every bit. Bit i is the cell at x offset i, so west means a left shift.
const SWAR: &str = r#"
fn life_word(
    tl: u32, tc: u32, tr: u32,
    ml: u32, mc: u32, mr: u32,
    bl: u32, bc: u32, br: u32,
) -> u32 {
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
    let ones = x ^ s2;
    let c3 = (s0 & s1) | (x & s2);
    let twos = c0 ^ c1 ^ c2 ^ c3;
    let many = (c0 & c1) | (c2 & c3) | ((c0 | c1) & (c2 | c3));
    return ~many & twos & (ones | mc);
}
"#;

const RANDOM_BITS: &str = r#"
fn random_word(i: u32) -> u32 {
    var word = 0u;
    for (var b = 0u; b < 32u; b += 8u) {
        let r = rand3(i, b, frame.seed);
        for (var k = 0u; k < 4u; k++) {
            let byte = (r >> (k * 8u)) & 255u;
            if (f32(byte) < P.density * 256.0) { word |= 1u << (b + k); }
        }
        let r2 = rand3(i, b + 1000u, frame.seed);
        for (var k = 0u; k < 4u; k++) {
            let byte = (r2 >> (k * 8u)) & 255u;
            if (f32(byte) < P.density * 256.0) { word |= 1u << (b + k + 4u); }
        }
    }
    return word;
}
"#;

const BIT_VIEW: &str = r#"
fn color(x: u32, y: u32) -> vec3f {
    let word = view_bits[y * P.wr + x / 32u];
    let alive = (word >> (x & 31u)) & 1u;
    return select(vec3f(0.035, 0.04, 0.07), vec3f(0.62, 0.96, 0.78), alive == 1u);
}
"#;

fn bit_init() -> String {
    format!(
        r#"{RANDOM_BITS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let wg = P.wr * P.h;
    let i = id.x + id.y * 16384u * 64u;
    if (i < wg) {{ cur[i] = random_word(i); }}
}}
"#
    )
}

fn bit_init_pass(words: u64) -> Pass {
    let per = 16384u64 * 64;
    Pass::new(
        "life init",
        bit_init(),
        vec![Bind::ReadWrite("bits", "cur")],
        [16384, words.div_ceil(per) as u32, 1],
    )
}

/// One generation per dispatch, 32 cells per invocation.
pub fn bits32(w: u32, h: u32, wg: [u32; 2]) -> Program {
    assert!(w % 32 == 0, "width must be a multiple of 32");
    let wr = w / 32;
    let words = wr as u64 * h as u64;
    let source = format!(
        r#"{SWAR}
@compute @workgroup_size({wx}, {wy})
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let wr = P.wr;
    let h = P.h;
    let x = id.x;
    let y = id.y;
    if (x >= wr || y >= h) {{ return; }}
    let up = select(y - 1u, h - 1u, y == 0u) * wr;
    let me = y * wr;
    let dn = select(y + 1u, 0u, y + 1u == h) * wr;
    let xl = select(x - 1u, wr - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == wr);
    next[me + x] = life_word(
        cur[up + xl], cur[up + x], cur[up + xr],
        cur[me + xl], cur[me + x], cur[me + xr],
        cur[dn + xl], cur[dn + x], cur[dn + xr],
    );
}}
"#,
        wx = wg[0],
        wy = wg[1],
    );
    Program {
        id: format!("life.bits32.{}x{}", wg[0], wg[1]),
        width: w,
        height: h,
        fields: vec![Field::double("bits", "u32", 4, words)],
        params_wgsl: PARAMS.into(),
        params: params(w, h, wr, 0.35),
        init: vec![bit_init_pass(words)],
        step: vec![
            Pass::new(
                "life bits32",
                source,
                vec![Bind::Read("bits", "cur"), Bind::Write("bits", "next")],
                [groups(wr, wg[0]), groups(h, wg[1]), 1],
            )
            .swap(&["bits"]),
        ],
        view: BIT_VIEW.into(),
        view_binds: vec![Bind::Read("bits", "view_bits")],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: 1,
    }
}

/// `T` generations per dispatch, computed in workgroup shared memory.
///
/// Each workgroup copies a tile plus a halo out of the big buffer, runs `t`
/// generations in shared memory, and writes the tile's middle back. Global memory
/// is touched once per `t` generations, which is what limits dense Life at scale.
///
/// One invocation owns each row and slides along it, so each word costs three
/// shared memory reads rather than nine. Horizontally a one word halo is enough
/// for `t <= 32`: wrong bits creep in from the outer edge one per generation.
pub fn ghost(w: u32, h: u32, t: u32, tile_w: u32, rows: u32) -> Program {
    assert!(w % 32 == 0, "width must be a multiple of 32");
    assert!(
        (1..=32).contains(&t),
        "the horizontal halo only covers 32 generations"
    );
    assert!(rows > 2 * t, "tile too short for the halo");
    let wr = w / 32;
    let words = wr as u64 * h as u64;
    let tile_h = rows - 2 * t;
    let cols = tile_w + 2;
    let source = format!(
        r#"{SWAR}
const T: u32 = {t}u;
const TW: u32 = {tile_w}u;
const TH: u32 = {tile_h}u;
const W: u32 = {cols}u;
const R: u32 = {rows}u;
var<workgroup> buf: array<array<u32, {shared}>, 2>;

@compute @workgroup_size({rows})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) row: u32) {{
    let wr = P.wr;
    let h = P.h;
    let x0 = wg.x * TW;
    let y0 = wg.y * TH + row;

    // Load this row of the tile and its halo.
    let y_world = (y0 + h * (T / h + 1u) - T) % h;
    let base = y_world * wr;
    var xw = (x0 + wr * (1u / wr + 1u) - 1u) % wr;
    let me = row * W;
    for (var c = 0u; c < W; c++) {{
        buf[0][me + c] = cur[base + xw];
        xw = select(xw + 1u, 0u, xw + 1u == wr);
    }}
    workgroupBarrier();

    for (var g = 1u; g <= T; g++) {{
        if (row >= g && row < R - g) {{
            let a = (g - 1u) & 1u;
            let b = g & 1u;
            let up = me - W;
            let dn = me + W;
            var tl = 0u; var ml = 0u; var bl = 0u;
            var tc = buf[a][up]; var mc = buf[a][me]; var bc = buf[a][dn];
            for (var c = 0u; c < W; c++) {{
                var tr = 0u; var mr = 0u; var br = 0u;
                if (c + 1u < W) {{
                    tr = buf[a][up + c + 1u];
                    mr = buf[a][me + c + 1u];
                    br = buf[a][dn + c + 1u];
                }}
                buf[b][me + c] = life_word(tl, tc, tr, ml, mc, mr, bl, bc, br);
                tl = tc; ml = mc; bl = bc;
                tc = tr; mc = mr; bc = br;
            }}
        }}
        workgroupBarrier();
    }}

    // Store the valid middle.
    if (row >= T && row < T + TH && y0 - T < h) {{
        let out = (y0 - T) * wr;
        let last = T & 1u;
        for (var c = 0u; c < TW; c++) {{
            let x = x0 + c;
            if (x < wr) {{ next[out + x] = buf[last][me + c + 1u]; }}
        }}
    }}
}}
"#,
        shared = cols * rows,
    );
    Program {
        id: format!("life.ghost.t{t}.w{tile_w}.r{rows}"),
        width: w,
        height: h,
        fields: vec![Field::double("bits", "u32", 4, words)],
        params_wgsl: PARAMS.into(),
        params: params(w, h, wr, 0.35),
        init: vec![bit_init_pass(words)],
        step: vec![
            Pass::new(
                "life ghost",
                source,
                vec![Bind::Read("bits", "cur"), Bind::Write("bits", "next")],
                [groups(wr, tile_w), groups(h, tile_h), 1],
            )
            .swap(&["bits"]),
        ],
        view: BIT_VIEW.into(),
        view_binds: vec![Bind::Read("bits", "view_bits")],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: t,
    }
}

/// The textbook version: one u32 per cell, eight reads per cell. The baseline.
pub fn naive(w: u32, h: u32) -> Program {
    let cells = w as u64 * h as u64;
    let source = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let w = P.w;
    let h = P.h;
    let x = id.x;
    let y = id.y;
    if (x >= w || y >= h) { return; }
    let xl = select(x - 1u, w - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == w);
    let up = select(y - 1u, h - 1u, y == 0u) * w;
    let me = y * w;
    let dn = select(y + 1u, 0u, y + 1u == h) * w;
    let n = cur[up + xl] + cur[up + x] + cur[up + xr]
          + cur[me + xl] + cur[me + xr]
          + cur[dn + xl] + cur[dn + x] + cur[dn + xr];
    next[me + x] = u32(n == 3u || (n == 2u && cur[me + x] == 1u));
}
"#;
    let init = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.w || id.y >= P.h) { return; }
    let i = id.y * P.w + id.x;
    cur[i] = u32(unit_f(rand3(i, 7u, frame.seed)) < P.density);
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    return select(vec3f(0.035, 0.04, 0.07), vec3f(0.62, 0.96, 0.78), view_cells[y * P.w + x] == 1u);
}
"#;
    Program {
        id: "life.naive".into(),
        width: w,
        height: h,
        fields: vec![Field::double("cells", "u32", 4, cells)],
        params_wgsl: PARAMS.into(),
        params: params(w, h, w, 0.35),
        init: vec![Pass::new(
            "life naive init",
            init,
            vec![Bind::ReadWrite("cells", "cur")],
            [groups(w, 64), h, 1],
        )],
        step: vec![
            Pass::new(
                "life naive",
                source,
                vec![Bind::Read("cells", "cur"), Bind::Write("cells", "next")],
                [groups(w, 64), h, 1],
            )
            .swap(&["cells"]),
        ],
        view: view.into(),
        view_binds: vec![Bind::Read("cells", "view_cells")],
        cells_per_step: cells,
        generations_per_step: 1,
    }
}

/// 64 cells per invocation. Needs SHADER_INT64, so native only.
pub fn bits64(w: u32, h: u32) -> Program {
    assert!(w % 64 == 0, "width must be a multiple of 64");
    let wr = w / 64;
    let words = wr as u64 * h as u64;
    let swar = SWAR.replace("u32", "u64").replace("31u", "63u");
    let source = format!(
        r#"{swar}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let wr = P.wr;
    let h = P.h;
    let x = id.x;
    let y = id.y;
    if (x >= wr || y >= h) {{ return; }}
    let up = select(y - 1u, h - 1u, y == 0u) * wr;
    let me = y * wr;
    let dn = select(y + 1u, 0u, y + 1u == h) * wr;
    let xl = select(x - 1u, wr - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == wr);
    next[me + x] = life_word(
        cur[up + xl], cur[up + x], cur[up + xr],
        cur[me + xl], cur[me + x], cur[me + xr],
        cur[dn + xl], cur[dn + x], cur[dn + xr],
    );
}}
"#
    );
    let init = format!(
        r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x + id.y * 16384u * 64u;
    if (i >= P.wr * P.h) {{ return; }}
    let a = rand3(i, 1u, frame.seed) & rand3(i, 2u, frame.seed);
    let b = rand3(i, 3u, frame.seed) & rand3(i, 4u, frame.seed);
    cur[i] = (u64(a) << 32u) | u64(b);
}}
"#
    );
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let word = view_bits[y * P.wr + x / 64u];
    let alive = (word >> (x & 63u)) & 1lu;
    return select(vec3f(0.035, 0.04, 0.07), vec3f(0.62, 0.96, 0.78), alive == 1lu);
}
"#;
    let per = 16384u64 * 64;
    Program {
        id: "life.bits64".into(),
        width: w,
        height: h,
        fields: vec![Field::double("bits", "u64", 8, words)],
        params_wgsl: PARAMS.into(),
        params: params(w, h, wr, 0.25),
        init: vec![Pass::new(
            "life64 init",
            init,
            vec![Bind::ReadWrite("bits", "cur")],
            [16384, words.div_ceil(per) as u32, 1],
        )],
        step: vec![
            Pass::new(
                "life bits64",
                source,
                vec![Bind::Read("bits", "cur"), Bind::Write("bits", "next")],
                [groups(wr, 64), h, 1],
            )
            .swap(&["bits"]),
        ],
        view: view.into(),
        view_binds: vec![Bind::Read("bits", "view_bits")],
        cells_per_step: w as u64 * h as u64,
        generations_per_step: 1,
    }
}

/// Temporal blocking on a column-major grid, in the style of Dijkstra and
/// Brouwer's fast-game-of-life: a workgroup owns one tall column of words, kept
/// in two overlapping windows so up to 16 generations stay valid sideways. Rows
/// are contiguous in memory, and each invocation holds `k` rows in registers
/// between barriers.
pub fn column(w: u32, h: u32, t: u32, threads: u32, k: u32) -> Program {
    sharded(w, h, t, threads, k, 1)
}

const SHARDS: [&str; 8] = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"];

/// The column kernel over several buffers, for worlds past the single buffer
/// limit. Each shard owns a run of columns and reads one column from each
/// neighbouring shard, so a dispatch binds at most 3 read buffers and 1 write.
pub fn sharded(w: u32, h: u32, t: u32, threads: u32, k: u32, shards: u32) -> Program {
    assert!(w % 32 == 0, "width must be a multiple of 32");
    assert!(
        (1..=16).contains(&t),
        "the two windows only cover 16 generations"
    );
    assert!((1..=8).contains(&shards));
    let wr = w / 32;
    assert!(wr % shards == 0, "columns must split evenly across shards");
    let cols = wr / shards;
    let shard_words = cols as u64 * h as u64;
    let rows = threads * k;
    assert!(rows > 2 * t, "tile too short for the halo");
    let sim = rows - 2 * t;
    let mut compute = String::new();
    let mut publish = String::new();
    for row in 0..k {
        compute += &format!(
            r#"
            let m{row} = py + {row}u;
            let u{row} = select(m{row} - 1u, 0u, m{row} == 0u);
            let d{row} = select(m{row} + 1u, N - 1u, m{row} + 1u == N);
            let la{row} = win_l[u{row}]; let ra{row} = win_r[u{row}];
            let lm{row} = win_l[m{row}]; let rm{row} = win_r[m{row}];
            let lb{row} = win_l[d{row}]; let rb{row} = win_r[d{row}];
            let nl{row} = substep(lm{row},
                la{row} << 1u, la{row}, (la{row} >> 1u) | (ra{row} << 31u),
                lm{row} << 1u, (lm{row} >> 1u) | (rm{row} << 31u),
                lb{row} << 1u, lb{row}, (lb{row} >> 1u) | (rb{row} << 31u));
            let nr{row} = substep(rm{row},
                (ra{row} << 1u) | (la{row} >> 31u), ra{row}, ra{row} >> 1u,
                (rm{row} << 1u) | (lm{row} >> 31u), rm{row} >> 1u,
                (rb{row} << 1u) | (lb{row} >> 31u), rb{row}, rb{row} >> 1u);
"#
        );
        publish += &format!("            win_l[m{row}] = nl{row}; win_r[m{row}] = nr{row};\n");
    }
    let make_source = |index: u32| {
        format!(
            r#"
const SHARD: u32 = {index}u;
const COLS: u32 = {cols}u;
const T: u32 = {t}u;
const K: u32 = {k}u;
const N: u32 = {rows}u;
const SIM: u32 = {sim}u;
var<workgroup> win_l: array<u32, N>;
var<workgroup> win_r: array<u32, N>;

fn substep(c: u32, n0: u32, n1: u32, n2: u32, n3: u32, n4: u32, n5: u32, n6: u32, n7: u32) -> u32 {{
    let x01 = n0 ^ n1;
    let s0 = x01 ^ n2;
    let c0 = (n0 & n1) | (x01 & n2);
    let x34 = n3 ^ n4;
    let s1 = x34 ^ n5;
    let c1 = (n3 & n4) | (x34 & n5);
    let s2 = n6 ^ n7;
    let c2 = n6 & n7;
    let x = s0 ^ s1;
    let ones = x ^ s2;
    let c3 = (s0 & s1) | (x & s2);
    let y = c0 ^ c1;
    let twos = y ^ c2 ^ c3;
    let many = (c0 & c1) | (y & c2) | ((y ^ c2) & c3);
    return ((c | ones) & twos) & ~many;
}}

@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) index: u32) {{
    let h = P.h;
    let col = wg.y;
    let py = index * K;
    // The neighbouring columns may live in the neighbouring shards' buffers.
    let xl = select(col - 1u, COLS - 1u, col == 0u) * h;
    let xm = col * h;
    let xr = select(col + 1u, 0u, col + 1u == COLS) * h;
    let y0 = wg.x * SIM + py;
    var y = (y0 + h * (T / h + 1u) - T) % h;
    for (var r = 0u; r < K; r++) {{
        var a: u32;
        var c: u32;
        if (col == 0u) {{ a = cur_l[xl + y]; }} else {{ a = cur[xl + y]; }}
        if (col + 1u == COLS) {{ c = cur_r[xr + y]; }} else {{ c = cur[xr + y]; }}
        let b = cur[xm + y];
        win_l[py + r] = (a >> 16u) | (b << 16u);
        win_r[py + r] = (b >> 16u) | (c << 16u);
        y = select(y + 1u, 0u, y + 1u == h);
    }}
    for (var s = 0u; s < T; s++) {{
        workgroupBarrier();
{compute}
        workgroupBarrier();
{publish}
    }}
    workgroupBarrier();
    for (var r = 0u; r < K; r++) {{
        let m = py + r;
        let out_y = y0 + r;
        if (m >= T && m < N - T && out_y - T < h) {{
            next[xm + out_y - T] = (win_l[m] >> 16u) | (win_r[m] << 16u);
        }}
    }}
}}
"#
        )
    };
    let mut arms = String::new();
    let mut view_binds = Vec::new();
    const VIEW_VARS: [&str; 8] = ["v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7"];
    for i in 0..shards as usize {
        view_binds.push(Bind::Read(SHARDS[i], VIEW_VARS[i]));
        arms += &format!("        case {i}u: {{ word = {}[i]; }}\n", VIEW_VARS[i]);
    }
    let view = format!(
        r#"
fn color(x: u32, y: u32) -> vec3f {{
    let col = x / 32u;
    let i = (col % {cols}u) * P.h + y;
    var word = 0u;
    switch (col / {cols}u) {{
{arms}        default: {{ }}
    }}
    let alive = (word >> (x & 31u)) & 1u;
    return select(vec3f(0.035, 0.04, 0.07), vec3f(0.62, 0.96, 0.78), alive == 1u);
}}
"#
    );
    let mut fields = Vec::new();
    let mut init = Vec::new();
    let mut step = Vec::new();
    for i in 0..shards as usize {
        let name = SHARDS[i];
        fields.push(Field::double(name, "u32", 4, shard_words));
        init.push(Pass::new(
            "life init",
            format!(
                r#"{RANDOM_BITS}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x + id.y * 16384u * 64u;
    if (i < {shard_words}u) {{ cur[i] = random_word(i + {i}u * {shard_words}u); }}
}}
"#
            ),
            vec![Bind::ReadWrite(name, "cur")],
            [16384, shard_words.div_ceil(16384 * 64) as u32, 1],
        ));
        let left = SHARDS[(i + shards as usize - 1) % shards as usize];
        let right = SHARDS[(i + 1) % shards as usize];
        let pass = Pass::new(
            "life column",
            make_source(i as u32),
            vec![
                Bind::Read(left, "cur_l"),
                Bind::Read(name, "cur"),
                Bind::Read(right, "cur_r"),
                Bind::Write(name, "next"),
            ],
            [groups(h, sim), cols, 1],
        );
        // Every shard reads the old generation, so all swap after the last one.
        step.push(if i + 1 == shards as usize {
            pass.swap(&SHARDS[..shards as usize])
        } else {
            pass
        });
    }
    Program {
        id: if shards == 1 {
            format!("life.column.t{t}.g{threads}.k{k}")
        } else {
            format!("life.sharded{shards}.t{t}")
        },
        width: w,
        height: h,
        fields,
        params_wgsl: PARAMS.into(),
        params: params(w, h, wr, 0.35),
        init,
        step,
        view,
        view_binds,
        cells_per_step: w as u64 * h as u64,
        generations_per_step: t,
    }
}

/// Life kernel variants by id, for the benchmark.
pub fn kernel(id: &str, w: u32, h: u32) -> Option<Program> {
    let rest = id.strip_prefix("life.")?;
    if rest == "naive" {
        return Some(naive(w, h));
    }
    if rest == "bits64" {
        return Some(bits64(w.next_multiple_of(64), h));
    }
    let w32 = w.next_multiple_of(32);
    if let Some(shape) = rest.strip_prefix("bits32.") {
        let (a, b) = shape.split_once('x')?;
        return Some(bits32(w32, h, [a.parse().ok()?, b.parse().ok()?]));
    }
    if let Some(spec) = rest.strip_prefix("column.") {
        let mut t = 16;
        let mut threads = 256;
        let mut k = 3;
        for part in spec.split('.') {
            let value = part.get(1..)?.parse().ok()?;
            match part.as_bytes().first()? {
                b't' => t = value,
                b'g' => threads = value,
                b'k' => k = value,
                _ => return None,
            }
        }
        return Some(column(w32, h, t, threads, k));
    }
    if let Some(n) = rest.strip_prefix("sharded") {
        return Some(sharded(w32, h, 16, 256, 3, n.parse().ok()?));
    }
    if let Some(spec) = rest.strip_prefix("ghost.") {
        let mut t = 8;
        let mut tile_w = 30;
        let mut rows = 256;
        for part in spec.split('.') {
            let value = part.get(1..)?.parse().ok()?;
            match part.as_bytes().first()? {
                b't' => t = value,
                b'w' => tile_w = value,
                b'r' => rows = value,
                _ => return None,
            }
        }
        return Some(ghost(w32, h, t, tile_w, rows));
    }
    None
}
