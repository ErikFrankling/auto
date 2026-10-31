//! Beyond the grid: agents moving over a field, and agents with their own brains.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

pub const PHYSARUM: Entry = Entry {
    id: "physarum",
    name: "Physarum slime mould",
    blurb: "Jones's agents: millions of particles follow and lay trails. Particles scatter into a grid with atomics.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: |w, h| physarum(w, h, (w as u64 * h as u64 / 2) as u32),
};

pub fn physarum(w: u32, h: u32, agents: u32) -> Program {
    let cells = w as u64 * h as u64;
    let params = "w: u32, h: u32, n: u32, _a: u32,";
    let param_bytes = bytemuck::cast_slice(&[w, h, agents, 0u32]).to_vec();
    let agent_step = r#"
const SENSE_ANGLE: f32 = 0.5;
const SENSE_DIST: f32 = 12.0;
const TURN: f32 = 0.32;
const SPEED: f32 = 1.2;
fn trail_at(p: vec2f) -> f32 {
    let x = u32(fract(p.x / f32(P.w)) * f32(P.w)) % P.w;
    let y = u32(fract(p.y / f32(P.h)) * f32(P.h)) % P.h;
    return trail[y * P.w + x];
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 65536u;
    if (i >= P.n) { return; }
    let a = agents[i];
    var pos = a.xy;
    var heading = a.z;
    let s = |a: f32| 0.0;
}
"#;
    // WGSL has no closures, so sensing is spelled out.
    let agent_step = agent_step.replace(
        "    let s = |a: f32| 0.0;\n",
        r#"    let left = trail_at(pos + SENSE_DIST * vec2f(cos(heading - SENSE_ANGLE), sin(heading - SENSE_ANGLE)));
    let mid = trail_at(pos + SENSE_DIST * vec2f(cos(heading), sin(heading)));
    let right = trail_at(pos + SENSE_DIST * vec2f(cos(heading + SENSE_ANGLE), sin(heading + SENSE_ANGLE)));
    let r = rand3(i, frame.tick, frame.seed);
    if (mid < left || mid < right) {
        if (left > right) { heading -= TURN; }
        else if (right > left) { heading += TURN; }
        else { heading += select(-TURN, TURN, (r & 1u) == 1u); }
    }
    pos += SPEED * vec2f(cos(heading), sin(heading));
    pos = fract(pos / vec2f(f32(P.w), f32(P.h))) * vec2f(f32(P.w), f32(P.h));
    agents[i] = vec4f(pos, heading, 0.0);
    let cell = (u32(pos.y) % P.h) * P.w + (u32(pos.x) % P.w);
    atomicAdd(&deposit[cell], 1u);
"#,
    );
    let diffuse = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let w = P.w; let h = P.h; let x = id.x; let y = id.y;
    if (x >= w || y >= h) { return; }
    let xl = select(x - 1u, w - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == w);
    let up = select(y - 1u, h - 1u, y == 0u) * w;
    let me = y * w;
    let dn = select(y + 1u, 0u, y + 1u == h) * w;
    let blur = (trail[up + xl] + trail[up + x] + trail[up + xr]
              + trail[me + xl] + trail[me + x] + trail[me + xr]
              + trail[dn + xl] + trail[dn + x] + trail[dn + xr]) / 9.0;
    let laid = f32(atomicExchange(&deposit[me + x], 0u));
    next[me + x] = blur * 0.88 + laid * 0.1;
}
"#;
    let init_agents = r#"
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 65536u;
    if (i >= P.n) { return; }
    let pos = vec2f(unit_f(rand3(i, 1u, frame.seed)) * f32(P.w), unit_f(rand3(i, 2u, frame.seed)) * f32(P.h));
    agents[i] = vec4f(pos, unit_f(rand3(i, 3u, frame.seed)) * 6.2831853, 0.0);
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let t = 1.0 - exp(-view_trail[y * P.w + x] * 0.55);
    return ramp(t, vec3f(0.01, 0.01, 0.03), vec3f(0.18, 0.06, 0.42), vec3f(0.75, 0.2, 0.45), vec3f(0.98, 0.62, 0.25), vec3f(1.0, 0.97, 0.8));
}
"#;
    Program {
        id: "physarum".into(),
        width: w,
        height: h,
        fields: vec![
            Field::single("agents", "vec4<f32>", 16, agents as u64),
            Field::double("trail", "f32", 4, cells),
            Field::single("deposit", "atomic<u32>", 4, cells),
        ],
        params_wgsl: params.into(),
        params: param_bytes,
        init: vec![Pass::new(
            "init agents",
            init_agents,
            vec![Bind::ReadWrite("agents", "agents")],
            [256, groups(agents, 65536), 1],
        )],
        step: vec![
            Pass::new(
                "agents",
                agent_step,
                vec![
                    Bind::ReadWrite("agents", "agents"),
                    Bind::Read("trail", "trail"),
                    Bind::ReadWrite("deposit", "deposit"),
                ],
                [256, groups(agents, 65536), 1],
            ),
            Pass::new(
                "diffuse",
                diffuse,
                vec![
                    Bind::Read("trail", "trail"),
                    Bind::Write("trail", "next"),
                    Bind::ReadWrite("deposit", "deposit"),
                ],
                [groups(w, 64), h, 1],
            )
            .swap(&["trail"]),
        ],
        view: view.into(),
        view_binds: vec![Bind::Read("trail", "view_trail")],
        cells_per_step: agents as u64,
        generations_per_step: 1,
    }
}

/// Throughput of many independent RNNs: every agent has its own weights.
/// `bits` 32 stores f32 weights, 8 stores packed 8-bit weights.
///
/// This is not a simulation. It answers what a brain costs on this GPU.
pub fn brains(agents: u32, hidden: u32, bits: u32) -> Program {
    assert!(bits == 32 || bits == 8);
    const INPUTS: u32 = 8;
    let weights = hidden * hidden + INPUTS * hidden;
    let params = "n: u32, hidden: u32, inputs: u32, weights: u32,";
    let param_bytes = bytemuck::cast_slice(&[agents, hidden, INPUTS, weights]).to_vec();
    let (elem, elem_bytes, words_per_agent) = if bits == 32 {
        ("f32", 4, weights)
    } else {
        ("u32", 4, weights.div_ceil(4))
    };
    let weight_fn = if bits == 32 {
        "fn weight(base: u32, k: u32) -> f32 { return genome[base + k]; }".to_string()
    } else {
        r#"fn weight(base: u32, k: u32) -> f32 {
    let word = genome[base + k / 4u];
    let raw = extractBits(i32(word), (k & 3u) * 8u, 8u);
    return f32(raw) * (1.0 / 64.0);
}"#
        .to_string()
    };
    let step = format!(
        r#"
{weight_fn}
const H: u32 = {hidden}u;
const K: u32 = {INPUTS}u;
const STRIDE: u32 = {words_per_agent}u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let a = id.x;
    if (a >= P.n) {{ return; }}
    let base = a * STRIDE;
    var h: array<f32, H>;
    for (var i = 0u; i < H; i++) {{ h[i] = state[a * H + i]; }}
    let seed = rand3(a, frame.tick, frame.seed);
    var w_i = 0u;
    for (var i = 0u; i < H; i++) {{
        var sum = 0.0;
        for (var j = 0u; j < H; j++) {{
            sum += weight(base, w_i) * h[j];
            w_i += 1u;
        }}
        for (var j = 0u; j < K; j++) {{
            let input_ = f32((seed >> j) & 1u);
            sum += weight(base, w_i) * input_;
            w_i += 1u;
        }}
        // A cheap tanh shape: sum / (1 + |sum|).
        state[a * H + i] = sum / (1.0 + abs(sum));
    }}
}}
"#
    );
    let init_weight = if bits == 32 {
        "(unit_f(r) - 0.5) * 0.5".to_string()
    } else {
        "r".to_string()
    };
    let init = format!(
        r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x + id.y * 65536u;
    if (i >= P.n * {words_per_agent}u) {{ return; }}
    let r = rand3(i, 3u, frame.seed);
    genome[i] = {init_weight};
}}
"#
    );
    let total = agents as u64 * words_per_agent as u64;
    Program {
        id: format!("brains.h{hidden}.b{bits}"),
        width: agents,
        height: 1,
        fields: vec![
            Field::single("genome", elem, elem_bytes, total),
            Field::single("state", "f32", 4, agents as u64 * hidden as u64),
        ],
        params_wgsl: params.into(),
        params: param_bytes,
        init: vec![
            Pass::new(
                "init",
                init,
                vec![Bind::ReadWrite("genome", "genome")],
                [1024, (total.div_ceil(65536)) as u32, 1],
            )
            .swap(&[]),
        ],
        step: vec![Pass::new(
            "brains",
            step,
            vec![
                Bind::Read("genome", "genome"),
                Bind::ReadWrite("state", "state"),
            ],
            [groups(agents, 64), 1, 1],
        )],
        view: "fn color(x: u32, y: u32) -> vec3f { return vec3f(0.0); }".into(),
        view_binds: vec![],
        cells_per_step: agents as u64,
        generations_per_step: 1,
    }
}

/// The same RNN, written the way the GPU wants it: weights in 4x4 blocks, state
/// in vec4s, one invocation per four neurons, no scratch copy of the brain.
/// Each block load gives 16 multiply-adds. `bits` 8 packs a block into 16 bytes.
pub fn brains_blocked(agents: u32, hidden: u32, bits: u32) -> Program {
    assert!(bits == 32 || bits == 8);
    assert!(hidden % 4 == 0, "hidden size must be a multiple of 4");
    let rows = hidden / 4;
    // Each row of 4 neurons reads `rows` blocks of state and 2 blocks of inputs.
    let blocks = rows as u64 * (rows as u64 + 2);
    let total_blocks = agents as u64 * blocks;
    let params = "n: u32, hidden: u32, _a: u32, _b: u32,";
    let param_bytes = bytemuck::cast_slice(&[agents, hidden, 0u32, 0u32]).to_vec();
    let (elem, elem_bytes, load) = if bits == 32 {
        (
            "mat4x4<f32>",
            64,
            "fn block_times(i: u32, v: vec4f) -> vec4f { return genome[i] * v; }".to_string(),
        )
    } else {
        (
            "vec4<u32>",
            16,
            r#"fn block_times(i: u32, v: vec4f) -> vec4f {
    let g = genome[i];
    return vec4f(
        dot(unpack4x8snorm(g.x), v),
        dot(unpack4x8snorm(g.y), v),
        dot(unpack4x8snorm(g.z), v),
        dot(unpack4x8snorm(g.w), v),
    ) * 2.0;
}"#
            .to_string(),
        )
    };
    let step = format!(
        r#"{load}
const ROWS: u32 = {rows}u;
const STRIDE: u32 = ROWS + 2u;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let a = id.x;
    let row = id.y;
    if (a >= P.n || row >= ROWS) {{ return; }}
    let base = (a * ROWS + row) * STRIDE;
    let state_base = a * ROWS;
    var acc = vec4f(0.0);
    for (var j = 0u; j < ROWS; j++) {{
        acc += block_times(base + j, cur[state_base + j]);
    }}
    let r = rand3(a, frame.tick, frame.seed);
    let in0 = vec4f(f32(r & 1u), f32((r >> 1u) & 1u), f32((r >> 2u) & 1u), f32((r >> 3u) & 1u));
    let in1 = vec4f(f32((r >> 4u) & 1u), f32((r >> 5u) & 1u), f32((r >> 6u) & 1u), f32((r >> 7u) & 1u));
    acc += block_times(base + ROWS, in0);
    acc += block_times(base + ROWS + 1u, in1);
    next[state_base + row] = acc / (vec4f(1.0) + abs(acc));
}}
"#
    );
    let init_value = if bits == 32 {
        r#"mat4x4f(
        vec4f(unit_f(rand3(i, 1u, frame.seed)), unit_f(rand3(i, 2u, frame.seed)), unit_f(rand3(i, 3u, frame.seed)), unit_f(rand3(i, 4u, frame.seed))) - 0.5,
        vec4f(unit_f(rand3(i, 5u, frame.seed)), unit_f(rand3(i, 6u, frame.seed)), unit_f(rand3(i, 7u, frame.seed)), unit_f(rand3(i, 8u, frame.seed))) - 0.5,
        vec4f(unit_f(rand3(i, 9u, frame.seed)), unit_f(rand3(i, 10u, frame.seed)), unit_f(rand3(i, 11u, frame.seed)), unit_f(rand3(i, 12u, frame.seed))) - 0.5,
        vec4f(unit_f(rand3(i, 13u, frame.seed)), unit_f(rand3(i, 14u, frame.seed)), unit_f(rand3(i, 15u, frame.seed)), unit_f(rand3(i, 16u, frame.seed))) - 0.5)"#
    } else {
        "vec4u(rand3(i, 1u, frame.seed), rand3(i, 2u, frame.seed), rand3(i, 3u, frame.seed), rand3(i, 4u, frame.seed))"
    };
    let init = format!(
        r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let i = id.x + id.y * 65536u;
    if (u64_lt(i)) {{ genome[i] = {init_value}; }}
}}
fn u64_lt(i: u32) -> bool {{ return i < {total_blocks}u; }}
"#
    );
    Program {
        id: format!("brainsb.h{hidden}.b{bits}"),
        width: agents,
        height: 1,
        fields: vec![
            Field::single("genome", elem, elem_bytes, total_blocks),
            Field::double("state", "vec4<f32>", 16, agents as u64 * rows as u64),
        ],
        params_wgsl: params.into(),
        params: param_bytes,
        init: vec![Pass::new(
            "init",
            init,
            vec![Bind::ReadWrite("genome", "genome")],
            [1024, total_blocks.div_ceil(65536) as u32, 1],
        )],
        step: vec![
            Pass::new(
                "brains blocked",
                step,
                vec![
                    Bind::Read("genome", "genome"),
                    Bind::Read("state", "cur"),
                    Bind::Write("state", "next"),
                ],
                [groups(agents, 64), rows, 1],
            )
            .swap(&["state"]),
        ],
        view: "fn color(x: u32, y: u32) -> vec3f { return vec3f(0.0); }".into(),
        view_binds: vec![],
        cells_per_step: agents as u64,
        generations_per_step: 1,
    }
}
