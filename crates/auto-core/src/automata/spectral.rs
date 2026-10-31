//! Automata with big smooth neighbourhoods, computed in frequency space.
//!
//! Lenia's neighbourhood covers hundreds of cells. Reading them one at a time
//! scales with radius squared; an FFT costs the same at any radius. Each kernel
//! is an annulus described in WGSL, and `update` turns the convolved fields into
//! the next state, so Lenia, SmoothLife and MNCA are mostly a few lines each.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

const WG: u32 = 64;

/// Radix-2 Stockham FFT over one axis, one stage per repeat of the pass.
fn fft_pass(
    label: &'static str,
    field: &'static str,
    w: u32,
    h: u32,
    axis_x: bool,
    inverse: bool,
) -> Pass {
    let (n, rows) = if axis_x { (w, h) } else { (h, w) };
    let log2 = n.trailing_zeros();
    let sign = if inverse { "1.0" } else { "-1.0" };
    let (ri, rj) = if axis_x {
        ("r * P.w + (p)", "r * P.w + (p)")
    } else {
        ("(p) * P.w + r", "(p) * P.w + r")
    };
    let index = |p: &str| ri.replace("(p)", &format!("({p})"));
    let _ = rj;
    let src0 = index("p");
    let src1 = index(&format!("p + {half}u", half = n / 2));
    let dst0 = index("2u * p - k");
    let dst1 = index("2u * p - k + m");
    let source = format!(
        r#"
@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) id: vec3u) {{
    let p = id.x;
    let r = id.y;
    if (p >= {half}u || r >= {rows}u) {{ return; }}
    let m = 1u << frame.phase;
    let k = p & (m - 1u);
    let a = cur[{src0}];
    let b = cur[{src1}];
    let angle = {sign} * 3.14159265358979 * f32(k) / f32(m);
    let tw = vec2f(cos(angle), sin(angle));
    let t = vec2f(b.x * tw.x - b.y * tw.y, b.x * tw.y + b.y * tw.x);
    next[{dst0}] = a + t;
    next[{dst1}] = a - t;
}}
"#,
        half = n / 2,
    );
    Pass::new(
        label,
        source,
        vec![Bind::Read(field, "cur"), Bind::Write(field, "next")],
        [groups(n / 2, WG), rows, 1],
    )
    .repeat(log2)
    .swap(&[field])
}

/// One spatial kernel. `shape` defines `fn shape(r: f32) -> f32` over distance in cells.
pub struct Kernel {
    pub shape: String,
    /// Sum of `shape` over the grid, so the convolution comes out as an average.
    pub sum: f32,
    /// Largest distance `shape` reaches, in cells.
    pub reach: i32,
}

fn kernel_sum(shape: impl Fn(f32) -> f32, reach: i32) -> f32 {
    let mut sum = 0.0f64;
    for y in -reach..=reach {
        for x in -reach..=reach {
            sum += shape(((x * x + y * y) as f32).sqrt()) as f64;
        }
    }
    sum as f32
}

pub struct Spectral<'a> {
    pub id: &'static str,
    pub w: u32,
    pub h: u32,
    pub kernels: Vec<Kernel>,
    /// `fn seed(x: u32, y: u32) -> f32`
    pub seed: &'a str,
    /// `fn update(s: f32, n: array<f32, K>, x, y) -> f32`
    pub update: &'a str,
    /// `fn shade(s: f32, x: u32, y: u32) -> vec3f`
    pub view: &'a str,
}

const KERN_FIELDS: [&str; 3] = ["kern0", "kern1", "kern2"];
const CONV_FIELDS: [&str; 3] = ["conv0", "conv1", "conv2"];

pub fn build(s: Spectral) -> Program {
    let (w, h) = (s.w, s.h);
    assert!(
        w.is_power_of_two() && h.is_power_of_two(),
        "FFT grids must be powers of two"
    );
    let cells = w as u64 * h as u64;
    let k = s.kernels.len();
    assert!((1..=3).contains(&k));
    for kernel in &s.kernels {
        // The kernel is stored once per torus position, so it must fit without wrapping.
        assert!(
            2 * kernel.reach < w.min(h) as i32,
            "world smaller than its kernel"
        );
    }
    let dispatch = [groups(w, 64), h, 1];

    let mut fields = vec![
        Field::double("state", "f32", 4, cells),
        Field::double("work", "vec2<f32>", 8, cells),
        Field::single("spec", "vec2<f32>", 8, cells),
    ];
    for i in 0..k {
        fields.push(Field::single(KERN_FIELDS[i], "vec2<f32>", 8, cells));
        fields.push(Field::single(CONV_FIELDS[i], "f32", 4, cells));
    }

    let copy = |label: &'static str, body: &str, binds: Vec<Bind>| {
        Pass::new(
            label,
            format!(
                r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x >= P.w || id.y >= P.h) {{ return; }}
    let i = id.y * P.w + id.x;
    {body}
}}
"#
            ),
            binds,
            dispatch,
        )
    };

    // Init: build each kernel in space, transform it, keep the spectrum.
    let mut init = Vec::new();
    for (i, kernel) in s.kernels.iter().enumerate() {
        init.push(
            Pass::new(
                "kernel",
                format!(
                    r#"{shape}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x >= P.w || id.y >= P.h) {{ return; }}
    let dx = f32(min(id.x, P.w - id.x));
    let dy = f32(min(id.y, P.h - id.y));
    let v = shape(sqrt(dx * dx + dy * dy)) / {sum:?};
    next[id.y * P.w + id.x] = vec2f(v, 0.0);
}}
"#,
                    shape = kernel.shape,
                    sum = kernel.sum,
                ),
                vec![Bind::Write("work", "next")],
                dispatch,
            )
            .swap(&["work"]),
        );
        init.push(fft_pass("kernel fft x", "work", w, h, true, false));
        init.push(fft_pass("kernel fft y", "work", w, h, false, false));
        init.push(copy(
            "keep",
            "out[i] = cur[i];",
            vec![
                Bind::Read("work", "cur"),
                Bind::ReadWrite(KERN_FIELDS[i], "out"),
            ],
        ));
    }
    init.push(Pass::new(
        "seed",
        format!(
            r#"{seed}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x >= P.w || id.y >= P.h) {{ return; }}
    out[id.y * P.w + id.x] = seed(id.x, id.y);
}}
"#,
            seed = s.seed
        ),
        vec![Bind::ReadWrite("state", "out")],
        dispatch,
    ));

    // Step: forward transform once, then one multiply and inverse per kernel.
    let mut step = vec![
        copy(
            "load",
            "next[i] = vec2f(state[i], 0.0);",
            vec![Bind::Read("state", "state"), Bind::Write("work", "next")],
        )
        .swap(&["work"]),
        fft_pass("fft x", "work", w, h, true, false),
        fft_pass("fft y", "work", w, h, false, false),
        copy(
            "spectrum",
            "spec[i] = cur[i];",
            vec![Bind::Read("work", "cur"), Bind::ReadWrite("spec", "spec")],
        ),
    ];
    let scale = 1.0 / cells as f32;
    for i in 0..k {
        step.push(
            copy(
                "multiply",
                "let a = spec[i]; let b = kern[i]; next[i] = vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);",
                vec![Bind::Read("spec", "spec"), Bind::Read(KERN_FIELDS[i], "kern"), Bind::Write("work", "next")],
            )
            .swap(&["work"]),
        );
        step.push(fft_pass("ifft y", "work", w, h, false, true));
        step.push(fft_pass("ifft x", "work", w, h, true, true));
        step.push(copy(
            "convolved",
            &format!("out[i] = cur[i].x * {scale:?};"),
            vec![
                Bind::Read("work", "cur"),
                Bind::ReadWrite(CONV_FIELDS[i], "out"),
            ],
        ));
    }
    let mut args = String::new();
    let mut binds = vec![Bind::Read("state", "cur"), Bind::Write("state", "next")];
    const VARS: [&str; 3] = ["c0", "c1", "c2"];
    for i in 0..k {
        binds.push(Bind::Read(CONV_FIELDS[i], VARS[i]));
        if i > 0 {
            args.push_str(", ");
        }
        args.push_str(&format!("{}[i]", VARS[i]));
    }
    step.push(
        Pass::new(
            "update",
            format!(
                r#"{update}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {{
    if (id.x >= P.w || id.y >= P.h) {{ return; }}
    let i = id.y * P.w + id.x;
    next[i] = update(cur[i], array<f32, {k}>({args}), id.x, id.y);
}}
"#,
                update = s.update
            ),
            binds,
            dispatch,
        )
        .swap(&["state"]),
    );

    Program {
        id: s.id.into(),
        width: w,
        height: h,
        fields,
        params_wgsl: "w: u32, h: u32, _a: u32, _b: u32,".into(),
        params: bytemuck::cast_slice(&[w, h, 0u32, 0u32]).to_vec(),
        init,
        step,
        view: format!(
            "{}\nfn color(x: u32, y: u32) -> vec3f {{ return shade(view_state[y * P.w + x], x, y); }}\n",
            s.view
        ),
        view_binds: vec![Bind::Read("state", "view_state")],
        cells_per_step: cells,
        generations_per_step: 1,
    }
}

/// Lenia with a ring kernel of radius `radius`.
pub fn lenia(w: u32, h: u32, radius: f32) -> Program {
    const ORBIUM: &str = "0u,0u,0u,0u,0u,0u,0u,13u,4u,0u,0u,0u,0u,0u,0u,60u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,35u,53u,54u,21u,18u,25u,26u,18u,1u,65u,0u,0u,0u,0u,0u,0u,0u,0u,0u,22u,68u,91u,98u,87u,48u,47u,44u,47u,45u,27u,89u,0u,0u,0u,0u,0u,0u,0u,3u,17u,89u,116u,119u,97u,33u,20u,14u,16u,31u,55u,180u,0u,0u,0u,0u,0u,0u,9u,33u,47u,87u,103u,98u,72u,34u,0u,0u,0u,0u,12u,102u,84u,0u,0u,1u,0u,4u,35u,43u,34u,28u,63u,69u,67u,53u,0u,0u,0u,0u,0u,5u,220u,0u,0u,60u,0u,26u,44u,20u,0u,0u,51u,79u,94u,95u,61u,0u,0u,0u,0u,0u,116u,40u,0u,0u,31u,47u,28u,0u,0u,0u,69u,109u,129u,137u,130u,0u,0u,0u,0u,0u,0u,132u,0u,0u,150u,55u,8u,0u,0u,0u,48u,135u,162u,174u,174u,107u,0u,0u,0u,0u,0u,109u,0u,0u,141u,60u,0u,0u,0u,0u,7u,158u,193u,214u,218u,206u,27u,0u,0u,0u,0u,72u,25u,0u,0u,152u,0u,0u,0u,0u,0u,170u,223u,245u,253u,248u,143u,0u,0u,0u,0u,57u,36u,0u,0u,214u,0u,0u,0u,0u,0u,129u,247u,255u,255u,255u,233u,71u,0u,0u,6u,56u,34u,0u,0u,141u,19u,0u,0u,0u,0u,85u,255u,255u,250u,255u,248u,142u,40u,13u,30u,62u,22u,0u,0u,8u,114u,0u,0u,0u,0u,45u,217u,255u,228u,220u,221u,166u,85u,50u,55u,59u,10u,0u,0u,0u,108u,29u,0u,0u,0u,29u,158u,224u,210u,199u,189u,156u,104u,75u,68u,42u,0u,0u,0u,0u,20u,85u,19u,0u,0u,30u,108u,172u,184u,173u,160u,134u,103u,79u,57u,16u,0u,0u,0u,0u,0u,42u,66u,38u,30u,44u,86u,127u,142u,139u,127u,109u,86u,62u,30u,0u,0u,0u,0u,0u,0u,0u,37u,59u,60u,66u,81u,99u,108u,105u,92u,78u,58u,32u,5u,0u,0u,0u,0u,0u,0u,0u,0u,18u,43u,58u,64u,70u,71u,66u,59u,42u,24u,5u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,0u,15u,26u,33u,34u,30u,20u,11u,0u,0u,0u,0u,0u";
    let shape_fn = move |r: f32| {
        let q = r / radius;
        if q <= 0.0 || q >= 1.0 {
            0.0
        } else {
            (4.0 - 1.0 / (q * (1.0 - q))).exp()
        }
    };
    let sum = kernel_sum(shape_fn, radius.ceil() as i32 + 1);
    let shape = format!(
        r#"fn shape(r: f32) -> f32 {{
    let q = r / {radius:?};
    if (q <= 0.0 || q >= 1.0) {{ return 0.0; }}
    return exp(4.0 - 1.0 / (q * (1.0 - q)));
}}"#
    );
    build(Spectral {
        id: "lenia",
        w,
        h,
        kernels: vec![Kernel {
            shape,
            sum,
            reach: radius.ceil() as i32,
        }],
        seed: &format!(
            r#"
// Orbium unicaudatus from Bert Chan's catalogue (MIT), scaled to 0..255.
const ORBIUM = array<u32, 400>({ORBIUM});
const SPACING: u32 = 64u;
fn seed(x: u32, y: u32) -> f32 {{
    let tx = x / SPACING; let ty = y / SPACING;
    let r = rand3(tx, ty, frame.seed);
    if ((r & 1u) != 0u) {{ return 0.0; }}
    let px = x % SPACING; let py = y % SPACING;
    if (px >= 20u || py >= 20u) {{ return 0.0; }}
    // A random quarter turn per creature, so they head off in different directions.
    var ix = px; var iy = py;
    switch ((r >> 8u) & 3u) {{
        case 1u: {{ ix = py; iy = 19u - px; }}
        case 2u: {{ ix = 19u - px; iy = 19u - py; }}
        case 3u: {{ ix = 19u - py; iy = px; }}
        default: {{ }}
    }}
    var cells = ORBIUM;
    return f32(cells[iy * 20u + ix]) / 255.0;
}}
"#
        ),
        update: r#"
const MU: f32 = 0.15;
const SIGMA: f32 = 0.015;
const DT: f32 = 0.1;
fn update(s: f32, n: array<f32, 1>, x: u32, y: u32) -> f32 {
    let d = (n[0] - MU) / SIGMA;
    return clamp(s + DT * (2.0 * exp(-0.5 * d * d) - 1.0), 0.0, 1.0);
}
"#,
        view: r#"
fn shade(s: f32, x: u32, y: u32) -> vec3f {
    return ramp(s, vec3f(0.01, 0.01, 0.05), vec3f(0.08, 0.12, 0.45), vec3f(0.1, 0.6, 0.7), vec3f(0.95, 0.85, 0.4), vec3f(1.0, 0.98, 0.9));
}
"#,
    })
}

/// Rafler's SmoothLife: an inner disc and an outer ring.
pub fn smoothlife(w: u32, h: u32, ra: f32) -> Program {
    let ri = ra / 3.0;
    let aa = 0.5f32;
    let inner = move |r: f32| ((ri + aa - r) / (2.0 * aa)).clamp(0.0, 1.0);
    let outer = move |r: f32| {
        ((ra + aa - r) / (2.0 * aa)).clamp(0.0, 1.0) * ((r - ri + aa) / (2.0 * aa)).clamp(0.0, 1.0)
    };
    let reach = ra.ceil() as i32 + 2;
    build(Spectral {
        id: "smoothlife",
        w,
        h,
        kernels: vec![
            Kernel {
                shape: format!(
                    "fn shape(r: f32) -> f32 {{ return clamp(({ri:?} + {aa:?} - r) / {:?}, 0.0, 1.0); }}",
                    2.0 * aa
                ),
                sum: kernel_sum(inner, reach),
                reach,
            },
            Kernel {
                shape: format!(
                    "fn shape(r: f32) -> f32 {{ return clamp(({ra:?} + {aa:?} - r) / {d:?}, 0.0, 1.0) * clamp((r - {ri:?} + {aa:?}) / {d:?}, 0.0, 1.0); }}",
                    d = 2.0 * aa
                ),
                sum: kernel_sum(outer, reach),
                reach,
            },
        ],
        seed: r#"
fn seed(x: u32, y: u32) -> f32 {
    // Scattered discs a little wider than the outer radius.
    let tile = 96u;
    let c = rand3(x / tile, y / tile, frame.seed);
    if ((c & 1u) == 0u) { return 0.0; }
    let dx = f32(x % tile) - f32(tile) / 2.0;
    let dy = f32(y % tile) - f32(tile) / 2.0;
    return select(0.0, 1.0, dx * dx + dy * dy < 24.0 * 24.0);
}
"#,
        update: r#"
const B1: f32 = 0.278; const B2: f32 = 0.365;
const D1: f32 = 0.267; const D2: f32 = 0.445;
const AN: f32 = 0.028; const AM: f32 = 0.147;
const DT: f32 = 0.1;
fn sig(x: f32, a: f32, alpha: f32) -> f32 { return 1.0 / (1.0 + exp((a - x) * 4.0 / alpha)); }
fn sign2(x: f32, a: f32, b: f32) -> f32 { return sig(x, a, AN) * (1.0 - sig(x, b, AN)); }
fn update(s: f32, c: array<f32, 2>, x: u32, y: u32) -> f32 {
    let m = sig(c[0], 0.5, AM);
    let target_ = sign2(c[1], mix(B1, D1, m), mix(B2, D2, m));
    return clamp(s + DT * (2.0 * target_ - 1.0), 0.0, 1.0);
}
"#,
        view: r#"
fn shade(s: f32, x: u32, y: u32) -> vec3f {
    return ramp(s, vec3f(0.02, 0.02, 0.03), vec3f(0.3, 0.07, 0.25), vec3f(0.85, 0.25, 0.25), vec3f(0.98, 0.7, 0.3), vec3f(1.0, 0.98, 0.85));
}
"#,
    })
}

/// Multiple Neighbourhood CA, in the style of Slackermanz: threshold bands on
/// two neighbourhood averages, each able to overrule the one before.
pub fn mnca(w: u32, h: u32) -> Program {
    mnca_shaped(w, h, &MNCA_RINGS)
}

const MNCA_RINGS: [f32; 4] = [12.0, 18.0, 3.0, 8.0];

/// `rings` is [inner0, outer0, inner1, outer1] in cells.
pub fn mnca_shaped(w: u32, h: u32, rings: &[f32]) -> Program {
    let ring = |lo: f32, hi: f32| {
        (
            format!(
                "fn shape(r: f32) -> f32 {{ return select(0.0, 1.0, r >= {lo:?} && r < {hi:?}); }}"
            ),
            kernel_sum(
                move |r| if r >= lo && r < hi { 1.0 } else { 0.0 },
                hi.ceil() as i32 + 1,
            ),
        )
    };
    let (s0, c0) = ring(rings[0], rings[1]);
    let (s1, c1) = ring(rings[2], rings[3]);
    build(Spectral {
        id: "mnca",
        w,
        h,
        kernels: vec![
            Kernel {
                shape: s0,
                sum: c0,
                reach: rings[1].ceil() as i32,
            },
            Kernel {
                shape: s1,
                sum: c1,
                reach: rings[3].ceil() as i32,
            },
        ],
        seed: r#"
fn seed(x: u32, y: u32) -> f32 {
    return select(0.0, 1.0, unit_f(rand3(x / 3u, y / 3u, frame.seed)) < 0.32);
}
"#,
        update: r#"
fn band(v: f32, lo: f32, hi: f32) -> bool { return v >= lo && v <= hi; }
fn update(s: f32, c: array<f32, 2>, x: u32, y: u32) -> f32 {
    var o = s;
    if (band(c[0], 0.210, 0.220)) { o = 1.0; }
    if (band(c[0], 0.350, 0.500)) { o = 0.0; }
    if (band(c[0], 0.750, 0.850)) { o = 0.0; }
    if (band(c[0], 0.120, 0.150)) { o = 0.0; }
    if (band(c[1], 0.100, 0.280)) { o = 0.0; }
    if (band(c[1], 0.430, 0.550)) { o = 1.0; }
    return o;
}
"#,
        view: r#"
fn shade(s: f32, x: u32, y: u32) -> vec3f {
    return mix3(vec3f(0.02, 0.02, 0.07), vec3f(0.75, 0.95, 0.85), s);
}
"#,
    })
}

pub const LENIA: Entry = Entry {
    id: "lenia",
    name: "Lenia",
    blurb: "Continuous life. Each cell senses a ring of radius 13 through an FFT, so radius costs nothing.",
    default_size: (2048, 1024),
    needs_pot: true,
    width_step: 1,
    build: |w, h| lenia(w, h, 13.0),
};

pub const SMOOTHLIFE: Entry = Entry {
    id: "smoothlife",
    name: "SmoothLife",
    blurb: "Rafler's continuous Game of Life: an inner disc and an outer ring instead of 8 neighbours.",
    default_size: (2048, 1024),
    needs_pot: true,
    width_step: 1,
    build: |w, h| smoothlife(w, h, 21.0),
};

pub const MNCA: Entry = Entry {
    id: "mnca",
    name: "Multiple Neighbourhood CA",
    blurb: "Slackermanz's idea: threshold bands on several neighbourhoods at once.",
    default_size: (2048, 1024),
    needs_pot: true,
    width_step: 1,
    build: mnca,
};
