// Draws part of the world for a person looking at it: any measurement as the
// colour of the cells, any field under them. Reads the world and changes nothing.
// The cases of the two switches below are filled in from the tables in explore.rs.

struct ViewU {
    origin: vec2f, scale: f32, samples: u32,
    out: vec2u, row0: u32, show: u32,
    cell_mode: u32, field_mode: u32, field_opacity: f32, cell_opacity: f32,
    cell_lo: f32, cell_hi: f32, tau: f32, clade: u32,
    hi: array<vec4f, 4>,
    sketch: vec4u,
    center: vec2f, sun_k: f32, season: f32,
    season_on: f32, ground_cap: f32, sun_keep: f32, pad0: f32,
    drift: vec2f, pad1: vec2f,
}
@group(1) @binding(0) var<uniform> V: ViewU;
@group(1) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

const SHOW_CELLS: u32 = 1u;
const SHOW_SIMILAR: u32 = 2u;
const SHOW_FOUNDER: u32 = 4u;

// Viridis, as a polynomial (Matt Zucker's fit). explore.rs has the same one for the legend.
fn seq(t0: f32) -> vec3f {
    let t = clamp(t0, 0.0, 1.0);
    let c0 = vec3f(0.2777273, 0.0054073, 0.3340998);
    let c1 = vec3f(0.1050930, 1.4046135, 1.3845901);
    let c2 = vec3f(-0.3308618, 0.2148476, 0.0950952);
    let c3 = vec3f(-4.6342305, -5.7991005, -19.3324410);
    let c4 = vec3f(6.2282699, 14.1799333, 56.6905526);
    let c5 = vec3f(4.7763850, -13.7451460, -65.3530326);
    let c6 = vec3f(-5.4354559, 4.6458523, 26.3124352);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3f(0.0), vec3f(1.0));
}
// Two hues and a grey middle, for readings that can go either way.
fn div(t0: f32) -> vec3f {
    let t = clamp(t0, 0.0, 1.0) * 2.0 - 1.0;
    let mid = vec3f(0.62, 0.62, 0.64);
    if (t < 0.0) { return mix(mid, vec3f(0.13, 0.47, 0.95), -t); }
    return mix(mid, vec3f(0.98, 0.42, 0.10), t);
}
fn hue(h: f32, s: f32, v: f32) -> vec3f {
    let k = fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3f(1.0), clamp(abs(k - 3.0) - vec3f(1.0), vec3f(0.0), vec3f(1.0)), s);
}
fn norm(v: f32) -> f32 { return (v - V.cell_lo) / max(V.cell_hi - V.cell_lo, 1e-6); }
fn flag(on: bool) -> vec3f {
    if (on) { return vec3f(1.0, 0.86, 0.2); }
    return vec3f(0.16, 0.18, 0.26);
}
fn founder(c: u32) -> vec3f {
    let h = pcg(c ^ 0x9E3779B9u);
    return hue(unit_f(h), 0.55 + 0.45 * f32((h >> 3u) & 7u) / 7.0, 0.72 + 0.28 * f32((h >> 6u) & 7u) / 7.0);
}
const ACTION_TINT = array<vec3f, 12>(
    vec3f(0.45, 0.47, 0.55), vec3f(0.95, 0.95, 0.95), vec3f(0.20, 0.80, 0.30), vec3f(0.85, 0.20, 0.25), vec3f(0.95, 0.55, 0.75),
    vec3f(0.30, 0.65, 1.00), vec3f(1.00, 0.55, 0.10), vec3f(0.60, 0.05, 0.10), vec3f(0.65, 0.40, 1.00), vec3f(1.00, 0.90, 0.20),
    vec3f(0.55, 0.80, 0.85), vec3f(0.55, 0.40, 0.20),
);
fn main_action(s: u32) -> vec3f {
    var best = 0u;
    var most = -1.0;
    let b0 = stat[s].b0;
    let b1 = stat[s].b1;
    let b2 = stat[s].b2;
    var v = array<f32, 12>(b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w, b2.x, b2.y, b2.z, b2.w);
    for (var k = 0u; k < 12u; k++) {
        if (v[k] > most) {
            most = v[k];
            best = k;
        }
    }
    if (most <= 0.0) { return vec3f(0.3); }
    var tints = ACTION_TINT;
    return tints[best];
}
fn cell_colour(s: u32) -> vec3f {
    switch (V.cell_mode) {
        //@CELLS
        default: { return vec3f(1.0); }
    }
}
fn cell_shown(s: u32) -> vec3f {
    var c = cell_colour(s);
    var keep = true;
    if ((V.show & SHOW_FOUNDER) != 0u) { keep = cell[s].clade == V.clade; }
    if ((V.show & SHOW_SIMILAR) != 0u) {
        let k = sketch[s];
        let d0 = unpack4x8snorm(k.x) - unpack4x8snorm(V.sketch.x);
        let d1 = unpack4x8snorm(k.y) - unpack4x8snorm(V.sketch.y);
        let d2 = unpack4x8snorm(k.z) - unpack4x8snorm(V.sketch.z);
        let d3 = unpack4x8snorm(k.w) - unpack4x8snorm(V.sketch.w);
        keep = sqrt((dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3)) / 16.0) * 4.0 < V.tau;
    }
    if ((V.show & (SHOW_SIMILAR | SHOW_FOUNDER)) != 0u) {
        if (keep) { return mix(c, vec3f(1.0), 0.25); }
        return vec3f(0.2, 0.2, 0.23);
    }
    return c;
}
fn level(v: f32, hi: f32) -> f32 { return sqrt(clamp(v / max(hi, 1e-6), 0.0, 1.0)); }
// As sun_at in sweep.wgsl. With moving weather, the map has slid by V.drift.
fn sun_now(p: u32) -> f32 {
    let x = (p + u32(V.drift.x)) & MASK;
    let y = ((p >> LOG_W) + u32(V.drift.y)) & MASK;
    let t = terrain[(y << LOG_W) | x];
    let base = f32(t & 0xFFFFu) / 65535.0 * SUN_RANGE;
    let phase = f32((t >> 16u) & 255u) / 256.0 * 6.2831853;
    let m = select(1.0, 0.55 + 0.45 * sin(phase + V.season), V.season_on > 0.5);
    return base * V.sun_k * m * V.sun_keep;
}
// The kind of place a square is, when walls or biomes are on.
fn place_tint(t: u32) -> vec3f {
    switch (place_of_room(t >> 28u)) {
        case 0u: { return vec3f(0.30, 0.55, 1.00); }
        case 1u: { return vec3f(1.00, 0.35, 0.20); }
        case 2u: { return vec3f(0.25, 0.75, 0.40); }
        default: { return vec3f(0.75, 0.65, 0.40); }
    }
}
// What's on a square under any cell, before the opacity is applied.
fn field_colour(p: u32) -> vec3f {
    switch (V.field_mode) {
        //@FIELDS
        default: { return vec3f(0.0); }
    }
}
const BASE = vec3f(0.035, 0.04, 0.055);
fn ground_at(p: u32) -> vec3f {
    if (V.field_mode == 0u) { return BASE; }
    return BASE + field_colour(p) * V.field_opacity;
}

struct Seen { colour: vec3f, cell: u32 }
fn square(x: u32, y: u32) -> Seen {
    let p = (y << LOG_W) | x;
    let o = occ[p];
    if (o == ROCK) { return Seen(vec3f(0.23, 0.235, 0.27), 0u); }
    let under = ground_at(p);
    if (o == 0u || (V.show & SHOW_CELLS) == 0u) { return Seen(under, 0u); }
    let c = cell_shown(o - 1u);
    return Seen(mix(under, c, V.cell_opacity), o);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let py = id.y + V.row0;
    if (id.x >= V.out.x || py >= V.out.y) { return; }
    var acc = vec3f(0.0);
    let n = V.samples;
    for (var sy = 0u; sy < n; sy++) {
        for (var sx = 0u; sx < n; sx++) {
            let sub = (vec2f(f32(sx), f32(sy)) + 0.5) / f32(n);
            let q = V.origin + (vec2f(f32(id.x), f32(py)) + sub) * V.scale;
            let c = square(u32(i32(floor(q.x))) & MASK, u32(i32(floor(q.y))) & MASK).colour;
            // Average in light rather than in gamma, or sparse bright cells go dark when zoomed out.
            acc += c * c;
        }
    }
    var colour = sqrt(acc / f32(n * n));
    let q = V.origin + (vec2f(f32(id.x), f32(py)) + 0.5) * V.scale;
    let cx = u32(i32(floor(q.x))) & MASK;
    let cy = u32(i32(floor(q.y))) & MASK;

    // Close up, a cell is a square with a gap around it and a mark on the side it faces.
    if (V.scale < 0.2) {
        let at = q - floor(q);
        let o = occ[(cy << LOG_W) | cx];
        if (o != 0u && o != ROCK && (V.show & SHOW_CELLS) != 0u) {
            let edge = min(min(at.x, 1.0 - at.x), min(at.y, 1.0 - at.y));
            if (edge < V.scale) {
                colour = ground_at((cy << LOG_W) | cx);
            } else if (V.scale < 0.08) {
                let u = unit8(cell[o - 1u].flags & 7u);
                let from_mark = at - vec2f(0.5) - u * 0.3;
                if (dot(from_mark, from_mark) < 0.012) {
                    let lum = dot(colour, vec3f(0.3, 0.6, 0.1));
                    colour = select(vec3f(0.0), vec3f(1.0), lum < 0.35);
                }
            }
        }
    }
    // The world wraps. One whole world around the middle is bright, and what repeats past it is dim.
    let away = abs(q - V.center);
    if (max(away.x, away.y) > f32(W) * 0.5) { colour *= 0.32; }
    textureStore(out_tex, vec2u(id.x, py), vec4f(colour, 1.0));
}
