// Draws part of the world for a person looking at it. Reads the world and changes nothing.
//
// Up close, each cell is a soft polygon: every pixel belongs to whichever nearby cell it's most
// inside of, as far as a little past that cell's radius, with a darker edge where it meets a
// neighbour or the open. Where touching cells overlap, the line between them is what makes a
// body read as cells. From far away a pixel is too big for any of that, so it shows a cell in
// its bin, or nothing.

struct ViewU {
    origin: vec2f, scale: f32, cell_mode: u32,
    out: vec2u, row0: u32, field_mode: u32,
    field_opacity: f32, cell_opacity: f32, age_top: f32, gen_top: f32,
    center: vec2f, e_top: f32, flags: u32,
    hi: array<vec4f, 2>,
    axes: array<vec4f, 12>,
    mean: array<vec4f, 4>,
    gain: vec4f,
}
@group(1) @binding(0) var<uniform> V: ViewU;
@group(1) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

const SHOW_CELLS: u32 = 1u;

// Viridis, as a polynomial (Matt Zucker's fit).
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
fn hue(h: f32, s: f32, v: f32) -> vec3f {
    let k = fract(vec3f(h) + vec3f(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3f(1.0), clamp(abs(k - 3.0) - vec3f(1.0), vec3f(0.0), vec3f(1.0)), s);
}

// One colour for each budget action, in budget order.
const ACTION_TINT = array<vec3f, 12>(
    vec3f(0.45, 0.47, 0.55), vec3f(0.95, 0.95, 0.95), vec3f(0.20, 0.80, 0.30), vec3f(0.85, 0.20, 0.25),
    vec3f(0.95, 0.55, 0.75), vec3f(0.30, 0.65, 1.00), vec3f(1.00, 0.55, 0.10), vec3f(0.60, 0.05, 0.10),
    vec3f(0.65, 0.40, 1.00), vec3f(1.00, 0.90, 0.20), vec3f(0.20, 0.85, 0.85), vec3f(0.75, 0.60, 0.35),
);
fn share_of(s: u32, k: u32) -> f32 {
    if (k < 4u) { return plan[s].s0[k]; }
    if (k < 8u) { return plan[s].s1[k - 4u]; }
    return plan[s].s2[k - 8u];
}
fn main_action(s: u32) -> vec3f {
    var best = 0u;
    var most = -1.0;
    for (var k = 0u; k < 12u; k++) {
        let v = share_of(s, k);
        if (v > most) {
            most = v;
            best = k;
        }
    }
    var tints = ACTION_TINT;
    return tints[best];
}
fn genome_colour(s: u32) -> vec3f {
    let k = sketch[s];
    let v0 = unpack4x8snorm(k.x) * 4.0 - V.mean[0];
    let v1 = unpack4x8snorm(k.y) * 4.0 - V.mean[1];
    let v2 = unpack4x8snorm(k.z) * 4.0 - V.mean[2];
    let v3 = unpack4x8snorm(k.w) * 4.0 - V.mean[3];
    let pc = vec3f(
        dot(V.axes[0], v0) + dot(V.axes[1], v1) + dot(V.axes[2], v2) + dot(V.axes[3], v3),
        dot(V.axes[4], v0) + dot(V.axes[5], v1) + dot(V.axes[6], v2) + dot(V.axes[7], v3),
        dot(V.axes[8], v0) + dot(V.axes[9], v1) + dot(V.axes[10], v2) + dot(V.axes[11], v3),
    ) * V.gain.xyz;
    let t = pc / (vec3f(1.0) + abs(pc));
    return vec3f(0.5) + 0.5 * t;
}
fn cell_colour(s: u32) -> vec3f {
    let m = V.cell_mode;
    // Modes 0 to 11 are the share of the budget for each action, after the others below.
    if (m >= 16u) { return seq(share_of(s, m - 16u) * 2.0); }
    switch (m) {
        case 0u: { return genome_colour(s); }
        case 1u: { return seq(f32(cell[s].e) * E_INV / V.e_top); }
        case 2u: { return main_action(s); }
        case 3u: { return seq(f32(cell[s].contacts) / 8.0); }
        case 4u: { return seq(f32(cell[s].age) / V.age_top); }
        case 5u: {
            let h = pcg(cell[s].clade ^ 0x9E3779B9u);
            return hue(unit_f(h), 0.55 + 0.45 * f32((h >> 3u) & 7u) / 7.0, 0.72 + 0.28 * f32((h >> 6u) & 7u) / 7.0);
        }
        case 6u: { return seq(f32(cell[s].generation) / V.gen_top); }
        case 7u: { return seq(cell[s].stick); }
        case 8u: { return seq(f32(cell[s].womb) * E_INV / 5.0); }
        case 9u: { return seq((log(max(cell[s].mu, 1e-4)) + 7.0) / 6.5); }
        default: { return vec3f(1.0); }
    }
}

fn level(v: f32, hi: f32) -> f32 { return sqrt(clamp(v / max(hi, 1e-6), 0.0, 1.0)); }
fn field_colour(p: u32) -> vec3f {
    let a = vec4f(fa0[p]) * E_INV;
    let b = vec4f(fb0[p]) * E_INV;
    switch (V.field_mode) {
        case 1u: { return vec3f(0.25, 0.95, 0.30) * level(a.x, V.hi[0].x); }
        case 2u: { return vec3f(1.00, 0.25, 0.30) * level(a.y, V.hi[0].y); }
        case 3u: { return vec3f(1.00, 0.50, 0.08) * level(a.z, V.hi[0].z); }
        case 4u: { return vec3f(0.90, 0.35, 1.00) * level(a.w, V.hi[0].w); }
        case 5u: { return vec3f(1.00, 0.22, 0.28) * level(b.x, V.hi[1].x); }
        case 6u: { return vec3f(0.20, 1.00, 0.32) * level(b.y, V.hi[1].y); }
        case 7u: { return vec3f(0.25, 0.55, 1.00) * level(b.z, V.hi[1].z); }
        case 8u: { return vec3f(1.00, 0.85, 0.10) * level(b.w, V.hi[1].w); }
        case 9u: {
            let sum = vec3f(1.0, 0.22, 0.28) * level(b.x, V.hi[1].x) + vec3f(0.2, 1.0, 0.32) * level(b.y, V.hi[1].y)
                + vec3f(0.25, 0.55, 1.0) * level(b.z, V.hi[1].z) + vec3f(1.0, 0.85, 0.1) * level(b.w, V.hi[1].w);
            return vec3f(1.0) - exp(-sum * 1.1);
        }
        case 10u: {
            let t = terrain[p];
            return vec3f(1.00, 0.80, 0.25) * sqrt(clamp(f32(t & 0xFFFFu) / 65535.0, 0.0, 1.0));
        }
        default: { return vec3f(0.0); }
    }
}

const BASE = vec3f(0.0, 0.0, 0.0);
const ROCK_COLOUR = vec3f(0.23, 0.235, 0.27);

fn wrap_near(d: f32) -> f32 {
    let w = f32(W);
    return d - round(d / w) * w;
}

fn shade(q: vec2f) -> vec3f {
    let w = f32(W);
    let qw = q - floor(q / w) * w;
    let p = (min(u32(qw.y), MASK) << LOG_W) | min(u32(qw.x), MASK);
    var under = BASE;
    if ((terrain[p] & T_ROCK) != 0u) {
        under = ROCK_COLOUR;
    } else if (V.field_mode != 0u) {
        under = BASE + field_colour(p) * V.field_opacity;
    }
    if ((V.flags & SHOW_CELLS) == 0u) { return under; }

    let bx = min(u32(qw.x * 0.5), BIN_MASK);
    let by = min(u32(qw.y * 0.5), BIN_MASK);
    // From far away, whichever cell is in this bin stands for it.
    if (V.scale >= 1.0) {
        let o = bb[((by << LOG_B) | bx) * BIN_CAP];
        if (o.slot == 0u) { return under; }
        return mix(under, cell_colour(o.slot - 1u), V.cell_opacity);
    }
    var best = 1e9;
    var second = 1e9;
    var slot = 0u;
    for (var dy = 0u; dy < 3u; dy++) {
        let oy = (by + dy + BIN_MASK) & BIN_MASK;
        for (var dx = 0u; dx < 3u; dx++) {
            let b2 = ((oy << LOG_B) | ((bx + dx + BIN_MASK) & BIN_MASK)) * BIN_CAP;
            for (var k = 0u; k < BIN_CAP; k++) {
                let o = bb[b2 + k];
                if (o.slot == 0u) { break; }
                let ox = wrap_near(f32(o.x) * POS_TO_SQ - qw.x);
                let oy2 = wrap_near(f32(o.y) * POS_TO_SQ - qw.y);
                let inside = sqrt(ox * ox + oy2 * oy2) - o.r;
                if (inside < best) {
                    second = best;
                    best = inside;
                    slot = o.slot;
                } else if (inside < second) {
                    second = inside;
                }
            }
        }
    }
    // A little past its radius, so cells that touch close up into one surface.
    let swell = 0.07;
    if (slot == 0u || best > swell) { return under; }
    var c = cell_colour(slot - 1u);
    if (V.scale < 0.25) {
        // How far to this cell's edge: half way to the next cell's, or out to its limit.
        let edge = min((second - best) * 0.5, swell - best);
        let width = max(0.09, V.scale * 2.0);
        c *= 0.32 + 0.68 * smoothstep(0.0, width, edge);
        // Soft against the ground.
        return mix(under, c, V.cell_opacity * smoothstep(0.0, V.scale * 1.5, swell - best));
    }
    return mix(under, c, V.cell_opacity);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let py = id.y + V.row0;
    if (id.x >= V.out.x || py >= V.out.y) { return; }
    let q = V.origin + (vec2f(f32(id.x), f32(py)) + 0.5) * V.scale;
    var colour = shade(q);
    // The world wraps. One whole world around the middle is bright, and what repeats past it is dim.
    let away = abs(q - V.center);
    if (max(away.x, away.y) > f32(W) * 0.5) { colour *= 0.32; }
    textureStore(out_tex, vec2u(id.x, py), vec4f(colour, 1.0));
}
