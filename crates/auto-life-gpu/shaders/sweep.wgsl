// World physics, run twice per tick: once along x, once along y. Neighbouring
// squares trade field values in pairs, in place. Nothing is created or lost by
// the trading, which keeps the energy ledger exact and needs no second buffer.

const FLOW_A = vec4f(0.0, MEAT_FLOW, HEAT_FLOW, NOISE_FLOW);
const KEEP_A = vec4f(1.0, 1.0, HEAT_DECAY_KEEP, NOISE_DECAY_KEEP);
const FLOW_B = vec4f(P0_FLOW, P1_FLOW, P2_FLOW, P3_FLOW);
const KEEP_B = vec4f(P0_DECAY_KEEP, P1_DECAY_KEEP, P2_DECAY_KEEP, P3_DECAY_KEEP);
const FLOW_C = vec4f(ENZYME_FLOW, FCY_FLOW, FCZ_FLOW, FCW_FLOW);
const KEEP_C = vec4f(ENZYME_DECAY_KEEP, FCY_DECAY_KEEP, FCZ_DECAY_KEEP, FCW_DECAY_KEEP);

fn sun_at(t: u32) -> f32 {
    let base = f32(t & 0xFFFFu) / 65535.0 * SUN_RANGE;
    let phase = f32((t >> 16u) & 255u) / 256.0 * 6.2831853;
    let m = select(1.0, 0.55 + 0.45 * sin(phase + F.season), SEASON_TICKS > 0.0);
    return base * F.sun_k * m;
}

// Sun grows ground food, and meat slowly rots into it.
fn feed(v: vec4f, t: u32) -> vec4f {
    var a = v;
    let got = min(sun_at(t), max(GROUND_CAP - a.x, 0.0));
    a.x += got;
    let rot = a.y * MEAT_ROT;
    a.y -= rot;
    let kept = min(rot, max(GROUND_CAP - a.x, 0.0));
    a.x += kept;
    if (LEDGER) {
        atomicAdd(&ctr[C_SUN], u32(got * LEDGER_SCALE + 0.5));
        lose(rot - kept);
    }
    return a;
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    var pa: u32;
    var pb: u32;
    if (F.axis == 0u) {
        let i = id.x * 2u + F.off;
        if (i >= W) { return; }
        let j = id.y + F.row0;
        pa = (j << LOG_W) | i;
        pb = (j << LOG_W) | ((i + 1u) & MASK);
    } else {
        if (id.x >= W) { return; }
        let j = (id.y + F.row0) * 2u + F.off;
        pa = (j << LOG_W) | id.x;
        pb = (((j + 1u) & MASK) << LOG_W) | id.x;
    }
    let ta = terrain[pa];
    let tb = terrain[pb];
    let rock_a = (ta & T_ROCK) != 0u;
    let rock_b = (tb & T_ROCK) != 0u;
    if (rock_a && rock_b) { return; }
    var a0 = fa[pa];
    var b0 = fa[pb];
    var a1 = fb[pa];
    var b1 = fb[pb];
    var a2 = fc[pa];
    var b2 = fc[pb];
    if (!rock_a && !rock_b) {
        let f0 = FLOW_A * (a0 - b0);
        a0 -= f0;
        b0 += f0;
        let f1 = FLOW_B * (a1 - b1);
        a1 -= f1;
        b1 += f1;
        let f2 = FLOW_C * (a2 - b2);
        a2 -= f2;
        b2 += f2;
    }
    a0 *= KEEP_A;
    b0 *= KEEP_A;
    a1 *= KEEP_B;
    b1 *= KEEP_B;
    a2 *= KEEP_C;
    b2 *= KEEP_C;
    if (F.axis == 0u) {
        if (!rock_a) { a0 = feed(a0, ta); }
        if (!rock_b) { b0 = feed(b0, tb); }
    }
    if (!rock_a) {
        fa[pa] = a0;
        fb[pa] = a1;
        fc[pa] = a2;
    }
    if (!rock_b) {
        fa[pb] = b0;
        fb[pb] = b1;
        fc[pb] = b2;
    }
}
