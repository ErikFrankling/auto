// Fields: deposit, exchange, decay, regrow. Jacobi on two buffers, in integers. Each side of a
// border works out the same flow with opposite signs, truncated towards zero, so exchange neither
// makes nor loses a single unit of food or meat.
//
// fa = food, meat, heat, enzyme. fb = four scents.

const FLOW_A = vec4f(FOOD_FLOW, MEAT_FLOW, HEAT_FLOW, ENZYME_FLOW);
const DECAY_A = vec4f(0.0, 0.0, HEAT_DECAY, ENZYME_DECAY);
const FLOW_B = vec4f(S0_FLOW, S1_FLOW, S2_FLOW, S3_FLOW);
const DECAY_B = vec4f(S0_DECAY, S1_DECAY, S2_DECAY, S3_DECAY);

fn route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

// Sun with the map slid along by the drift, when patches move. Its total stays the same.
fn sun_of(t: u32) -> f32 { return f32(t & 0xFFFFu) / 65535.0 * SUN_RANGE * F.sun_k; }
fn sun_here(x: u32, y: u32, t: u32) -> f32 {
    if (SUN_DRIFT <= 0.0) { return sun_of(t); }
    let x0 = (x + u32(F.drift_x)) & MASK;
    let x1 = (x0 + 1u) & MASK;
    let y0 = (y + u32(F.drift_y)) & MASK;
    let y1 = (y0 + 1u) & MASK;
    let near = mix(sun_of(terrain[(y0 << LOG_W) | x0]), sun_of(terrain[(y0 << LOG_W) | x1]), fract(F.drift_x));
    let far = mix(sun_of(terrain[(y1 << LOG_W) | x0]), sun_of(terrain[(y1 << LOG_W) | x1]), fract(F.drift_x));
    return mix(near, far, fract(F.drift_y));
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    if (x >= W) { return; }
    let y = id.y + F.row0;
    let p = (y << LOG_W) | x;
    let t = terrain[p];
    if ((t & T_ROCK) != 0u) { return; }
    let metered = LEDGER || ((y + F.tick) & 63u) == 0u;
    let ix = i32(x);
    let iy = i32(y);
    var a = fa0[p];
    var fa_in = vec4i(0);
    if ((t & T_E) == 0u) { fa_in += vec4i(vec4f(fa0[wrap_sq(ix + 1, iy)] - a) * FLOW_A); }
    if ((t & T_W) == 0u) { fa_in += vec4i(vec4f(fa0[wrap_sq(ix - 1, iy)] - a) * FLOW_A); }
    if ((t & T_S) == 0u) { fa_in += vec4i(vec4f(fa0[wrap_sq(ix, iy + 1)] - a) * FLOW_A); }
    if ((t & T_N) == 0u) { fa_in += vec4i(vec4f(fa0[wrap_sq(ix, iy - 1)] - a) * FLOW_A); }

    let d = p * DEP_STRIDE;
    let want_food = dep[d];
    let want_meat = dep[d + 1u];
    let meat_in = dep[d + 2u];
    let heat_in = dep[d + 3u];
    let enzyme_in = dep[d + 4u];
    dep[d] = 0u;
    dep[d + 1u] = 0u;
    dep[d + 2u] = 0u;
    dep[d + 3u] = 0u;
    dep[d + 4u] = 0u;

    // Supply and demand. Only what can't flow away this tick is on offer, so a square can't go
    // below zero by being both eaten and drained at once.
    let eaten = min(want_food, u32(f32(a.x) * FOOD_REACHABLE));
    let chewed = min(want_meat, u32(f32(a.y) * MEAT_REACHABLE));

    var next = a + fa_in;
    next.x -= i32(eaten);
    next.y -= i32(chewed);
    // Decay comes after exchange, as in predictions.py.
    next -= vec4i(vec4f(next) * DECAY_A);
    next.y += i32(meat_in);
    next.z += i32(heat_in);
    next.w += i32(enzyme_in);

    // Meat slowly rots into food, and sun grows food up to a limit.
    let rot = i32(f32(next.y) * MEAT_ROT);
    next.y -= rot;
    let cap = i32(FOOD_CAP * E_ONE);
    let kept = min(rot, max(cap - next.x, 0));
    next.x += kept;
    let sun = i32(sun_here(x, y, t) * E_ONE);
    let grown = min(sun, max(cap - next.x, 0));
    next.x += grown;
    fa1[p] = next;

    if (metered) {
        route(R_SUN, u32(grown));
        route(R_ROT_LOST, u32(rot - kept));
        route(R_EATEN, eaten);
        route(R_MEAT_EATEN, chewed);
    }

    if (SCENT_ON) {
        var b = fb0[p];
        var fb_in = vec4i(0);
        if ((t & T_E) == 0u) { fb_in += vec4i(vec4f(fb0[wrap_sq(ix + 1, iy)] - b) * FLOW_B); }
        if ((t & T_W) == 0u) { fb_in += vec4i(vec4f(fb0[wrap_sq(ix - 1, iy)] - b) * FLOW_B); }
        if ((t & T_S) == 0u) { fb_in += vec4i(vec4f(fb0[wrap_sq(ix, iy + 1)] - b) * FLOW_B); }
        if ((t & T_N) == 0u) { fb_in += vec4i(vec4f(fb0[wrap_sq(ix, iy - 1)] - b) * FLOW_B); }
        var nb = b + fb_in;
        nb -= vec4i(vec4f(nb) * DECAY_B);
        nb += vec4i(i32(dep[d + 5u]), i32(dep[d + 6u]), i32(dep[d + 7u]), i32(dep[d + 8u]));
        dep[d + 5u] = 0u;
        dep[d + 6u] = 0u;
        dep[d + 7u] = 0u;
        dep[d + 8u] = 0u;
        fb1[p] = nb;
    }
}
