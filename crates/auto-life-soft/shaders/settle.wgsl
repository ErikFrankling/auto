// A cell's own books for the tick: trades, harvest, harm, death, and the step it takes. Reads the
// old fields and the totals on its square, and writes only itself, apart from atomics on its square
// when it dies and one bit when it wants a slot or gives one back.

fn route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x + F.row0;
    if (s >= SLOTS) { return; }
    var c = cell[s];
    if ((c.flags & F_ALIVE) == 0u) { return; }
    let metered = LEDGER || ((s + F.tick) & 63u) == 0u;
    let a = acc[s];
    var blank: Acc;
    acc[s] = blank;

    // ---- Trades ----
    let left = c.e - a.give_out;
    let asked = mail[s * 2u];
    let paid = min(asked, left);
    let inbox = mail[s * 2u + 1u];
    mail[s * 2u] = 0u;
    mail[s * 2u + 1u] = 0u;
    let bitten_out = paid == left && paid > 0u;
    var e = left - paid + inbox + a.bite_got;
    let fee = min(u32(a.fee * E_ONE), e);
    e -= fee;

    // ---- Harvest ----
    let p = square_of(c.x, c.y);
    let d = p * DEP_STRIDE;
    let here = fa0[p];
    var got_food = 0u;
    if (a.wish_food > 0u) {
        let supply = f32(here.x) * FOOD_REACHABLE;
        let want = f32(atomicLoad(&dep[d]));
        got_food = u32(f32(a.wish_food) * min(1.0, supply / max(want, 1.0)) * 0.999999);
    }
    var got_meat = 0u;
    if (a.wish_meat > 0u) {
        let supply = f32(here.y) * MEAT_REACHABLE;
        let want = f32(atomicLoad(&dep[d + 1u]));
        got_meat = u32(f32(a.wish_meat) * min(1.0, supply / max(want, 1.0)) * 0.999999 * MEAT_YIELD);
    }
    e += got_food + got_meat;

    // ---- Heat ----
    // Hits thin cells hardest, since rich ones have reserves to spread the damage over.
    var harm = 0u;
    if (HEAT_HARM > 0.0) {
        let h = f32(here.z) * E_INV;
        let burn = HEAT_HARM * h / (1.0 + f32(e) * E_INV / HEAT_SCALE);
        harm = min(u32(burn * E_ONE), e);
        e -= harm;
    }
    let burned_out = harm > 0u && e == 0u;

    // ---- Reserves have a limit ----
    let cap_u = u32(MAX_ENERGY * E_ONE);
    let over = e - min(e, cap_u);
    e -= over;

    if (metered) {
        route(R_BITE_PAID, paid);
        route(R_STICK, fee);
        route(R_EAT_GOT, got_food);
        route(R_MEAT_GOT, got_meat);
        route(R_HARM, harm);
        route(R_CAP, over);
    }
    c.contacts = a.contacts;
    c.felt += f32(harm + paid) * E_INV;
    c.bitten = f32(paid) * E_INV;

    // ---- Death ----
    var r = pcg(s ^ (F.tick * 2654435761u) ^ (F.seed * 40503u) ^ 0x51ed270bu);
    r = pcg(r);
    let old_age = unit_f(r) < HAZARD;
    if (e == 0u || old_age) {
        // Its body and everything it held becomes meat.
        let meat = e + c.womb + u32(BODY * E_ONE);
        atomicAdd(&dep[d + 2u], meat);
        if (metered) { route(R_DEATH, meat); }
        var which = C_STARVED;
        if (old_age && e > 0u) {
            which = C_OLD;
        } else if (burned_out) {
            which = C_BURNED;
        } else if (bitten_out) {
            which = C_BITTEN;
        }
        atomicAdd(&ctr[which], 1u);
        atomicOr(&bits[FREE_BASE + (s >> 5u)], 1u << (s & 31u));
        c.flags = 0u;
        c.e = 0u;
        c.womb = 0u;
        c.unit = 0.0;
        cell[s] = c;
        return;
    }
    c.e = e;

    // ---- Move ----
    // Overdamped: velocity is force over drag, with a limit on the step.
    let t = terrain[p];
    var f = vec2f(f32(a.fx), f32(a.fy)) / FORCE_ONE;
    if (THRUST_ON) {
        f += vec2f(c.hx, c.hy) * effort(plan[s].s0.y * c.unit) * THRUST_RATE;
    }
    // Rock pushes a disc that overlaps it, like another cell would.
    let at_x = f32(c.x & POS_LOW) * POS_TO_SQ;
    let at_y = f32(c.y & POS_LOW) * POS_TO_SQ;
    if ((t & T_E) != 0u) { f.x -= K_PUSH * max(c.r - (1.0 - at_x), 0.0); }
    if ((t & T_W) != 0u) { f.x += K_PUSH * max(c.r - at_x, 0.0); }
    if ((t & T_S) != 0u) { f.y -= K_PUSH * max(c.r - (1.0 - at_y), 0.0); }
    if ((t & T_N) != 0u) { f.y += K_PUSH * max(c.r - at_y, 0.0); }
    var v = f / DRAG;
    let speed = length(v);
    if (speed > STEP_MAX) { v *= STEP_MAX / speed; }
    var nx = c.x + u32(i32(v.x * SQ_TO_POS));
    var ny = c.y + u32(i32(v.y * SQ_TO_POS));
    // Nothing enters rock, however hard it's pushed. Rock is at least 5 squares thick, and no
    // step is more than one square, so nothing can get to the far side of a wall.
    let cross_x = (nx >> POS_SHIFT) != (c.x >> POS_SHIFT);
    let cross_y = (ny >> POS_SHIFT) != (c.y >> POS_SHIFT);
    if (cross_x && (t & select(T_W, T_E, v.x > 0.0)) != 0u) {
        nx = c.x;
    }
    if (cross_y && (t & select(T_N, T_S, v.y > 0.0)) != 0u) {
        ny = c.y;
    }
    if ((nx >> POS_SHIFT) != (c.x >> POS_SHIFT) && (ny >> POS_SHIFT) != (c.y >> POS_SHIFT)
        && (terrain[square_of(nx, ny)] & T_ROCK) != 0u) {
        ny = c.y;
    }
    c.x = nx;
    c.y = ny;

    // ---- Split ----
    // When the womb covers a body and the brain says so. alloc.wgsl finds it a slot.
    if (SPLIT_ON && c.womb >= u32(BODY * E_ONE) && plan[s].split.w > 0.0) {
        atomicOr(&bits[s >> 5u], 1u << (s & 31u));
    }
    cell[s] = c;
}
