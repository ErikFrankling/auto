// Field wishes and emissions: what each cell wants to eat from its square and what it puts into
// it. All by atomics on the square, in fixed point, so totals can't depend on thread order. Runs
// in bin order, so the squares it touches are close together in memory.

@compute @workgroup_size(32, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bx = id.x;
    if (bx >= BINS) { return; }
    let b = (((id.y + F.row0) << LOG_B) | bx) * BIN_CAP;
    for (var k = 0u; k < BIN_CAP; k++) {
        let me = bb[b + k];
        if (me.slot == 0u) { break; }
        let s = me.slot - 1u;
        let p = square_of(me.x, me.y);
        let unit = cell[s].unit;
        let pl = plan[s];
        let d = p * DEP_STRIDE;

        // Locked food: how much of what's under it a cell can reach, given the enzyme there.
        let z = f32(fa0[p].w) * E_INV;
        let reach = (1.0 - LOCKED) + LOCKED * z / (z + ENZYME_HALF);
        let food = u32(min(effort(pl.s0.z * unit) * EAT_RATE, EAT_MAX) * reach * E_ONE);
        if (food > 0u) { atomicAdd(&dep[d], food); }
        acc[s].wish_food = food;
        if (MEAT_ON) {
            let meat = u32(min(effort(pl.s0.w * unit) * MEAT_RATE, MEAT_MAX) * E_ONE);
            if (meat > 0u) { atomicAdd(&dep[d + 1u], meat); }
            acc[s].wish_meat = meat;
        }

        // Every action sheds heat, and a cell can also make it on purpose.
        var heat = HEAT_RATE * f32(cell[s].shed) * E_INV;
        if (HEAT_ON) { heat += effort(pl.s1.z * unit) * HEAT_POWER; }
        let heat_u = u32(heat * E_ONE);
        if (heat_u > 0u) { atomicAdd(&dep[d + 3u], heat_u); }

        if (ENZYME_ON) {
            let e_u = u32(effort(pl.s2.y * unit) * ENZYME_POWER * E_ONE);
            if (e_u > 0u) { atomicAdd(&dep[d + 4u], e_u); }
        }
        if (SCENT_ON) {
            let amount = pl.s1.y * unit * EMIT_POWER * E_ONE;
            let m = vec4u(pl.mix * amount);
            if (m.x > 0u) { atomicAdd(&dep[d + 5u], m.x); }
            if (m.y > 0u) { atomicAdd(&dep[d + 6u], m.y); }
            if (m.z > 0u) { atomicAdd(&dep[d + 7u], m.z); }
            if (m.w > 0u) { atomicAdd(&dep[d + 8u], m.w); }
        }
    }
}
