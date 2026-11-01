// A random cell lands now and then, so the world can't die for good. Each tile
// picks one square, so no two landings can collide.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= TILES || id.y >= TILES) { return; }
    var r = pcg(((id.y << 16u) | id.x) ^ (F.tick * 2654435761u) ^ (F.seed * 40503u) ^ 0x5bd1e995u);
    let burst = f32(F.tick) < RAIN_BURST_TICKS && ctl[1] < ctl[3] / 100u;
    let chance = select(RAIN_P * f32(TILE * TILE) / 4096.0, 1.0, burst);
    r = pcg(r);
    if (unit_f(r) >= chance) { return; }
    r = pcg(r);
    let x = id.x * TILE + (r & (TILE - 1u));
    r = pcg(r);
    let y = id.y * TILE + (r & (TILE - 1u));
    let p = (y << LOG_W) | x;
    if (occ[p] != 0u) { return; }
    let t = atomicAdd(&ctr[C_TAIL], 1u);
    if (i32(t - ctl[0]) >= 0) { return; }
    let child = ring[t & RING_MASK];
    r = pcg(r);
    var c: Cell;
    c.e = RAIN_ENERGY;
    c.stamp = F.tick;
    c.pos = p;
    c.flags = r & 7u;
    r = pcg(r);
    c.clade = r;
    cell[child] = c;
    for (var k = 0u; k < HR; k++) { mem[child * HR + k] = vec4f(0.0); }
    occ[p] = child + 1u;
    seen_next[p] = 0x80u | (r & 7u);
    let j = atomicAdd(&ctr[C_JOBS], 1u);
    r = pcg(r);
    jobs[j] = vec4u(child, NONE, 0u, r);
    atomicAdd(&ctr[C_RAINED], 1u);
    if (LEDGER) { atomicAdd(&ctr[C_RAIN_E], u32((RAIN_ENERGY + BODY) * LEDGER_SCALE + 0.5)); }
}
