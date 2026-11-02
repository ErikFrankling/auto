// A random cell lands now and then, so the world can't die for good. Each tile rolls once a tick,
// and a hit asks for a slot the same way a split does.

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= TILES || id.y >= TILES) { return; }
    var r = pcg(((id.y << 16u) | id.x) ^ (F.tick * 2654435761u) ^ (F.seed * 40503u) ^ 0x5bd1e995u);
    let burst = f32(F.tick) < RAIN_BURST_TICKS && ctl[L_ALIVE] < ctl[L_OPEN] / 100u;
    let chance = select(RAIN_P * f32(TILE * TILE) / 4096.0, 1.0, burst);
    r = pcg(r);
    if (unit_f(r) >= chance) { return; }
    let k = SLOTS + id.y * TILES + id.x;
    atomicOr(&bits[k >> 5u], 1u << (k & 31u));
}
