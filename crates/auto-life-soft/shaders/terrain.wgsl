// Builds the land on the GPU. The same scale-free sun and rock as the lattice engine.

//@NOISE

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let world_x = id.x;
    let world_y = id.y + F.row0;
    if (world_x >= W) { return; }
    let p = (world_y << LOG_W) | world_x;
    // A batch: many separate worlds on one map, each with its own land, with rock between them.
    var x = world_x;
    var y = world_y;
    var seed = F.seed;
    var between = false;
    let tile = max(u32(BATCH_TILE), 1u);
    span = W;
    if (BATCH_TILE > 0.0) {
        span = tile;
        x = world_x & (tile - 1u);
        y = world_y & (tile - 1u);
        seed = pcg(seed ^ ((world_y / tile) * 65536u + world_x / tile + 1u));
        between = x < u32(BATCH_WALL) || y < u32(BATCH_WALL);
    }
    let base = min(scale_free_sun(f32(x), f32(y), seed), SUN_RANGE * 0.999);
    var rock = between;
    if (ROCK_ON) { rock = rock || scale_free_rock(f32(x), f32(y), seed); }
    var t = u32(clamp(base / SUN_RANGE, 0.0, 1.0) * 65535.0);
    if (rock) { t |= T_ROCK; }
    terrain[p] = t;
}

// Once the mean of the sun is known: each square learns which neighbours are rock, so nothing
// later has to read five terrain words, and the food starts as if some sun had already fallen.
@compute @workgroup_size(64, 1)
fn finish(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y + F.row0;
    if (x >= W) { return; }
    let p = (y << LOG_W) | x;
    let ix = i32(x);
    let iy = i32(y);
    var t = terrain[p];
    if ((terrain[wrap_sq(ix + 1, iy)] & T_ROCK) != 0u) { t |= T_E; }
    if ((terrain[wrap_sq(ix - 1, iy)] & T_ROCK) != 0u) { t |= T_W; }
    if ((terrain[wrap_sq(ix, iy + 1)] & T_ROCK) != 0u) { t |= T_S; }
    if ((terrain[wrap_sq(ix, iy - 1)] & T_ROCK) != 0u) { t |= T_N; }
    terrain[p] = t;
    if ((t & T_ROCK) != 0u) { return; }
    let base = f32(t & 0xFFFFu) / 65535.0 * SUN_RANGE;
    let food = min(FOOD_CAP, base * F.sun_k * FOOD_START_TICKS);
    fa0[p] = vec4i(i32(food * E_ONE), 0, 0, 0);
}

// Slots start free, in order.
@compute @workgroup_size(64)
fn fill_ring(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 65536u;
    if (i >= SLOTS) { return; }
    ring[i] = i;
}
