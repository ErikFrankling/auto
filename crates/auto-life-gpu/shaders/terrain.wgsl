// Builds the world on the GPU, so a huge world never needs a CPU copy. Patches
// nest inside regions inside continents, so there are places of every size to
// live in, and rock ridges cut them apart with gaps to pass through.

fn hash2(x: u32, y: u32, seed: u32) -> f32 {
    return unit_f(pcg(pcg(x ^ (seed * 2654435761u)) ^ y));
}

// Smooth noise that wraps at the world edge. `size` is a power of two.
fn vnoise(x: u32, y: u32, size: u32, seed: u32) -> f32 {
    let m = W / size - 1u;
    let ix = x / size;
    let iy = y / size;
    let tx0 = f32(x & (size - 1u)) / f32(size);
    let ty0 = f32(y & (size - 1u)) / f32(size);
    let tx = tx0 * tx0 * (3.0 - 2.0 * tx0);
    let ty = ty0 * ty0 * (3.0 - 2.0 * ty0);
    let a = hash2(ix & m, iy & m, seed);
    let b = hash2((ix + 1u) & m, iy & m, seed);
    let c = hash2(ix & m, (iy + 1u) & m, seed);
    let d = hash2((ix + 1u) & m, (iy + 1u) & m, seed);
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

fn scale_of(want: u32) -> u32 { return min(want, W / 2u); }

// The same noise at any point, not just on squares, so coordinates can be bent.
fn fnoise(x: f32, y: f32, size: u32, seed: u32) -> f32 {
    let m = W / size - 1u;
    let fs = f32(size);
    let w = f32(W);
    let px = x - floor(x / w) * w;
    let py = y - floor(y / w) * w;
    let ix = u32(px / fs);
    let iy = u32(py / fs);
    let tx0 = fract(px / fs);
    let ty0 = fract(py / fs);
    let tx = tx0 * tx0 * (3.0 - 2.0 * tx0);
    let ty = ty0 * ty0 * (3.0 - 2.0 * ty0);
    let a = hash2(ix & m, iy & m, seed);
    let b = hash2((ix + 1u) & m, iy & m, seed);
    let c = hash2(ix & m, (iy + 1u) & m, seed);
    let d = hash2((ix + 1u) & m, (iy + 1u) & m, seed);
    return mix(mix(a, b, tx), mix(c, d, tx), ty);
}

// Plain lattice noise always peaks on its grid, which lines rich land up in rows.
// Bending the coordinates by noise of the same size breaks the grid and keeps the wrap.
fn bent(x: f32, y: f32, size: u32, seed: u32) -> f32 {
    let fs = f32(size);
    let wx = x + (fnoise(x, y, size, seed ^ 201u) - 0.5) * fs * 2.5;
    let wy = y + (fnoise(x, y, size, seed ^ 202u) - 0.5) * fs * 2.5;
    return fnoise(wx, wy, size, seed);
}

// Sun with patches of every size, from a few squares to continents, with most of the
// variation in the big ones. A lot of different places to live, none the same size.
fn scale_free_sun(x: f32, y: f32, seed: u32) -> f32 {
    var sum = 0.0;
    var total = 0.0;
    var size = 8u;
    var weight = 0.45;
    var k = 0u;
    loop {
        if (size > W / 2u) { break; }
        sum += (bent(x, y, size, seed ^ (300u + k * 7u)) * 2.0 - 1.0) * weight;
        total += weight * weight;
        weight *= 1.3;
        size *= 2u;
        k += 1u;
    }
    return exp(TERRAIN_CONTRAST * sum / sqrt(max(total, 1e-6)));
}

// How rich a square is in mineral: patches of every size, from noise of its own, pushed away
// from rich sun by mineral_anti. Only the shape matters: fill_mineral sets how much there is.
fn mineral_richness(x: f32, y: f32, seed: u32, sun: f32) -> f32 {
    var sum = 0.0;
    var total = 0.0;
    var size = 8u;
    var weight = 0.45;
    var k = 0u;
    loop {
        if (size > W / 2u) { break; }
        sum += (bent(x, y, size, seed ^ (700u + k * 19u)) * 2.0 - 1.0) * weight;
        total += weight * weight;
        weight *= 1.3;
        size *= 2u;
        k += 1u;
    }
    let own = exp(MINERAL_CONTRAST * sum / sqrt(max(total, 1e-6)));
    return max(own * pow(max(sun, 0.02), -MINERAL_ANTI), 1e-3);
}

// Ridges of mixed size and thickness, instead of one size of wall everywhere.
fn scale_free_rock(x: f32, y: f32, seed: u32) -> bool {
    var size = 16u;
    var k = 0u;
    loop {
        if (size > W / 2u) { break; }
        // Thin at every size, and only in some places, so regions vary from open to walled in.
        let ridge = bent(x, y, size, seed ^ (400u + k * 11u));
        let where_ = bent(x, y, min(size * 4u, W / 2u), seed ^ (500u + k * 13u));
        let width = 0.008 + 0.05 * where_ * where_ * where_ * where_;
        let gap = fnoise(x, y, max(size / 2u, 8u), seed ^ (600u + k * 17u));
        if (abs(ridge - 0.5) < width && gap > 0.35 && where_ > 0.45) {
            return true;
        }
        size *= 4u;
        k += 1u;
    }
    return false;
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y + F.row0;
    if (x >= W) { return; }
    let p = (y << LOG_W) | x;
    let seed = F.seed;
    let new_style = TERRAIN_STYLE >= 1.0;
    let fx = f32(x);
    let fy = f32(y);
    var base: f32;
    if (new_style) {
        base = min(scale_free_sun(fx, fy, seed), SUN_RANGE * 0.999);
    } else {
        let patch_ = vnoise(x, y, scale_of(32u), seed ^ 13u);
        let region = vnoise(x, y, scale_of(128u), seed ^ 21u);
        let land = vnoise(x, y, scale_of(512u), seed ^ 31u);
        let fine = vnoise(x, y, scale_of(8u), seed ^ 14u);
        base = patch_ * patch_ * patch_ * 4.0 * (0.25 + 1.5 * region) * (0.25 + 1.5 * land) + fine * 0.3 + 0.02;
    }
    var rock = false;
    if (ROCK_ON && new_style) {
        rock = scale_free_rock(fx, fy, seed);
    } else if (ROCK_ON) {
        let small_ridge = vnoise(x, y, scale_of(32u), seed ^ 11u);
        let small_gap = vnoise(x, y, scale_of(16u), seed ^ 12u);
        let big_ridge = vnoise(x, y, scale_of(256u), seed ^ 41u);
        let big_gap = vnoise(x, y, scale_of(64u), seed ^ 42u);
        rock = (abs(small_ridge - 0.5) < 0.035 && small_gap > 0.42)
            || (abs(big_ridge - 0.5) < 0.008 && big_gap > 0.3);
    }
    let phase_noise = select(vnoise(x, y, scale_of(256u), seed ^ 15u), bent(fx, fy, scale_of(128u), seed ^ 15u), new_style);
    let phase = u32(phase_noise * 2.0 * 255.0) & 255u;
    var biome = u32(vnoise(x, y, scale_of(1024u), seed ^ 51u) * 15.99);
    var place = 0u;
    if (BIOMES_ON || WALLS_ON) {
        // Rooms with wandering edges. The wander is gentle enough that a wall never thins to nothing.
        let room_w = W / 4u;
        let wander = scale_of(room_w / 2u);
        let u = f32(x) / f32(room_w) + 0.2 * (vnoise(x, y, wander, seed ^ 61u) - 0.5);
        let v = f32(y) / f32(room_w) + 0.2 * (vnoise(x, y, wander, seed ^ 62u) - 0.5);
        let fu = u - floor(u);
        let fv = v - floor(v);
        biome = (u32(i32(floor(u)) & 3) | (u32(i32(floor(v)) & 3) << 2u));
        place = place_of_room(biome);
        if (WALLS_ON) {
            let r = f32(room_w);
            let du = min(fu, 1.0 - fu) * r;
            let dv = min(fv, 1.0 - fv) * r;
            // One pass through the middle of each wall between two rooms.
            let wall_u = du < WALL_THICK * 0.5;
            let wall_v = dv < WALL_THICK * 0.5;
            let pass_u = abs(fv - 0.5) * r < WALL_PASS * 0.5;
            let pass_v = abs(fu - 0.5) * r < WALL_PASS * 0.5;
            if ((wall_u && !pass_u) || (wall_v && !pass_v)) {
                rock = true;
            } else if (wall_u || wall_v) {
                rock = false;
            }
        }
    }
    if (BIOMES_ON) { base *= SUNX_V[place]; }
    var t = u32(clamp(base / SUN_RANGE, 0.0, 1.0) * 65535.0) | (phase << 16u) | (biome << 28u);
    if (rock) { t |= T_ROCK; }
    terrain[p] = t;
    occ[p] = select(0u, ROCK, rock);
    seen[p] = select(0u, 1u, rock);
    seen_next[p] = select(0u, 1u, rock);
    fa[p] = vec4f(select(min(GROUND_CAP, base * 2.0 * SUN / 0.05), 0.0, rock), 0.0, 0.0, 0.0);
    fb[p] = vec4f(0.0);
    fc[p] = vec4f(0.0);
    fd[p] = vec4f(0.0);
    if (MINERAL_ON && !rock) { fd[p].z = mineral_richness(fx, fy, seed, base); }
}

// Lays mineral out as richness times F.a, in whole grains. Runs once, when the world is built.
@compute @workgroup_size(64, 1)
fn fill_mineral(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y + F.row0;
    if (x >= W) { return; }
    let p = (y << LOG_W) | x;
    fd[p].y = grains(fd[p].z * bitcast<f32>(F.a));
}

@compute @workgroup_size(64)
fn fill_ring(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 65536u;
    if (i >= SLOTS) { return; }
    ring[i] = i;
    cell[i].e = -1.0;
    cell[i].pos = NONE;
}
