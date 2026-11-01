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

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let x = id.x;
    let y = id.y + F.row0;
    if (x >= W) { return; }
    let p = (y << LOG_W) | x;
    let seed = F.seed;
    let patch_ = vnoise(x, y, scale_of(32u), seed ^ 13u);
    let region = vnoise(x, y, scale_of(128u), seed ^ 21u);
    let land = vnoise(x, y, scale_of(512u), seed ^ 31u);
    let fine = vnoise(x, y, scale_of(8u), seed ^ 14u);
    let base = patch_ * patch_ * patch_ * 4.0 * (0.25 + 1.5 * region) * (0.25 + 1.5 * land) + fine * 0.3 + 0.02;
    var rock = false;
    if (ROCK_ON) {
        let small_ridge = vnoise(x, y, scale_of(32u), seed ^ 11u);
        let small_gap = vnoise(x, y, scale_of(16u), seed ^ 12u);
        let big_ridge = vnoise(x, y, scale_of(256u), seed ^ 41u);
        let big_gap = vnoise(x, y, scale_of(64u), seed ^ 42u);
        rock = (abs(small_ridge - 0.5) < 0.035 && small_gap > 0.42)
            || (abs(big_ridge - 0.5) < 0.008 && big_gap > 0.3);
    }
    let phase = u32(vnoise(x, y, scale_of(256u), seed ^ 15u) * 2.0 * 255.0) & 255u;
    let biome = u32(vnoise(x, y, scale_of(1024u), seed ^ 51u) * 15.99);
    var t = u32(clamp(base / SUN_RANGE, 0.0, 1.0) * 65535.0) | (phase << 16u) | (biome << 28u);
    if (rock) { t |= T_ROCK; }
    terrain[p] = t;
    occ[p] = select(0u, ROCK, rock);
    seen[p] = select(0u, 1u, rock);
    seen_next[p] = select(0u, 1u, rock);
    fa[p] = vec4f(select(min(GROUND_CAP, base * 2.0 * SUN / 0.05), 0.0, rock), 0.0, 0.0, 0.0);
    fb[p] = vec4f(0.0);
    fc[p] = vec4f(0.0);
}

@compute @workgroup_size(64)
fn fill_ring(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x + id.y * 65536u;
    if (i >= SLOTS) { return; }
    ring[i] = i;
    cell[i].e = -1.0;
    cell[i].pos = NONE;
}
