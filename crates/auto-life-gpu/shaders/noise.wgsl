// Land shapes that wrap: sun with patches of every size, and rock ridges. Shared by the
// lattice engine and the soft engine. Needs TERRAIN_CONTRAST, pcg and unit_f.

// How wide the noise is before it wraps: the whole world, or one world of a batch.
var<private> span: u32 = 0u;

fn hash2(x: u32, y: u32, seed: u32) -> f32 {
    return unit_f(pcg(pcg(x ^ (seed * 2654435761u)) ^ y));
}

// Smooth noise that wraps at the world edge. `size` is a power of two.
fn vnoise(x: u32, y: u32, size: u32, seed: u32) -> f32 {
    let m = span / size - 1u;
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

fn scale_of(want: u32) -> u32 { return min(want, span / 2u); }

// The same noise at any point, not just on squares, so coordinates can be bent.
fn fnoise(x: f32, y: f32, size: u32, seed: u32) -> f32 {
    let m = span / size - 1u;
    let fs = f32(size);
    let w = f32(span);
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
        if (size > span / 2u) { break; }
        sum += (bent(x, y, size, seed ^ (300u + k * 7u)) * 2.0 - 1.0) * weight;
        total += weight * weight;
        weight *= 1.3;
        size *= 2u;
        k += 1u;
    }
    return exp(TERRAIN_CONTRAST * sum / sqrt(max(total, 1e-6)));
}

// Ridges of mixed size and thickness, instead of one size of wall everywhere.
fn scale_free_rock(x: f32, y: f32, seed: u32) -> bool {
    var size = 16u;
    var k = 0u;
    loop {
        if (size > span / 2u) { break; }
        // Thin at every size, and only in some places, so regions vary from open to walled in.
        let ridge = bent(x, y, size, seed ^ (400u + k * 11u));
        let where_ = bent(x, y, min(size * 4u, span / 2u), seed ^ (500u + k * 13u));
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
