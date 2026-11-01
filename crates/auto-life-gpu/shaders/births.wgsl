// Writes the genomes of this tick's new cells. A separate pass, because a birth
// touches every weight and most cells aren't giving birth.

fn gauss(r: ptr<function, u32>) -> f32 {
    *r = pcg(*r);
    let u = max(unit_f(*r), 1e-7);
    *r = pcg(*r);
    let v = unit_f(*r);
    return sqrt(-2.0 * log(u)) * cos(6.2831853 * v);
}

fn mutate_word(w: u32, r: ptr<function, u32>) -> u32 {
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        var v = extractBits(i32(w), k * 8u, 8u);
        *r = pcg(*r);
        if (unit_f(*r) < MUT_RATE) {
            // One byte step is 2/127 of a weight.
            v = clamp(v + i32(round(gauss(r) * MUT_SIZE * 63.5)), -127, 127);
        }
        out |= (u32(v) & 255u) << (k * 8u);
    }
    return out;
}

fn random_word(r: ptr<function, u32>) -> u32 {
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        let v = clamp(i32(round(gauss(r) * GENE_INIT * 63.5)), -127, 127);
        out |= (u32(v) & 255u) << (k * 8u);
    }
    return out;
}

// Keeps the bytes of a word whose hidden unit comes from the mate.
fn byte_mask(bits: u32) -> u32 {
    var m = 0u;
    for (var k = 0u; k < 4u; k++) {
        if (((bits >> k) & 1u) == 1u) { m |= 255u << (k * 8u); }
    }
    return m;
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ctl[2]) { return; }
    let job = jobs[id.x];
    let child = job.x;
    let parent = job.y;
    let mate = job.z;
    var r = job.w;
    r = pcg(r);
    // Whole hidden units come from one parent or the other, so a unit's wiring stays together.
    let units = select(0u, r, mate != 0u);
    var s0 = vec4f(0.0);
    var s1 = vec4f(0.0);
    var s2 = vec4f(0.0);
    var s3 = vec4f(0.0);
    for (var b = 0u; b < BLOCKS; b++) {
        var g: vec4u;
        if (parent == NONE) {
            g = vec4u(random_word(&r), random_word(&r), random_word(&r), random_word(&r));
        } else {
            g = gene(parent * BLOCKS + b);
            if (mate != 0u) {
                let h = gene((mate - 1u) * BLOCKS + b);
                if (b < HO_BASE) {
                    let row = select((b - HH_BASE) / HR, b / IN_BLOCKS, b < HH_BASE);
                    let bits = (units >> (row * 4u)) & 15u;
                    if ((bits & 1u) != 0u) { g.x = h.x; }
                    if ((bits & 2u) != 0u) { g.y = h.y; }
                    if ((bits & 4u) != 0u) { g.z = h.z; }
                    if ((bits & 8u) != 0u) { g.w = h.w; }
                } else {
                    let col = (b - HO_BASE) % HR;
                    let m = byte_mask((units >> (col * 4u)) & 15u);
                    g = (g & vec4u(~m)) | (h & vec4u(m));
                }
            }
            g = vec4u(mutate_word(g.x, &r), mutate_word(g.y, &r), mutate_word(g.z, &r), mutate_word(g.w, &r));
        }
        set_gene(child * BLOCKS + b, g);
        // A fixed random projection of the weights, used for colour and for measuring similarity.
        for (var w = 0u; w < 4u; w++) {
            let h = pcg(b * 4u + w + 12345u);
            let signs = vec4f(f32(h & 1u), f32((h >> 1u) & 1u), f32((h >> 2u) & 1u), f32((h >> 3u) & 1u)) * 2.0 - vec4f(1.0);
            let v = unpack4x8snorm(g[w]) * signs;
            switch ((h >> 8u) & 3u) {
                case 0u: { s0 += v; }
                case 1u: { s1 += v; }
                case 2u: { s2 += v; }
                default: { s3 += v; }
            }
        }
    }
    // Scaled so one unit is one unit of rms weight difference, stored in +-4.
    let k = 2.0 / sqrt(f32(BLOCKS)) / 4.0;
    sketch[child] = vec4u(pack4x8snorm(s0 * k), pack4x8snorm(s1 * k), pack4x8snorm(s2 * k), pack4x8snorm(s3 * k));
}
