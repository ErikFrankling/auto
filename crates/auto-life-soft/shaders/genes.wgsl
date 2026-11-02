// Writes the genomes of this tick's new cells. A separate pass, because a birth touches every
// weight and most cells aren't giving birth. The same mutation and colour projection as the
// lattice engine, except that each cell carries its own mutation rate.

fn gauss(r: ptr<function, u32>) -> f32 {
    *r = pcg(*r);
    let u = max(unit_f(*r), 1e-7);
    *r = pcg(*r);
    let v = unit_f(*r);
    return sqrt(-2.0 * log(u)) * cos(6.2831853 * v);
}

fn mutate_word(w: u32, rate: f32, r: ptr<function, u32>) -> u32 {
    var out = 0u;
    for (var k = 0u; k < 4u; k++) {
        var v = extractBits(i32(w), k * 8u, 8u);
        *r = pcg(*r);
        if (unit_f(*r) < rate) {
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

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= ctl[L_BIRTHS]) { return; }
    let job = jobs[id.x];
    let child = job.x;
    let parent = job.y;
    var r = job.w;
    r = pcg(r);
    let rate = cell[child].mu;
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
            g = vec4u(mutate_word(g.x, rate, &r), mutate_word(g.y, rate, &r), mutate_word(g.z, rate, &r), mutate_word(g.w, rate, &r));
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
