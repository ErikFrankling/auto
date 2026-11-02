// Draws part of the world into a panel of the output image.

struct ViewU { origin: vec2f, scale: f32, samples: u32, out: vec2u, mode: u32, x0: u32 }
@group(1) @binding(0) var<uniform> V: ViewU;
@group(1) @binding(1) var out_tex: texture_storage_2d<rgba8unorm, write>;

fn color(x: u32, y: u32) -> vec3f {
    let p = (y << LOG_W) | x;
    let o = occ[p];
    if (o == ROCK) { return vec3f(0.23, 0.235, 0.27); }
    if (o != 0u) {
        let s = o - 1u;
        if (V.mode == 0u) {
            return unpack4x8unorm(cell[s].rgb).xyz;
        }
        let d = stat[s].diet;
        let t = max(d.x + d.y + d.z + d.w, 1e-3);
        return vec3f(0.16) + 0.84 * vec3f((d.y + d.z) / t, d.x / t, d.w / t);
    }
    let a = fa[p];
    let b = fb[p];
    let g = min(a.x / GROUND_CAP, 1.0);
    let m = min(a.y / 3.0, 1.0);
    let heat = min(a.z / 2.0, 1.0);
    let scent = min((b.x + b.y + b.z + b.w) / 4.0, 1.0);
    var ground = vec3f(0.04 + m * 0.27 + heat * 0.78, 0.047 + g * 0.15 + heat * 0.35, 0.063 + scent * 0.17 + g * 0.03);
    if (BLOOM_SHARE > 0.0) { ground += vec3f(0.8, 0.7, 0.0) * min(max(a.x - GROUND_CAP, 0.0) / 20.0, 1.0); }
    return ground;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let py = id.y + F.row0;
    if (id.x >= V.out.x || py >= V.out.y) { return; }
    var acc = vec3f(0.0);
    let n = V.samples;
    for (var sy = 0u; sy < n; sy++) {
        for (var sx = 0u; sx < n; sx++) {
            let sub = (vec2f(f32(sx), f32(sy)) + 0.5) / f32(n);
            let q = V.origin + (vec2f(f32(id.x), f32(py)) + sub) * V.scale;
            // Average in light rather than in gamma, or sparse bright cells go dark when zoomed out.
            let c = color(u32(i32(floor(q.x))) & MASK, u32(i32(floor(q.y))) & MASK);
            acc += c * c;
        }
    }
    textureStore(out_tex, vec2u(id.x + V.x0, py), vec4f(sqrt(acc / f32(n * n)), 1.0));
}
