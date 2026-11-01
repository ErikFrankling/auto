// Turns each cell's genome into a colour, using the main axes along which the
// population's genomes differ. Similar weights get similar colours.

struct Bake { m: array<vec4f, 12>, mean: array<vec4f, 4>, gain: vec4f }
@group(1) @binding(0) var<uniform> B: Bake;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x + id.y * 65536u;
    if (s >= SLOTS) { return; }
    if (cell[s].e <= 0.0 && cell[s].pos == NONE) { return; }
    let k = sketch[s];
    let v0 = unpack4x8snorm(k.x) - B.mean[0];
    let v1 = unpack4x8snorm(k.y) - B.mean[1];
    let v2 = unpack4x8snorm(k.z) - B.mean[2];
    let v3 = unpack4x8snorm(k.w) - B.mean[3];
    let pc = vec3f(
        dot(B.m[0], v0) + dot(B.m[1], v1) + dot(B.m[2], v2) + dot(B.m[3], v3),
        dot(B.m[4], v0) + dot(B.m[5], v1) + dot(B.m[6], v2) + dot(B.m[7], v3),
        dot(B.m[8], v0) + dot(B.m[9], v1) + dot(B.m[10], v2) + dot(B.m[11], v3),
    ) * B.gain.xyz;
    let t = pc / (vec3f(1.0) + abs(pc));
    cell[s].rgb = pack4x8unorm(vec4f(vec3f(0.5) + 0.5 * t, 1.0));
}
