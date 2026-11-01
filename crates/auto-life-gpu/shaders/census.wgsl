// Sums each tile of the world for the report and for choosing where to zoom.

@compute @workgroup_size(16, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= TILES) { return; }
    let ty = id.y + F.row0;
    var a = vec4f(0.0);
    var c = vec4f(0.0);
    var c2 = vec4f(0.0);
    for (var y = 0u; y < TILE; y++) {
        for (var x = 0u; x < TILE; x++) {
            let o = occ[((ty * TILE + y) << LOG_W) | (id.x * TILE + x)];
            if (o == 0u || o == ROCK) { continue; }
            let s = o - 1u;
            let d = stat[s].diet;
            let total = max(d.x + d.y + d.z + d.w, 1e-6);
            let rgb = unpack4x8unorm(cell[s].rgb);
            a += vec4f(1.0, cell[s].e, (d.y + d.z) / total, stat[s].chase.w);
            c += rgb;
            c2 += rgb * rgb;
        }
    }
    tiles[(ty * TILES + id.x) * 3u] = a;
    tiles[(ty * TILES + id.x) * 3u + 1u] = c;
    tiles[(ty * TILES + id.x) * 3u + 2u] = c2;
}
