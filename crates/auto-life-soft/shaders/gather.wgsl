// Copies what neighbours can see of each cell into its bin, so the contact pass reads memory in
// the order of the world, not of the slots. Empties the bins for the next tick as it goes.

@compute @workgroup_size(32, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bx = id.x;
    if (bx >= BINS) { return; }
    let b = (((id.y + F.row0) << LOG_B) | bx) * BIN_CAP;
    for (var k = 0u; k < BIN_CAP; k++) {
        let v = bins[b + k];
        if (v == 0u) {
            bb[b + k].slot = 0u;
            return;
        }
        bins[b + k] = 0u;
        let s = v - 1u;
        var o: BinBody;
        o.slot = v;
        o.x = cell[s].x;
        o.y = cell[s].y;
        o.r = cell[s].r;
        o.hd = pack2x16snorm(vec2f(cell[s].hx, cell[s].hy));
        o.stick = cell[s].stick;
        o.push = cell[s].push;
        o.gv = cell[s].gv;
        bb[b + k] = o;
    }
}
