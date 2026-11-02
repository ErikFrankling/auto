// Adds up where the energy is, exactly, in 64-bit counters. Only runs at log points. Each
// invocation adds up a stretch in a plain u32, which can't overflow at these lengths, then touches
// the shared total once.

fn wide(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&tot[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&tot[k * 2u + 1u], 1u); }
}

// One row of squares at a time, flushed every 256 squares, so a plain u32 is safe unless 256
// squares in a row average over 256 energy.
@compute @workgroup_size(8)
fn squares(@builtin(global_invocation_id) id: vec3u) {
    let y = id.x + F.row0;
    if (y >= W) { return; }
    var x = 0u;
    loop {
        if (x >= W) { break; }
        var food = 0u;
        var meat = 0u;
        var heat = 0u;
        var enzyme = 0u;
        for (var k = x; k < min(x + 256u, W); k++) {
            let v = fa0[(y << LOG_W) | k];
            food += u32(v.x);
            meat += u32(v.y);
            heat += u32(v.z);
            enzyme += u32(v.w);
        }
        wide(0u, food);
        wide(1u, meat);
        wide(2u, heat);
        wide(3u, enzyme);
        x += 256u;
    }
}

// 256 slots at a time.
@compute @workgroup_size(8)
fn slots(@builtin(global_invocation_id) id: vec3u) {
    let first = (id.x + F.row0) * 256u;
    if (first >= SLOTS) { return; }
    var e = 0u;
    var womb = 0u;
    var alive = 0u;
    for (var s = first; s < min(first + 256u, SLOTS); s++) {
        if ((cell[s].flags & F_ALIVE) == 0u) { continue; }
        e += cell[s].e;
        womb += cell[s].womb;
        alive += 1u;
    }
    wide(4u, e);
    wide(5u, womb);
    wide(6u, alive);
}
