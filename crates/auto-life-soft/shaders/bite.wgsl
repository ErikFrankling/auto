// Bites, settled as supply and demand. Every biter has added what it wants to its target's word.
// A target can lose at most what it has left, so each biter gets its wish scaled by
// min(1, what's left / total asked). The target works out the same total in settle.wgsl. Some of
// what's taken goes to the biter and the rest is shed.

fn route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x + F.row0;
    if (s >= SLOTS) { return; }
    let t1 = acc[s].target_;
    if (t1 == 0u) { return; }
    let t = t1 - 1u;
    let asked = mail[t * 2u];
    let left = cell[t].e - acc[t].give_out;
    // A hair under 1, so rounding can't hand out more than the target pays.
    let share = min(1.0, f32(left) / max(f32(asked), 1.0)) * 0.999999;
    let take = u32(f32(acc[s].wish) * share);
    let got = u32(f32(take) * BITE_YIELD);
    acc[s].bite_got = got;
    if (LEDGER || ((s + F.tick) & 63u) == 0u) { route(R_BITE_GOT, got); }
}
