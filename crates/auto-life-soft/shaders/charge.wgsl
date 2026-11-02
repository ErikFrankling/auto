// The start of a tick, for every living cell: pay upkeep, pay this tick's budget, work out what
// that buys, and join a bin. Only touches the cell's own record and the bins, so everything that
// reads energy later in the tick sees it already paid.
//
// A bin keeps its cells sorted. Each insert carries a value down the bin with atomicMax: each place
// keeps the larger of what it holds and what arrives, and passes the other on. Whatever order the
// threads run in, the bin ends up with its 8 highest slots, sorted, so a full bin drops the same
// cells every time and a run stays the same.

fn route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let s = id.x + F.row0;
    if (s >= SLOTS) { return; }
    var c = cell[s];
    if ((c.flags & F_ALIVE) == 0u) { return; }
    let metered = LEDGER || ((s + F.tick) & 63u) == 0u;

    let pl = plan[s];
    let rest = pl.s0.x;
    let e0 = f32(c.e) * E_INV;
    let upkeep = (UPKEEP + UPKEEP_E * e0) * (1.0 - REST_SAVING * rest);
    let pay = min(u32(upkeep * E_ONE + 0.5), c.e);
    c.e -= pay;
    // A share of what it holds, up to a limit. Resting is the only way to spend less.
    let budget = min(SPEND, SPEND_SHARE * e0);
    let want = budget * (1.0 - rest);
    let spent = min(u32(want * E_ONE), c.e);
    // Only scaled down when it can't afford it all, so a cell that can gets exactly its budget.
    let want_u = u32(want * E_ONE);
    let unit = budget * select(1.0, f32(spent) / max(f32(want_u), 1.0), spent < want_u);
    c.unit = unit;
    // What goes to growing fills the womb, and comes back if the womb is full.
    var grow = min(u32(pl.s1.x * unit * E_ONE), spent);
    var back = 0u;
    if (c.womb + grow > u32(WOMB_CAP * E_ONE)) {
        back = grow;
        grow = 0u;
    }
    c.e -= spent - back;
    c.womb += grow;
    let shed = spent - back - grow;
    c.shed = pay + shed;
    if (metered) {
        route(R_UPKEEP, pay);
        route(R_SPEND, shed);
        route(R_GROW, grow);
    }

    // What the others will feel of it this tick.
    let e1 = f32(c.e) * E_INV;
    c.r = min(R0 * sqrt(1.0 + e1 / E_SIZE), R_MAX);
    var stick = 0.0;
    if (STICK_ON) {
        let w = effort(pl.s2.z * unit);
        stick = w / (w + STICK_HALF);
    }
    c.stick = stick;
    c.push = select(0.0, effort(pl.s2.w * unit) * PUSH_POWER, PUSH_ON);
    c.age += 1u;
    cell[s] = c;

    let bx = c.x >> (POS_SHIFT + 1u);
    let by = c.y >> (POS_SHIFT + 1u);
    let b = ((by << LOG_B) | bx) * BIN_CAP;
    var v = s + 1u;
    for (var k = 0u; k < BIN_CAP; k++) {
        let old = atomicMax(&bins[b + k], v);
        if (old == 0u) {
            v = 0u;
            break;
        }
        v = min(old, v);
    }
    if (v != 0u) { atomicAdd(&ctr[C_OVERFLOW], 1u); }
}
