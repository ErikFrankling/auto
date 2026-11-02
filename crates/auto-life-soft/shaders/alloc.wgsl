// Hands out slots in a fixed order, so which slot a new cell gets never depends on which thread ran
// first. Slot numbers decide when a cell thinks and which cell a full bin drops, so a world can only
// repeat exactly if they do.
//
// Everyone wanting a slot, or giving one back, has set a bit. A cell's place in the queue is how
// many bits come before its own. Counting that for every cell would be slow, so words are grouped
// in chunks of 64: one pass counts each chunk, one thread adds up the chunks, and a cell only has
// to look at the words of its own chunk.

fn set_route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

// How many set bits come before bit `bit` of word `w`.
fn rank_of(w: u32, bit: u32, word: u32) -> u32 {
    let c = w >> 6u;
    var n = chunks[c];
    for (var k = c << 6u; k < w; k++) { n += countOneBits(bits[k]); }
    return n + countOneBits(word & ((1u << bit) - 1u));
}

@compute @workgroup_size(16)
fn count(@builtin(global_invocation_id) id: vec3u) {
    let c = id.x;
    if (c >= CHUNKS) { return; }
    var n = 0u;
    for (var k = 0u; k < 64u; k++) { n += countOneBits(bits[(c << 6u) + k]); }
    chunks[c] = n;
}

@compute @workgroup_size(1)
fn prefix() {
    var claims = 0u;
    for (var c = 0u; c < (FREE_BASE >> 6u); c++) {
        let n = chunks[c];
        chunks[c] = claims;
        claims += n;
    }
    var frees = 0u;
    for (var c = (FREE_BASE >> 6u); c < CHUNKS; c++) {
        let n = chunks[c];
        chunks[c] = frees;
        frees += n;
    }
    let head = ctl[L_HEAD];
    let tail = ctl[L_TAIL];
    let births = min(claims, head + frees - tail);
    ctl[L_HEAD_NOW] = head;
    ctl[L_TAIL_NOW] = tail;
    ctl[L_BIRTHS] = births;
    ctl[L_HEAD] = head + frees;
    ctl[L_TAIL] = tail + births;
    ctl[L_ALIVE] += births;
    ctl[L_ALIVE] -= frees;
    ctl[L_UID_NOW] = ctl[L_NEXT_UID];
    ctl[L_NEXT_UID] += births;
    args[0] = births;
    args[1] = 1u;
    args[2] = 1u;
    atomicAdd(&ctr[C_NO_SLOT], claims - births);
}

@compute @workgroup_size(32)
fn give_back(@builtin(global_invocation_id) id: vec3u) {
    let w = FREE_BASE + id.x;
    if (w >= TOTAL_WORDS) { return; }
    var word = bits[w];
    if (word == 0u) { return; }
    let all = word;
    let base = ctl[L_HEAD_NOW];
    loop {
        if (word == 0u) { break; }
        let bit = firstTrailingBit(word);
        word &= word - 1u;
        let rank = rank_of(w, bit, all);
        ring[(base + rank) & RING_MASK] = id.x * 32u + bit;
    }
}

@compute @workgroup_size(32)
fn birth(@builtin(global_invocation_id) id: vec3u) {
    let w = id.x;
    if (w >= FREE_BASE) { return; }
    var word = bits[w];
    if (word == 0u) { return; }
    let all = word;
    let base = ctl[L_TAIL_NOW];
    let allowed = ctl[L_BIRTHS];
    loop {
        if (word == 0u) { break; }
        let bit = firstTrailingBit(word);
        word &= word - 1u;
        let rank = rank_of(w, bit, all);
        if (rank >= allowed) { continue; }
        let child = ring[(base + rank) & RING_MASK];
        let k = w * 32u + bit;
        var r = pcg(k ^ (F.tick * 2654435761u) ^ (F.seed * 40503u) ^ 0x9e3779b9u);
        r = pcg(r);
        var kid: Cell;
        kid.flags = F_ALIVE;
        kid.uid = ctl[L_UID_NOW] + rank;
        kid.born = F.tick;
        var parent = NONE;
        var pl: Plan;
        if (k < SLOTS) {
            // ---- A split ----
            parent = k;
            var c = cell[k];
            pl = plan[k];
            // Whatever is in the womb beyond the child's body, and the share the brain chose.
            let share = u32(f32(c.e) * pl.split.z * SPLIT_SHARE_MAX);
            c.e -= share;
            kid.e = c.womb - u32(BODY * E_ONE) + share;
            c.womb = 0u;
            cell[k] = c;
            let e_f = f32(kid.e) * E_INV;
            let kid_r = min(R0 * sqrt(1.0 + e_f / E_SIZE), R_MAX);
            let cs = pl.split.x;
            let sn = pl.split.y;
            kid.hx = c.hx * cs - c.hy * sn;
            kid.hy = c.hx * sn + c.hy * cs;
            kid.r = kid_r;
            // At contact distance, at the angle the brain chose. If that's rock, the other way.
            let off = (c.r + kid_r) * SQ_TO_POS;
            var nx = c.x + u32(i32(kid.hx * off));
            var ny = c.y + u32(i32(kid.hy * off));
            if ((terrain[square_of(nx, ny)] & T_ROCK) != 0u) {
                nx = c.x - u32(i32(kid.hx * off));
                ny = c.y - u32(i32(kid.hy * off));
            }
            if ((terrain[square_of(nx, ny)] & T_ROCK) != 0u) {
                // Hemmed in. It starts on the parent, and crowding sorts them out.
                nx = c.x + u32(i32(kid.hx * SQ_TO_POS * 0.01));
                ny = c.y + u32(i32(kid.hy * SQ_TO_POS * 0.01));
            }
            kid.x = nx;
            kid.y = ny;
            // A child starts with its parent's outputs, so it sticks to its parent with no
            // recognition rule at all.
            kid.gv = c.gv;
            kid.parent = c.uid;
            kid.gen = c.gen + 1u;
            kid.clade = c.clade;
            var mu = c.mu;
            if (MUT_HERITABLE) {
                r = pcg(r);
                let g = sqrt(-2.0 * log(max(unit_f(r), 1e-7)));
                r = pcg(r);
                mu = clamp(mu * exp(MUT_RATE_STEP * g * cos(6.2831853 * unit_f(r))), 1e-4, 0.5);
            }
            kid.mu = mu;
            for (var q = 0u; q < HR; q++) { mem[child * HR + q] = mem[k * HR + q] * MEMORY_ON_F; }
            atomicAdd(&ctr[C_BIRTHS], 1u);
        } else {
            // ---- Rain ----
            let tile = k - SLOTS;
            let tx = tile % TILES;
            let ty = tile / TILES;
            r = pcg(r);
            var x = (tx * TILE) << POS_SHIFT;
            x += r >> (LOG_W - LOG_TILE);
            r = pcg(r);
            var y = (ty * TILE) << POS_SHIFT;
            y += r >> (LOG_W - LOG_TILE);
            // Rock is rare enough that sliding along a row or two nearly always finds open ground.
            for (var q = 0u; q < 8u; q++) {
                if ((terrain[square_of(x, y)] & T_ROCK) == 0u) { break; }
                r = pcg(r);
                x = ((tx * TILE) << POS_SHIFT) + (r >> (LOG_W - LOG_TILE));
                r = pcg(r);
                y = ((ty * TILE) << POS_SHIFT) + (r >> (LOG_W - LOG_TILE));
            }
            kid.x = x;
            kid.y = y;
            r = pcg(r);
            let a = unit_f(r) * 6.2831853;
            kid.hx = cos(a);
            kid.hy = sin(a);
            kid.e = u32(RAIN_ENERGY * E_ONE);
            kid.r = min(R0 * sqrt(1.0 + RAIN_ENERGY / E_SIZE), R_MAX);
            kid.parent = NONE;
            kid.clade = kid.uid;
            kid.mu = MUT_RATE;
            // It rests until it first thinks.
            pl.s0 = vec4f(1.0, 0.0, 0.0, 0.0);
            pl.mix = vec4f(0.25);
            for (var q = 0u; q < HR; q++) { mem[child * HR + q] = vec4f(0.0); }
            atomicAdd(&ctr[C_RAINED], 1u);
            if (LEDGER || (F.tick & 63u) == 0u) {
                set_route(R_RAIN, u32(RAIN_ENERGY * E_ONE) + u32(BODY * E_ONE));
            }
        }
        cell[child] = kid;
        plan[child] = pl;
        var blank: Sense;
        sense[child] = blank;
        r = pcg(r);
        jobs[rank] = vec4u(child, parent, 0u, r);
    }
}
