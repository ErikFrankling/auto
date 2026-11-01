// One cell's tick: sense, think, spend. Each dispatch handles a 4x4 lattice of
// squares, so cells running at the same time are at least 4 squares apart. A cell
// may write up to 1 square away and read up to 2 with no locks, and the result
// doesn't depend on timing.
//
// Cells are never told who is related to them. They see bodies, energy and what
// neighbours choose to display, smell what others chose to emit, and that's all.
// Fields: fa = ground, meat, heat, noise. fb = four scents with no built-in
// meaning. fc = enzyme, then three channels free for experiments.

const EN0 = vec4f(1.0, 1.0, 1.0, 1.0);
const EN1 = vec4f(1.0, PHEROMONES_ON_F, HEAT_ON_F, BITE_ON_F);
const EN2 = vec4f(GIVE_ON_F, ENZYME_ON_F, X0_ON_F, X1_ON_F);

fn bt(i: u32, v: vec4f) -> vec4f {
    let g = genome[i];
    return vec4f(
        dot(unpack4x8snorm(g.x), v),
        dot(unpack4x8snorm(g.y), v),
        dot(unpack4x8snorm(g.z), v),
        dot(unpack4x8snorm(g.w), v),
    ) * 2.0;
}

// Only for measuring, never sensed: how far apart two genomes are.
fn genome_gap(a: u32, b: u32) -> f32 {
    let x = sketch[a];
    let y = sketch[b];
    let d0 = unpack4x8snorm(x.x) - unpack4x8snorm(y.x);
    let d1 = unpack4x8snorm(x.y) - unpack4x8snorm(y.y);
    let d2 = unpack4x8snorm(x.z) - unpack4x8snorm(y.z);
    let d3 = unpack4x8snorm(x.w) - unpack4x8snorm(y.w);
    return sqrt((dot(d0, d0) + dot(d1, d1) + dot(d2, d2) + dot(d3, d3)) / 16.0) * 4.0;
}

@compute @workgroup_size(32, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let lx = id.x * 4u + F.px;
    if (lx >= W) { return; }
    let ly = (id.y + F.row0) * 4u + F.py;
    let p0 = (ly << LOG_W) | lx;
    let o = occ[p0];
    if (o == 0u || o == ROCK) { return; }
    let s = o - 1u;
    var c = cell[s];
    // Acts once a tick, even if it moved onto a square that runs later. A cell born
    // this tick has this stamp too, and its genome is written when the tick ends.
    if (c.stamp == F.tick) { return; }
    c.stamp = F.tick;

    var r = pcg(p0 ^ (F.tick * 2654435761u) ^ (F.seed * 40503u));
    r = pcg(r);
    let old = unit_f(r) < HAZARD;
    if (c.e <= 0.0 || old) {
        var cause = (c.flags >> 8u) & 3u;
        if (c.e > 0.0) { cause = 3u; }
        atomicAdd(&ctr[C_STARVED + cause], 1u);
        fa[p0].y += max(c.e, 0.0) + c.womb + BODY;
        occ[p0] = 0u;
        seen_next[p0] = 0u;
        cell[s].e = -1.0;
        cell[s].pos = NONE;
        let h = atomicAdd(&ctr[C_HEAD], 1u);
        ring[h & RING_MASK] = s;
        return;
    }

    c.age += 1.0;
    let here_a = fa[p0];
    let burn = min(here_a.z * HEAT_BURN, c.e);
    c.e -= burn;
    lose(burn);
    let felt = c.hit;
    c.hit = burn;
    if (c.e <= 0.0) {
        c.flags = (c.flags & ~0x300u) | (2u << 8u);
        cell[s] = c;
        return;
    }

    // ---- Sense ----
    let here_b = fb[p0];
    let here_c = fc[p0];
    let pe = sq(lx, ly, 1, 0);
    let pw = sq(lx, ly, -1, 0);
    let ps = sq(lx, ly, 0, 1);
    let pn = sq(lx, ly, 0, -1);
    let gxa = contrast4(fa[pe], fa[pw]);
    let gya = contrast4(fa[ps], fa[pn]);
    let gxb = contrast4(fb[pe], fb[pw]);
    let gyb = contrast4(fb[ps], fb[pn]);
    let gxc = contrast4(fc[pe], fc[pw]);
    let gyc = contrast4(fc[ps], fc[pn]);

    let me8 = sat(c.e, 10.0);
    var count = 0.0;
    var said = vec2f(0.0);
    var bigger = vec2f(0.0);
    var smaller = vec2f(0.0);
    var said0 = vec2f(0.0);
    var said1 = vec2f(0.0);
    // Sight: bodies at distance 1, 2 and 4 in eight directions, from last tick's snapshot.
    for (var d = 0u; d < 8u; d++) {
        let dx = dx8(d);
        let dy = dy8(d);
        let u = unit8(d);
        for (var k = 0u; k < u32(VISION_STEPS); k++) {
            let reach = i32(1u << k);
            let w = seen[sq(lx, ly, dx * reach, dy * reach)];
            if ((w & 0x80u) == 0u) { continue; }
            let close = 1.0 / f32(reach);
            let their = f32(w >> 24u) / 255.0;
            let v0 = f32((w >> 16u) & 255u) / 127.5 - 1.0;
            let v1 = f32((w >> 8u) & 255u) / 127.5 - 1.0;
            if (k == 0u) {
                count += 1.0;
                said += vec2f(v0, v1);
            }
            if (their > me8 + 0.02) {
                bigger += u * close;
            } else if (their < me8 - 0.02) {
                smaller += u * close;
            }
            said0 += u * close * v0;
            said1 += u * close * v1;
        }
    }
    let old_dir = c.flags & 7u;
    let w_front = seen[sq(lx, ly, dx8(old_dir), dy8(old_dir))];
    var front_kind = 0.0;
    var front_gap = 0.0;
    var front_v0 = 0.0;
    var front_v1 = 0.0;
    var front_faces = 0.0;
    if (w_front == 1u) {
        front_kind = -1.0;
    } else if ((w_front & 0x80u) != 0u) {
        front_kind = 1.0;
        let their = f32(w_front >> 24u) / 255.0;
        front_gap = (their - me8) / (their + me8 + 0.05);
        front_v0 = f32((w_front >> 16u) & 255u) / 127.5 - 1.0;
        front_v1 = f32((w_front >> 8u) & 255u) / 127.5 - 1.0;
        if (((w_front & 7u) + 4u) % 8u == old_dir) { front_faces = 1.0; }
    }
    let heard = said / max(count, 1.0);

    r = pcg(r);
    let i0 = vec4f(me8, c.womb / WOMB, sat(c.age, 1000.0), sat(felt, 0.5));
    let i1 = vec4f(unit_f(r) * 2.0 - 1.0, 1.0, count / 8.0, c.x0);
    let i2 = sat4(here_a, 2.0);
    let i3 = sat4(here_b, 2.0);
    let i4 = gxa;
    let i5 = gya;
    let i6 = gxb;
    let i7 = gyb;
    let i8 = vec4f(heard, front_kind, front_gap);
    let i9 = vec4f(front_v0, front_v1, front_faces, sat(here_c.x, 2.0));
    let i10 = clamp(vec4f(bigger, smaller) * 0.5, vec4f(-1.0), vec4f(1.0));
    let i11 = clamp(vec4f(said0, said1) * 0.5, vec4f(-1.0), vec4f(1.0));
    let i12 = vec4f(sat(here_c.y, 2.0), gxc.y, gyc.y, c.x1);

    // ---- Think ----
    //@BRAIN

    // ---- Spend ----
    // o0 rest move ground meat, o1 grow emit heat bite, o2 give enzyme x0 x1 are the budget.
    // o3 dir dir voice voice, o4 scent mix, o5 mate share y0 y1.
    let top = max(maxv(o0), max(maxv(o1), maxv(o2)));
    var s0 = exp(clamp(o0 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN0;
    var s1 = exp(clamp(o1 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN1;
    var s2 = exp(clamp(o2 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN2;
    let z = max(dot(s0, vec4f(1.0)) + dot(s1, vec4f(1.0)) + dot(s2, vec4f(1.0)), 1e-9);
    s0 /= z;
    s1 /= z;
    s2 /= z;

    // Resting is the only way to spend less, and it nearly stops the clock.
    let rest = s0.x;
    let upkeep = (UPKEEP + UPKEEP_E * c.e) * (1.0 - REST_SAVING * rest);
    let pay = min(upkeep, c.e);
    c.e -= pay;
    lose(pay);
    let want = SPEND * (1.0 - rest);
    let scale = select(1.0, c.e / max(want, 1e-9), want > c.e);
    let a0 = s0 * SPEND * scale;
    let a1 = s1 * SPEND * scale;
    let a2 = s2 * SPEND * scale;
    // Spending a little on something barely works, so dabbling in everything is a waste.
    let w0 = a0 * a0 / (a0 + vec4f(COMMITMENT));
    let w1 = a1 * a1 / (a1 + vec4f(COMMITMENT));
    let w2 = a2 * a2 / (a2 + vec4f(COMMITMENT));

    let boost = 1.0 + min(here_c.x, ENZYME_CAP);
    let ate_ground = here_a.x * min(w0.z * GROUND_RATE * boost, 1.0);
    let ate_meat = here_a.y * min(w0.w * MEAT_RATE, 1.0);
    fa[p0].x = here_a.x - ate_ground;
    fa[p0].y = here_a.y - ate_meat;
    let meat_gain = ate_meat * MEAT_YIELD;
    lose(ate_meat - meat_gain);
    let spent = want * scale;
    let grow = select(0.0, a1.x, c.womb < WOMB);
    c.e += ate_ground + meat_gain - spent + (a1.x - grow);
    c.womb += grow;
    lose(max(spent - a1.x, 0.0));

    if (PHEROMONES_ON && a1.y > 0.0) {
        let m = exp(clamp(o4 - vec4f(maxv(o4)), vec4f(-20.0), vec4f(0.0)));
        fb[p0] = here_b + m / dot(m, vec4f(1.0)) * a1.y * EMIT_POWER;
    }
    if (ENZYME_ON) { fc[p0].x = here_c.x + w2.y * ENZYME_POWER; }
    if (HEAT_ON && a1.z > 0.0) {
        // Heat lands on the four squares around the emitter, so it hurts its neighbours, not itself.
        let flame = w1.z * HEAT_POWER / 4.0;
        if (occ[pe] != ROCK) { fa[pe].z += flame; }
        if (occ[pw] != ROCK) { fa[pw].z += flame; }
        if (occ[ps] != ROCK) { fa[ps].z += flame; }
        if (occ[pn] != ROCK) { fa[pn].z += flame; }
    }
    var voice = vec2f(0.0);
    if (VOICE_ON) { voice = squash(vec4f(o3.z, o3.w, 0.0, 0.0)).xy; }

    let d = u32(i32(round(atan2(o3.y, o3.x) / 0.78539816)) + 8) & 7u;
    var flags = d;
    if (SEX_ON && o5.x > 0.0) { flags |= F_MATE; }
    if (SHARE_ON && o5.y > 0.0) { flags |= F_SHARE; }

    let front = sq(lx, ly, dx8(d), dy8(d));
    let tq = occ[front];
    var bite_take = 0.0;
    var bite_who = 0u;
    var gain_bite = 0.0;
    if (tq != 0u && tq != ROCK) {
        let t = tq - 1u;
        var te = cell[t].e;
        if (te > 0.0) {
            if (BITE_ON && a1.w > 0.0) {
                // The stronger cell bites harder, so stored energy is also power.
                let force = 2.0 * c.e / (c.e + te + 1e-3);
                var open = 1.0;
                if (EXPOSURE) {
                    // Only open sides can be reached, so a cell inside a clump is protected.
                    var sides = 1.0;
                    let tx = front & MASK;
                    let ty = front >> LOG_W;
                    for (var k = 0u; k < 8u; k++) {
                        if (occ[sq(tx, ty, dx8(k), dy8(k))] == 0u) { sides += 1.0; }
                    }
                    open = sides / 8.0;
                }
                let take = min(w1.w * BITE_POWER * force * open, te);
                let keep = take * BITE_YIELD;
                let spill = min(take - keep, keep);
                te -= take;
                cell[t].hit = max(cell[t].hit, take);
                c.e += keep;
                fa[front].y += spill;
                lose(take - keep - spill);
                gain_bite = keep;
                bite_take = take;
                bite_who = t;
                if (take > 0.0) {
                    flags |= F_BIT;
                    if (NOISE_ON) { fa[front].w += NOISE_POWER; }
                }
                if (te <= 0.0) { cell[t].flags = (cell[t].flags & ~0x300u) | (1u << 8u); }
            }
            if (GIVE_ON && a2.x > 0.0 && c.e > 0.0 && te > 0.0) {
                let amount = min(w2.x * GIVE_POWER, c.e);
                c.e -= amount;
                te += amount * GIVE_YIELD;
                lose(amount * (1.0 - GIVE_YIELD));
                stat[t].diet.w += amount * GIVE_YIELD * 0.01;
            }
            if (SHARE_ON && o5.y > 0.0 && (cell[t].flags & F_SHARE) != 0u && te > 0.0) {
                // Two cells that both agree level out their energy.
                let flow = (c.e - te) * SHARE_RATE;
                c.e -= flow;
                te += flow;
            }
            cell[t].e = te;
        }
    }

    if (c.e > MAX_ENERGY) {
        lose(c.e - MAX_ENERGY);
        c.e = MAX_ENERGY;
    }

    // ---- Birth ----
    if (c.e > 0.0 && c.womb >= WOMB) {
        r = pcg(r);
        let first = r & 7u;
        var free_q = NONE;
        for (var k = 0u; k < 8u; k++) {
            let q = sq(lx, ly, dx8((first + k) & 7u), dy8((first + k) & 7u));
            if (occ[q] == 0u) {
                free_q = q;
                break;
            }
        }
        if (free_q != NONE) {
            let tk = atomicAdd(&ctr[C_TAIL], 1u);
            if (i32(tk - ctl[0]) < 0) {
                let child = ring[tk & RING_MASK];
                // A willing neighbour can be the other parent. Any willing neighbour:
                // who mates with whom is up to their brains.
                var mate = 0u;
                if (SEX_ON && (flags & F_MATE) != 0u) {
                    for (var k = 0u; k < 8u; k++) {
                        let j = occ[sq(lx, ly, dx8((first + k) & 7u), dy8((first + k) & 7u))];
                        if (j == 0u || j == ROCK) { continue; }
                        let other = j - 1u;
                        if (cell[other].e > 0.0 && (cell[other].flags & F_MATE) != 0u
                            && (COMPAT <= 0.0 || genome_gap(s, other) < COMPAT)) {
                            mate = j;
                            break;
                        }
                    }
                }
                var kid: Cell;
                kid.e = c.womb - BODY;
                kid.stamp = F.tick;
                kid.pos = free_q;
                r = pcg(r);
                kid.flags = r & 7u;
                kid.clade = c.clade;
                kid.gen = c.gen + 1u;
                kid.x0 = c.x0;
                kid.x1 = c.x1;
                cell[child] = kid;
                //@MEMCOPY
                occ[free_q] = child + 1u;
                seen_next[free_q] = 0x80u | (r & 7u);
                let j = atomicAdd(&ctr[C_JOBS], 1u);
                r = pcg(r);
                jobs[j] = vec4u(child, s, mate, r);
                c.womb = 0.0;
                atomicAdd(&ctr[C_BIRTHS], 1u);
                if (mate != 0u) { atomicAdd(&ctr[C_MIXED], 1u); }
            }
        }
    }

    // ---- Move ----
    // Last, so nothing above reaches more than one square from where this tick began.
    var p = p0;
    r = pcg(r);
    if (c.e > 0.0 && unit_f(r) < w0.y * MOVE_RATE && occ[front] == 0u) {
        occ[front] = o;
        occ[p] = 0u;
        seen_next[p] = 0u;
        p = front;
        c.pos = p;
        flags |= F_MOVED;
        // Moving makes noise anyone can hear. Heavier bodies are louder.
        if (NOISE_ON) { fa[p].w += NOISE_POWER * (0.5 + me8); }
    }

    // ---- Finish ----
    c.voice = voice;
    c.flags = flags;
    cell[s] = c;
    let e8 = u32(sat(max(c.e, 0.0), 10.0) * 255.0);
    let v8 = u32((voice.x + 1.0) * 127.5);
    let u8_ = u32((voice.y + 1.0) * 127.5);
    let moved_bit = (flags >> 5u) & 3u;
    seen_next[p] = (e8 << 24u) | (v8 << 16u) | (u8_ << 8u) | 0x80u | (moved_bit << 3u) | d;

    // Measuring only. One tick in eight per cell, weighted to match.
    if (((F.tick + s) & 7u) == 0u) {
        const K = 0.08;
        var st = stat[s];
        st.diet = st.diet * (1.0 - K) + vec4f(ate_ground, meat_gain, gain_bite, 0.0) * K;
        st.b0 += (s0 - st.b0) * K;
        st.b1 += (s1 - st.b1) * K;
        st.b2 += (s2 - st.b2) * K;
        let moved = (flags & F_MOVED) != 0u;
        st.chase.w += (f32(moved) - st.chase.w) * K;
        if (moved) {
            let u = unit8(d);
            let lb = length(bigger);
            let ls = length(smaller);
            let food = vec2f(gxa.x, gya.x);
            let lf = length(food);
            if (lb > 0.0) { st.chase.x += (dot(u, bigger) / lb - st.chase.x) * K; }
            if (ls > 0.0) { st.chase.y += (dot(u, smaller) / ls - st.chase.y) * K; }
            if (lf > 1e-3) { st.chase.z += (dot(u, food) / lf - st.chase.z) * K; }
            let along = gxb * u.x + gyb * u.y;
            st.follow += (along - st.follow) * K;
        }
        var kin_bite = 0.0;
        var other_bite = 0.0;
        if (bite_take > 0.0) {
            if (genome_gap(s, bite_who) < REPORT_TAU) { kin_bite = bite_take; } else { other_bite = bite_take; }
        }
        st.misc += (vec4f(count / 8.0, kin_bite, other_bite, f32(front_kind > 0.0)) - st.misc) * K;
        stat[s] = st;
    }
}
