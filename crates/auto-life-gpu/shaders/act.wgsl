// One cell's tick: sense, think, spend. Each dispatch handles a 4x4 lattice of
// squares, so cells running at the same time are at least 4 squares apart. A cell
// may write up to 1 square away and read up to 2 with no locks, and the result
// doesn't depend on timing.
//
// Cells are never told who is related to them. They see bodies, energy and what
// neighbours choose to display, smell what others chose to emit, and that's all.
// Fields: fa = ground, meat, heat, noise. fb = four scents with no built-in
// meaning. fc = enzyme, waste, fire (sweep.wgsl only), one free. fd = tracks, mineral on the
// ground, how rich the square is in mineral (never changes), steps taken here (measuring only).
//
// Inputs that belong to a rule read 0 while it's off, and in config.rs a block with
// nothing live isn't computed at all. research/life-gpu/RULES.md lists every rule.

const EN0 = vec4f(1.0, 1.0, 1.0, 1.0);
const EN1 = vec4f(1.0, PHEROMONES_ON_F, HEAT_ON_F, BITE_ON_F);
const EN2 = vec4f(GIVE_ON_F, ENZYME_ON_F, ARMOUR_ON_F, WASTE_ON_F * WASTE_CLEAN_F);
// Mineral uptake is the first spare action, so its switch turns that action on.
const EN3 = vec4f(max(X0_ON_F, MINERAL_ON_F), X1_ON_F, X2_ON_F, X3_ON_F);
const SPARE_ACTIONS = X0_ON || X1_ON || X2_ON || X3_ON || MINERAL_ON;
// How much a womb can hold. With birth_choice a cell decides when to empty it.
const WOMB_LIMIT = WOMB + (WOMB_CAP - WOMB) * BIRTH_CHOICE_F;
// Mineral a birth takes, in whole grains. Zero when the rule is off, and so is every store.
const MINERAL_NEED = round(MINERAL_BIRTH * MINERAL_ON_F * GRAIN) / GRAIN;
// The share of a meal's energy the eater keeps. The rest stays in its waste.
const MEAL_KEPT = 1.0 - WASTE_HOLDS * WASTE_ON_F;
// Energy that a unit of waste holds, and how much of it clearing brings back.
const WASTE_BACK = WASTE_HOLDS / max(WASTE_DEPOSIT, 1e-6) * WASTE_YIELD;

fn bt(i: u32, v: vec4f) -> vec4f {
    let g = gene(i);
    return vec4f(
        dot(unpack4x8snorm(g.x), v),
        dot(unpack4x8snorm(g.y), v),
        dot(unpack4x8snorm(g.z), v),
        dot(unpack4x8snorm(g.w), v),
    ) * 2.0;
}

// Energy from a snapshot word. The same squashing as when it was written, undone.
fn seen_energy(w: u32) -> f32 {
    let x = f32(w >> 24u) / 255.0;
    return min(10.0 * x / max(1.0 - x, 1e-3), MAX_ENERGY);
}

// Whether a snapshot word 8-neighbour `k` of a square shows a body that really bit that square.
fn bit_this_square(w: u32, k: u32) -> bool {
    return (w & (0x80u | SEEN_ATTACK)) == (0x80u | SEEN_ATTACK) && (w & 7u) == ((k + 4u) & 7u);
}

// Neighbours of a square that really bit it last tick, leaving out square `me`: how many,
// and their summed energy. Reads only the snapshot, so it's safe at any distance.
fn attackers(at: u32, me: u32) -> vec2f {
    var out = vec2f(0.0);
    let x = at & MASK;
    let y = at >> LOG_W;
    for (var k = 0u; k < 8u; k++) {
        let q = sq(x, y, dx8(k), dy8(k));
        let w = seen[q];
        if (q != me && bit_this_square(w, k)) {
            out += vec2f(1.0, seen_energy(w));
        }
    }
    return out;
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
    // The kind of place sets how fast life runs here and how hard it is to walk.
    var tempo = 1.0;
    var place_drag = 1.0;
    if (BIOMES_ON) {
        let place = place_of_room(terrain[p0] >> 28u);
        tempo = TEMPO_V[place];
        place_drag = DRAG_V[place];
    }
    // With age_metabolic the roll was made last tick, once the cell's spending was known.
    var old = unit_f(r) < HAZARD * tempo;
    if (AGE_METABOLIC) { old = (c.flags & F_OLD) != 0u; }
    if (c.e <= 0.0 || old) {
        var cause = (c.flags >> 8u) & 3u;
        if (c.e > 0.0) { cause = 3u; }
        atomicAdd(&ctr[C_STARVED + cause], 1u);
        fa[p0].y += max(c.e, 0.0) + c.womb + BODY;
        // A body gives back everything it held. This is the only way bound mineral leaves one.
        if (MINERAL_ON) { fd[p0].y += c.mineral + c.bound; }
        occ[p0] = 0u;
        seen_next[p0] = 0u;
        // The whole record at once. Writing just these two fields stored 0 for pos on the software
        // driver once this shader got large enough, with every rule on.
        c.e = -1.0;
        c.pos = NONE;
        cell[s] = c;
        let h = atomicAdd(&ctr[C_HEAD], 1u);
        ring[h & RING_MASK] = s;
        return;
    }

    // Wounds and armour hold susceptibility, not energy, so their fading moves none.
    if (PACK_ON) { c.wound *= PACK_WOUND_KEEP; }
    if (ARMOUR_ON) { c.hard *= 1.0 - ARMOUR_FADE; }
    let hardness = select(0.0, c.hard, ARMOUR_ON);
    c.age += 1.0;
    let here_a = fa[p0];
    let burn = min(here_a.z * HEAT_BURN / (1.0 + ARMOUR_RESIST * hardness), c.e);
    c.e -= burn;
    lose(burn);
    // Sitting in waste wears a cell down whatever it is doing. Waste deaths count as burned.
    if (WASTE_ON) {
        let poison = min(fc[p0].y * WASTE_HARM, c.e);
        c.e -= poison;
        lose(poison);
    }
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

    let me8 = sat(c.e, 10.0);
    var count = 0.0;
    var said = vec2f(0.0);
    var bigger = vec2f(0.0);
    var smaller = vec2f(0.0);
    var said0 = vec2f(0.0);
    var said1 = vec2f(0.0);
    var biting = vec2f(0.0);
    var biting_near = 0.0;
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
            // Seeing a body bite is seeing what it's doing, like seeing where it faces.
            if (GANG_ON && GANG_SENSE && (w & SEEN_ATTACK) != 0u) {
                biting += u * close;
                if (k == 0u) { biting_near += 1.0; }
            }
        }
    }
    let old_dir = c.flags & 7u;
    let front_old = sq(lx, ly, dx8(old_dir), dy8(old_dir));
    let w_front = seen[front_old];
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
    let i0 = vec4f(me8, select(c.womb / WOMB, sat(c.womb, WOMB), BIRTH_CHOICE), sat(c.age, 1000.0), sat(felt, 0.5));
    let i1 = vec4f(unit_f(r) * 2.0 - 1.0, 1.0, count / 8.0, hardness);
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
    // Rule inputs. i12.w and i13.w are free.
    var i12 = vec4f(0.0);
    if (WASTE_ON) {
        i12 = vec4f(sat(here_c.y, 2.0), contrast1(fc[pe].y, fc[pw].y), contrast1(fc[ps].y, fc[pn].y), 0.0);
    }
    var i13 = vec4f(0.0);
    if (ROADS_ON) {
        i13 = vec4f(sat(fd[p0].x, 2.0), contrast1(fd[pe].x, fd[pw].x), contrast1(fd[ps].x, fd[pn].x), 0.0);
    }
    var i14 = vec4f(0.0);
    if (SKILL_ON && SKILL_SENSE) { i14 = vec4f(c.skill_g, c.skill_f, 0.0, 0.0); }
    if (BIOMES_ON && BIOME_FEEL) { i14 = vec4f(i14.xy, c.feel_u, c.feel_d); }
    // The coat of the body in front, and a cell's own, as points on the circle.
    var i15 = vec4f(0.0);
    if (COAT_ON && COAT_SENSE) {
        i15 = vec4f(0.0, 0.0, cos(c.coat * TAU), sin(c.coat * TAU));
        let q = occ[front_old];
        if (q != 0u && q != ROCK) {
            let a = cell[q - 1u].coat * TAU;
            i15 = vec4f(cos(a), sin(a), i15.zw);
        }
    }
    // Mineral on the ground here, east and south, and in a cell's own free store.
    var i16 = vec4f(0.0);
    if (MINERAL_ON) {
        i16 = vec4f(sat(fd[p0].y, 1.0), contrast1(fd[pe].y, fd[pw].y), contrast1(fd[ps].y, fd[pn].y),
            sat(c.mineral, max(MINERAL_BIRTH, 1e-3)));
    }
    // Bodies seen biting anything, by direction, how many touch this cell, and how many are on the body in front.
    var i17 = vec4f(0.0);
    if (GANG_ON && GANG_SENSE) {
        var on_front = 0.0;
        if (front_kind > 0.0) { on_front = attackers(front_old, p0).x / 8.0; }
        i17 = vec4f(clamp(biting * 0.5, vec2f(-1.0), vec2f(1.0)), biting_near / 8.0, on_front);
    }

    // ---- Think ----
    //@BRAIN

    // ---- Spend ----
    // o0 rest move ground meat, o1 grow emit heat bite, o2 give enzyme armour clean are the budget.
    // o3 dir dir voice voice, o4 scent mix, o5 mate share coat key.
    // o6 is four spare budget actions, x0 to x3, when their switches are on. x0 is mineral uptake.
    // Rules also read o6.y (a gift carries mineral) and o6.z (give birth now), which aren't budget.
    var top = max(maxv(o0), max(maxv(o1), maxv(o2)));
    if (SPARE_ACTIONS) { top = max(top, maxv(o6 * EN3 - (vec4f(1.0) - EN3) * 1e6)); }
    var s0 = exp(clamp(o0 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN0;
    var s1 = exp(clamp(o1 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN1;
    var s2 = exp(clamp(o2 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN2;
    var s3 = exp(clamp(o6 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN3;
    let z = max(dot(s0, vec4f(1.0)) + dot(s1, vec4f(1.0)) + dot(s2, vec4f(1.0)) + dot(s3, vec4f(1.0)), 1e-9);
    s0 /= z;
    s1 /= z;
    s2 /= z;
    s3 /= z;

    // Resting is the only way to spend less, and it nearly stops the clock.
    let rest = s0.x;
    let upkeep = ((UPKEEP + UPKEEP_E * c.e) * (1.0 - REST_SAVING * rest) + ARMOUR_UPKEEP * hardness) * tempo;
    let pay = min(upkeep, c.e);
    c.e -= pay;
    lose(pay);
    // A coat creeps toward what the brain asks for, and each bit of the way costs energy.
    if (COAT_ON) {
        let arc = fract(fract(o5.z * COAT_SPAN) - c.coat + 0.5) - 0.5;
        let step_ = clamp(arc, -COAT_RATE, COAT_RATE);
        let cost = min(abs(step_) * COAT_COST, c.e);
        c.coat = fract(c.coat + step_);
        c.e -= cost;
        lose(cost);
    }
    // What a cell can feel of its place next tick: the upkeep it just paid, and how hard a step is.
    if (BIOMES_ON && BIOME_FEEL) {
        c.feel_u = sat(upkeep, max(UPKEEP, 1e-6));
        c.feel_d = sat(place_drag - 1.0, 1.0);
    }
    // A faster place lets a cell spend more per tick, so all it does happens faster.
    let spend_here = SPEND * tempo;
    let want = spend_here * (1.0 - rest);
    let scale = select(1.0, c.e / max(want, 1e-9), want > c.e);
    let a0 = s0 * spend_here * scale;
    let a1 = s1 * spend_here * scale;
    let a2 = s2 * spend_here * scale;
    let a3 = s3 * spend_here * scale;
    // Spending a little on something barely works, so dabbling in everything is a waste.
    let w0 = a0 * a0 / (a0 + vec4f(COMMITMENT));
    let w1 = a1 * a1 / (a1 + vec4f(COMMITMENT));
    let w2 = a2 * a2 / (a2 + vec4f(COMMITMENT));
    let w3 = a3 * a3 / (a3 + vec4f(COMMITMENT));

    // Practice sets how much of what a cell eats it can use. A convex curve means half the
    // practice is worth less than half the skill. The best case is the plain yield, so
    // practice only ever wastes food and can never make energy.
    var skill_ground = 1.0;
    var skill_flesh = 1.0;
    if (SKILL_ON) {
        skill_ground = SKILL_NAIVE + (1.0 - SKILL_NAIVE) * pow(c.skill_g, SKILL_CURVE);
        skill_flesh = SKILL_NAIVE + (1.0 - SKILL_NAIVE) * pow(c.skill_f, SKILL_CURVE);
    }

    let boost = 1.0 + min(here_c.x, ENZYME_CAP);
    // A deep pile can't be swallowed any faster than a full square, so one cell can't finish a bloom.
    let within_reach = select(here_a.x, min(here_a.x, GROUND_CAP), BLOOM_SHARE > 0.0);
    let ate_ground = within_reach * min(w0.z * GROUND_RATE * boost, 1.0);
    let ate_meat = here_a.y * min(w0.w * MEAT_RATE, 1.0);
    fa[p0].x = here_a.x - ate_ground;
    fa[p0].y = here_a.y - ate_meat;
    let ground_gain = ate_ground * skill_ground;
    lose(ate_ground - ground_gain);
    let meat_gain = ate_meat * MEAT_YIELD * skill_flesh;
    lose(ate_meat - meat_gain);
    let spent = want * scale;
    let grow = select(0.0, a1.x, c.womb < WOMB_LIMIT);
    let meal = ground_gain + meat_gain;
    // Waste comes from what was eaten. The energy it holds leaves the ledger as it is made and
    // comes back only when cleared, so waste that fades takes its energy with it.
    var waste_made = meal * WASTE_DEPOSIT;
    lose(meal * (1.0 - MEAL_KEPT));
    var cleaned = 0.0;
    var reclaimed = 0.0;
    if (WASTE_ON) {
        cleaned = here_c.y * min(w2.w * WASTE_RATE, 1.0);
        reclaimed = cleaned * WASTE_BACK;
        if (LEDGER) { atomicAdd(&ctr[C_SUN], u32(reclaimed * LEDGER_SCALE + 0.5)); }
    }
    c.e += meal * MEAL_KEPT + reclaimed - spent + (a1.x - grow);
    c.womb += grow;
    lose(max(spent - a1.x, 0.0));

    // Old age comes faster the more a cell burns. It takes effect at the start of next tick.
    var aged = false;
    if (AGE_METABOLIC) {
        r = pcg(r);
        let burned = spent / max(spend_here, 1e-9);
        aged = unit_f(r) < HAZARD * tempo * (AGE_REST + (1.0 - AGE_REST) * burned);
    }

    // Uptake is paid for in spent and lose above. Mineral moves in whole grains, so its total is exact.
    if (MINERAL_ON) {
        let here_m = fd[p0].y;
        let room = max(MINERAL_CAP - c.mineral, 0.0);
        let took = grains(min(here_m * min(w3.x * MINERAL_RATE, 1.0), room));
        fd[p0].y = here_m - took;
        c.mineral += took;
    }

    // Building is already paid for in spent and lose above. Hardness holds no energy.
    if (ARMOUR_ON) { c.hard = min(1.0, c.hard + w2.z * ARMOUR_BUILD); }

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
    if (aged) { flags |= F_OLD; }

    let front = sq(lx, ly, dx8(d), dy8(d));
    let tq = occ[front];
    var bite_take = 0.0;
    var bite_who = 0u;
    var gain_bite = 0.0;
    var bite_te = 0.0;
    var bite_fit = 0.0;
    if (tq != 0u && tq != ROCK) {
        let t = tq - 1u;
        var te = cell[t].e;
        if (te > 0.0) {
            if (BITE_ON && a1.w > 0.0) {
                // The stronger cell bites harder, so stored energy is also power.
                var force = 2.0 * c.e / (c.e + te + 1e-3);
                var open = 1.0;
                // Whoever bit this target last tick adds its energy, and doesn't block a side.
                var pack_e = c.e;
                if (EXPOSURE || GANG_ON) {
                    // Only open sides can be reached, so a cell inside a clump is protected.
                    var sides = 1.0;
                    let tx = front & MASK;
                    let ty = front >> LOG_W;
                    for (var k = 0u; k < 8u; k++) {
                        let q = sq(tx, ty, dx8(k), dy8(k));
                        var helps = false;
                        if (GANG_ON && q != p0) {
                            let w = seen[q];
                            helps = bit_this_square(w, k);
                            if (helps) { pack_e += seen_energy(w); }
                        }
                        if (occ[q] == 0u || helps) { sides += 1.0; }
                    }
                    if (EXPOSURE) { open = sides / 8.0; }
                }
                if (GANG_ON) {
                    let ours = pow(max(pack_e, 0.0), GANG_SHARP);
                    force = 2.0 * ours / (ours + pow(max(te, 0.0), GANG_SHARP) + 1e-3);
                }
                // A bite is only as good as its key's match with the target's coat.
                var fit = 1.0;
                var key = 0.0;
                if (COAT_ON) {
                    key = fract(o5.w * COAT_SPAN);
                    // Rounding can put this a hair under 0, where pow gives no number at all.
                    let m = max(0.5 + 0.5 * cos((key - cell[t].coat) * TAU), 0.0);
                    bite_fit = m;
                    fit = COAT_FLOOR + (1.0 - COAT_FLOOR) * pow(m, COAT_SHARP);
                }
                // Paid bite work opens the surface for later bites, up to the plain damage.
                var wound_factor = 1.0;
                if (PACK_ON) {
                    wound_factor = PACK_INTACT + (1.0 - PACK_INTACT) * cell[t].wound;
                    cell[t].wound = min(1.0, cell[t].wound + w1.w * PACK_WOUND_RATE);
                }
                let target_hard = select(0.0, cell[t].hard, ARMOUR_ON);
                let take = min(w1.w * BITE_POWER * force * open * wound_factor * fit / (1.0 + ARMOUR_RESIST * target_hard), te);
                bite_te = te;
                // The same share of the target's free mineral as of its energy. What won't fit falls to the ground.
                if (MINERAL_ON && take > 0.0) {
                    let spoil = grains(cell[t].mineral * (take / max(te, 1e-6)));
                    let kept = min(spoil, grains(max(MINERAL_CAP - c.mineral, 0.0)));
                    cell[t].mineral -= spoil;
                    c.mineral += kept;
                    fd[front].y += spoil - kept;
                }
                let keep = take * BITE_YIELD * skill_flesh;
                let spill = min(take - keep, keep);
                te -= take;
                cell[t].hit = max(cell[t].hit, take);
                c.e += keep * MEAL_KEPT;
                waste_made += keep * WASTE_DEPOSIT;
                fa[front].y += spill;
                lose(take - keep * MEAL_KEPT - spill);
                gain_bite = keep * MEAL_KEPT;
                bite_take = take;
                bite_who = t;
                if (take > 0.0) {
                    flags |= F_BIT;
                    if (NOISE_ON) { fa[front].w += NOISE_POWER; }
                    // A real attack takes real effort. A resting cell's tiny bite doesn't count.
                    if (w1.w >= GANG_EFFORT) {
                        flags |= F_ATTACK;
                        if (COAT_ON) { c.key = key; }
                    }
                }
                if (te <= 0.0) { cell[t].flags = (cell[t].flags & ~0x300u) | (1u << 8u); }
            }
            if (MINERAL_ON && GIVE_ON && a2.x > 0.0 && c.e > 0.0 && te > 0.0 && o6.y > 0.0) {
                // This gift carries mineral in place of energy. Its effort is paid for either way.
                let room = max(MINERAL_CAP - cell[t].mineral, 0.0);
                let given = grains(min(min(w2.x * MINERAL_GIVE, c.mineral), room));
                c.mineral -= given;
                cell[t].mineral += given;
            } else if (GIVE_ON && a2.x > 0.0 && c.e > 0.0 && te > 0.0) {
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

    if (WASTE_ON) { fc[p0].y = here_c.y + waste_made - cleaned; }

    if (c.e > MAX_ENERGY) {
        lose(c.e - MAX_ENERGY);
        c.e = MAX_ENERGY;
    }

    // Practice drifts toward the mix of what this cell really did this tick, so the two
    // can never sum past 1: gaining one takes it from the other. Unused practice fades.
    if (SKILL_ON) {
        let did_ground = select(0.0, s0.z, here_a.x > SKILL_MIN_FOOD);
        let did_flesh = select(0.0, s0.w, here_a.y > SKILL_MIN_FOOD) + select(0.0, s1.w, bite_take > 0.0);
        let did = did_ground + did_flesh;
        if (did > 1e-6) {
            let rate = SKILL_LEARN * min(did, 1.0);
            c.skill_g += (did_ground / did - c.skill_g) * rate;
            c.skill_f += (did_flesh / did - c.skill_f) * rate;
        }
        c.skill_g *= 1.0 - SKILL_FADE;
        c.skill_f *= 1.0 - SKILL_FADE;
    }

    // ---- Birth ----
    // With mate_pay on, a willing cell can give birth early if a willing neighbour pays the
    // rest of the child from its own womb. Alone it still has to fill the whole womb.
    // With birth_choice the brain says when, and the child is whatever is in the womb.
    let womb_ready = c.womb >= BIRTH_MIN && c.womb > BODY;
    let womb_full = select(c.womb >= WOMB, womb_ready && o6.z > 0.0, BIRTH_CHOICE);
    let can_pool = MATE_PAY > 0.0 && SEX_ON && (flags & F_MATE) != 0u && c.womb >= WOMB * (1.0 - MATE_PAY);
    // Measuring only: what stands between this cell and a child, this tick.
    var limit = G_LIM_READY;
    if (DEPTH_STATS) {
        let energy_ok = womb_full || can_pool || (BIRTH_CHOICE && womb_ready);
        let mineral_ok = c.mineral >= MINERAL_NEED;
        if (!energy_ok && !mineral_ok) {
            limit = G_LIM_BOTH;
        } else if (!energy_ok) {
            limit = G_LIM_ENERGY;
        } else if (!mineral_ok) {
            limit = G_LIM_MINERAL;
        } else if (!(womb_full || can_pool)) {
            limit = G_LIM_WAIT;
        }
    }
    if (c.e > 0.0 && (womb_full || can_pool)) {
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
        if (free_q == NONE && limit == G_LIM_READY) { limit = G_LIM_ROOM; }
        if (free_q != NONE) {
            // A willing neighbour can be the other parent. Any willing neighbour:
            // who mates with whom is up to their brains. compat acts here and is never sensed.
            let price = MATE_PAY * WOMB;
            let mate_mineral = grains(MINERAL_NEED * MATE_PAY);
            var mate = 0u;
            if (SEX_ON && (flags & F_MATE) != 0u) {
                for (var k = 0u; k < 8u; k++) {
                    let j = occ[sq(lx, ly, dx8((first + k) & 7u), dy8((first + k) & 7u))];
                    if (j == 0u || j == ROCK) { continue; }
                    let other = j - 1u;
                    if (cell[other].e > 0.0 && (cell[other].flags & F_MATE) != 0u
                        && cell[other].womb >= price
                        && cell[other].mineral >= mate_mineral
                        && (COMPAT <= 0.0 || genome_gap(s, other) < COMPAT)) {
                        mate = j;
                        break;
                    }
                }
            }
            // A birth takes mineral from the parents, and it stays bound in the child until it dies.
            let my_mineral = MINERAL_NEED - select(0.0, mate_mineral, mate != 0u);
            if ((womb_full || mate != 0u) && c.mineral >= my_mineral) {
                let tk = atomicAdd(&ctr[C_TAIL], 1u);
                if (i32(tk - ctl[0]) < 0) {
                    let child = ring[tk & RING_MASK];
                    // The energy only moves: from the two wombs into the child.
                    var paid = 0.0;
                    if (mate != 0u) {
                        paid = price;
                        cell[mate - 1u].womb -= price;
                        cell[mate - 1u].mineral -= mate_mineral;
                    }
                    c.mineral -= my_mineral;
                    var kid: Cell;
                    kid.e = c.womb + paid - BODY;
                    kid.stamp = F.tick;
                    kid.pos = free_q;
                    r = pcg(r);
                    kid.flags = r & 7u;
                    kid.clade = c.clade;
                    kid.gen = c.gen + 1u;
                    kid.bound = MINERAL_NEED;
                    // Like memory, a coat starts as the parent's and the brain can move it.
                    kid.coat = c.coat;
                    if (DEPTH_STATS) {
                        let bin = u32(clamp(floor(log2(max(kid.e, 0.5))) + 1.0, 0.0, 5.0));
                        atomicAdd(&ctr[G_CHILD + bin], 1u);
                        atomicAdd(&ctr[G_CHILD_E], u32(max(kid.e, 0.0) * 16.0 + 0.5));
                    }
                    // Armour and wounds are physical, so a child starts without them.
                    kid.skill_g = c.skill_g * SKILL_INHERIT;
                    kid.skill_f = c.skill_f * SKILL_INHERIT;
                    // Measuring only, never sensed. See research/life-gpu/exp/mating/.
                    if (MATE_TRACK) {
                        var near = 0.0;
                        for (var k = 0u; k < 8u; k++) {
                            let j = occ[sq(lx, ly, dx8((first + k) & 7u), dy8((first + k) & 7u))];
                            if (j == 0u || j == ROCK || cell[j - 1u].age <= 0.0) { continue; }
                            near = 1.0 + round(genome_gap(s, j - 1u) * 100.0);
                            break;
                        }
                        kid.mate_near = -near - 1.0;
                        if (mate != 0u) {
                            kid.mate_near = near * 1000.0 + round(genome_gap(s, mate - 1u) * 100.0);
                            cell[mate - 1u].mate_kids += 1.0;
                        }
                        c.mate_kids += 1.0;
                    }
                    cell[child] = kid;
                    // The slot's last owner's measurements mustn't colour this cell's.
                    var blank: Stat;
                    stat[child] = blank;
                    //@MEMCOPY
                    occ[free_q] = child + 1u;
                    seen_next[free_q] = 0x80u | (r & 7u);
                    let j = atomicAdd(&ctr[C_JOBS], 1u);
                    r = pcg(r);
                    jobs[j] = vec4u(child, s, mate, r);
                    c.womb = 0.0;
                    atomicAdd(&ctr[C_BIRTHS], 1u);
                    if (mate != 0u) { atomicAdd(&ctr[C_MIXED], 1u); }
                } else {
                    atomicAdd(&ctr[C_NO_SLOT], 1u);
                }
            }
        }
    }

    // ---- Move ----
    // Last, so nothing above reaches more than one square from where this tick began.
    var p = p0;
    r = pcg(r);
    var move_chance = w0.y * MOVE_RATE;
    var mass = 0.0;
    if (MASS_ON) {
        mass = max(c.e, 0.0) / max(MASS_SCALE, 1e-6);
        move_chance /= 1.0 + MASS_DRAG * mass;
    }
    if (ARMOUR_ON) { move_chance /= 1.0 + ARMOUR_DRAG * c.hard; }
    if (BIOMES_ON) { move_chance /= place_drag; }
    // Trodden ground is easier: the same effort goes further. Costs and creates no energy.
    var track_ahead = 0.0;
    if (ROADS_ON) {
        track_ahead = fd[front].x;
        let road = track_ahead * track_ahead / (track_ahead * track_ahead + TRACK_HALF * TRACK_HALF);
        move_chance *= 1.0 + ROAD_EASE * road;
    }
    if (c.e > 0.0 && unit_f(r) < move_chance) {
        let body = occ[front];
        var can_move = body == 0u;
        var cost = 0.0;
        if (MASS_ON) {
            cost = MASS_STEP_COST * (1.0 + mass);
            // A heavier cell can shove a lighter living one into the square it leaves.
            if (body != 0u && body != ROCK) {
                let their_e = cell[body - 1u].e;
                can_move = their_e > 0.0 && c.e > their_e;
                cost += MASS_PUSH_COST * (1.0 + mass + max(their_e, 0.0) / max(MASS_SCALE, 1e-6));
            }
        }
        if (can_move && c.e > cost) {
            c.e -= cost;
            lose(cost);
            occ[front] = o;
            occ[p] = body;
            seen_next[p] = 0u;
            if (body != 0u) {
                // Keeps its stamp and its moved flag: a shove isn't a chosen step. One that
                // hasn't acted yet and lands on a square already done waits for next tick.
                let t = body - 1u;
                cell[t].pos = p;
                let tc = cell[t];
                let te8 = u32(sat(max(tc.e, 0.0), 10.0) * 255.0);
                let tv8 = u32((tc.voice.x + 1.0) * 127.5);
                let tu8 = u32((tc.voice.y + 1.0) * 127.5);
                seen_next[p] = (te8 << 24u) | (tv8 << 16u) | (tu8 << 8u)
                    | 0x80u | (((tc.flags >> 5u) & 3u) << 3u) | (tc.flags & 7u)
                    | select(0u, SEEN_ATTACK, (tc.flags & F_ATTACK) != 0u);
            }
            p = front;
            c.pos = p;
            flags |= F_MOVED;
            // Moving makes noise anyone can hear. Heavier bodies are louder.
            if (NOISE_ON) { fa[p].w += NOISE_POWER * (0.5 + me8); }
            // Every step wears a track into the square it lands on.
            if (ROADS_ON) { fd[p].x = min(fd[p].x + TRACK_LAY, TRACK_CAP); }
            // Measuring only: steps landing on each square, read at log points.
            if (TRAIL_STATS) { fd[p].w += 1.0; }
        }
    }

    // ---- Finish ----
    c.voice = voice;
    c.flags = flags;
    cell[s] = c;
    let e8 = u32(sat(max(c.e, 0.0), 10.0) * 255.0);
    let v8 = u32((voice.x + 1.0) * 127.5);
    let u8_ = u32((voice.y + 1.0) * 127.5);
    let moved_bit = (flags >> 5u) & 3u;
    seen_next[p] = (e8 << 24u) | (v8 << 16u) | (u8_ << 8u) | 0x80u | (moved_bit << 3u) | d
        | select(0u, SEEN_ATTACK, (flags & F_ATTACK) != 0u);

    // Measuring only, for the log and never sensed. One tick in 64 per cell, so that even a giant
    // world's counters last a log step. Counts by group size come from last tick's snapshot.
    if (DEPTH_STATS && ((F.tick + s) & 63u) == 0u) {
        atomicAdd(&ctr[G_CELLS], 1u);
        atomicAdd(&ctr[limit], 1u);
        if ((flags & F_ATTACK) != 0u) {
            let n = u32(attackers(front, p0).x);
            atomicAdd(&ctr[G_BITES + n], 1u);
            atomicAdd(&ctr[G_GAIN + n], u32(gain_bite * 64.0 + 0.5));
            atomicAdd(&ctr[G_TARGET + n], u32(bite_te * 4.0 + 0.5));
            if (COAT_ON) {
                atomicAdd(&ctr[G_MATCH_N], 1u);
                atomicAdd(&ctr[G_MATCH], u32(bite_fit * 256.0 + 0.5));
            }
            // The control: another body this cell could have bitten and didn't.
            let spin = (F.tick ^ s) & 7u;
            for (var k = 0u; k < 8u; k++) {
                let q = sq(lx, ly, dx8((spin + k) & 7u), dy8((spin + k) & 7u));
                if (q != front && (seen[q] & 0x80u) != 0u) {
                    atomicAdd(&ctr[G_CTRL_N], 1u);
                    atomicAdd(&ctr[G_CTRL_EXTRA], u32(attackers(q, p0).x));
                    break;
                }
            }
        }
        // Scent ahead against scent to the sides, for cells that hardly emit: what it follows
        // can't be its own. Directed at 0 by chance, whatever the shape of the field.
        if (w0.y * MOVE_RATE > 0.05 && stat[s].b1.y < 0.02) {
            let side_a = sq(lx, ly, dx8((d + 2u) & 7u), dy8((d + 2u) & 7u));
            let side_b = sq(lx, ly, dx8((d + 6u) & 7u), dy8((d + 6u) & 7u));
            let along = contrast4(fb[front], (fb[side_a] + fb[side_b]) * 0.5);
            atomicAdd(&ctr[G_FOLLOW_N], 1u);
            atomicAdd(&ctr[G_FOLLOW], u32(i32(along.x * 256.0)));
            atomicAdd(&ctr[G_FOLLOW + 1u], u32(i32(along.y * 256.0)));
            atomicAdd(&ctr[G_FOLLOW + 2u], u32(i32(along.z * 256.0)));
            atomicAdd(&ctr[G_FOLLOW + 3u], u32(i32(along.w * 256.0)));
        }
    }

    // Measuring only. One tick in eight per cell, weighted to match.
    if (((F.tick + s) & 7u) == 0u) {
        const K = 0.08;
        var st = stat[s];
        let old_follow = st.follow;
        // What the cell itself kept from each food, after poor skill and what its waste holds.
        st.diet = st.diet * (1.0 - K) + vec4f(ground_gain * MEAL_KEPT, meat_gain * MEAL_KEPT, gain_bite, 0.0) * K;
        st.b0 += (s0 - st.b0) * K;
        st.b1 += (s1 - st.b1) * K;
        st.b2 += (s2 - st.b2) * K;
        st.b3 += (s3 - st.b3) * K;
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
        // Measures for the roads experiment, never sensed. Facing counts whether or not the step
        // worked, and the square ahead is compared with the two beside it, so neither easier
        // steps nor a cell's own trail behind it count. Mode 2 also makes follows_scent unbiased.
        if (ROADS_STATS > 0.0) {
            let side_a = sq(lx, ly, dx8((d + 2u) & 7u), dy8((d + 2u) & 7u));
            let side_b = sq(lx, ly, dx8((d + 6u) & 7u), dy8((d + 6u) & 7u));
            let tries = w0.y * MOVE_RATE > 0.05;
            let rich = f32(here_a.x > GROUND_CAP);
            var m = st.measures;
            if (tries) {
                let beside = (fd[side_a].x + fd[side_b].x) * 0.5;
                m.x += ((track_ahead - beside) / (0.05 + track_ahead + beside) - m.x) * K;
            }
            m.y += (rich - m.y) * K;
            m.z += (s1.y * rich - m.z) * K;
            m.w += (s1.y * (1.0 - rich) - m.w) * K;
            st.measures = m;
            if (ROADS_STATS > 1.5) {
                st.follow = old_follow;
                if (tries) {
                    let along = contrast4(fb[front], (fb[side_a] + fb[side_b]) * 0.5);
                    st.follow += (along - st.follow) * K;
                }
            }
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
