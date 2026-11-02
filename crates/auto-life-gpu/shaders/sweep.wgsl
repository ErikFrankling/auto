// World physics, run twice per tick: once along x, once along y. Neighbouring
// squares trade field values in pairs, in place. Nothing is created or lost by
// the trading, which keeps the energy ledger exact and needs no second buffer.

const FLOW_A = vec4f(select(0.0, PLANTS_FLOW, PLANTS_ON), MEAT_FLOW, HEAT_FLOW, NOISE_FLOW);
const KEEP_A = vec4f(1.0, 1.0, HEAT_DECAY_KEEP, NOISE_DECAY_KEEP);
const FLOW_B = vec4f(P0_FLOW, P1_FLOW, P2_FLOW, P3_FLOW);
const KEEP_B = vec4f(P0_DECAY_KEEP, P1_DECAY_KEEP, P2_DECAY_KEEP, P3_DECAY_KEEP);
// fc.z is fire: it spreads and fades by its own rule below, not by flow and decay.
const FLOW_C = vec4f(ENZYME_FLOW, WASTE_FLOW, 0.0, FCW_FLOW);
const KEEP_C = vec4f(ENZYME_DECAY_KEEP, WASTE_DECAY_KEEP, 1.0, FCW_DECAY_KEEP);
// fd.x is tracks. fd.y and fd.z are mineral and how rich a square is in it, traded below.
// fd.w only changes in act.wgsl. A rule that needs fd to be swept must also switch this on.
const FD_ON = ROADS_ON || MINERAL_ON;
const FLOW_D = vec4f(TRACK_FLOW, 0.0, 0.0, 0.0);
const KEEP_D = vec4f(TRACK_DECAY_KEEP, 1.0, 1.0, 1.0);
// Peak of a bloom: its volume is share of the sun, over one tile and one period.
const BLOOM_HEIGHT = 2.0 * BLOOM_SHARE * SUN * BLOOM_GRID * BLOOM_GRID * BLOOM_EVERY
    / (3.14159265 * BLOOM_RADIUS * BLOOM_RADIUS);

fn sun_at(t: u32) -> f32 {
    let base = f32(t & 0xFFFFu) / 65535.0 * SUN_RANGE;
    let phase = f32((t >> 16u) & 255u) / 256.0 * 6.2831853;
    let m = select(1.0, 0.55 + 0.45 * sin(phase + F.season), SEASON_TICKS > 0.0);
    return base * F.sun_k * m * (1.0 - BLOOM_SHARE);
}

// Weather: the whole sun map slides across the torus, so its total stays the same
// and only where it falls changes. The two speeds differ so it never retraces a line.
// Reads terrain at any distance, which is safe because terrain never changes.
fn sun_here(p: u32, t: u32) -> f32 {
    if (!WEATHER_SUN_ON) { return sun_at(t); }
    let sx = f32(F.tick) * WEATHER_DRIFT;
    let sy = sx * 0.618034;
    let x0 = (p + u32(sx)) & MASK;
    let x1 = (x0 + 1u) & MASK;
    let y0 = ((p >> LOG_W) + u32(sy)) & MASK;
    let y1 = (y0 + 1u) & MASK;
    let near = mix(sun_at(terrain[(y0 << LOG_W) | x0]), sun_at(terrain[(y0 << LOG_W) | x1]), fract(sx));
    let far = mix(sun_at(terrain[(y1 << LOG_W) | x0]), sun_at(terrain[(y1 << LOG_W) | x1]), fract(sx));
    return mix(near, far, fract(sy));
}

// Weather: fire, held in fc.z. Above 0 a square is alight and fading. Below 0 it has
// just burned and can't light again until the value climbs back to 0, so fronts die
// out instead of circling for ever. At 0 it lights only on thick ground, from a
// burning neighbour or, rarely, by itself.
fn flame_next(own: f32, other: f32, fuel: f32, p: u32) -> f32 {
    if (own > 0.0) {
        let f = own * WEATHER_FIRE_FADE;
        return select(-1.0, f, f >= 0.05);
    }
    if (own < 0.0) { return min(own + 0.5 / max(WEATHER_FIRE_REST, 1.0), 0.0); }
    if (fuel < WEATHER_FIRE_FUEL) { return 0.0; }
    let r = pcg(p ^ (F.tick * 2654435761u) ^ (F.seed * 40503u) ^ F.axis);
    let chance = WEATHER_FIRE_RATE + select(0.0, WEATHER_FIRE_SPREAD, other > 0.0);
    return select(0.0, 1.0, f32(r) * 2.3283064e-10 < chance);
}

// Each tile of the world has one bloom per period, at a random time and place. What it
// lands on any square depends only on the square and the tick, so nothing needs storing.
fn bloom_at(p: u32) -> f32 {
    let x = p & MASK;
    let y = p >> LOG_W;
    let g = u32(BLOOM_GRID);
    let ix = x / g;
    let iy = y / g;
    let h = pcg(pcg(ix ^ (F.seed * 2654435761u)) ^ (iy * 40503u));
    let period = u32(BLOOM_EVERY);
    let t = F.tick + h % period;
    if (t % period != 0u) { return 0.0; }
    let h1 = pcg(h ^ (t / period));
    let h2 = pcg(h1);
    let room = BLOOM_GRID - 2.0 * BLOOM_RADIUS;
    let dx = f32(x - ix * g) - BLOOM_RADIUS - unit_f(h1) * room;
    let dy = f32(y - iy * g) - BLOOM_RADIUS - unit_f(h2) * room;
    return BLOOM_HEIGHT * max(1.0 - (dx * dx + dy * dy) / (BLOOM_RADIUS * BLOOM_RADIUS), 0.0);
}

// Sun grows ground food, and meat slowly rots into it.
fn feed(v: vec4f, t: u32, p: u32) -> vec4f {
    var a = v;
    if (BLOOM_SHARE > 0.0) {
        // Food piled above the cap spoils fast. Spoiled food is gone.
        let spoiled = max(a.x - GROUND_CAP, 0.0) * BLOOM_FADE;
        a.x -= spoiled;
        lose(spoiled);
        let landed = bloom_at(p);
        a.x += landed;
        if (LEDGER) { atomicAdd(&ctr[C_SUN], u32(landed * LEDGER_SCALE + 0.5)); }
    }
    var light = sun_here(p, t);
    if (PLANTS_ON) {
        // Living ground: growth is set by how much is already growing. Middling cover
        // grows fastest, bare ground only by a trickle, and full ground not at all.
        let d = clamp(a.x / GROUND_CAP, 0.0, 1.0);
        light *= (PLANTS_PEAK * 4.0 * d + PLANTS_TRICKLE) * (1.0 - d);
    }
    let got = min(light, max(GROUND_CAP - a.x, 0.0));
    a.x += got;
    // Measuring only, one tick in 256 at 256 counts per unit, so it's the sun taken up per tick.
    if (PLANTS_METER && (F.tick & 255u) == 0u) {
        atomicAdd(&ctr[C_SUN_TAKEN], u32(got * 256.0 + 0.5));
    }
    var rot = a.y * MEAT_ROT;
    if (BIOMES_ON) { rot *= ROTX_V[place_of_room(t >> 28u)]; }
    a.y -= rot;
    let kept = min(rot, max(GROUND_CAP - a.x, 0.0));
    a.x += kept;
    if (LEDGER) {
        atomicAdd(&ctr[C_SUN], u32(got * LEDGER_SCALE + 0.5));
        lose(rot - kept);
    }
    return a;
}

@compute @workgroup_size(64, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    var pa: u32;
    var pb: u32;
    if (F.axis == 0u) {
        let i = id.x * 2u + F.off;
        if (i >= W) { return; }
        let j = id.y + F.row0;
        pa = (j << LOG_W) | i;
        pb = (j << LOG_W) | ((i + 1u) & MASK);
    } else {
        if (id.x >= W) { return; }
        let j = (id.y + F.row0) * 2u + F.off;
        pa = (j << LOG_W) | id.x;
        pb = (((j + 1u) & MASK) << LOG_W) | id.x;
    }
    let ta = terrain[pa];
    let tb = terrain[pb];
    let rock_a = (ta & T_ROCK) != 0u;
    let rock_b = (tb & T_ROCK) != 0u;
    if (rock_a && rock_b) { return; }
    var a0 = fa[pa];
    var b0 = fa[pb];
    var a1 = fb[pa];
    var b1 = fb[pb];
    var a2 = fc[pa];
    var b2 = fc[pb];
    if (!rock_a && !rock_b) {
        let f0 = FLOW_A * (a0 - b0);
        a0 -= f0;
        b0 += f0;
        let f1 = FLOW_B * (a1 - b1);
        a1 -= f1;
        b1 += f1;
        let f2 = FLOW_C * (a2 - b2);
        a2 -= f2;
        b2 += f2;
    }
    if (WEATHER_FIRE_ON) {
        // Rock never burns and never passes fire on. Flames heat the square, and heat
        // is what hurts: cells can already feel it, and its damage is already counted.
        let flame_a = a2.z;
        let flame_b = b2.z;
        if (!rock_a) { a2.z = flame_next(flame_a, select(flame_b, 0.0, rock_b), a0.x, pa); }
        if (!rock_b) { b2.z = flame_next(flame_b, select(flame_a, 0.0, rock_a), b0.x, pb); }
        if (a2.z > 0.0) {
            let burned = a0.x * WEATHER_FIRE_BURN;
            a0.x -= burned;
            a0.z += a2.z * WEATHER_FIRE_HEAT;
            lose(burned);
        }
        if (b2.z > 0.0) {
            let burned = b0.x * WEATHER_FIRE_BURN;
            b0.x -= burned;
            b0.z += b2.z * WEATHER_FIRE_HEAT;
            lose(burned);
        }
    }
    a0 *= KEEP_A;
    b0 *= KEEP_A;
    a1 *= KEEP_B;
    b1 *= KEEP_B;
    a2 *= KEEP_C;
    b2 *= KEEP_C;
    if (F.axis == 0u) {
        if (!rock_a) { a0 = feed(a0, ta, pa); }
        if (!rock_b) { b0 = feed(b0, tb, pb); }
    }
    if (!rock_a) {
        fa[pa] = a0;
        fb[pa] = a1;
        fc[pa] = a2;
    }
    if (!rock_b) {
        fa[pb] = b0;
        fb[pb] = b1;
        fc[pb] = b2;
    }
    if (FD_ON) {
        var a3 = fd[pa];
        var b3 = fd[pb];
        if (!rock_a && !rock_b) {
            let f3 = FLOW_D * (a3 - b3);
            a3 -= f3;
            b3 += f3;
            // Mineral drains toward each square's richness, so patches last for ever. The
            // trade is in whole grains between a pair, so the total can't change by a single bit.
            if (MINERAL_ON) {
                let wa = a3.z;
                let wb = b3.z;
                let m = round(MINERAL_FLOW * (a3.y * wb - b3.y * wa) / (wa + wb) * GRAIN) / GRAIN;
                a3.y -= m;
                b3.y += m;
            }
        }
        if (!rock_a) { fd[pa] = a3 * KEEP_D; }
        if (!rock_b) { fd[pb] = b3 * KEEP_D; }
    }
}
