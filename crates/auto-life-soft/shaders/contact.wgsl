// Every cell looks at what's near it and adds up force, senses and trades. Reads only last
// tick's picture of its neighbours and writes only its own record, except for two atomic words a
// neighbour owns: what's being asked of it by biters, and what it's being given.
//
// A pair's force is worked out with the lower slot first, on both sides, so each side gets
// exactly the same numbers and the two forces cancel to the last bit.

const FRONT_COS: f32 = 0.70710678;

fn route(k: u32, x: u32) {
    if (x == 0u) { return; }
    let old = atomicAdd(&ctr[k * 2u], x);
    if (old > 0xFFFFFFFFu - x) { atomicAdd(&ctr[k * 2u + 1u], 1u); }
}

// Radial force along the line from the lower slot to the higher, positive apart.
// a is the lower slot, v the vector from a to b, d its length.
struct Pair { crowd: f32, pull: f32, shove: f32, strength: f32 }
fn pair_force(ar: f32, br: f32, astick: f32, bstick: f32, ag: vec2f, bg: vec2f,
              apush: f32, bpush: f32, ah: vec2f, bh: vec2f, v: vec2f, d: f32) -> Pair {
    var out: Pair;
    let rr = ar + br;
    if (CROWD_ON) { out.crowd = K_PUSH * max(rr - d, 0.0); }
    if (STICK_ON && astick > 0.0 && bstick > 0.0) {
        out.strength = astick * bstick * glue_match(ag, bg);
        // A pull up to gap beyond contact, fading to nothing there, so a pair pulled further
        // apart stops sticking. Nothing is stored about the pair, so this is how it breaks.
        out.pull = out.strength * STICK_MAX * clamp(1.0 - (d - rr) / GAP, 0.0, 1.0);
    }
    if (PUSH_ON && d < rr) {
        if (apush > 0.0 && dot(ah, v) > FRONT_COS * d) { out.shove += apush; }
        if (bpush > 0.0 && -dot(bh, v) > FRONT_COS * d) { out.shove += bpush; }
    }
    return out;
}

// Hands a gift to each contact in proportion to match. Runs only for a cell that's giving.
fn hand_out(me: BinBody, s: u32, bx: u32, by: u32, amount: f32, msum: f32, metered: bool) -> u32 {
    let mg = unpack4x8snorm(me.gv).xy;
    var out = 0u;
    var inn = 0u;
    for (var dy = 0u; dy < 3u; dy++) {
        let oy = (by + dy + BIN_MASK) & BIN_MASK;
        for (var dx = 0u; dx < 3u; dx++) {
            let b2 = ((oy << LOG_B) | ((bx + dx + BIN_MASK) & BIN_MASK)) * BIN_CAP;
            for (var q = 0u; q < BIN_CAP; q++) {
                let o = bb[b2 + q];
                if (o.slot == 0u) { break; }
                if (o.slot == me.slot) { continue; }
                let v = vec2f(f32(i32(o.x - me.x)), f32(i32(o.y - me.y))) * POS_TO_SQ;
                let rr = me.r + o.r;
                if (dot(v, v) >= rr * rr) { continue; }
                let m = glue_match(mg, unpack4x8snorm(o.gv).xy);
                let x = amount * m / msum;
                let x_u = u32(x * E_ONE);
                let y_u = u32(x * GIVE_YIELD * E_ONE);
                out += x_u;
                inn += y_u;
                if (y_u > 0u) { atomicAdd(&mail[(o.slot - 1u) * 2u + 1u], y_u); }
            }
        }
    }
    if (metered) {
        route(R_GIVE_OUT, out);
        route(R_GIVE_IN, inn);
    }
    return out;
}

@compute @workgroup_size(32, 1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let bx = id.x;
    if (bx >= BINS) { return; }
    let by = id.y + F.row0;
    let b = ((by << LOG_B) | bx) * BIN_CAP;
    for (var k = 0u; k < BIN_CAP; k++) {
        let me = bb[b + k];
        if (me.slot == 0u) { break; }
        let s = me.slot - 1u;
        let mh = unpack2x16snorm(me.hd);
        let mgv = unpack4x8snorm(me.gv);
        let mg = mgv.xy;
        let mv = mgv.zw;
        // Senses are only worth adding up for a cell that thinks next tick.
        let due = BRAIN_EVERY_U == 1u || (s % BRAIN_EVERY_U) == ((F.tick + 1u) % BRAIN_EVERY_U);
        var biting = false;
        var giving = false;
        if (BITE_ON || GIVE_ON) {
            let u = cell[s].unit;
            biting = BITE_ON && plan[s].s1.w * u > 0.0;
            giving = GIVE_ON && plan[s].s2.x * u > 0.0;
        }

        var fx = 0i;
        var fy = 0i;
        var fee = 0.0;
        var contacts = 0u;
        var pressure = 0.0;
        var msum = 0.0;
        var best = -2.0;
        var target_ = 0u;
        var count = vec4f(0.0);
        var gap = vec4f(0.0);
        var v0 = vec4f(0.0);
        var v1 = vec4f(0.0);
        var mt = vec4f(0.0);

        for (var dy = 0u; dy < 3u; dy++) {
            let oy = (by + dy + BIN_MASK) & BIN_MASK;
            for (var dx = 0u; dx < 3u; dx++) {
                let b2 = ((oy << LOG_B) | ((bx + dx + BIN_MASK) & BIN_MASK)) * BIN_CAP;
                for (var q = 0u; q < BIN_CAP; q++) {
                    let o = bb[b2 + q];
                    if (o.slot == 0u) { break; }
                    if (o.slot == me.slot) { continue; }
                    let dv = vec2f(f32(i32(o.x - me.x)), f32(i32(o.y - me.y))) * POS_TO_SQ;
                    let d2 = dot(dv, dv);
                    let rr = me.r + o.r;
                    let reach = rr + GAP;
                    if (d2 >= reach * reach) { continue; }
                    let d = max(sqrt(d2), 1e-6);
                    let ogv = unpack4x8snorm(o.gv);
                    let oh = unpack2x16snorm(o.hd);
                    let lower = me.slot < o.slot;
                    let v = select(-dv, dv, lower);
                    var pf: Pair;
                    if (lower) {
                        pf = pair_force(me.r, o.r, me.stick, o.stick, mg, ogv.xy, me.push, o.push, mh, oh, v, d);
                    } else {
                        pf = pair_force(o.r, me.r, o.stick, me.stick, ogv.xy, mg, o.push, me.push, oh, mh, v, d);
                    }
                    let f = pf.crowd + pf.shove - pf.pull;
                    let fixed = vec2i(v / d * f * FORCE_ONE);
                    if (lower) {
                        fx -= fixed.x;
                        fy -= fixed.y;
                    } else {
                        fx += fixed.x;
                        fy += fixed.y;
                    }
                    fee += pf.strength;
                    if (d >= rr) { continue; }
                    // Touching.
                    contacts += 1u;
                    let m = glue_match(mg, ogv.xy);
                    msum += m;
                    if (biting) {
                        let cs = dot(mh, dv) / d;
                        if (cs > FRONT_COS && cs > best) {
                            best = cs;
                            target_ = o.slot;
                        }
                    }
                    if (due) {
                        pressure += pf.crowd + pf.shove;
                        // Which side of the body it's on.
                        let along = dot(mh, dv);
                        let side = mh.x * dv.y - mh.y * dv.x;
                        var mask: vec4f;
                        if (abs(side) <= along) {
                            mask = vec4f(1.0, 0.0, 0.0, 0.0);
                        } else if (abs(side) <= -along) {
                            mask = vec4f(0.0, 0.0, 1.0, 0.0);
                        } else if (side > 0.0) {
                            mask = vec4f(0.0, 1.0, 0.0, 0.0);
                        } else {
                            mask = vec4f(0.0, 0.0, 0.0, 1.0);
                        }
                        count += mask;
                        gap += mask * (o.r - me.r);
                        v0 += mask * ogv.z;
                        v1 += mask * ogv.w;
                        mt += mask * m;
                    }
                }
            }
        }

        // Bites are settled as supply and demand once everyone has asked: see bite.wgsl.
        var wish = 0u;
        if (target_ != 0u) {
            wish = u32(effort(plan[s].s1.w * cell[s].unit) * BITE_POWER * E_ONE);
            if (wish > 0u) { atomicAdd(&mail[(target_ - 1u) * 2u], wish); }
        }
        let metered = LEDGER || ((s + F.tick) & 63u) == 0u;
        var give_out = 0u;
        if (giving && msum > 0.0) {
            let e_f = f32(cell[s].e) * E_INV;
            let amount = min(effort(plan[s].s2.x * cell[s].unit) * GIVE_POWER, e_f);
            give_out = hand_out(me, s, bx, by, amount, msum, metered);
        }

        var a: Acc;
        a.fx = fx;
        a.fy = fy;
        a.give_out = give_out;
        a.target_ = select(0u, target_, wish > 0u);
        a.wish = wish;
        a.fee = fee * STICK_FEE;
        a.contacts = contacts;
        acc[s] = a;
        if (due) {
            var sn: Sense;
            sn.count = count;
            sn.gap = gap;
            sn.v0 = v0;
            sn.v1 = v1;
            sn.mt = mt;
            sn.pressure = pressure;
            sn.contacts = f32(contacts);
            sense[s] = sn;
        }
    }
}
