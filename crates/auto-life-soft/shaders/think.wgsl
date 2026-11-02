// One cell thinks: senses, runs its brain, and decides how to divide its budget until it next
// thinks. A share of the cells run each tick, staggered by slot, so the load per tick stays even.
//
// Inputs, each a row of four:
//  i0 own energy, womb, age, harm felt      i1 noise, bias, size, pressure
//  i2 contacts by sector                    i3 size gap by sector
//  i4 first voice by sector                 i5 second voice by sector
//  i6 glue match by sector                  i7 food, meat, heat, enzyme here
//  i8 scents here                           i9 gradients of i7, forward
//  i10 gradients of i7, to the left         i11 gradients of i8, forward
//  i12 gradients of i8, to the left         i13 own stick, own shove, contacts, bites felt
// Sectors are front, left, back, right in the body's frame. A cell is never told who it's
// related to: everything it senses of a neighbour is what that neighbour's brain chose.
//
// Outputs:
//  o0 rest thrust eat-food eat-meat         o1 grow emit heat bite
//  o2 give enzyme stick push                o3 turn turn voice voice
//  o4 mix of the four scents                o5 glue glue split share
//  o6 split-angle split-angle spare spare

const EN0 = vec4f(1.0, THRUST_ON_F, 1.0, MEAT_ON_F);
const EN1 = vec4f(1.0, SCENT_ON_F, HEAT_ON_F, BITE_ON_F);
const EN2 = vec4f(GIVE_ON_F, ENZYME_ON_F, STICK_ON_F, PUSH_ON_F);

fn bt(i: u32, v: vec4f) -> vec4f {
    let g = gene(i);
    return vec4f(
        dot(unpack4x8snorm(g.x), v),
        dot(unpack4x8snorm(g.y), v),
        dot(unpack4x8snorm(g.z), v),
        dot(unpack4x8snorm(g.w), v),
    ) * 2.0;
}

fn fa_f(p: u32) -> vec4f { return vec4f(fa0[p]) * E_INV; }
fn fb_f(p: u32) -> vec4f { return vec4f(fb0[p]) * E_INV; }

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let s = (id.x + F.row0) * BRAIN_EVERY_U + F.phase;
    if (s >= SLOTS) { return; }
    var c = cell[s];
    if ((c.flags & F_ALIVE) == 0u) { return; }

    // ---- Sense ----
    let p = square_of(c.x, c.y);
    let ix = i32(p & MASK);
    let iy = i32(p >> LOG_W);
    let t = terrain[p];
    let here_a = fa_f(p);
    let here_b = fb_f(p);
    // Rock has no field, so a wall reads as nothing there, the same as an empty square.
    let ae = fa_f(wrap_sq(ix + 1, iy));
    let aw = fa_f(wrap_sq(ix - 1, iy));
    let as_ = fa_f(wrap_sq(ix, iy + 1));
    let an = fa_f(wrap_sq(ix, iy - 1));
    let be = fb_f(wrap_sq(ix + 1, iy));
    let bw = fb_f(wrap_sq(ix - 1, iy));
    let bs = fb_f(wrap_sq(ix, iy + 1));
    let bn = fb_f(wrap_sq(ix, iy - 1));
    let gxa = contrast4(ae, aw);
    let gya = contrast4(as_, an);
    let gxb = contrast4(be, bw);
    let gyb = contrast4(bs, bn);
    let sn = sense[s];

    var r = pcg(s ^ (F.tick * 2654435761u) ^ (F.seed * 40503u));
    r = pcg(r);
    let e_f = f32(c.e) * E_INV;
    let i0 = vec4f(sat(e_f, 10.0), sat(f32(c.womb) * E_INV, BODY), sat(f32(c.age), 1000.0), sat(c.felt, 0.05));
    let i1 = vec4f(unit_f(r) * 2.0 - 1.0, 1.0, (c.r - R0) / max(R_MAX - R0, 1e-6), sat(sn.pressure, 1.0));
    let i2 = sn.count / 4.0;
    let i3 = sn.gap;
    let i4 = sn.v0 / 4.0;
    let i5 = sn.v1 / 4.0;
    let i6 = sn.mt / 4.0;
    let i7 = sat4(here_a, 2.0);
    let i8 = sat4(here_b, 2.0);
    let i9 = gxa * c.hx + gya * c.hy;
    let i10 = -gxa * c.hy + gya * c.hx;
    let i11 = gxb * c.hx + gyb * c.hy;
    let i12 = -gxb * c.hy + gyb * c.hx;
    let i13 = vec4f(c.stick, sat(c.push, 0.2), sat(sn.contacts, 6.0), sat(c.bitten, 0.1));

    // ---- Think ----
    //@BRAIN

    // ---- Decide ----
    let top = max(maxv(o0), max(maxv(o1), maxv(o2)));
    var s0 = exp(clamp(o0 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN0;
    var s1 = exp(clamp(o1 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN1;
    var s2 = exp(clamp(o2 - vec4f(top), vec4f(-20.0), vec4f(0.0))) * EN2;
    let z = max(dot(s0, vec4f(1.0)) + dot(s1, vec4f(1.0)) + dot(s2, vec4f(1.0)), 1e-9);
    s0 /= z;
    s1 /= z;
    s2 /= z;
    let m = exp(clamp(o4 - vec4f(maxv(o4)), vec4f(-20.0), vec4f(0.0)));

    // The new heading, turning at most so far in the body's own frame.
    let turn = clamp(atan2(o3.y, o3.x + 1e-6), -TURN_MAX, TURN_MAX);
    let ct = cos(turn);
    let st = sin(turn);
    let hx = c.hx * ct - c.hy * st;
    let hy = c.hx * st + c.hy * ct;
    let len = max(sqrt(hx * hx + hy * hy), 1e-6);
    c.hx = hx / len;
    c.hy = hy / len;

    var voice = vec2f(0.0);
    if (VOICE_ON) { voice = squash(vec4f(o3.z, o3.w, 0.0, 0.0)).xy; }
    let glue = squash(vec4f(o5.x, o5.y, 0.0, 0.0)).xy;
    c.gv = pack4x8snorm(vec4f(glue, voice));
    c.felt = 0.0;

    // The direction a child would appear in, relative to the heading.
    let al = atan2(o6.y, o6.x + 1e-6);
    var pl: Plan;
    pl.s0 = s0;
    pl.s1 = s1;
    pl.s2 = s2;
    pl.mix = m / dot(m, vec4f(1.0));
    pl.split = vec4f(cos(al), sin(al), clamp(0.5 + 0.5 * o5.w, 0.0, 1.0), o5.z);
    plan[s] = pl;
    cell[s] = c;
}
