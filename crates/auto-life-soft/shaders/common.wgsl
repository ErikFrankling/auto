// Shared by every pass. Buffers and physics constants are declared before this.
//
// Energy, food and every field are integers, in units of 1/65536. Anything that adds up across
// cells or squares does so in integers, so the order threads run in can't change a total.
// Positions are integers too: a u32 is one trip round the world, so differences wrap by themselves.

const E_ONE: f32 = 65536.0;
const E_INV: f32 = 1.0 / 65536.0;
const NONE: u32 = 0xFFFFFFFFu;
const FORCE_ONE: f32 = 65536.0;

// Bits of a cell's flags.
const F_ALIVE: u32 = 1u;
// Bits of a terrain word. Low 16 are sun, then 8 of phase.
const T_ROCK: u32 = 1u << 24u;
const T_E: u32 = 1u << 25u;
const T_W: u32 = 1u << 26u;
const T_S: u32 = 1u << 27u;
const T_N: u32 = 1u << 28u;

// Energy flows, each a 64-bit counter of two words. See Ledger in world.rs.
const R_SUN: u32 = 0u;
const R_RAIN: u32 = 1u;
const R_UPKEEP: u32 = 2u;
const R_SPEND: u32 = 3u;
const R_STICK: u32 = 4u;
const R_HARM: u32 = 5u;
const R_EATEN: u32 = 6u;
const R_EAT_GOT: u32 = 7u;
const R_MEAT_EATEN: u32 = 8u;
const R_MEAT_GOT: u32 = 9u;
const R_BITE_PAID: u32 = 10u;
const R_BITE_GOT: u32 = 11u;
const R_GIVE_OUT: u32 = 12u;
const R_GIVE_IN: u32 = 13u;
const R_CAP: u32 = 14u;
const R_ROT_LOST: u32 = 15u;
const R_DEATH: u32 = 16u;
const R_GROW: u32 = 17u;
const R_COUNT: u32 = 18u;
// Plain counters come after the flows.
const C_BASE: u32 = R_COUNT * 2u;
const C_BIRTHS: u32 = C_BASE + 0u;
const C_RAINED: u32 = C_BASE + 1u;
const C_STARVED: u32 = C_BASE + 2u;
const C_BITTEN: u32 = C_BASE + 3u;
const C_BURNED: u32 = C_BASE + 4u;
const C_OLD: u32 = C_BASE + 5u;
const C_OVERFLOW: u32 = C_BASE + 6u;
const C_NO_SLOT: u32 = C_BASE + 7u;
const C_CONTACTS: u32 = C_BASE + 8u;

// Words of ctl, the allocator's state. Only the one-invocation pass in alloc.wgsl writes it.
const L_HEAD: u32 = 0u;
const L_TAIL: u32 = 1u;
const L_ALIVE: u32 = 2u;
const L_OPEN: u32 = 3u;
const L_HEAD_NOW: u32 = 4u;
const L_TAIL_NOW: u32 = 5u;
const L_BIRTHS: u32 = 6u;
const L_NEXT_UID: u32 = 7u;
const L_UID_NOW: u32 = 8u;

// PCG hash (Jarzynski and Olano 2020).
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit_f(r: u32) -> f32 { return f32(r >> 8u) * (1.0 / 16777216.0); }
fn sat(v: f32, s: f32) -> f32 { return v / max(s + v, 1e-9); }
fn sat4(v: vec4f, s: f32) -> vec4f { return v / (vec4f(s) + v); }
fn contrast4(a: vec4f, b: vec4f) -> vec4f { return (a - b) / (vec4f(0.05) + a + b); }
fn squash(x: vec4f) -> vec4f {
    let c = clamp(x, vec4f(-3.0), vec4f(3.0));
    return c * (vec4f(27.0) + c * c) / (vec4f(27.0) + 9.0 * c * c);
}
fn maxv(v: vec4f) -> f32 { return max(max(v.x, v.y), max(v.z, v.w)); }

// Spending a little on something barely works, so dabbling in everything is a waste.
fn effort(a: f32) -> f32 { return a * a / max(a + COMMITMENT, 1e-12); }

fn square_of(x: u32, y: u32) -> u32 { return ((y >> POS_SHIFT) << LOG_W) | (x >> POS_SHIFT); }
fn wrap_sq(x: i32, y: i32) -> u32 {
    return ((u32(y) & MASK) << LOG_W) | (u32(x) & MASK);
}

// How alike two glues are, from 1 down towards 0. Symmetric, so both sides of a pair agree.
fn glue_match(a: vec2f, b: vec2f) -> f32 {
    let d = a - b;
    return 1.0 / (1.0 + GLUE_SHARP * dot(d, d));
}
