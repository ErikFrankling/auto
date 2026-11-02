// Shared by every pass. Buffers and physics constants are declared before this.

const NONE: u32 = 0xFFFFFFFFu;
const ROCK: u32 = 0xFFFFFFFFu;
const T_ROCK: u32 = 1u << 24u;

const C_HEAD: u32 = 0u;
const C_TAIL: u32 = 1u;
const C_JOBS: u32 = 2u;
const C_BIRTHS: u32 = 3u;
const C_MIXED: u32 = 4u;
const C_STARVED: u32 = 5u;
const C_BITTEN: u32 = 6u;
const C_BURNED: u32 = 7u;
const C_OLD: u32 = 8u;
const C_RAINED: u32 = 9u;
const C_LOST: u32 = 10u;
const C_SUN: u32 = 11u;
const C_RAIN_E: u32 = 12u;
// Births that had a free square but no free slot, so the world is at its memory limit.
const C_NO_SLOT: u32 = 13u;
// Sun taken up, sampled, when plants_meter is on.
const C_SUN_TAKEN: u32 = 14u;
// 16..31 are free for experiments.
const C_X0: u32 = 16u;

const F_MATE: u32 = 8u;
const F_SHARE: u32 = 16u;
const F_MOVED: u32 = 32u;
const F_BIT: u32 = 64u;

// PCG hash (Jarzynski and Olano 2020).
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn unit_f(r: u32) -> f32 { return f32(r >> 8u) * (1.0 / 16777216.0); }
fn sat(v: f32, s: f32) -> f32 { return v / (s + v); }
fn sat4(v: vec4f, s: f32) -> vec4f { return v / (vec4f(s) + v); }
// Contrast between two readings, so a gradient can be followed at any strength.
fn contrast4(a: vec4f, b: vec4f) -> vec4f { return (a - b) / (vec4f(0.05) + a + b); }
fn squash(x: vec4f) -> vec4f {
    let c = clamp(x, vec4f(-3.0), vec4f(3.0));
    return c * (vec4f(27.0) + c * c) / (vec4f(27.0) + 9.0 * c * c);
}
fn contrast1(a: f32, b: f32) -> f32 { return (a - b) / (0.05 + a + b); }
fn maxv(v: vec4f) -> f32 { return max(max(v.x, v.y), max(v.z, v.w)); }
fn dx8(d: u32) -> i32 { return i32((0x901Au >> (d * 2u)) & 3u) - 1; }
fn dy8(d: u32) -> i32 { return i32((0x01A9u >> (d * 2u)) & 3u) - 1; }
fn unit8(d: u32) -> vec2f {
    let k = select(1.0, 0.70710678, (d & 1u) == 1u);
    return vec2f(f32(dx8(d)), f32(dy8(d))) * k;
}
fn sq(x: u32, y: u32, dx: i32, dy: i32) -> u32 {
    let nx = u32(i32(x) + dx) & MASK;
    let ny = u32(i32(y) + dy) & MASK;
    return (ny << LOG_W) | nx;
}
fn lose(x: f32) {
    if (LEDGER) { atomicAdd(&ctr[C_LOST], u32(x * LEDGER_SCALE + 0.5)); }
}

// Kinds of place: 0 cold, 1 hot, 2 swamp, 3 barren. When walls or biomes are on, the
// four biome bits of a terrain word hold the room, 4 rooms by 4, and a room's kind
// follows from it, so two rooms sharing a wall are always different kinds of place.
const TEMPO_V = vec4f(TEMPO_COLD, TEMPO_HOT, TEMPO_SWAMP, TEMPO_BARREN);
const DRAG_V = vec4f(DRAG_COLD, DRAG_HOT, DRAG_SWAMP, DRAG_BARREN);
const SUNX_V = vec4f(SUNX_COLD, SUNX_HOT, SUNX_SWAMP, SUNX_BARREN);
const ROTX_V = vec4f(ROTX_COLD, ROTX_HOT, ROTX_SWAMP, ROTX_BARREN);
fn place_of_room(room: u32) -> u32 { return ((room & 3u) + (room >> 2u)) & 3u; }
