//! Reaction-diffusion: float state, one 3x3 Laplacian per tick.

use super::Entry;
use crate::program::{Bind, Field, Pass, Program, groups};

pub const GRAY_SCOTT: Entry = Entry {
    id: "grayscott",
    name: "Gray-Scott reaction-diffusion",
    blurb: "Two chemicals, feed and kill rates varying across the world, so every pattern type appears at once.",
    default_size: (1920, 1080),
    needs_pot: false,
    width_step: 1,
    build: gray_scott,
};

pub fn gray_scott(w: u32, h: u32) -> Program {
    let cells = w as u64 * h as u64;
    let step = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let w = P.w; let h = P.h; let x = id.x; let y = id.y;
    if (x >= w || y >= h) { return; }
    let xl = select(x - 1u, w - 1u, x == 0u);
    let xr = select(x + 1u, 0u, x + 1u == w);
    let up = select(y - 1u, h - 1u, y == 0u) * w;
    let me = y * w;
    let dn = select(y + 1u, 0u, y + 1u == h) * w;
    let c = cur[me + x];
    // Karl Sims's Laplacian weights: edges 0.2, corners 0.05, centre -1.
    let lap = (cur[me + xl] + cur[me + xr] + cur[up + x] + cur[dn + x]) * 0.2
            + (cur[up + xl] + cur[up + xr] + cur[dn + xl] + cur[dn + xr]) * 0.05 - c;
    let u = f32(x) / f32(w);
    let v = f32(y) / f32(h);
    // Pearson's map of the parameter space across the world.
    let feed = 0.022 + 0.040 * v;
    let kill = 0.056 + 0.0095 * u;
    let reaction = c.x * c.y * c.y;
    next[me + x] = vec2f(
        c.x + (1.0 * lap.x - reaction + feed * (1.0 - c.x)),
        c.y + (0.5 * lap.y + reaction - (feed + kill) * c.y),
    );
}
"#;
    let init = r#"
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= P.w || id.y >= P.h) { return; }
    // Blocks of the second chemical scattered through the first.
    let blob = rand3(id.x / 16u, id.y / 16u, frame.seed) % 8u == 0u;
    cur[id.y * P.w + id.x] = select(vec2f(1.0, 0.0), vec2f(0.5, 0.25), blob);
}
"#;
    let view = r#"
fn color(x: u32, y: u32) -> vec3f {
    let c = view_cells[y * P.w + x];
    let t = clamp(c.y * 2.6, 0.0, 1.0);
    return ramp(t, vec3f(0.01, 0.015, 0.05), vec3f(0.03, 0.16, 0.34), vec3f(0.05, 0.55, 0.6), vec3f(0.6, 0.9, 0.6), vec3f(1.0, 0.98, 0.85));
}
"#;
    Program {
        id: "grayscott".into(),
        width: w,
        height: h,
        fields: vec![Field::double("cells", "vec2<f32>", 8, cells)],
        params_wgsl: "w: u32, h: u32, _a: u32, _b: u32,".into(),
        params: bytemuck::cast_slice(&[w, h, 0u32, 0u32]).to_vec(),
        init: vec![Pass::new(
            "init",
            init,
            vec![Bind::ReadWrite("cells", "cur")],
            [groups(w, 64), h, 1],
        )],
        step: vec![
            Pass::new(
                "gray-scott",
                step,
                vec![Bind::Read("cells", "cur"), Bind::Write("cells", "next")],
                [groups(w, 64), h, 1],
            )
            .swap(&["cells"]),
        ],
        view: view.into(),
        view_binds: vec![Bind::Read("cells", "view_cells")],
        cells_per_step: cells,
        generations_per_step: 1,
    }
}
