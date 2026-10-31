//! What an automaton is, from the engine's point of view.
//!
//! Nothing here assumes a 3x3 neighbourhood or even a cell grid. An automaton is
//! a set of buffers and a list of compute passes over them. Life is one pass over
//! two bit buffers, Lenia is two dozen FFT passes over complex numbers, falling
//! sand is one in-place pass that runs on a shifted block grid each tick.

/// A GPU buffer the automaton owns.
#[derive(Clone, Debug)]
pub struct Field {
    pub name: &'static str,
    /// WGSL element type, for example `u32` or `vec2<f32>`.
    pub elem: &'static str,
    pub elem_bytes: u32,
    pub len: u64,
    /// Double buffered fields have a front and a back buffer that passes can swap.
    pub double: bool,
}

impl Field {
    pub fn single(name: &'static str, elem: &'static str, elem_bytes: u32, len: u64) -> Self {
        Self {
            name,
            elem,
            elem_bytes,
            len,
            double: false,
        }
    }
    pub fn double(name: &'static str, elem: &'static str, elem_bytes: u32, len: u64) -> Self {
        Self {
            name,
            elem,
            elem_bytes,
            len,
            double: true,
        }
    }
    pub fn bytes(&self) -> u64 {
        self.len * self.elem_bytes as u64
    }
}

/// How a pass sees a field. The second string is the WGSL variable name.
#[derive(Clone, Copy, Debug)]
pub enum Bind {
    /// The front buffer, read only.
    Read(&'static str, &'static str),
    /// The back buffer, writable.
    Write(&'static str, &'static str),
    /// The front buffer, or a single buffer, read and write.
    ReadWrite(&'static str, &'static str),
}

impl Bind {
    pub fn field(&self) -> &'static str {
        match self {
            Bind::Read(f, _) | Bind::Write(f, _) | Bind::ReadWrite(f, _) => f,
        }
    }
}

/// One compute dispatch, possibly repeated. Each repeat gets `frame.phase` = 0..repeat.
#[derive(Clone, Debug)]
pub struct Pass {
    pub label: &'static str,
    pub source: String,
    pub binds: Vec<Bind>,
    pub dispatch: [u32; 3],
    pub repeat: u32,
    /// Fields to swap front and back after each repeat of this pass.
    pub swap: Vec<&'static str>,
}

impl Pass {
    pub fn new(
        label: &'static str,
        source: impl Into<String>,
        binds: Vec<Bind>,
        dispatch: [u32; 3],
    ) -> Self {
        Self {
            label,
            source: source.into(),
            binds,
            dispatch,
            repeat: 1,
            swap: vec![],
        }
    }
    pub fn swap(mut self, fields: &[&'static str]) -> Self {
        self.swap = fields.to_vec();
        self
    }
    pub fn repeat(mut self, n: u32) -> Self {
        self.repeat = n;
        self
    }
}

/// A complete automaton on a specific grid.
#[derive(Clone, Debug)]
pub struct Program {
    pub id: String,
    pub width: u32,
    pub height: u32,
    pub fields: Vec<Field>,
    /// WGSL struct body for the `P` uniform, and its bytes.
    pub params_wgsl: String,
    pub params: Vec<u8>,
    pub init: Vec<Pass>,
    pub step: Vec<Pass>,
    /// WGSL defining `fn color(x: u32, y: u32) -> vec3f`.
    pub view: String,
    pub view_binds: Vec<Bind>,
    /// Cells updated by one call of the step passes, for throughput numbers.
    pub cells_per_step: u64,
    /// Generations advanced per step. Above 1 for multi-generation kernels.
    pub generations_per_step: u32,
}

impl Program {
    pub fn total_bytes(&self) -> u64 {
        self.fields
            .iter()
            .map(|f| f.bytes() * if f.double { 2 } else { 1 })
            .sum()
    }
}

/// Group count needed to cover `n` items with workgroups of `size`.
pub fn groups(n: u32, size: u32) -> u32 {
    n.div_ceil(size)
}

/// Shared helpers every automaton's WGSL can use.
pub const WGSL_COMMON: &str = r#"
fn wrap_add(a: u32, b: u32, n: u32) -> u32 { let s = a + b; return select(s, s - n, s >= n); }
fn wrap_sub(a: u32, b: u32, n: u32) -> u32 { return select(a - b, a + n - b, a < b); }
// PCG hash, a good per-cell random source (Jarzynski & Olano 2020).
fn pcg(v: u32) -> u32 {
    let s = v * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn rand3(a: u32, b: u32, c: u32) -> u32 { return pcg(pcg(pcg(a) ^ b) ^ c); }
fn unit_f(r: u32) -> f32 { return f32(r) * (1.0 / 4294967296.0); }
fn mix3(a: vec3f, b: vec3f, t: f32) -> vec3f { return a + (b - a) * clamp(t, 0.0, 1.0); }
// Multi-stop gradient over a 5-colour palette.
fn ramp(t: f32, c0: vec3f, c1: vec3f, c2: vec3f, c3: vec3f, c4: vec3f) -> vec3f {
    let s = clamp(t, 0.0, 1.0) * 4.0;
    if (s < 1.0) { return mix3(c0, c1, s); }
    if (s < 2.0) { return mix3(c1, c2, s - 1.0); }
    if (s < 3.0) { return mix3(c2, c3, s - 2.0); }
    return mix3(c3, c4, s - 3.0);
}
"#;
