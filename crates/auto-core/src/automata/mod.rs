//! The automata. Each module builds a [`Program`] for a grid size.

use crate::program::Program;

pub mod agents;
pub mod bits;
pub mod diffusion;
pub mod life;
pub mod sand;
pub mod spectral;
pub mod stencil;

/// Everything needed to list and build one automaton.
pub struct Entry {
    pub id: &'static str,
    pub name: &'static str,
    pub blurb: &'static str,
    /// Default grid width and height for live viewing.
    pub default_size: (u32, u32),
    /// Width and height must be powers of two.
    pub needs_pot: bool,
    /// Smallest step of width.
    pub width_step: u32,
    pub build: fn(u32, u32) -> Program,
}

pub const ALL: &[Entry] = &[
    life::ENTRY,
    bits::RULE30,
    bits::RULE110,
    bits::BRAIN,
    bits::WIREWORLD,
    stencil::CYCLIC,
    stencil::HODGEPODGE,
    stencil::ANTS,
    stencil::FOREST,
    stencil::SANDPILE,
    sand::ENTRY,
    diffusion::GRAY_SCOTT,
    spectral::LENIA,
    spectral::SMOOTHLIFE,
    spectral::MNCA,
    agents::PHYSARUM,
];

pub fn find(id: &str) -> Option<&'static Entry> {
    ALL.iter().find(|e| e.id == id)
}

/// Round a requested size to one an automaton accepts.
pub fn fit_size(entry: &Entry, w: u32, h: u32) -> (u32, u32) {
    if entry.needs_pot {
        (w.next_power_of_two(), h.next_power_of_two())
    } else {
        (
            w.next_multiple_of(entry.width_step),
            h.next_multiple_of(if entry.id == "sand" { 2 } else { 1 }),
        )
    }
}

/// Build an automaton by id, or a specific kernel variant like `life.naive`.
pub fn build(id: &str, w: u32, h: u32) -> Option<Program> {
    if let Some(entry) = find(id) {
        let (w, h) = fit_size(entry, w, h);
        return Some((entry.build)(w, h));
    }
    if let Some(spec) = id.strip_prefix("brainsb.") {
        // brainsb.h32.b8, with w as the agent count.
        let (hidden, bits) = spec.split_once('.')?;
        return Some(agents::brains_blocked(
            w,
            hidden.strip_prefix('h')?.parse().ok()?,
            bits.strip_prefix('b')?.parse().ok()?,
        ));
    }
    if let Some(spec) = id.strip_prefix("brains.") {
        // brains.h32.b8, with w as the agent count.
        let (hidden, bits) = spec.split_once('.')?;
        return Some(agents::brains(
            w,
            hidden.strip_prefix('h')?.parse().ok()?,
            bits.strip_prefix('b')?.parse().ok()?,
        ));
    }
    if let Some(spec) = id.strip_prefix("mnca.") {
        let v: Vec<f32> = spec.split(',').filter_map(|p| p.parse().ok()).collect();
        return Some(spectral::mnca_shaped(
            w.next_power_of_two(),
            h.next_power_of_two(),
            &v,
        ));
    }
    life::kernel(id, w, h)
}
