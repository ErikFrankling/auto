//! Soft life on the GPU: discs in continuous 2D that can push, stick and share, so multicellular
//! bodies become physically possible. One kind of cell, one RNN each, energy as the only currency.
//! See research/life-soft/DESIGN.md.

pub mod config;
pub mod dump;
pub mod view;
pub mod measure;
pub mod world;

pub use auto_life_gpu::world::Pacing;
pub use auto_life_gpu::{VRAM_MARGIN, free_vram, gpu_busy_elsewhere, lock_gpu};
pub use config::Config;
pub use world::World;
