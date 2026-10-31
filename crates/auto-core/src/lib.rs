//! The engine: describe an automaton as buffers and WGSL passes, and this runs,
//! renders and reads it back on any wgpu backend, native or in the browser.

pub mod automata;
pub mod gpu;
pub mod program;
pub mod sim;

pub use gpu::{Checks, Gpu, GpuOptions};
pub use program::{Bind, Field, Pass, Program};
pub use sim::{Camera, Sim};
