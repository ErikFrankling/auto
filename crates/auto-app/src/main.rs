//! `auto-view [automaton]`: watch an automaton in a window.
//! Space pauses, left and right switch automaton, up and down change speed,
//! the wheel zooms, r reseeds, q quits.

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    auto_app::run_native(std::env::args().nth(1));
}
