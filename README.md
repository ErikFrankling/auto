# auto

Very fast cellular automata in Rust and WebGPU. The same code runs natively through
Vulkan and in a browser, and most automata are checked against a plain CPU version.

On an RX 7900 XT, Conway's Game of Life reaches about 5 trillion cell updates per
second, stays there up to tens of billions of cells, and runs just as fast in Chrome.
See [the report](https://erikfrankling.github.io/auto/) for the measurements and
videos, or [run it in your browser](https://erikfrankling.github.io/auto/app/).

## Run it

Everything needs the dev shell (`nix develop`, or Rust plus a Vulkan driver).

```sh
cargo run --release -p auto-app --bin auto-view -- lenia   # a window
cargo run --release -p auto-cli --bin auto -- list          # what exists
cargo run --release -p auto-cli --bin auto -- bench life --sizes 16384x16384
cargo run --release -p auto-cli --bin auto -- record lenia --out lenia.mp4
cargo test --release -p auto-core -p auto-cpu               # needs any Vulkan device

cd crates/auto-web && trunk serve --release                 # the browser build
```

In the window: space pauses, left and right switch automaton, up and down change
speed, the wheel zooms, `r` reseeds. The browser build needs WebGPU. On Linux it has
to be switched on: `dom.webgpu.enabled` in Firefox, or `--enable-unsafe-webgpu
--enable-features=Vulkan --use-webgpu-adapter=vulkan` for Chrome.

`./life` opens the artificial life program, building it first if anything changed.
It starts on an evolved world if one is in `/tmp/lg/saves`, otherwise a new one.

## How the code is laid out

| Crate | What it does |
|---|---|
| `auto-core` | The engine: an automaton is a list of buffers and WGSL compute passes. Builds for native and wasm. |
| `auto-core/src/automata` | The automata, one family per file. |
| `auto-cpu` | Life on the CPU (scalar, AVX2, AVX-512) and Hashlife. |
| `auto-cli` | `bench`, `record`, `cpu`, `list`, `probe`. |
| `auto-app` | The live viewer, shared by the window and the browser. |
| `auto-life-gpu` | The artificial life world: one cell per square, one small brain per cell. The `life` binary runs it headless. |
| `auto-life-app` | A window onto a life world: pan, zoom, every physics setting, every colouring, inspect any cell. Start it with `./life`. |
| `auto-web` | Browser entry points and the page. |
| `auto-vk` | The same programs through hand-written Vulkan, to measure what wgpu costs. |

## Adding an automaton

A rule that reads its 8 neighbours is three WGSL functions in
[`stencil.rs`](crates/auto-core/src/automata/stencil.rs). Anything else builds a
`Program` directly, which assumes nothing about neighbourhoods or even a grid:

- Life and Brian's Brain use bit planes and adder logic ([`life.rs`](crates/auto-core/src/automata/life.rs), [`bits.rs`](crates/auto-core/src/automata/bits.rs)).
- Rule 30 only updates one row per tick, over a circular history.
- Falling sand is one in-place pass on a block grid that shifts every tick ([`sand.rs`](crates/auto-core/src/automata/sand.rs)).
- Lenia, SmoothLife and MNCA are about 25 FFT passes, and their rule is one function ([`spectral.rs`](crates/auto-core/src/automata/spectral.rs)).
- Physarum has agents that scatter into a grid with atomics ([`agents.rs`](crates/auto-core/src/automata/agents.rs)).

Add a test in [`tests/automata.rs`](crates/auto-core/tests/automata.rs) against a CPU
version of the rule. The tests have already caught real bugs that the benchmarks
could not see.

## Reproducing the numbers

```sh
nix develop -c cargo build --release --workspace
nix develop -c scripts/bench_all.sh      # results/*.jsonl
nix develop -c scripts/record_all.sh     # videos/
nix develop -c python3 scripts/make_report.py
```

## Research

`research/notes/` has the literature and project surveys behind this, including
[the fastest published Life kernels](research/notes/fastest-life.md),
[WebGPU against Vulkan](research/notes/webgpu-vs-vulkan.md) and
[large-scale neural evolution](research/notes/neural-evolution-academic.md).
