# Working in this repo

This repo's GPU is the one Erik's desktop draws with. Long or huge GPU work makes his screen stutter or freeze.

- Scale every experiment from small to big, and measure each step before the next. `auto bench` and `auto record` enforce this: they check free VRAM (3 GB stays spare) and refuse anything predicted to hold the GPU for over 50 ms per step.
- Never use `--force` while Erik may be using the machine, unless he asked for that run.
- Keep each GPU submission short. A big dispatch blocks the compositor until it finishes.
- One GPU job at a time. Leave pauses between long runs.
- CPU jobs over about 10 minutes go to the homelab with `heavy`.

Every automaton needs a test against a plain CPU version of its rule in `crates/auto-core/tests/`.
