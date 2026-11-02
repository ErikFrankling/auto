# Weather: drifting sun and fire

**No verdict. The data is too thin to support one.** The experiment was stopped early when the project moved to a large screen. Only 5 GPU runs at 512 finished, all seed 1, all 100,000 ticks, one setting each. The 300,000-tick runs on seeds 1, 2 and 3, with their controls and the combined rule, never got a GPU turn and were cancelled.

## The rule
Two separate switches in `sweep.wgsl`, both off by default. `act.wgsl` is untouched.

- **Drifting sun** (`weather_sun_on`): the whole sun map slides across the wrapped world at `weather_drift` squares per tick along x and 0.618 times that along y.
- **Fire** (`weather_fire_on`, in `fc.z`): a square lights only if its ground food is at least `weather_fire_fuel`, either by itself or from a burning neighbour. It then burns its ground fast and gives off heat, and the heat does the hurting. Cells can already sense heat, and its damage already goes through `lose()` and counts as `burned`. Burned ground food also goes through `lose()`. Rock and thin ground stop a front. A burned square can't relight for `weather_fire_rest` ticks, so fronts die out rather than circling. Food does regrow in that time. Meat doesn't burn.
## Settings
Defaults are the values to start from, not proven ones: `weather_drift` 0.01, `weather_fire_rate` 1e-8 (chance per square per sweep, two sweeps a tick), `weather_fire_spread` 0.5, `weather_fire_fuel` 3, `weather_fire_burn` 0.5, `weather_fire_fade` 0.7, `weather_fire_heat` 8, `weather_fire_rest` 2000.
For the screen, try `weather_drift` 0.003 to 0.03 and fire rates 1e-9 to 1e-8 at 512. Drift 0.3 looks too strong. In smaller worlds, multiply the fire rate by (512/size)² to keep the same number of sparks per world. One spark can burn all the connected thick ground, so this is a guess that needs measuring. Drift 0.003 and 0.01 never ran on the GPU.
## Results: seed 1 at 100,000 ticks, one run each
`burned` covers the last 25,000 ticks and includes ordinary heat fights.

| Run | cells | burned | species at 0.3 | species at 0.5 | explain behaviour | groups ≥1% |
|---|---|---:|---:|---:|---:|---:|
| baseline (older binary, see below) | 49,514 | 303,193 | 3.8 | 1.3 | 0.06 | 2 |
| sun, drift 0.03 | 87,404 | 2,215,738 | 2.4 | 1.0 | 0.00 | 1 |
| sun, drift 0.3 | 50,363 | 939,277 | 1.1 | 1.0 | 0.00 | 1 |
| fire, rate 1e-9 | 48,091 | 111,308 | 2.4 | 2.0 | 0.16 | 3 |
| fire, rate 1e-8 | 54,638 | 544,938 | 2.2 | 1.2 | 0.06 | 2 |
| fire, rate 1e-7, rest 500 | 50,372 | 1,487,267 | 6.1 | 1.0 | 0.04 | 3 |
## What I saw
- Fire works mechanically. On the software driver at 128, snapshots show rings spreading through thick ground, stopping at thin ground and rock, and a second wave only after the rest period.
- Drift 0.3 had about 1.1 species at every log line, against 1.3 to 2.4 for drift 0.03. That hints at lost diversity at the strong end, since sun this fast averages out to nearly uniform. One seed only.
- Fire at 1e-9 scored highest here, but baseline swings as much by itself on seed 1: explain behaviour went from 0.00 at 125,000 ticks to 0.18 at 245,238. I read this as noise.
- Baseline diversity on seed 1 mostly appears after 150,000 ticks, so 100,000 ticks is probably too short to show any effect.
## What this can't tell you
- Whether weather keeps several ways of living going, or creates more change over time. I didn't measure change over time, and there's no second seed.
- The baseline row came from an older build. The binary was rebuilt before my runs, and none of my controls ran, so even this comparison isn't like for like.
- How much actually burns. The log has no fire count, and `burned` can't separate fire from heat fights.
- Whether drift keeps total sun the same in a real world. The CPU test shows the total is preserved on a plain wrapped grid, but rock squares block some sun, and the ledger isn't read out. Drift may also deliver more food, because sun wastes less on ground already at its cap. Watch for this: drift 0.03 had 87,404 cells against 49,514, on one seed.
- No turn-off test on a saved world was done.
## Checks
- Switches off match baseline: same binary, seed 1, 128, 300 ticks on the software driver, against the shaders from before this change. Every summary number matched (cells, births, deaths, species, world averages). The per-group tables differ, but they differ just as much between two runs of the old shaders.
- `crates/auto-core/tests/weather.rs` is a CPU version of the rule. All 5 tests passed once, built with bare `rustc` on the code as committed. It wasn't run through cargo. A later rebuild failed because the linker had gone from this machine, so it wasn't re-run after the last settings change, which it doesn't depend on. It pins expressions in the shader but doesn't compare against GPU output.
- `check256.jsonl`: short GPU check with both switches on. It ran without errors.
