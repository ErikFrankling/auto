# Waste

**Very little data, no verdict.** The project changed plan before the main runs got GPU time. This rests on four 100,000-tick runs at 512, all seed 1. The planned 300,000-tick runs on seeds 1 to 3, the video and the turn-it-off tests never ran.
## The rule
Eating leaves waste in the free field `fc.y`, in proportion to the energy gained from ground, meat and bites. Waste spreads and fades slowly. Every cell loses energy in proportion to the waste on its square, whatever it is doing, and that energy is gone (`lose()`). Optionally, a share of each meal's energy stays in the waste instead of reaching the eater. A cell can then spend budget action x0 to clear waste from its square and get part of that energy back. Conversion can never return more than was eaten: the share leaves the ledger when the waste is made, comes back only when converted, and is gone if the waste fades first. Every cell has the same options and nothing is told who made the waste.
## Settings
Switches in `defaults.json`, off by default. Other values are defaults unless listed.
| Key | Default | Meaning |
|---|---|---|
| `waste_on` | false | The whole rule |
| `waste_deposit` | 0.1 | Waste per unit of energy eaten. Must not be 0, since it is divided by |
| `waste_harm` | 0.03 | Energy lost per tick per unit of waste on the square |
| `waste_holds` | 0 | Share of a meal's energy kept in the waste |
| `waste_rate`, `waste_yield` | 6, 0.5 | How fast x0 clears waste, and the share of its energy recovered |
| `fcy_flow`, `fcy_decay` | 0.025, 0.002 | Spread and fading. Changed from 0, with no effect while nothing writes `fc.y` |
**Most promising, for the screen:** `waste_on=true`. For conversion add `waste_holds=0.2` and `x0_on=true`. Good range for `waste_harm`: 0.01 to 0.08. At a whole-world level, `waste_harm * waste_deposit / fcy_decay` is roughly the share of income lost at full density. An earlier draft had this at 12.5, and its small test worlds fell to 1% density. The harm values below give 0.5, 1.5 and 4.
## Results at tick 100,000, 512 squares wide
Lifestyle numbers are averaged over cells. `burned` includes waste deaths, since a death has only four possible causes.
| Run | Seed | Density | moving | spread | neighbours | heads_to_food | x0 budget | Bitten % | Burned % | species_explain_behaviour |
|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 1 | 0.201 | 0.239 | 0.748 | 4.11 | 0.445 | 0 | 59 | 5 | 0.059 |
| Baseline | 2 | 0.359 | 0.097 | 0.946 | 5.98 | 0.601 | 0 | 7 | 19 | 0 |
| Baseline | 3 | 0.262 | 0.222 | 0.834 | 3.92 | 0.525 | 0 | 42 | 10 | 0 |
| harm 0.01 | 1 | 0.201 | 0.251 | 0.766 | 3.81 | 0.600 | 0 | 4 | 82 | 0 |
| harm 0.03 | 1 | 0.135 | 0.405 | 0.828 | 1.68 | 0.671 | 0 | 2 | 88 | 0 |
| harm 0.08 | 1 | 0.082 | 0.624 | 0.858 | 0.76 | 0.502 | 0 | 0 | 86 | 0 |
| harm 0.03 + conversion | 1 | 0.155 | 0.303 | 0.764 | 3.05 | 0.464 | 0.084 | 54 | 28 | 0 |
## What I saw
- Off matches baseline: at tick 20,000 on seed 1, cells (61,284) and all death counts are identical.
- More poison gives more movement and fewer neighbours, in order, and no world died out. Seeds differ a lot in baseline, so one seed shows a trend, not a size.
- Biting deaths nearly vanish with poison alone, from 59% to 4% or less. I don't know why. Thinner worlds may be part of it.
- With conversion, biting is back at 54% and cells spend 8% of their budget on x0. Of the two large behaviour groups, one (69% of cells) moves 0.37 with x0 0.06, the other (31%) moves 0.15 with x0 0.13. That hints at a second way to live, but I haven't shown that x0 matters, and it may be a side effect.
- Species did not separate: `species_explain_behaviour` is 0 in every waste run. 100,000 ticks may be too short, since baseline seed 1 reaches 0.2 to 0.45 later in its run.
- Not tested: whether anything actually travels. `moving` counts single cells stepping, not moving fronts or swarms, and I watched no video.
## Checks
`perl crates/auto-core/tests/waste_reference.pl` gives every cell a fixed brain and compares one shader tick on the software driver with plain arithmetic, then checks a full energy ledger with waste on. `crates/auto-life-gpu/tests/world.rs` has matching Rust tests, which have never been compiled or run because the brief forbids cargo.
## Verdict
None. Waste clearly raises `moving` on one seed and the world survives at lower density, which is enough to include it in the screen. There is no evidence yet for travelling fronts, for real species, or that conversion is a viable way of life.
