# biomes: many kinds of place

**Very little data: six 512 runs, two seeds per arm, 200,000 ticks, cut short by the plan change. No verdict. It's a lead for the screen.**

**Rule.** With `biomes_on`, each of 16 rooms (4 by 4) is one of four kinds of land, and the kind sets physics every cell is subject to. `tempo` scales spending per tick, upkeep and the old-age hazard, so life runs faster and costs more. `drag` divides the chance a paid-for step succeeds. `sunx` scales sun and `rotx` scales meat rot. The kind sits in `fc.w` and the room in the terrain biome bits. Cells aren't told the kind. With `biome_feel`, input `x0` is the upkeep just paid and `x1` the cost of a step. `walls_on` adds rock walls between rooms with one pass per wall, `wall_pass` squares wide.

**Settings.** Defaults are off: `biomes_on=false`, `walls_on=false`. Use `biome_feel=true`, `wall_thick=6`. Most promising: `biomes_on=true walls_on=true wall_pass=64`, plus the kinds below, which are the `defaults.json` values. Nearly isolated: `wall_pass=4`.

| Kind | tempo | sunx | drag | rotx |
|---|---|---|---|---|
| cold | 0.5 | 0.5 | 1 | 0.4 |
| hot | 2 | 2 | 1 | 2 |
| swamp | 1 | 1.2 | 4 | 1 |
| barren | 0.7 | 0.15 | 3 | 1 |

Fair sun: sun is scaled per open square, so walls cut the total. Raise `sun` by old open squares / new open squares. At 512: 0.0546 for `wall_pass=4`, 0.0521 for `wall_pass=64`. Total sun then matched baseline within 0.1% at seed 1, the one seed checked against it. At 128 walls take about 30% of the world, so re-measure or thin them. The rule uses `fc.w`, `x0` and `x1`, so it clashes with any other rule using those.

**Results.** Species columns are means of the logs from tick 120,000 to 200,000. Baseline is `base-v2`, same ticks. Room columns are from the world saved at 200,000: distance between room averages (excluding rooms sharing a wall), same kind / different kind. Labels are a formula on the room, so in identical rooms they mean nothing physical, which makes those runs a control. Baseline has no saved rooms.

| Arm (seed) | Species 0.4 | Species 0.5 | `species_explain_behaviour` | `founder_lines` | Genome gap | Behaviour gap |
|---|---|---|---|---|---|---|
| Biomes, wide passes (1) | 7.7 | 4.4 | 0.30 | 5.0 | 2.02 / 2.96 | 0.23 / 0.31 |
| Biomes, wide passes (2) | 6.1 | 3.3 | 0.20 | 3.4 | 2.11 / 3.17 | 0.19 / 0.36 |
| Biomes, narrow passes (1) | 8.1 | 3.2 | 0.30 | 3.4 | 3.51 / 3.52 | 0.27 / 0.37 |
| Biomes, narrow passes (2) | 11.9 | 8.7 | 0.33 | 4.4 | 4.13 / 4.05 | 0.29 / 0.44 |
| Identical rooms, narrow passes (1) | 2.0 | 1.3 | 0.13 | 2.4 | 1.55 / 1.72 | 0.31 / 0.32 |
| Identical rooms, narrow passes (2) | 1.8 | 1.4 | 0.06 | 2.2 | 1.98 / 1.95 | 0.21 / 0.23 |
| Baseline (1, 2, 3) | 2.3, 1.1, 2.5 | 1.5, 1.0, 1.8 | 0.17, 0.01, 0.09 | 1.6, 1.8, 2.2 | | |

**What I saw.** Biome worlds held more genome groups, also at threshold 0.5, explained more behaviour by species and kept more founder lines than identical rooms and baseline, on both seeds. Same-kind rooms behaved more alike in all four biome runs but not in identical rooms. With narrow passes genomes split by room, and same-kind rooms were no closer than others, so that split looks like drift. With wide passes same-kind rooms had closer genomes on both seeds, which is what place shaping genes under migration would look like. In those worlds 32% and 55% of cells had genomes nearest another room's average, against 25% and 27% with narrow passes.

**Why this is not proof.** Some behaviour differences may be direct physics: swamp cells move less because drag blocks steps, and `moving` can't separate that from evolved choice. Barren land stays thin (density 0.04 to 0.08). Lifestyles remain mostly ground eating. `species_explain_behaviour` swings a lot between log lines.

**Not done, because these runs were stopped in the queue:** seed 3 for biomes, the identical-rooms control with wide passes, turning the rule off in a saved world, a longer run to see whether differences last, and the video. So the wide-pass genome result has no matching control, whether they stay different has no evidence, and no test shows the behaviours matter to the population. `biome_feel=false` was never run. Only one end snapshot per run exists, so nothing here tracks migrants over time.

**Checks.** On the software driver at 128, the switches off give the same population, birth, death, founder and species numbers as the built-in shaders, rechecked on the final shaders. Sealed walls at 512 gave 23 separate regions. The energy ledger at 128 (before the merge with `main`) had the same small negative gap of about 1,000 units on and off, so no energy appears from nothing.

**Verdict.** None yet. Worth including in the screen, with both switches tested separately. `analyse.py` produced the room numbers.
