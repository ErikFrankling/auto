# roads: blooms and trodden paths, to give scent a reason to exist

**Not a verdict.** The experiment was stopped early when the project moved to a wide screen. Data: five 512 runs, all seed 1, 100,000 ticks each, plus one 256 check. The planned 300,000-tick runs on seeds 1-3, the longer-range scent runs and every pheromone knockout never ran, so "does switching scent off hurt?" is unanswered.
## The rule
Blooms: part of the sun's food instead lands as a sudden deep pile. Each tile of the world gets one per period, at a random time and place. Food above `ground_cap` spoils fast, and a cell can't eat a deep pile faster than a full square, so one cell can't finish a bloom. Total food deposited matches baseline, though spoiled food is lost, so food actually available is lower. Blooms also land on barren ground. Tracks: each step adds to free field `fc.y`, already sensed, and stepping onto a worn square gets more movement from the same effort. Tracks cost and create no energy. Scent is unchanged and has no built-in meaning.
## Switches (`defaults.json`, all off by default)
| Key | Off | Most promising so far | Meaning |
|---|---|---|---|
| `bloom_share` | 0 | 0.3 | Share of sun moved into blooms. Above 0 turns on blooms, spoiling and the meal limit |
| `bloom_grid`, `bloom_every`, `bloom_radius`, `bloom_fade` | 32, 500, 6, 0.01 | same | Tile size, period, radius, spoil rate per tick. Keep radius under half the grid |
| `roads_on` | false | true | Tracks and easier steps |
| `fcy_decay`, `fcy_flow` | 0, 0 | 0.002, 0.005 | **Set these with `roads_on`**, or tracks never fade |
| `track_lay`, `track_half`, `track_cap`, `road_ease` | 0.01, 1, 4, 3 | same | Wear per step, half-effect level, most wear, extra movement on a full road |
| `roads_stats` | 0 | 1 | Measuring only. See below |

`roads_stats=1` reports in budget `x0` a choice measure: track on the faced square against the two beside it, free of own-trail and easier-step bias. `x1` is the share of cells on a bloom square. `roads_stats=2` replaces `follows_scent` with the same unbiased measure and splits emit into `x0` on blooms and `x1` elsewhere; it has only run on the software driver. Both take over x0/x1, so don't combine them with `x0_on` or `x1_on`. `roads_on` uses `fc.y`, so it can't combine with another rule using that field.

Off matches upstream: on the software driver at 128, seed 5, 1,500 ticks, population, births, deaths, budgets, diets and species hash the same as the shaders from main at 3d71e8a. This branch also carries main's shader changes since abfed62, needed to compare with `base-v2`.
## Results: 512, seed 1, at tick 100,000
| Run | Settings | Cells | Emit share | `follows_scent` (4 channels) | x0 track choice | On blooms | Lives on ground | Bite share | Behaviour explained |
|---|---|---|---|---|---|---|---|---|---|
| base-v2 | all off | 49,514 | 0.006 | -0.001 -0.003 -0.006 -0.004 | n/a | n/a | 0.78 | 0.05 | 0.05 |
| `tune/full` | share 0.3, roads | 45,884 | 0.003 | -0.000 -0.002 -0.005 -0.006 | -0.007 | 0.053 | 0.82 | 0.03 | 0.03 |
| `tune/emit` | full, `emit_power=100` | 39,675 | 0.003 | -0.003 -0.005 -0.009 -0.012 | -0.006 | 0.040 | 0.71 | 0.08 | 0.00 |
| `tune/bloomonly` | share 0.3, no roads | 54,234 | 0.004 | -0.001 -0.003 -0.005 -0.001 | 0 | 0.022 | 0.94 | 0.00 | 0.00 |
| `tune/wide` | share 0.5, grid 64, every 1000, radius 10, roads | 33,041 | 0.004 | -0.003 -0.004 -0.009 -0.004 | -0.004 | 0.028 | 0.86 | 0.02 | 0.05 |
| `tune/roadsonly` | roads, no blooms | 56,228 | 0.002 | 0.001 -0.000 -0.002 -0.002 | -0.003 | 0 | 0.75 | 0.04 | 0.12 |
## What I saw
- Cells did not start spending on emit in any arm. Share stayed at 0.2-0.4%, no higher than baseline, including with emission over three times cheaper. So in this one seed, emission cost doesn't look like the limit.
- `follows_scent` stayed near zero and mostly negative, like baseline. A moving emitter leaves scent behind it, which pushes this measure negative, so small negatives don't show avoidance.
- Cells did not choose worn squares: x0 is slightly negative in every arm with tracks.
- The only hint: with tracks, 5.3% of cells stood on blooms against 2.2% without. Treat it as a hint, since it is one seed and `wide` got 2.8%. The only chance-level reading is about 3% of squares being bloom, from the 256 check at tick 8,000 under the same bloom settings as `full`, not from these runs.
- Deception can't be judged. With no real emit or follow, there is nothing to deceive with, and the per-channel measures never ran on evolved worlds.
- Road networks can't be claimed. The only track measurement is the 256 check at tick 8,000 (`tracks.pl` on a saved world), which showed no concentration: top 5% of squares held 10.5% of the track. No 512 world was saved.
- Worlds at 100,000 ticks are young: 1 to 2 effective species everywhere, so these runs say little about coexisting ways of life.
## Verdict
None. Scent went unused in the little that ran, but one seed at 100,000 ticks with no knockout can't show that this rule fails. The untested stronger setting was `bloom_share=0.7 bloom_fade=0.02`, with and without `p2_decay=0.01 p3_decay=0.03` for longer scent reach.
