# The rules in one engine

Every rule below is off by default and can be combined with any other using `--set`. Each has its own state, so nothing is shared. Reports and logs are in `exp/NAME/`. Their numbers came from shaders before this merge, and none of the combinations has been run at scale, so treat every range as a place to start looking. Worlds saved before the merge won't load, and scripts in `exp/*/` that read saves (`snapshot.pl`, `snapshot.py`, `analyse.py`, `tracks.pl`) still expect the old layout.

## Where state lives

| Thing | Layout |
|---|---|
| Cell, 96 bytes | Words 0-15 as before, with `x0`-`x3` now `hard wound skill_g skill_f`. 16-19 `feel_u feel_d mate_near mate_kids`. 20-23 `mineral coat bound key`. None free |
| Stat, 144 bytes | As before, plus `b3` (spare actions) and `measures` |
| Fields | `fc` enzyme, waste, fire, free. `fd` tracks, mineral on the ground, mineral richness of the square (never changes), steps taken here (measuring only) |
| Terrain | Biome bits hold the room when biomes or walls are on |
| Inputs, 18 blocks | `i1.w` hardness. `i12` waste level, east, south. `i13` track level, east, south. `i14` practice ground, flesh, felt upkeep, felt drag. `i15` coat in front, own coat, each as cos and sin. `i16` mineral here, east, south, own store. `i17` bites seen east, south, touching this cell, on the body in front. Free: `i12.w`, `i13.w` |
| Outputs, 7 rows | `o2.z` armour, `o2.w` clean. `o5.z` coat, `o5.w` key. `o6` is spare actions `x0`-`x3` behind `x0_on`-`x3_on`, where `x0` is also mineral uptake. Rules also read `o6.y` (a gift carries mineral) and `o6.z` (give birth now), outside the budget, so don't turn on `x1_on` or `x2_on` with them. Free: `o6.w` |
| Counters | `ctr[14]` sun taken. 16-31 free. 32-95 are gauges, for measuring only and not saved with a world |
| Snapshot word | As before, plus bit 5: this body really bit last tick, at least `gang_effort` of work |

An input whose rule is off reads 0 and isn't computed. An action whose rule is off gets no budget. To add an input or output, also update `input_live` or `output_live` in `config.rs`.

## Switches

**`mass_on`.** Stored energy divides move chance by `1 + mass_drag * e / mass_scale`, and a step costs `mass_step_cost * (1 + m)`. A heavier cell can shove a lighter living one into its old square for `mass_push_cost * (1 + m + their m)`. Screen: `mass_drag` 0.3-3. Costs are guesses, 0.01-0.1 and 0.02-0.2. A shoved cell that hadn't acted can lose that tick's turn. Ran at 128 and 256 only.

**`armour_on`.** Action `o2.z` builds hardness (0-1) at `armour_build` per unit of work, and it fades by `armour_fade` a tick. Hardness divides bites and heat by `1 + armour_resist * h`, divides move chance by `1 + armour_drag * h`, and costs `armour_upkeep * h` a tick. Children start soft. Input `i1.w`. Screen: all five settings within a factor of 2 to 3 of default. Ran at 128 only, and at half the fade its report claimed (fixed, see below).

**`pack_on`.** A bite does `pack_intact` of its damage on an intact body, rising to full on a wounded one. Bite work raises the wound by `pack_wound_rate * work`, and wounds keep `pack_wound_keep` a tick. Not sensed. Screen: intact 0.05-0.5, keep 0.5-0.9, rate 4-24. Ran at 128 only, where it cut diversity.

**`waste_on`.** Eating leaves waste in `fc.y` at `waste_deposit` per unit eaten. It spreads and fades (`waste_flow`, `waste_decay`), and takes `waste_harm` per unit from any cell on it. Waste deaths count as `burned`. Input `i12`. **`waste_clean`** enables action `o2.w`, which clears waste at `waste_rate`. **`waste_holds`** keeps that share of each meal's energy in the waste, and clearing returns `waste_yield` of it. Screen: harm 0.01-0.08, holds 0.1-0.4 with clean on. At full density `waste_harm * waste_deposit / waste_decay` is about the share of income lost, so keep it under about 4.

**`plants_on`.** Growth is `sun * (plants_peak * 4d + plants_trickle) * (1 - d)` with `d` food over cap, and ground spreads at `plants_flow`. Screen: peak 1.5-3, since 1.5 took up 24-38% less sun than controls. `plants_meter` logs `sun_taken`, which over ticks in a log step is sun per tick. It used to borrow `rained`.

**`weather_sun_on`.** The sun map slides `weather_drift` squares a tick along x and 0.618 times that along y. Screen: 0.003-0.03. Drift 0.3 looks too strong, and 0.003 and 0.01 have never run. Sliding off rock can change total sun, so watch `sun_taken`.

**`weather_fire_on`.** Fire in `fc.z` lights at `weather_fire_rate` per square per sweep, only on food of at least `weather_fire_fuel`. It spreads at `weather_fire_spread`, burns `weather_fire_burn` of the ground and gives `weather_fire_heat` of heat per sweep, and fades by `weather_fire_fade`. A burned square rests `weather_fire_rest` ticks. Cells feel it as heat. Screen: rate 1e-9 to 1e-8 at 512, times (512/size)², rest 500-4000. Nobody has measured how much really burns.

**`mate_pay`** (number, 0 off). A willing cell with `1 - mate_pay` of a womb gives birth if a willing neighbour holds `mate_pay` of a womb, which moves to the child. Screen: 0.3-0.7. 0.5 alone looked poor, above 0.5 is untried. **`compat`** (number, 0 off) blocks mixing above that genome gap, the cell can still clone, and it's never sensed. Screen: 0.25-0.8, with 0.4 the lead on one seed of two, perhaps with `mut_rate=0.03`. `mate_track` is measuring only.

**`skill_on`.** Practice in `skill_g` and `skill_f` moves toward what the cell really ate, at `skill_learn`, and fades by `skill_fade`. It sets the share of food used: `skill_naive + (1 - skill_naive) * practice^skill_curve`, and the rest is lost. **`skill_sense`** feeds it to `i14.xy`. `skill_inherit` is how much a child gets. Screen: naive 0.6-0.95, curve 1-4, inherit 0-1. The old default of 0.3 killed fresh worlds, so it's now 0.6. Naive 0.6 at 512 removed predators.

**`biomes_on`.** 16 rooms of four kinds. `tempo_*` scales spending, upkeep and hazard, `drag_*` divides move chance, `sunx_*` scales sun and `rotx_*` meat rot. **`biome_feel`** (default on, needs `biomes_on`) feeds last tick's upkeep and drag to `i14.zw`. **`walls_on`** adds rock walls between rooms, `wall_thick` wide, with a pass `wall_pass` wide. Screen: `wall_pass` 4-64 at 512, scaled to world size, with thickness 2-8 scaled the same. Walls cut total sun, so raise `sun` to match (0.052 at 512). At 128 walls take about 30% of the world. This has the best lead so far, more species on both seeds, but no turn-off test or seed 3.

**`bloom_share`** (number, 0 off). That share of the sun lands as piles, one per `bloom_grid` tile per `bloom_every` ticks, `bloom_radius` wide. Food over the cap spoils at `bloom_fade`, and a cell eats a pile no faster than a full square. Screen: share 0.3-0.7, fade 0.005-0.02. Keep radius under half the grid. **`roads_on`.** Each step adds `track_lay` to `fd.x`, up to `track_cap`, and move chance rises by up to `road_ease` on worn ground. It spreads and fades by `track_flow` and `track_decay`. Input `i13`. Screen: decay 0.0005-0.01, flow 0.001-0.02, ease 1-6. `roads_stats` 1 logs `measures` (track choice, on bloom, emit on bloom, emit off bloom), and 2 also makes `follows_scent` unbiased.

**`x0_on` to `x3_on`.** Spare actions for the next rule. On their own they burn energy for nothing, so don't screen them.

## Rules for depth of strategy

`PREDICTIONS.md` has the reasoning, the calculations and what would show each is wrong. It was written before any of them ran. Settings below are where it starts, and none has run at scale.

**`gang_on`.** A neighbour of the target that really bit it last tick, read from the snapshot, counts as an open side, not a blocked one, and its energy joins the attacker's in `force = 2 S^g / (S^g + te^g)`, with `g` as `gang_sharp`. "Really" means at least `gang_effort` of work, since a resting cell spends a little on everything. **`gang_sense`** feeds `i17`. Start: sharp 2, `max_energy=80`. Sharp 1 is the note's formula, and it lets a lone small cell profit from a fat one. Screen: sharp 1-3.

**`coat_on`.** A coat is a point on a circle, moved at most `coat_rate` turns a tick toward `o5.z * coat_span`, at `coat_cost` energy per turn moved. A bite's key is `o5.w * coat_span`, free and instant. Damage is multiplied by `coat_floor + (1 - coat_floor) * m^coat_sharp`, with `m` 1 at a perfect match and 0 half a turn off. A child starts with its parent's coat, and a newcomer with a random one. Neither is sensed. **`coat_sense`** feeds `i15`. Start: sharp 8, floor 0.05, rate 0.002, cost 2, no sense. Floor must stay under about 0.08, or mismatched bites pay anyway. Screen: sharp 2-32, floor 0.01-0.15.

**`mineral_on`.** Mineral lies on the ground in `fd.y`, at a mean of `mineral_start` per open square, shaped by richness in `fd.z`: noise of its own, contrast `mineral_contrast`, pushed away from sun by `mineral_anti`. Action `x0` takes it up at `mineral_rate`, into a store of at most `mineral_cap`. A birth takes `mineral_birth` from the parents, however big the child, and it stays bound in the child until it dies. Death returns store and body to the ground. A bite moves the same share of the target's store as of its energy, and what the biter can't hold falls to the ground. With `o6.y > 0` a gift carries `mineral_give` per unit of work in place of energy. Ground mineral trades between pairs of squares toward each one's richness at `mineral_flow`, so patches last. Newcomers bring none. All amounts are whole grains of 1/4096, so the total is exact. Input `i16`. Start: start 0.25, birth 1, flow 0.005, anti 0.5, which limits rich land by mineral and poor land by energy. Screen: start 0.08-1.2, flow 0.001-0.05.

**`birth_choice`.** With `o6.z > 0` and a womb over `birth_min` and over a body, a cell gives birth and the child gets the womb, less a body. A womb fills up to `womb_cap`, and `i0.y` becomes `womb / (womb + WOMB)`. Start: min 1.5, cap 40. Screen: min 1.2-3, cap 10-60.

**`age_metabolic`.** The old-age chance is `hazard * (age_rest + (1 - age_rest) * spent / spend)`, using what the cell spent, rolled at the end of its tick and acted on at the start of the next. Start: rest 0.05. Screen: 0.02-0.5.

**`depth_stats`** (on) and **`trail_stats`** (off). Measures, so don't screen them. They have no effect on a world. `depth_stats` samples each cell one tick in 64, and `trail_stats` adds one write per step and reads `fd` at each log point.

### What the log and analysers say about depth

| Where | Field | Meaning |
|---|---|---|
| Log | `group_hunting` | By `n`, others that bit the same target last tick: `bites`, `gain_per_bite`, `target_energy`. `extras` is the mean `n`, `extras_control` is the same count on a body next to the biter it didn't pick, and `extras_chance` is what bites at random across all bodies would give. `group_advantage` is gain with company over gain alone |
| Log | `birth_limit` | Share of sampled cells held back by energy, mineral, both, room, or waiting by choice |
| Log | `child_size_hist`, `child_energy` | Child energy in bins under 1, 1-2, 2-4, 4-8, 8-16 and over, and its mean |
| Log | `coat` | `effective` coats in use, `key_match` of real bites against `key_chance`, `key_tracking` from 0 for random to 1 for as good as any one key could do |
| Log | `mineral` | Per open square: on the ground, in stores, bound in bodies |
| Log | `trails` | With `trail_stats`, share of steps on the busiest 5% of squares since the last line, and that share for random steps |
| Log | `follows_others_scent` | Scent ahead against the sides, for cells that hardly emit, by channel. 0 is chance |
| `life-analyse` | `bite_links` | Who bites whom by strategy at one moment, and how many real links |
| `life-analyse` | strategies' `income`, `share_of_flow` | Where each strategy's energy comes from, and its share of the world's |
| `life-analyse` | `colony_sizes`, `colony_size_explained_by_sun`, `mineral_land` | `log10_sd`, power-law `alpha`, `ks` and `decades`. How much terrain explains size. Who lives where across sun and mineral |
| `life-history` | `fine_groups`, `persistent_1pct`, `persistent_3pct`, `distinct_3pct_over_time` | Counts from 12 clusters with near centres merged, so they don't lean to 2. Read these, not `strategies_k` |
| `life-history` | `coexistence` | At 16, 32 and 64 squares, the chance two neighbours differ in strategy over the chance two cells anywhere do. 1 is side by side, 0 is separate places |
| `life-history` | `colony_travel`, `strategy_travel` | Whether colonies and strategies' centres move, in squares per 10,000 ticks and own radii |

`scripts/life_screen.py` runs them all, with binaries from `LIFE_BIN`, `LIFE_ANALYSE` and `LIFE_HISTORY`, and `rank` keeps the best worlds per measure, with no total score.

## Combinations that make no sense

- `waste_holds` without `waste_clean`: a flat tax on meals, since nobody can get the energy back. Allowed, but it isn't the conversion rule.
- `mate_pay` or `compat` with `sex_on=false`: nothing happens.
- `skill_sense` without `skill_on`, `biome_feel` without `biomes_on`, `waste_clean` without `waste_on`: their inputs and actions stay off.
- `weather_sun_on` with `biomes_on`: sun slides off the rooms it was scaled for, so `sunx_*` stops being a property of place, while tempo and drag stay put. It runs, but it's a different experiment.
- `armour_on` with `pack_on` at strong settings: both shrink bites and they multiply, so expect predators to vanish.
- `walls_on` without `biomes_on`: identical rooms, which is the control for biomes rather than a rule of its own.
- `plants_on` with low `ground_cap`: the peak moves with the cap, so sun taken up changes a lot.
- `roads_stats`, `mate_track` and `plants_meter`: measures, so don't screen them.
- `coat_sense` or `gang_sense` without their rules: their inputs stay off.
- `x1_on` or `x2_on` with `mineral_on` or `birth_choice`: a gift's carry or the birth signal would also be budget. It runs, but one output means two things.
- `mineral_on` with `--seed-from`: the planted bodies add to the new world's mineral. The total is then fixed from there on, not equal to `mineral_start`.
- `birth_choice` with `mate_pay`: pooling still goes by `WOMB`, not by the brain's signal.
- `gang_on` with `exposure=false`: force still adds up, and nothing counts as blocked.

## Changed from the experiment branches

- Armour faded at 0.25% a tick, not 0.5%, because a key ending in `_decay` gets a square root meant for fields updated twice a tick. It's now `armour_fade`, and it does what it says.
- `plants_meter` overwrote the `rained` count, and skills and `roads_stats` overwrote budget slots in the log. All three now have their own: `sun_taken`, `hardness`, `wound`, `practice` and `measures`.
- Waste used `x0` and skills `x2`/`x3`, as did others, so they clashed. Each rule now has its own words and inputs. `fcy_*` is now `waste_*` and `track_*`, and `waste_clean` replaces `x0_on` for clearing.
- Biomes kept the kind of place in `fc.w`, which `fcw_flow` could smear. It now reads terrain. Fire no longer has flow and decay settings that could smear it.
- `weather_fire_rest=0` and `waste_deposit=0` no longer divide by zero.
- `mate_pay` was written on older shaders that had no count of births with no slot and no blank measurements at birth. Both are back.
- The diet log counted ground and meat before waste's share and bites after. Now all three are what the cell kept, after poor skill too.
- With biomes on, armour upkeep scales with the tempo of the place, like all other upkeep. Mass step costs do not.

## Still unverified

- `gang_on`, `coat_on`, `mineral_on`, `birth_choice` and `age_metabolic` have only run on the software driver at small sizes. `act.wgsl` now holds bigger brains and more reads, so check `gpu_ms_per_tick` on the real GPU.
- On the software driver, once the shader was large enough, a death that wrote `cell[s].e` and `cell[s].pos` as two fields stored 0 for `pos`. Writing the whole record fixes it there, and the cause isn't known, so the real GPU needs `dead_cells_are_marked_dead_with_everything_on` run on it.
- No combination has been run on the real GPU or at 512. This merge ran only on the software driver, at 128 and below, which shows that it works and adds up, not what it does to life. `act.wgsl` now binds 18 storage buffers. That passed on the software driver, and hasn't been tried on the real one.
- Every rule's effect on species comes from its own thin runs in `exp/*/README.md`, and nearly all of those say no verdict.
- `waste_harm`, armour, pack and mass ranges are mostly guesses around one tested value.
