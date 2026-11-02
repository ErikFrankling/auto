# The rules in one engine

Every rule below is off by default and can be combined with any other using `--set`. Each has its own state, so nothing is shared. Reports and logs are in `exp/NAME/`. Their numbers came from shaders before this merge, and none of the combinations has been run at scale, so treat every range as a place to start looking. Worlds saved before the merge won't load, and scripts in `exp/*/` that read saves (`snapshot.pl`, `snapshot.py`, `analyse.py`, `tracks.pl`) still expect the old layout.

## Where state lives

| Thing | Layout |
|---|---|
| Cell, 96 bytes | Words 0-15 as before, with `x0`-`x3` now `hard wound skill_g skill_f`. 16-19 `feel_u feel_d mate_near mate_kids`. 20-23 free (`x0`-`x3`) |
| Stat, 144 bytes | As before, plus `b3` (spare actions) and `measures` |
| Fields | `fc` enzyme, waste, fire, free. New `fd` tracks, three free |
| Terrain | Biome bits hold the room when biomes or walls are on |
| Inputs, 18 blocks | `i1.w` hardness. `i12` waste level, east, south. `i13` track level, east, south. `i14` practice ground, flesh, felt upkeep, felt drag. Free: `i12.w`, `i13.w`, `i15`-`i17` |
| Outputs, 7 rows | `o2.z` armour, `o2.w` clean. Free: `o5.z`, `o5.w`, `o6`, which is spare actions `x0`-`x3` behind `x0_on`-`x3_on` |
| Counters | `ctr[14]` sun taken. 16-31 free |

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

## Combinations that make no sense

- `waste_holds` without `waste_clean`: a flat tax on meals, since nobody can get the energy back. Allowed, but it isn't the conversion rule.
- `mate_pay` or `compat` with `sex_on=false`: nothing happens.
- `skill_sense` without `skill_on`, `biome_feel` without `biomes_on`, `waste_clean` without `waste_on`: their inputs and actions stay off.
- `weather_sun_on` with `biomes_on`: sun slides off the rooms it was scaled for, so `sunx_*` stops being a property of place, while tempo and drag stay put. It runs, but it's a different experiment.
- `armour_on` with `pack_on` at strong settings: both shrink bites and they multiply, so expect predators to vanish.
- `walls_on` without `biomes_on`: identical rooms, which is the control for biomes rather than a rule of its own.
- `plants_on` with low `ground_cap`: the peak moves with the cap, so sun taken up changes a lot.
- `roads_stats`, `mate_track` and `plants_meter`: measures, so don't screen them.

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

- No combination has been run on the real GPU or at 512. This merge ran only on the software driver, at 128 and below, which shows that it works and adds up, not what it does to life. `act.wgsl` now binds 18 storage buffers. That passed on the software driver, and hasn't been tried on the real one.
- Every rule's effect on species comes from its own thin runs in `exp/*/README.md`, and nearly all of those say no verdict.
- `waste_harm`, armour, pack and mass ranges are mostly guesses around one tested value.
