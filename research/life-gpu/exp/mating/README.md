# mating: what makes species form

**Very little data, so no verdict.** Five 512 runs of 200k ticks, stopped early by a change of plan: 2 seeds each for `compat` and `mate_pay`, 1 seed for low mutation. No combined runs, no seed 3, no turn-it-off tests.

**Rule.** `mate_pay` (share of a womb, default 0 = off): a willing cell can give birth once its womb holds `1 - mate_pay` of a full womb, if a willing neighbour has `mate_pay` of a womb to pay. That energy moves from the neighbour's womb into the child, and nothing is created. Every mate pays, so nobody spreads genes for free. Alone, a cell must still fill the whole womb, so clones stay possible. `compat` already existed: it only blocks mixing above a genome gap at birth, the cell then may clone, and it is never sensed. `mate_track` (default false) is measuring only: it writes `x2` and `x3`, which are never sensed, for `snapshot.py` to read from a saved world.

**Checks.** With the switches off, a 128 software run matches the original shader exactly (seed 4, 1500 ticks: 352 cells, 1140 births, 329 mixed, same deaths). Leftover ledger error is no worse with the rule on (-242) than off (-274). The 128 and 256 runs only prove the code runs. All switches on together has only run at 128. Baseline config matches apart from my keys and `report_tau`. No Rust test added, since the brief forbids cargo.

Results at tick 200k. Baseline is the shared `exp/base/`, not a paired control, and worlds are chaotic, so one seed against one seed is weak.

| Run | seed | cells | mixed births % | species_at 0.3 / 0.4 / 0.5 / 0.7 | explain |
|---|---|---|---|---|---|
| base | 1 | 47141 | 3 | 6.26 / 2.52 / 2.21 / 1.00 | 0.13 |
| base | 2 | 131002 | 29 | 19.22 / 1.18 / 1.00 / 1.00 | 0.00 |
| `compat=0.4` | 1 | 55615 | 17 | 5.66 / 4.28 / 1.71 / 1.00 | 0.07 |
| `compat=0.4` | 2 | 40127 | 27 | 3.76 / 3.06 / 3.01 / 2.76 | 0.28 |
| `mate_pay=0.5` | 1 | 46276 | 3 | 7.41 / 3.95 / 1.04 / 1.00 | 0.06 |
| `mate_pay=0.5` | 2 | 44338 | 3 | 5.83 / 4.99 / 1.93 / 1.00 | 0.18 |
| `mut_rate=0.03` | 1 | 56319 | 2 | 4.82 / 1.62 / 1.00 / 1.00 | 0.17 |

**What I saw**
- `compat=0.4`, seed 2 is the only run close to steady from 0.3 to 0.7, and it has the highest explain. It built up over time (0.7 threshold: 1.94 at 125k, 1.00 at 150k, 2.07 at 175k, 2.76 at 200k; explain 0.21, 0.25, 0.30, 0.28), but it has only 3 sizeable groups. Seed 1 did not repeat it: it has 1.00 at 0.7 and explain below baseline. That is one good seed out of two.
- `compat` kept sex common: 46-48% of cells were willing at 200k, against 3% with `mate_pay`. Seed 1 had 17% mixed births against 3% in base. Seed 2 base already had 29%, so this is not general.
- `mate_pay=0.5` did not make sex last. Mixed births started at 29-41% at 25k and fell to 3% by 200k. Paying half is exactly even with cloning in genes per energy, so it is not a net benefit. Species and explain are not clearly different from base.
- `mut_rate=0.03` slowed drift as expected (median gap 0.65 against 0.88 in base seed 1), but species did not get cleaner within 200k. One seed. It may need much longer runs, or sex and `compat` as well.

**Are wide hybrids already bad?** Possibly. In the two `mate_pay` runs, cells with distant parents are rarer among old cells and have fewer children. Old cells per young cell, closest parents (gap under 0.1) against farthest (over 0.8): 0.95 to 0.56 in seed 1, 0.72 to 0.13 in seed 2. Children per 1000 ticks: 5.83 to 3.58, and 6.85 to 2.60. Clones sit near 1.0, so age structure is not the cause. This is one snapshot per world and each wide bin has about 20-120 cells. Parent gap may also go along with place or lifestyle, and plain baseline was not tracked. `mut_rate=0.03` is too sparse to read. `compat` runs block these crosses, so they can't test it. No controlled cross was done.

**Do mates choose?** Not shown. Without `compat`, mates are closer than a random neighbour at the same birth (by 0.15 and 0.08 with `mate_pay`, 0.09 with low mutation, standard errors about 0.01). But willingness is inherited and relatives sit together, so this appears with no signal at all. The `compat` figures (0.29, 0.34) come from the rule itself. Voice does differ more between unrelated neighbours (gap over 0.45) than related ones in all five worlds, for example 0.71 against 1.37 in `mate_pay` seed 2, so a usable signal exists. Whether willingness responds to it was not measured.

**Verdict.** None supported. Evidence is suggestive for `compat`, with one seed of two. There is nothing yet for `mate_pay` as a payoff, or for evolved mate choice.

**For the screen, most promising first.** `compat=0.4`. Then `compat=0.4` with `mut_rate=0.03`, untested but the literature's case for low mutation. Try `compat` at 0.25 and 0.6 too, since only 0.4 was run. `mate_pay=0.5` alone looks unpromising. Above 0.5 it favours the initiator over the mate, which is untried. Keep `mate_track=false` unless reading saves.

Files: `*.jsonl` are run logs, except `snapshots.jsonl`, the `snapshot.py` output for the five saved worlds at tick 200k, in the order `mate_pay` seeds 1 and 2, `compat=0.4` seeds 1 and 2, `mut_rate=0.03` seed 1. Saves were left in `/tmp/mating/saves/`. No video was made.
