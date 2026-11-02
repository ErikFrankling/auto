# skills: practice makes specialists

**Very little data, and no verdict.** The plan changed before this finished. Five 512 runs completed: rule on at one strength on seeds 1 and 2, two variants on seed 1 only, one control on seed 1. No seed 3, no knock-out, no weaker strength, no test on an evolved world, no video. Those six runs were queued and then cancelled. All numbers are from one run per setting except the main one.

**Rule.** `c.x2` is practice at grazing, `c.x3` at eating bodies (meat and biting). Each tick practice moves toward the mix of what the cell really did, at a rate of `skill_learn` times the share of budget used on feeding, and all of it fades by `skill_fade`. Because practice moves toward a point whose parts sum to 1, `x2 + x3` can never pass 1: gaining one takes it from the other. Practice sets how much of the food eaten the cell can use: `skill = naive + (1 - naive) * practice^curve`. It multiplies ground gain, meat yield and bite yield. The maximum is the plain yield, so it only wastes food and cannot make energy; the waste goes through `lose()`, and a bite's waste through the existing spill and `lose()`. The victim loses the same either way. Failed bites and empty squares (below `skill_min_food`) earn no practice. No cell is given a role.

**Inheritance.** Default is none (`skill_inherit=0`): every cell starts unpractised. If the question is whether specialists split genetically, inherited practice would blur it, since a lineage could differ by culture alone. `skill_inherit=1` is the culture version.

**Switches, all in `defaults.json`** (JSON has no comments, so settings are noted here): `skill_on` false, `skill_sense` false (feeds x2 and x3 to inputs `i1.w` and `i12.w`), `skill_naive` 0.3, `skill_curve` 2, `skill_learn` 0.05, `skill_fade` 0.002, `skill_inherit` 0, `skill_min_food` 0.01.

**Settings I used:** `skill_on=true skill_sense=true skill_naive=0.6`, others default. Note the default naive of 0.3 is not viable from a fresh start: at 128 on the software driver the population fell to 2 cells by tick 4000 (naive 0.1, curve 3: 0 cells). Naive 0.6 and 0.8 survived there. That is a check that it runs, not a result. For the screen I suggest naive 0.6 to 0.95, curve 1 to 4, inherit 0 and 1. No setting looked promising for the goal. Naive above 0.6 is untested at 512, and that is where any effect short of wiping out predators would have to be.

| 512, last log line | Tick | Cells | Species (0.4) | Explain | Cells at 50%+ from others | Top group: from others, ground/flesh practice |
|---|---|---|---|---|---|---|
| control seed 1 (rule off, own shaders) | 297k | 36,670 | 2.13 | 0.271 | 20% | 0.43, none |
| base-v2 seed 1 (shared baseline) | 300k | 43,950 | 2.01 | 0.215 | 18% | 0.43, none |
| base-v2 seed 2 (shared baseline) | 300k | 131,004 | 1.89 | 0.115 | 1.6% | 0.06, none |
| main seed 1, naive 0.6 | 300k | 102,234 | 1.05 | 0.000 | 0.13% | 0.01, 0.54/0.02 |
| main seed 2, naive 0.6 | 250k | 103,746 | 1.04 | 0.000 | 0.08% | 0.01, 0.53/0.03 |
| inherit seed 1, naive 0.6, inherit 1 | 300k | 80,617 | 1.18 | 0.000 | 0.25% | 0.01, 0.60/0.04 |
| sharp seed 1, naive 0.6, curve 4 | 300k | 52,401 | 1.09 | 0.000 | 0.10% | 0.00, 0.70/0.01 |

Seed 2 has no control of my own, only base-v2, which ran at a different log interval. On seed 1 the control and base-v2 differ (explain 0.27 to 0.34 against 0.10 to 0.25 from tick 150k), so treat that gap as run-to-run noise.

**What I saw.** In all four runs with the rule on, from tick 150k, nearly every cell lives on grazing and groups do not differ in how they live. Explain was 0 in most log lines, with short spikes that did not last (0.293 on main seed 2 at 200k, then 0; 0.159 and 0.125 on inherit seed 1 at 200k and 225k, then 0). Two humps in `from_other_cells_hist` never appeared. On seed 1 the control keeps two equal groups, one at 43% from others and one at 8%, and the rule removes that split. Bite deaths in the 25k ticks up to tick 250k: 140,894 with the rule, 5,326,896 in the control. Population more than doubled and the top group's age rose from 633 to 2,290. On seed 2 the baseline was already a grazing world, and explain went from about 0.1 to 0.

Flesh specialists can exist: lifestyles with flesh practice of 0.5 to 0.84 appear, but at about 0.1% or less. Curve 4 pushed grazing practice up from 0.54 to 0.70, so selection does respond to the rule, toward more commitment to grazing. Inheriting practice changed little in this one run.

**What this supports.** At naive 0.6, on two seeds from a fresh start, practice gave everyone grazing, not disruptive selection. It does not show that the idea fails in general. A likely reason, not tested: grazing is always available, a would-be predator starts unpractised with a bite yield of 0.3 instead of 0.5, and the rule hits the 40-to-60% mixed feeders that make up the baseline's predators. So there is a valley that nothing crosses. How strong the effect must be before a split shows is unanswered: the one strength tested at 512 is already past the point of removing predators, and the weaker one never ran. Whether the rule would split a world that already has predators (load a saved world, then switch it on) is also unanswered, and is the most useful test for the screen.

**Checks.** Rule off matched unchanged shaders on cells, births and deaths at 128 on the software driver, on the binary current at the end (192 cells, 621 births at tick 1500), and the rule still runs on it. The rule compiled and ran on GPU at 256. `practice_matches_plain_rule` and `energy_adds_up_with_practice` are in `crates/auto-life-gpu/tests/world.rs` but have **not been run**, because the brief forbids cargo. So the energy ledger is checked by reasoning only.

**Combining with other rules.** The rule uses `x2` and `x3`. `skill_sense=true` takes over the inputs normally fed by `x0` and `x1`. With the rule on, the log's `budget.x0` and `budget.x1` show ground and flesh practice instead of those actions' budget, and x0 and x1 actions would be misreported (the behaviour measure leaves these two slots out, so it is unaffected). The screen should not combine this with another rule that uses those slots without changing that.
