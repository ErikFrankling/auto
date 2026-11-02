# plants: the ground is alive

**Thin data: 3 runs with the rule and 2 controls at 512, one setting, one final frame each, no video, no turn-off test. The owner ended the experiment for a wider screen, so this is a first look, not a verdict.**

Rule: with `plants_on`, growth per square is `sun * (plants_peak * 4d + plants_trickle) * (1 - d)`, where `d` is food over `ground_cap`. Middling cover grows fastest, bare ground only by a trickle, full ground not at all. Food also spreads between neighbouring squares at `plants_flow`. No roles: all ground behaves the same and all cells eat it the same way.

Settings, all in `defaults.json`, off by default: `plants_on` false, `plants_peak` 1, `plants_trickle` 0.001, `plants_flow` 0.001, `plants_meter` false. Runs used `plants_on=true plants_peak=1.5`, 300,000 ticks, seeds 1 to 3. `plants_meter` turns `rained` into a measure of sun taken up (`rained` / ticks in a log step), because the ledger isn't logged. Use it only for measuring.

Means from tick 150,000 on. Control is my rule-off run on seeds 1 and 2; seed 3 is the shared baseline, which has no meter and only 6 log lines in that range.

| Seed, run | cells (low to high) | explain | species at 0.5 | bite deaths | moving | age | sun per tick |
|---|---|---|---|---|---|---|---|
| 1 off | 47,357 (41,704 to 54,390) | 0.153 | 1.85 | 70% | 0.27 | 505 | 11,402 |
| 1 on | 91,338 (74,649 to 114,629) | 0.049 | 1.04 | 8% | 0.03 | 2,160 | 7,077 |
| 2 off | 130,052 (122,109 to 131,030) | 0.013 | 1.02 | 3% | 0.02 | 2,447 | 8,047 |
| 2 on | 36,204 (28,513 to 48,099) | 0.080 | 1.34 | 56% | 0.21 | 501 | 6,103 |
| 3 base | 53,044 (42,918 to 67,831) | 0.113 | 2.41 | 42% | 0.24 | 460 | not metered |
| 3 on | 25,504 (19,368 to 32,001) | 0.008 | 1.12 | 23% | 0.26 | 487 | 5,435 |

What I saw:

- **The comparison isn't fair yet.** Peak 1.5 took up 24% to 38% less sun than the controls on the same seed, against a nominal 12,339 per tick. I chose 1.5 from short 128 runs (peak 1: 444 per tick, peak 2: about 1,000, nominal 789). Fewer cells on seeds 2 and 3 may be partly this, not the rule's shape.
- **Behaviour measure:** lower with the rule on seeds 1 and 3, higher on seed 2. Seed 2 compares against a world of one sedentary species, the weakest control. No sign of more ways of living.
- **Predators:** no sign the rule makes room for them. Bite deaths fell on seeds 1 and 3 and rose on seed 2, the seed whose control has almost none. With seed 1 and 2 swapping character, this looks like the same seed-to-seed split as baseline: a mat of sitters or a moving, biting world.
- **Crashes:** no world-wide wipe. After tick 25,000 the lowest counts were 38,352, 25,918 and 19,368. The widest swing is seed 1 on, about 1.5 times low to high late on. Regional crashes were not measured.
- **Waves, refuges, prudent grazers:** not tested. The final frame of seed 1 on shows still, dense colonies on bright patches and nearly bare land between them, unlike its control. One frame can't show waves, and nothing logged separates prudent grazers from exploiters.
- `heads_to_food` was a little higher on (0.62, 0.54, 0.68) than the controls (0.45, 0.51, 0.58), which fits food mattering more, but it is weak.

Checks: off is baseline. Seed 1 off has exactly the shared baseline's cell counts at every shared log tick up to 225,000. Seed 2 off matches up to 100,000, then differs by under 4% from 125,000 on, as the world nears the 131,072 cell limit. I did not find out why; the rule is off in both. The growth rule matches a plain CPU version in `crates/auto-core/tests/plants-reference.pl`. Energy: growth goes through the existing `C_SUN` count and spreading is pair trading, but `ledger=true` totals could not be read from the log, so the books were not checked by a run. GPU cost was small: 0.65 ms per tick, longest submission 15 ms.

For the screen: the settings to try are `plants_on=true` with `plants_peak` 1.5, 2 and 3, since 1.5 is short of the control's sun, and `plants_meter=true` on a few worlds to check it. `--load` takes physics from the current defaults and `--set`, so pass the plants settings again when loading a world.

Verdict: none. These runs give no evidence that the rule helps, and they can't rule it out at a fair sun budget.
