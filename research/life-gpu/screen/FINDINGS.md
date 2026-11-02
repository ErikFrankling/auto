# What the first screen found

Every number here was checked against the data on 2026-11-02. The clean one-rule-at-a-time set (`b1-*`) and two more long runs are still running, so this will grow.

## The screen

- 384 worlds, 256 wide, 120,000 ticks (about 500 generations): 32 unmodified, and 22 sets of physics with 16 seeds each. Results in `results.jsonl`, each set read by a sub-agent with its maps in `reads/`.
- All 32 baseline worlds lived. Of the 22 rule mixes, 7 lost all 16 worlds and 2 more kept only 1 or 2. A rain of random cells can restart a dead world, by design, so some "regrown" worlds are new starts.
- Unmodified physics already varies hugely by seed: 1 to 4 lasting strategies, 3 to 16.5 species, 2% to 35% of energy from other cells, and 0% to 20% from biting. Three seeds could never have shown this.
- No rule mix raised lasting strategies above the baseline. Most made worlds duller: few species, less biting. The first screen changed many settings at once, so it can't say what one rule does. That's what `b1-*` is for.
- Kinds of baseline world (sub-agent's sorting of 32): 9 steady grazers, 8 drifting to rest, 7 part-time biters, 4 relays where the main strategy keeps being replaced, 4 hunting economies. The middle cuts are arbitrary, since biting runs in a continuum.
- No thin trails in any map that readers opened. Group hunting didn't appear. The gang rule pays per bite where it's on, but bites stay solo: in `b0-0012`, 90% of bites were alone.

## Biters don't choose

For the 205 worlds with enough biters at 120,000 ticks, the share of bite energy taken from close kin matches the share of close kin next to the biters (correlation 0.96, median difference 0.2 points, within 10 points in 200 of 205). Bites land on whoever is adjacent. Nothing has learned to tell kin from strangers. At 600,000 ticks in 16 baseline worlds it's the same.

## Biting isn't sustaining anyone

Turning biting off on saved worlds and running 30,000 more ticks barely changes population: a median of 6% more cells than the same world left alone, in 16 baseline worlds. None of the 16 lost more than 5%. One gained a lot: 59% more without biting. So biting is mostly a cost to the population even in worlds where it's 20% of the energy flow.

## Five times longer doesn't help

16 baseline worlds continued from 120,000 to 600,000 ticks. Strategies over 3% stay at a median of 2. Behaviour dimensions stay at 3.4. Worlds settle one of two ways: lawns where resting climbs to 50-60%, cells double, biting vanishes and species multiply as clonal patches (t006, t011, t012, t013, and t004 partly), or scrambles where biting rises to 15-23% of energy and cells stay low (t000, t002, t010, t014, t015).

## Things that looked real and weren't

- **Population cycles** in `b0-0001-t000` and others, about 6,400 to 13,300 cells, with a period of 4 samples. All worlds share that rhythm. Samples every 30,000 ticks alias against the built-in 8,000-tick season: it's seasons. Don't sample at a period that shares a low common multiple with the season.
- **Life hugging the outer wall in mineral worlds.** Real, but an artefact of the mineral rule. Mineral piles against rock at 11 to 35 times the density inside the world, at all tested flow settings, and cells follow it. Next to natural rock it's 6 to 10 times higher too, in the two worlds measured. So mineral worlds didn't test a second resource. Likely carried there by cells and released where they die, but that mechanism isn't tested.
- **"Traffic on lines"** is high wherever colonies are. No map showed lines. It's useless as a trail detector as built.
- **"Lives on other cells: 27%"** in rank tables is the average of per-cell ratios. Energy flows say 8% for the same world. Use flows.

## Why this engine can't make bodies

One cell per square, no bond, and force doesn't cross a group. Each cell can only shove one neighbour. So there is no way for a group to hold together, move together or push. See `research/notes/multicellularity.md` and `research/life-soft/DESIGN.md`.
