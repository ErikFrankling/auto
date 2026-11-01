# Thermodynamic life: first tests

Written 2026-11-01. Code is in [crates/auto-life](../../crates/auto-life), a CPU sim that can serve as the reference for a later GPU version. Worlds were 128² to 256², with up to about 17,000 cells and at most 37,000 ticks. That is far smaller and shorter than Fraser's runs, so "did not appear" here doesn't mean "cannot appear".

## The rules

One kind of cell. Each square holds at most one cell. There are no types, organs, roles, foods-for-types or scores.

Each cell has its own 504-weight RNN with 8 memory units. Each tick it splits its energy across rest, move, eat ground food, eat meat, grow a child, emit scent, emit heat, bite, give and emit enzyme.

| Physics | Rule |
|---|---|
| Sun | Patchy and seasonal. Fixed total. Rocks form regions joined by corridors |
| Death | Body and leftover energy become meat, which rots back into ground food |
| Bite | Pulls energy from a touching cell. Stronger cells bite harder. Only open sides can be reached, so being inside a clump protects you |
| Eating others | Loses energy. Biters keep 50% and the rest falls as meat, which digests at 70%, so 85% at best |
| Heat | Lands on the four squares around the emitter and burns whoever is there |
| Enzyme | Lets anyone nearby eat ground food up to 3× faster, so cheating is possible |
| Scent | Four channels, from long trails to short shouts, with no built-in meaning. Everyone also leaks an odour set by their genes |
| Touch signals | Two numbers that neighbours can hear |
| Sharing | Two touching cells that both agree level out their energy |
| Birth | Clone, or mix with a willing neighbour close enough in genome. Whole memory units come from one parent or the other |
| Commitment | Spending a little on something barely works, so dabbling is wasteful |
| Rain | One random cell lands every 200 ticks, so the world can't die for good |

## What happened

**Eating must lose energy.** With nearly lossless biting, one cannibal monoculture took over at 128²: 61% of energy from biting, 82% of bites on relatives, and giving, scent and enzyme close to zero (`1-lossless-128`). With biting far too weak, nearly everything grazed (`2-lossy-*`). The settings above give a middle, but not a wide one.

**Groups with different diets appeared without being programmed.** Groups are sets of cells close enough in genome to mate, at threshold 0.2.

| World | Group | Cells | Energy from other cells | Typical age | Neighbours |
|---|---|---|---|---|---|
| Seed 2 (`5-seed2`) | Open grazers | 73% | 8% | 788 | 5.9 |
| | Dense colony | 15% | 6% | 1,274 | 7.0 |
| | Hunters | 9% | 46% | 107 | 3.5 |
| Seed 1 (`5-seed1`) | Grazers | 78% | 15% | 385 | 2.3 |
| | Hunters | 20% | 44% | 164 | 4.4 |

A group that got over 40% of its energy from other cells appeared in 3 of 5 worlds at these settings: `5-seed1`, `5-seed2` and `7-blooms-seed2`. `7-blooms-seed1` stayed 98% one group, and at 256² the most predatory sizeable group got 24% by tick 30,000. These groups depend on the threshold: at 0.3 almost everything merges, and at 0.15 there are dozens.

**Removing abilities.** Each row copies an evolved world, removes one ability and compares mean population over the next 3,000 ticks with an untouched copy. There is one copy per world, so a few percent is noise.

| Removed | Seed 1 | Seed 2 | Blooms 1 | Blooms 2 |
|---|---|---|---|---|
| Sharing | −13% | −13% | −16% | −8% |
| Biting | +18% | +12% | 0% | +19% |
| Memory | −18% | +6% | −18% | −3% |
| Scent | +3% | +3% | −6% | −1% |
| Touch signals | −6% | −1% | +1% | −3% |
| Gifts | −4% | −1% | −5% | −1% |
| Enzyme | −2% | −1% | −2% | 0% |
| Heat | −1% | +1% | −1% | −2% |
| Kin sense | −2% | +4% | −1% | 0% |
| Mating | +1% | +2% | −1% | −3% |

- **Used:** sharing, in all four worlds.
- **A real burden:** biting, in three of four. Worlds hold more cells without it.
- **Unclear:** memory.
- **No effect yet:** scent, signals, gifts, enzyme, heat, kin sense and mating. Blooms didn't change that for scent.

**Kin recognition isn't shown.** Bites per meeting were 1.4 to 2.3 times higher on non-relatives. With kin sense off the ratio was 2.9 (`6-no-kin-sense`), so the sense isn't the cause. Odour stayed on in that control, and clump geometry could also explain it, since relatives sit inside clumps where bites reach fewer sides.

**Nothing died out.** No world went extinct, including 256². Every world ended with one surviving founder lineage, so rain isn't what kept them alive.

**Colonies formed, but travelling swarms aren't shown.** At 256², neighbours had 4.3 others around them against 1.9 if placed at random, and neighbouring headings aligned at 0.47. Most structure in the video is colonies sitting on rich ground, and I haven't confirmed that groups travel together.

## Bugs and limits

- Tests check the energy ledger, that the thread count doesn't change a run, that grid and cells agree, and that crossover keeps whole units.
- The 256² run was killed at tick 37,000 by the PC's load guard, which stopped it for using 4 cores for 5 minutes. Its last report is tick 30,000. Use `heavy`, or 1 to 2 threads for short runs.
- Results in `results/` came from several versions of the physics as it changed. Files numbered 1 to 3 predate later rules, and `1` and `2` used a build without commitment, exposure or odour.
- Everything has a few seeds at most. Treat these as leads.

## Run it

```
RUSTFLAGS="-C target-cpu=native" cargo build --release -p auto-life
RAYON_NUM_THREADS=1 target/release/auto-life --size 128 --ticks 30000 --log-every 10000 \
  --bite-power 30 --bite-yield 0.5 --meat-yield 0.7 --compat 0.2 --seed 2 \
  --out run.jsonl --ablate-at 30000 --ablate-ticks 3000
```

Add `--video run.mp4 --frame-every 60` for two views side by side: colour is roughly the genome on the left, and diet on the right.

## Sources

- [Higgs and Derrida 1992](https://experts.mcmaster.ca/scholarly-works/1425982): mating limited by genetic similarity splits a population into species without barriers.
- [de Aguiar et al. 2009](https://necsi.edu/global-patterns-of-speciation-and-diversity): limits on both mate distance and genetic distance.
- [Dieckmann and Doebeli 1999](https://pure.iiasa.ac.at/5700): competition plus assortative mating gives sympatric speciation.
- [Floreano et al. 2007](https://lis2.epfl.ch/CompletedResearchProjects/ECAgents) and [Mitri et al. 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2747196): communication evolves among relatives under group selection. Conflict brings deception and suppressed cues.
- [Wiley's replication of AntFarm](https://keithwiley.com/artificialLife/antColony.shtml): laying trails evolved, but following them didn't.
- [Griffin et al. 2004](https://era.ed.ac.uk/items/c781cb25-0e04-427e-9b71-07e59b491a2e): relatedness favours cooperation, and local competition weakens that.
- [Riolo et al. 2001](https://public.websites.umich.edu/~axe/research/EC_wo_reciprocity.pdf) and [Kaznatcheev and Shultz](https://www.cs.mcgill.ca/~akazna/KaznatcheevShultz20110501-2.pdf): tags and cooperation, and evidence that children staying near parents does the real work.
- [Goldsby et al. 2012](https://experts.arizona.edu/en/publications/task-switching-costs-promote-the-evolution-of-division-of-labor-a/): task-switching costs promote division of labour.
- [Rosenberg et al. 1977](https://pmc.ncbi.nlm.nih.gov/articles/PMC235010): Myxococcus grows faster at higher density on shared enzymes.
- [PolyWorld](https://en.wikipedia.org/wiki/Polyworld) and [a review of it](https://cs.tau.ac.il/~ruppin/ALseminar/naphtali.html): neural agents that eat, mate and fight, with species such as "Indolent Cannibals" reported.
- [Lennon and Jones, via MSU](https://msutoday.msu.edu/news/2011/the-undead-may-influence-biodiversity-greenhouse-gas-emissions): dormancy and seed banks support diversity.
