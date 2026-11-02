# Multicellular bodies

2026-11-02. Nothing was run. `[read]` may mean via summariser or keywords. Predictions are speculation.

**Short answer.** Our 384 worlds couldn't have made bodies: no shader has a bond, and force can't cross a body (`mass_on` only shoves one neighbour). Biters hit kin at chance, 0.424 against 0.425 (`screen/kin.jsonl`). I found no simulation where bodies arise without a fitness function or handed-out groups, then reproduce as units, inherit size and differentiate. Lab yeast and algae got unit reproduction under imposed selection or predators, undifferentiated.

## 1. Fraser Price

No paper, talk, simulator repo or site found. [GitHub "Fraser Price"](https://github.com/fraserprice) has CUDA inference repos; identity inferred. All `[said]`, from [earlier notes](../fraser-price/README.md), not refetched: closed CUDA, 1M particles at 250+ steps/s, per-particle RNNs, actions include [bonding, forces, sharing](https://x.com/fraserpricee/status/2100357688613908707). Particles deposit into force and scent fields, with [no programmed cell walls](neural-evolution-scene.md). Hierarchy unmeasured. He credited someone's rendered "cell" look, so outlines may be drawing. Unknown: whether a polygon is one particle or a clump, and what the moats are. [Origin](https://github.com/nvneik3d/Origin) credits him and says: "Turn the bonus off and almost no bodies form". That bonus is a designed family food reward.

## 2. Who got bodies

|System|Cells had|Selection|Actually showed|
|---|---|---|---|
|Hogeweg 2000, Potts|Gene network: adhesion, division, death|Fitness: count of cell types|Engulfing, budding as by-products [abstract](https://utrecht-main-test.atmire.com/items/87772324-2fa5-4d1f-9a02-b5a5244c3fb9)|
|Colizzi 2020, Potts|Only adhesion evolves; chemotaxis given|Fitness: nearness to moving peak|200 cells. Groups win long seasons, singles short. No group reproduction [read](https://elifesciences.org/articles/56349)|
|CHIMERA|Springs, evolving adhesion|None explicit|Aggregates [read](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0059664). Near-point physics fails [review](https://arxiv.org/abs/1403.3217)|
|Box2D gene networks|Adhesion receptors|None|Max 12 cells, one run, no unit reproduction [read](https://arxiv.org/abs/2305.12249)|
|van Gestel 2021|Adhesion probability|None|Polarity only with kin recognition given [read](https://doi.org/10.1371/journal.pbio.3001250)|
|Kaneko|Metabolic equations|None|Differentiation [review](https://arxiv.org/abs/1403.3217)|
|Avida, Goldsby|25-cell colonies|Rewards; replication given|Switching cost raises labour division, 0.40 to 1.07 [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC3427090)|
|Tierra 1994|CPUs|Hand-built ancestor|2 to 4 CPUs; different data, same code [read](https://tomray.me/pubs/alife4/)|
|Biomaker CA|Root, leaf, flower given|Flowers reproduce|Big plants [read](https://arxiv.org/abs/2307.09320)|
|ALIEN|15 types, genome blueprint|None|No transition claimed [readme](https://github.com/chrxh/alien)|
|EvoLife|No fixed types|None|"Filamentous algae", "no real cell differentiation" [said](https://store.steampowered.com/app/3291610)|

No bodies: Evoloops shrink [secondary](https://necsi.edu/a-new-structurally-dissolvable-selfreproducing-loop-evolving-in-a-simple-cellular-automata-space). Flow-Lenia has no cells. Microcosmos filaments use search [abstract](https://arxiv.org/abs/2607.02954). PolyWorld, Bibites, Primordial Life: one body per organism.

## 3. Biology

- **Staying together.** Selected for settling, all 10 yeast populations clustered within 60 transfers. Tension fractures clusters into propagules, unprogrammed; traits highly heritable. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC4309424/) Size selection later gave macroscopic, undifferentiated bodies. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC10425966)
- **Coming together.** Across 168 species, clonal groups more often became obligate. [abstract](https://www.lunduniversity.lu.se/lup/publication/ecbbc23d-48fd-4a28-ba91-275a46a68b9c) Aggregation favours short-lived groups. [abstract](https://sonar.ch/global/documents/47514)
- **Predation.** Algae: 2 of 5 populations, ~750 generations, 0 of 3 controls, stable four years. Boraas similar. [read](https://www.biorxiv.org/content/10.1101/247361v2.full-text)
- **Public goods.** Scarce sucrose: all 10 populations clumped, ~230-310 generations. [read](https://elifesciences.org/articles/00367)
- **FLO1.** Adhesion is recognition: carriers stick together regardless of other genes. Survival 2 to 100+ fold better, growth 4-fold slower. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC2703716)
- **Steinberg.** Cadherin density tracks aggregate tension, but measured tension far exceeds bond estimates. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC9547846) Bond properties don't predict sorting. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC2568938)
- **Volvocines.** Matrix embedding (223 Ma) preceded incomplete cytokinesis (211 Ma). [secondary](https://pmc.ncbi.nlm.nih.gov/articles/PMC2651347)
- **Scaffolding.** Patchy resources plus dispersal make patches Darwinian; single-cell founders favour reproductive labour division. Model only. [read](https://www.biorxiv.org/content/10.1101/656660v1)

Minimum, my reading: stay attached, a local benefit singles lack, break-up, a bottleneck. Differentiation comes later.

## 4. Physics needed

|Physics|Kind|When|
|---|---|---|
|Adhesion output, both pay, breaks under strain|Generic|First. Colizzi: evolves at cost 0.5, not 0.75|
|Overlap push, force through bonds|Generic|First. Gives fracture, body motion|
|Child starts bonded|Generic|First. Kin without recognition|
|Contact signalling|Generic; voice exists|Differentiation|
|Sharing, chemical field|Generic; `give`, scents exist|Have|
|Polarity, oriented division|Generic as free outputs|Later|
|Costly role switching|Generic|Labour division|
|Programmed death|Designed|Skip; give and starve suffices|
|Body IDs, family bonus, group replication, types|Designed|Never|

## 5. GPU substrates

Today, fitting `exp/giant`: 6.6 ms per million cells plus 0.5 per million squares. 5M at density 0.2 is 46 ms, 20 steps/s, at our 50 ms cap. Below is extrapolated.

|Substrate|Cost|Bodies|Pitfalls|
|---|---|---|---|
|Kawasaki swaps|Today plus a pass|[Sorts](https://pmc.ncbi.nlm.nih.gov/articles/PMC8624153); no force|[D~R⁻³ on metals](https://arxiv.org/abs/1303.1002): by analogy 1,000 cells crawl 6,000x slower|
|Sub-square offsets|Slightly above today|Deform, weak push|No compression. Leaving cells mid-update [bred false order](https://arxiv.org/abs/1211.1646); shift the grid|
|Soft particles, hash|[400M queries/s, 2013 GPU](https://ramakarl.com/pdfs/2014_Hoetzlein_Fast_Neighbors.pdf): 12+ ms|Fracture, rotation|Stiff springs explode; use overdamped motion. Sort indices, never 4.3 GB of weights; bonds need stable IDs|
|Position-based|[128k, 6 iterations, 10 ms, GTX680](https://cgweb.informatik.uni-freiburg.de/intern/seminar/positionBasedFluids_2014_SIGGRAPH.pdf)|Stiff bonds|Jacobi converges slowly|
|Potts|[3.36M cells, 3D, 0.53 sweeps/s](https://arxiv.org/html/2312.09317v1)|Real polygons|Needs 50x; Colizzi seasons take 5,000+ sweeps. Mutexes stop sorting; big checkerboards break motility|
|Particle-mesh|Fraser's, `[said]`|Unknown|Forces blurred to grid scale|

Conflicts: each cell writes only itself from a snapshot; two-sided deals compute identically on both sides. Births use `atomicAdd`. WGSL atomics are [integer only](https://www.w3.org/TR/WGSL/), so fixed point.

Pick: soft particles from 100k, with real volume (CHIMERA's warning), brains every 4-6 physics steps. My guess at 5M: 20-40 steps/s.

## 6. Experiments

1. **Locked food on islands, existing engine.** A lone cell eats without enzyme (`act.wgsl:333`), and the screen never varied enzyme. Require more enzyme than one cell builds, on islands that run out, with rare dispersal.
2. **Sticky births, fracture, biters. Soft particles.** Yeast plus Herron. Bites must break bonds first, or bodies eat themselves.
3. **Drifting food, noisy gradient.** Colizzi without his fitness function: untested.

Control: adhesion off. A clump fails tests 2, 3 or 5:

1. Body means bond-graph component; membership half-life beats control.
2. Ten-plus tracked body generations; relatedness higher within bodies.
3. Size at fragmentation: parent-offspring slope above zero.
4. Clonemates keep different budgets, predicted by position, beating shuffles.
5. Same genomes, adhesion off, on saved worlds: 50/50 and rare both ways, 3+ seeds.

**Not reached:** Ratcliff 2012, Boraas, Hogeweg, Steinberg full texts; Fraser's new post.
