# What it takes to get real depth of strategy, and what to change in our world

Research date 2026-11-02. No simulations were run for this note. Every prediction about our world below is reasoning, not a result.

**Short answer.** Our world is doing what theory says it should. One energy currency, one renewable supply, identical bodies and bites that add up badly can hold about two ways of living. A bigger world gives more copies of those two, not more kinds. Nobody has shown truly open-ended evolution, and I found nobody who evolved trail recruitment or group raiding from scratch with no fitness function. What the evidence does support is a short list of physics that raises the ceiling: several essential resources, attack and defence that must match, and fights where numbers pay.

**Labels.** `[read]` I read the paper's text. `[abstract]` I saw only the abstract, a repository record or a search summary of it. `[secondary]` I only saw someone else's account. `[not verified]` I could not reach it, or it is from memory. `[ours]` is from this repo's code or logs. `[reasoning]` is my own inference.

## A. What others built, and what it measurably did

| System | What it showed | Limit |
|---|---|---|
| **Tierra** | Parasites, immunity, parasites that beat immunity, hyperparasites, sociality, then cheats, all unprompted. Ray credits parasites with raising community diversity. [read](https://faculty.cc.gatech.edu/~turk/bio_sim/articles/tierra_thomas_ray.pdf) | Mostly described, not measured over long runs. I found no source on when it stalls. |
| **Avida, Lenski 2003** | EQU evolved in 23 of 50 populations when simpler functions were rewarded, none when only EQU was. Mutations harmful when they appeared were sometimes stepping stones. [read](https://www.cs.princeton.edu/courses/archive/fall24/cos597C/bib/598-2017/adami03Nature.pdf) | The rewards are designer-chosen tasks. |
| **Avida with parasites, Zaman 2014** | **The closest result to what Erik wants.** With no reward for complexity, EQU evolved in 17 of 50 populations with coevolving parasites and 0 of 50 without. Parasites infect by matching a host's functions. It took a diversity of coevolving lineages, not plain escalation, and hosts became more evolvable. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267771) | The matching is a built-in lock and key, though the content evolves. |
| **Echo, Holland** | Tags for combat, trade and mating allow non-transitive relations. [read](https://people.reed.edu/~mab/papers/alife6.pdf) | **The warning case.** Bedau's numbers: bounded activity, unlike the fossil record. Smith and Bedau: "genotypic diversity without phenotypic diversity", and populations collapsed into trading one resource. [read](https://people.reed.edu/~mab/publications/papers/smith.pdf) Species abundance curves still looked natural, so such curves prove little. [abstract](https://www.cs.unm.edu/~forrest/publications/echo-97.pdf) |
| **Sugarscape** | With sugar and spice, agents die if either runs out, which creates a reason to trade. Trade reportedly raises carrying capacity and inequality. [secondary](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/epaxrev.pdf) | Trade is a coded rule, and I did not read the book. |
| **PolyWorld** | Neural complexity is under selection, in different directions at different scales, shown against a run with selection removed. [read](https://arxiv.org/abs/1112.4906) Swarming and foraging are reported. [abstract](https://shinyverse.org/larryy/Polyworld.html) | Behaviour reports are qualitative. |
| **Geb, Channon** | Claimed to pass Bedau's unbounded test after changing how it is normalised. [abstract](https://eprints.keele.ac.uk/36/) **Maximum complexity hits a ceiling if you scale world size alone, or neurons alone. Only scaling both together removes it, and growth is then logarithmic.** [read summary](https://arxiv.org/abs/1909.04430), [abstract](https://eprints.keele.ac.uk/5900/) | Simple artificial worlds, years of runtime. |
| **Chromaria** | Stagnates when any of four hypothesised conditions is missing. [abstract](https://stars.library.ucf.edu/scopus2010/9117) A too-strict survival criterion freezes the world, none at all decays into noise, the middle changes most. [read](https://www.uvm.edu/neurobotics/pubs/pdf/2016_SorosCheneyStanley_HowTheStrictnessOfTheMinimalCriterionImpactsOpenEndedEvolution_ALIFE.pdf) | Tested in one world only. The four conditions as I remember them: non-trivial reproduction criterion, new individuals create new opportunities, individuals decide how to meet it, unbounded phenotype. [not verified beyond the first] |
| **EcoSim** | No fitness in the run. About 55 persistent species against 65,000 tiny "species" with selection removed. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC4570812/) | Prey and predators are designed types, 7 fixed actions, species counted by genome. Not a measure of strategies. |
| **Neural MMO** | Bigger populations explore more, and more separate policies spread into separate regions. Its key line: competition is "a curriculum magnifier, not a curriculum in and of itself", and the environment must support it. [read](https://arxiv.org/abs/1903.00784) | RL with rewards, up to 128 agents, and niches are mostly map regions. |
| **DISHTINY** | Ten multicellular forms in one chosen population of 40. Novelty, complexity and adaptation only loosely track each other. [read](https://arxiv.org/abs/2405.07241) | Hands out kin groups and a group-size food bonus up to 12 cells, which Erik's rule forbids. |
| **ALIEN** | Striking videos, a competition prize. [readme](https://github.com/chrxh/alien) | 15 designed cell types, and no measured result on the page. |
| **Flow-Lenia** | Conserved mass gives lasting forms. Evolutionary activity falls quickly at higher mutation rates. [read](https://arxiv.org/abs/2506.08569) | The authors call it a step toward open-endedness. |
| **JaxLife** | Terraforming, rudimentary communication. At 32 agents no patterns, and at 256 agents the communication channel is ignored. [read](https://arxiv.org/html/2409.00853v1) | Short runs, identical bodies, which its authors name as a limit. |
| **EcoEvoJax** | No resets. Behaviours group into **two types**, and 9 of 50 sampled agents were "sustainable foragers". [read](https://arxiv.org/abs/2302.09334) | Same two-way outcome as ours, on one resource. |
| **Harvard ecology, Bejjani 2025** | 60,000 agents. Mining, vision-based foraging and predation emerged. Larger worlds extinct far less (hazard ratio 0.04 at 512 against 128). Resource rules matter, starting population doesn't. [read](https://arxiv.org/html/2510.18221v3) | No count of strategies, no measured cooperation, memoryless policies, four seeds. |
| **Burtsev and Turchin 2006** | Energy only, on a 30x30 torus, strategies built from elementary actions: hawk, dove, bourgeois. Cooperative attack and defence appeared only with visible heritable markers. [abstract](https://ideas.repec.org/a/nat/nature/v440y2006i7087d10.1038_nature04470.html) **Pack hunting did not evolve, and the authors blame the lack of a way to move as a group.** [secondary](https://www.pandasthumb.org/archives/2006/05/the-evolution-o-4.html) | Markers are genome-given, which Erik rejects. |
| **Hide and seek, OpenAI** | Six phases of counter-strategy from competition alone, including tool use. [read](https://arxiv.org/abs/1909.07528) | RL with team rewards, shared policy, not evolution. Shows that a game with counters produces escalation, not that energy does. |

**Honest state of the art.** Parasite-style coevolution has the best controlled evidence for driving complexity. Everything else with rich strategies either has designed roles, external rewards or short runs, or is only demonstrated in videos. The field's own 2019 overview found no single test and no agreed categories. [read](https://arxiv.org/abs/1909.04430)

### How people measure it

- **Bedau's activity statistics.** Count how long each component persists, and compare with a **neutral shadow** that copies the real run's births and deaths but picks parents at random. Three classes: no adaptive activity, bounded, unbounded. [read](https://people.reed.edu/~mab/papers/alife6.pdf) Channon had to change the normalisation, so passing depends on method. [abstract](https://eprints.keele.ac.uk/36/) Flow-Lenia still counts every parameter point as a species regardless of phenotype, and says so. [read](https://arxiv.org/abs/2506.08569)
- **MODES (Dolson et al.).** Change, novelty, complexity and ecological potential, tested on NK landscapes and Avida. [abstract](https://peerj.com/preprints/27249v2) Full text was blocked for me. As I remember it, only lineages that persist past a filter are counted. [not verified]
- **Taylor 2016 and Packard 2019.** No single test. Several kinds of open-endedness exist: new kinds of things and interactions, evolvability, major transitions, and new meaning. [read](https://arxiv.org/abs/1909.04430), [abstract](https://research.aston.ac.uk/en/publications/open-ended-evolution-perspectives-from-the-oee-workshop-in-york/)
- **Competition assays.** DISHTINY scores a trait as real only if removing it loses a head-to-head fight. [read](https://arxiv.org/abs/2405.07241) This is the strongest and most expensive test.

### Has anyone evolved trail recruitment or raiding from scratch with no fitness function?

**I found no case.** Absence from a search isn't proof, but I looked directly in two searches and in the sources above.

- Ant-like robots: the fitness was foraging trips, and laying was hand-given. Only detection and traffic rules evolved. Detection alone did not help, since trails jammed. The traffic rule came first by stochastic tunnelling, about 158 generations to fixation. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC6338667/)
- Spiking-network ants: scent use was not coded, but there is an explicit fitness, a nest and food carrying. Mean fitness 8100 with scent against 5800 without, over 1700 generations. [read](https://arxiv.org/abs/2212.08484)
- A hobbyist AntFarm replication: rewarded every step, including dropping scent. Laying evolved and "the ability to follow the trail back to the food source never evolved". [read](https://keithwiley.com/artificialLife/antColony.shtml)
- Swarm predators and prey: grouping evolves when predators get confused or attack by density, but those predator rules are built in. [abstract](https://gpbib.cs.ucl.ac.uk/gp-html/olson_2013_RSIF.html), [abstract](https://arxiv.org/abs/1310.6012)
- A 3D swarm with energy-seeking agents reports evolved signalling. I saw only a search snippet. [secondary](https://arxiv.org/abs/1903.08228)

**What it took in real ants, which is very close to Erik's example.** Army ant mass raids evolved from scout-led group raids, in which a scout finds prey and lays a trail to recruit. In the lab, simply enlarging colonies turned group raids into mass raids without changing individual rules. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC8179172) The prey are often other social insects, and such predators are "unlikely to be solitary foragers, because ant colonies are well-defended and able to kill solitary intruders". So the driver is **big defended prey that kills lone attackers, plus colony size**. Trails also need numbers. Pharaoh's ant colonies switch from disordered to trail foraging above a size threshold, abruptly when food is hard to find. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC55516) Models give minimum colony sizes for trails to hold. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC2915909) Honeybee dances pay in some seasons and not others, apparently when food is patchy and short-lived. [abstract](https://ideas.repec.org/a/nat/nature/v419y2002i6910d10.1038_nature01127.html)

## B. What theory says a world can hold

- **Competitive exclusion.** Complete competitors can't coexist. [secondary](https://en.wikipedia.org/wiki/Competitive_exclusion_principle) **Levin 1970:** no stable equilibrium if r species are limited by fewer than r limiting factors, and "limiting factor" means anything that regulates growth, not only food. [abstract](https://www.journals.uchicago.edu/doi/10.1086/282676), [read summary](https://gezameszena.web.elte.hu/wp-content/pdf/ElmEvol/Szabo_and_Meszena_LimSimRev.pdf) **Loophole:** two consumers can share one resource if cycles sustain them. [read](https://www-users.cse.umn.edu/~mcgehee/publications/Armstrong1976a/Armstrong1976JTheorBiol56p499.pdf)
- **R\* and resource ratios (Tilman).** On one resource the type that can live at the lowest level wins. With two, coexistence needs each to be better on one, a trade-off. [secondary](https://en.wikipedia.org/wiki/R*_rule_(ecology)) Of 1,333 citing papers only 26 were good tests, and those supported it 75% of the time. [abstract](https://scholars.csus.edu/esploro/outputs/journalArticle/A-critical-review-of-twenty-years/99257847159101671) A reply argues support is stronger for R\* itself. [abstract](https://dare.uva.nl/id/4e4f2eef-d963-420b-ab63-0c2003be61f3)
- **Liebig and stoichiometry.** Growth is set by the scarcest essential input. Measured support: across 45 grasslands, each added limiting nutrient lost more species, even after accounting for biomass, which supports the idea that more limiting factors means more niches. [abstract](https://www.nature.com/articles/537042a)
- **Paradox of the plankton (Hutchinson).** Many species on few resources, so something else is going on, and there's no accepted single answer. [abstract](https://centaur.reading.ac.uk/36885/) With **three or more essential resources**, competition itself can oscillate and go chaotic, letting 12 species live on 5 resources in a model. [abstract](https://doi.org/10.1038/46540)
- **Storage effect and fluctuations (Chesson).** Coexistence needs stabilising forces, not just equal fitness. [abstract](https://experts.arizona.edu/en/publications/mechanisms-of-maintenance-of-species-diversity/) The storage effect needs species that **specialise on different aspects of a variable environment**. It has "little to do with" a hardy stage in itself. Relative nonlinearity can support species counts growing with the square of the resource count. [read](https://arxiv.org/abs/2201.07926) **So weather alone shouldn't add strategies unless cells can be good at different weather.**
- **Colonisation against competition (Tilman 1994).** Unlimited species on one resource, if strictly weaker competitors colonise sufficiently better. [secondary](https://link.springer.com/article/10.1007/s00442-011-2013-1) **Caveat:** Adler and Mosquera show high diversity comes from assuming a small mortality difference gives a large competitive one, not from space. [abstract](https://agris.fao.org/search/en/records/65de39fb7c7033e84bea177d)
- **Janzen-Connell.** Young die more near their own kind. A meta-analysis supports it, with effect sizes that vary widely, and asks for tests of whether it actually maintains diversity. [abstract](https://repository.si.edu/handle/10088/22199) It needs enemies that specialise, which ours can't.
- **Red Queen and matching.** In pond sediment, parasites infected hosts from their own time best: real ongoing tracking. [abstract](https://ideas.repec.org/a/nat/nature/v450y2007i7171d10.1038_nature06291.html) Zaman above is the digital version. [read](https://pmc.ncbi.nlm.nih.gov/articles/PMC4267771) Matching rewards being rare, a built-in reason never to settle. Formal matching-allele theory: [not verified].
- **Rock-paper-scissors.** Kerr 2002: three bacterial strains coexist when dispersal and interaction are local, and one wins when mixed. Patches chase each other. [read](https://www.bcp.fu-berlin.de/biologie/arbeitsgruppen/zoologie/ag_steppuhn/lehrveranstaltungen/1_Theoretische-Oekologie_Kerr-et-al-2002.pdf) Reichenbach 2007: spirals below a critical mobility, Mc = 4.5e-4 of the world's area per generation, and one survivor above. **Spirals outgrow small worlds.** [read](https://arxiv.org/abs/0709.0217) **Caveat:** in 18 field annual plants, intransitivity was rare and did not create coexistence without pairwise niche differences. [abstract](https://collaborate.princeton.edu/en/publications/intransitivity-is-infrequent-and-fails-to-promote-annual-plant-co/)
- **Predators and the Cambrian.** Efficient predators can raise diversity by stopping one species taking everything (Paine). [abstract](https://pubs.usgs.gov/publication/70175674) Stanley proposed the same for the Cambrian: high diversity depends on cropping. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC433525) Sperling links oxygen to carnivory. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC3746845) The causes remain debated, and I could not verify the vision-first idea. [not verified]
- **Major transitions.** New levels of individuality reshape what can evolve. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547294) Yeast clusters evolved in weeks, under selection designed to favour them. [abstract](https://experts.umn.edu/en/publications/experimental-evolution-of-multicellularity/) Transitions are contingent and may not recur. [read](https://arxiv.org/abs/1909.04430)
- **Lanchester.** If the larger side can focus attacks, strength goes as size squared. If fights are one on one, linear. [read](https://math.fsu.edu/~mesterto/NewCourses/MAP5932/2016/PDF/PDF03/Adams2003bFightsAmongSocialAnimals.pdf) **Caveat:** fire ant trials rejected the square law, finding ability roughly linear in group size. [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC1691006)
- **Group hunting.** Wolf success levels off beyond 4, and the study shows free riding is why. [read](https://www.wolf.org/wp-content/uploads/2013/08/326nonlineareffectsgroupsize.pdf) Lions do best alone or at 5-6 when prey is scarce, and form groups for more than food. [abstract](https://carta.anthropogeny.org/node/310628) A later Serengeti study reportedly found cooperation on bigger and harder prey. [not verified, search summary only] Payoff condition: a lone hunter must do badly and a group well, per head, which means large or dangerous prey.
- **Signalling.** Information is worth paying for only if it changes what you do, and the payoff of doing that exceeds the cost. Stephens 1989 is the usual citation. [not verified, citation only](https://www.journals.uchicago.edu/doi/10.1086/284969) In evolving robots, unrelated competitors evolved deceptive signals and suppressed cues, while related or group-selected ones signalled honestly. [abstract](https://www.epfl.ch/labs/lis/research/completed/communication2/), [abstract](https://pmc.ncbi.nlm.nih.gov/articles/PMC2747196) Zahavi's handicap idea is now disputed, with critics arguing Grafen's models don't support it. [abstract](https://real.mtak.hu/73702/) So don't build "costly signal" logic on it.
- **Helping neighbours.** In a crowded world, help to relatives can be cancelled by competition with those same relatives (Taylor 1992, as described by citing papers). [not verified] Empty sites restore the benefit, at an intermediate number. [abstract](https://bioone.org/journals/Evolution/volume-62/issue-6/j.1558-5646.2008.00369.x/Empty-Sites-can-Promote-Altruistic-Behavior/10.1111/j.1558-5646.2008.00369.x.full)
- **Error threshold and mutation.** Above a critical mutation rate a population loses its information (Eigen 1971). [secondary](https://en.wikipedia.org/wiki/Error_threshold_(evolution)) High mutation favours lower, flatter fitness peaks over tall, narrow ones: "survival of the flattest". [abstract](https://ideas.repec.org/a/nat/nature/v412y2001i6844d10.1038_35085569.html) Adaptation slows at least as 1/n as the number of independent traits grows. [secondary](https://en.wikipedia.org/wiki/Fisher%27s_geometric_model)

## C. Our world

### C1. The market analysis

All `[ours]`: `crates/auto-life-gpu/shaders/`, `research/life-gpu/analysis/deep/v2-seed{1,2,3}.json`, `research/life-gpu/exp/base-v2/`.

| | Seed 1 | Seed 2 | Seed 3 |
|---|---|---|---|
| Income: ground / bite / meat | 71 / 16 / 13% | 95 / 0.2 / 5% | 78 / 11 / 11% |
| Deaths: bitten / starved / old / burned | 71 / 22 / 4 / 2% | 2.5 / 35 / 55 / 7% | 33 / 43 / 5 / 20% |
| Budget on rest | 0.1% | 57% | 10% |
| Density, largest colony | 0.18, 2,909 | 0.54, 90,264 | 0.30, 9,695 |
| Behaviour clusters, silhouette | 2, 0.39 | 2, 0.32 | 2, 0.41 |

- **Supply.** One flow: about 12,300 energy per tick from sun on about 247,000 open squares, mean 0.05 per square. Ground does not flow, and caps at 4 per square. There is one currency and no stock worth fighting over except bodies. Meat is a by-product.
- **Demand.** Everything is priced in energy. Nothing needs two things, so no cell can be rich in one thing and poor in another, and nothing is worth carrying or trading.
- **Land is the real asset.** Squares with sun 0.2 or more are 69-100% occupied, with residents moving 3% of the time and holding 28-40 energy. Squares under 0.02 are at most 7% occupied, and those cells move 35-39% of the time. Sitting on a rich square is a monopoly with no rival claim except a bite.
- **Idle capital.** The richest residents sit at the 40 cap, wasting income, and a child needs an empty neighbour. There's nothing else to buy: no bigger body, no lasting building, no larger offspring (`womb` is fixed at 5).
- **No lasting skills.** Bodies are identical and a brain can switch from grazing to biting in one tick for free. A flexible brain does everything a specialist can, so nothing forces a split. Trade-offs are what let types coexist, and the only one here is `COMMITMENT` within a tick.
- **Bites are absurdly profitable where prey touches you.** A full-budget bite takes up to 6.7 energy per tick at even force. Even through a full clump's 1/8 exposure that keeps about 0.4 per tick, against mean income of 0.06-0.3 per cell. Prey comes to you, since children are born adjacent and half of all bites (`bites_on_similar` 0.48) land on near relatives. So sit and bite is the best use of a hunter's time, and chasing isn't needed. Top speed is equal for everyone, so a chase can't be won anyway unless the prey stops.
- **Moving is expensive and growth competes with it.** Top speed takes 60% of budget (0.15 energy per tick). A fed cell makes a child in 20 ticks, in which a walker covers 20 squares. Biology has this ratio thousands of times wider. Why walk when you can breed there? [reasoning]
- **Nothing lasts.** Apart from eaten ground, nothing a cell does outlives about 100 ticks. The world cannot accumulate history.

### C2. How many strategies should this world hold?

**About two, three at most, and that's what we measure.** [reasoning] Independent limiting factors: ground food, and prey availability. Meat is mostly tied to kills. Space matters, but only through bites. Levin's bound therefore says 2, plus perhaps one through cycles. Without predators it should fall to the single lowest-R\* type, which is seed 2: rests 57% of the time, packs in at density 0.54, a colony of 90,264 cells, bites 0.2% of income. Seeds 1 and 3 are the predicted second state, where biters keep any one type from monopolising (Paine). Seeds differing is two alternative outcomes of one simple system, not depth.

- **Same sessile colonies:** sizes do differ at 512 (p10 about 35, p90 about 1,100-3,600), but they follow terrain. Colony centres sit on the terrain's 32-square noise grid above chance in all three seeds (0.44 vs 0.19, 0.22 vs 0.13, 0.51 vs 0.19). The sun's correlation length (0.3 at lag 32) doesn't change with world size. So a larger world adds copies of the same patch. Geb's result says the same: size alone hits a ceiling.
- **The 4096 run is young.** `exp/giant/4096.jsonl` ends at tick 15,905, seeded with evolved life, about 75 generations on (generation 981 at tick 3,000, 1,045 at the end). On seed 1 at 512, most diversity appears only after 150,000 ticks, so what Erik saw at 4096 may partly reflect that. But theory says time alone won't fix it.
- **Removing scent, gifts, heat or sight changes nothing** because in a world like this none of them pays. They are unpriced goods. Theory agrees: no value of information without a decision that depends on it.
- **Is the sun amount a niche axis?** No. More sun is better for everybody, so there's no trade-off to split on. Evidence in-house: biomes where physics differs in kind (`tempo`, `drag`) gave 3.2-8.7 species at threshold 0.5 and 0.20-0.33 explained, against identical rooms at 1.3-1.4 and 0.06-0.13, and baseline 1.0-1.8. Very little data, no verdict, but the direction fits. [ours: exp/biomes on branch t3/509c6643]

### C3. Checking the mutation reasoning

The claim was: one birth changes behaviour by 10% of the gap between cells, so mutation load isn't the main limit. **I'd keep the conclusion for today and drop the argument.** [reasoning]

1. **The steps add up.** If two lines each drift, gap squared is about 2T times step squared. A 8-12% step gives T of about 40-80 generations. So the whole observed gap between cells could be recent mutation drift, with no selection needed. A small step relative to that gap says nothing about load.
2. **Rare behaviour is invisible.** Each birth changes about 67 of 672 weights. A weight under no selection is fully scrambled in about 160 generations (0.1 per generation times step 0.05, against starting spread 0.2), which is 40,000-80,000 ticks here, and we've run 600-1,170 generations. **Anything used rarely, like a raid against a prey that turns up seldom, is erased faster than selection can hold it.** The 96 situations mostly test common cases.
3. **Survival of the flattest.** High mutation favours brains that don't depend on exact weights. Reactivity only 2-3x random, and `bias` being the top input in seed 1 (0.43), fit that but don't show it.
4. **Sex isn't rescuing it:** 1% of births are mixed in seed 1.

Flow-Lenia also loses evolutionary activity at higher mutation. [read](https://arxiv.org/abs/2506.08569) The mutation table already says rate 0.01 gives 2-3% of the gap, not 10%. So mutation isn't the first limit but **will become binding as soon as anything richer pays**. Screen it from the start, since it costs nothing.

### C4. Why no trails and no raids: our physics disallows them, specifically

All `[ours]` or `[reasoning]`.

**1. Ganging up is punished by the rules.** `act.wgsl:240-249`: `open = (1 + empty neighbours of the target) / 8`. An attacker fills a neighbour, so it lowers everyone's exposure. On a lone target, N attackers do total damage N(9-N)/8:

| Attackers | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Total damage | 1 | 1.75 | 2.25 | 2.5 | 2.5 | 2.25 | 1.75 | 1 |
| Each attacker's share | 1 | 0.88 | 0.75 | 0.63 | 0.5 | 0.38 | 0.25 | 0.13 |

Every extra biter earns less, and past 5 total damage falls. That's the opposite of what group hunting needs. `force` at `act.wgsl:238` compares one attacker's energy with the target's, so numbers can't beat a fat cell either.

**2. No prey needs a team.** A lone biter can already kill anything, risking only that the target bites back. Colonies can't kill intruders: defenders can't tell an intruder from a relative, so biting back means biting family. Army ant logic has no foothold.

**3. Nothing is worth recruiting to.** Ground food is thin and everywhere. Nothing is bigger than what one cell can handle, and a cell can eat a full square in one tick. The biggest item is a fat body or carcass (about 46 energy), and one cell clears it. Local births fill any rich spot.

**4. No home, no return trip.** Ant trails connect a nest to food. Here nothing carries anything, and the recruits are whoever happens to be near. Relatedness comes only from being born adjacent, and competition with those same relatives may cancel the benefit.

**5. Scent is too short-lived.** Lifetimes per scent channel are about 100, 25, 6 and 2 ticks, with steady-state reach of about 1 square around a sitting emitter. A trail from p0 stays readable about 100-250 ticks, so at top speed the far end is gone by about 100-250 squares, and a recruit must arrive in time. The roads experiment's untested idea of longer-lived scent is right. Self-emission also biases `follows_scent` negative, so we'd not see a follower with that measure anyway.

**6. Telling friend from foe is hard.** A cell can show a voice and read a neighbour's, but it can't compare it with its own except through memory, so a family password needs several parts to arrive together. Burtsev and Turchin got cooperative attack only with markers. [abstract](https://ideas.repec.org/a/nat/nature/v440y2006i7087d10.1038_nature04470.html)

### C5. Changes, ranked

All physics, none handed to any type. Costs: no new buffers unless stated. Free now: `fc.y,z,w`, `c.x0..x3`, actions x0 and x1, outputs `o5.z,w`, biome bits. Predictions are `[reasoning]`.

**1. A second (then third) essential nutrient, in a closed cycle, patchy against sun.**
*Mechanism:* a mineral in `fc.y`, supplied slowly from patches that don't line up with rich sun, or a fixed total with none created. Cells hold it in `c.x0`. A child needs both energy and mineral (Liebig). Uptake is a budget action (x0). Death returns it to the ground, bites transfer it.
*Incentive:* no square is best for everything, so land rent splits. A cell short of mineral will pay to move, bite or receive a gift. A body becomes worth eating for what's in it, not just energy. Locked-up mineral gives moving waves of scarcity.
*Predict:* mineral-land and sun-land specialists, migrants along boundaries, predators choosing rich bodies, a use for `give`, boom and bust that doesn't settle, and more strategies on the Levin count.
*Risk:* everyone crowds the border where both are good, or minerals are plentiful and never limit. Harpole's result says adding plenty reduces diversity, so tune until both actually limit, and log which one blocks births. A flexible brain could stay a generalist, so keep `COMMITMENT`.
*Evidence:* Levin, Tilman, Harpole, Huisman with 3+ resources, Sugarscape, our biomes hint. Strongest theory, mid-strength data.
*Cost:* small. One field, one state float, one action.

**2. Attack must match defence: slow brain-set coat, fast brain-set key.**
*Mechanism:* each cell keeps a coat, a point on a circle in `c.x1`, moved slowly toward an output of its own brain at an energy cost. A bite carries a key (another output, or `o5.z`). Damage scales with how well key matches coat. Coat is not sensed at first, so nobody is told anything, and the brain chooses both, not the genome.
*Incentive:* a rare coat is safe. A common coat feeds whichever biter matches it. So a big uniform colony becomes a target, which sets a limit on colony size that depends on what's around (kill the winner), and it rewards keys that track the current common coat. There's no fixed best answer, only a moving one.
*Predict:* cycling of coats and keys in space and time, travelling fronts, colonies that bloom and collapse, biters who specialise and can't switch instantly, and later (once coats can be felt) mimicry.
*Risk:* keys may settle at a broad average, so tune sharpness. Too sharp and predators die out, giving the rest monoculture of seed 2. Spirals die above the critical mobility and in small worlds: at 256 with our movement we may already be over Mc (about 9e-4 against 4.5e-4, rough), so test at 512. Rough, because cells don.t random-walk. One for Erik to judge against his rule: both values come from the brain and neither is sensed, so I class it as physics.
*Evidence:* Zaman 17/50 vs 0/50 (strongest in the whole note), Tierra, Decaestecker, Kerr, Reichenbach. Godoy warns intransitivity isn't automatically coexistence.
*Cost:* small. Two floats read at bite time, one line.

**3. Fights where numbers pay, against prey that needs them.**
*Mechanism:* (a) count a square around the target as open if it's empty *or* holds a cell that bit this target last tick (`seen` holds direction and `F_BIT`, safe at any distance). (b) `force = 2·S/(S+te)`, S being the summed energy of those attackers, so several small cells can beat a fat one, and a lone small cell can't. (c) Raise `max_energy` so fat bodies exist. (d) Optional sense: difference between my voice and the voice in front. Both are brain-chosen, so it's no genome signal.
*Incentive:* per-head return rises with attackers against big prey and stays flat against small prey. Exactly the payoff shape group hunters need. A fat, rich resident or colony now has a price on it. Each small step pays on its own, so there are stepping stones (Lenski).
*Predict:* aggregations around fat prey, then attack coordination, then scent or voice use to attract others if food patchiness makes it pay, and mobbing of intruders with (d). Free riders and a plateau near a few attackers, like wolves.
*Risk:* colonies eat themselves faster, since neighbours are mostly family and half of bites hit relatives already. Density and diversity crash. The `pack` wound rule made diversity worse, though only at 128 and 6,000 ticks, and it kept the exposure penalty and let lone attackers exploit wounds, so it isn't this test. Square-law expectations may be too strong: fire ants are near linear.
*Evidence:* Chandra (defended prey), MacNulty, Burtsev and Turchin. Weak on the square law.
*Cost:* about 8 `seen` reads per bite. Very small.

**4. Settings that need no code: costlier births, fewer mutations, longer scent, stored food.** See C6. Reason: raises travel against birth, keeps rare behaviour, makes trails physically possible. This goes first because it's free, even though it's unlikely to be enough on its own.

**5. Terrain with no favoured scale, and land that differs in kind.**
*Mechanism:* replace the 32-square lattice noise (`terrain.wgsl:38`) with several octaves or a heavy-tailed mix, no dominant size. Add a few land kinds with real physics differences (the biomes branch has `tempo`, `drag`, `sunx`, `rotx`). Add a slow drift of sun over tens of thousands of ticks so old land stops paying.
*Incentive:* large territories and tiny niches both exist, and specialists on kind of land, not just amount.
*Predict:* heavy-tailed colony sizes, regional types that meet at passes, invasions when climate shifts.
*Risk:* heterogeneity gives only regional variety, not interaction. Fast drift averaged to nothing in the weather runs (1.1 species at drift 0.3, one seed). Chesson: weather without different responses adds nothing.
*Evidence:* biomes hint, Harvard (terrain changes behaviour), Neural MMO (regions). *Cost:* tiny, at world build.

**6. Life-history freedom.**
*Mechanism:* an output decides when to birth, so child size is whatever is in the womb, not fixed 5. Old-age hazard scales with metabolism, so a resting cell really ages slower (now `act.wgsl:53` hits it regardless). A wider womb cap gives wealth somewhere to go.
*Incentive:* cheap, many, weak against few, big, strong. Colonisers against holders.
*Predict:* weeds that fill empty land after a crash, big-offspring holders, seed banks over bad seasons, and varied colony size.
*Risk:* tiny children are eaten. Tilman's many-species outcome needs steep trade-offs (Adler). Dormancy without different responses isn't a storage effect.
*Evidence:* Tilman, Adler, Chesson. *Cost:* nearly nothing.

**7. Lasting structure and stores.**
*Mechanism:* spend energy to lay a wall or leave a store on a square, decaying slowly. Rock already blocks squares. Stores are `fa.y` deposits, with lower rot.
*Incentive:* a sink for idle wealth, and things worth raiding and defending. A place to come back to: a nest, without coding one. The world gets a history.
*Predict:* fortified colonies, tunnels, breaches. Paired with #3, the first thing a raid has reason to break.
*Risk:* walls freeze the world. Chromaria's too-strict case is a warning. Stores need defending, which needs friend-or-foe (#3d).
*Evidence:* Chromaria (opportunities), Geb (agents that transform their world), JaxLife and Harvard (terrain changes). *Cost:* moderate. Mostly `occ` handling and one more use of `seen`.

**8. Speed and body trade-offs.** The `mass` branch has drag and step costs already. Screen it combined with #3. Fat cells slow, thin cells fast, gives pursuit a winner. Armour alone: nothing at 128. Practice alone (`skills`): everyone grazed. Tests so far were thin, so no verdict.

**9. Not yet.** *Bonds and bodies:* a real new level, but the lattice can't move a body in one tick, and no system I found got there without handing out groups. *Bigger brains:* Geb says scale brain with world, but that matters only once there's something to compute. Ours is 672 weights, against about 22,000 in Harvard, so this may yet bite. *More senses, more actions, more size* without new incentives: no reason to expect anything, given the removal runs and Neural MMO's "magnifier" line.

### C6. Settings worth screening

| Setting (default) | Range | Why |
|---|---|---|
| `mut_rate` (0.1) | 0.003-0.1, log | Keep rare behaviour. C3. |
| `mut_size` (0.1) | 0.03-0.2 | Same. |
| `womb` (5), `body` (1) | 5-40, 1-10 | Birth against travel. |
| `max_energy` (40) | 20-200 | Fat prey, wealth. |
| `spend` (0.25) | 0.1-0.5 | Activity rate against income. |
| `move_rate` (8) | 4-16 | Price of travel. |
| `bite_power` (30) | 3-30 | 30 makes one bite worth many ticks of grazing. |
| `bite_yield` (0.5), `meat_yield` (0.7) | 0.2-0.9 | Length and strength of food chains. |
| `exposure` (on) | on, off | Removes the anti-teamwork rule. A free test of C4.1. |
| `commitment` (0.03) | 0.01-0.1 | Specialists against generalists. |
| `ground_cap` (4) | 1-16 | Stock for roamers against flow for sitters. |
| `sun` (0.05) | 0.02-0.1 | Thin worlds move more. Harvard: resources matter. |
| `season_ticks` (8000) | 0, 2k, 8k, 32k, 128k | Time niches. |
| `hazard` (0.00025) | 0-0.001 | Pace of turnover. |
| `upkeep_e` (0.0004) | 0-0.002 | Cost of holding wealth. |
| `meat_rot` (0.002) | 0.0005-0.01 | Scavenging. |
| `p0_decay` (0.005), `p1_decay` (0.02) | 0.0003-0.005, 0.001-0.02 | Trails physically possible. |
| `vision_steps` (3) | 3-5 | Seeing prey further than 4. Costs reads. |
| `rain_p` (0.00125) | 0-0.00125 | Whether outside newcomers are propping up "diversity". |

Method: random or Latin hypercube sampling, since many settings interact. Successive halving: 256 for 60,000 ticks drops dead and stagnant worlds. 512 for 300,000 on 3 seeds for the survivors, as seeds disagree hugely. 1024 only after. Don't discard physics of type #2 for failing at 256. Baseline diversity appears late, so a 30,000-tick cut would throw out everything. And no single score: DISHTINY shows novelty, complexity and adaptation diverge. Keep the best of each measure.

### C7. Spotting real depth automatically

Most useful first. Group counts by genome are not on this list. Echo and EcoSim both show lots of genomes with no behaviour to match.

1. **Persistent strategies.** Cluster behaviour on the 96 situations, count clusters over 1% that survive across log points. MODES-style filter. Add the same score for a neutral shadow.
2. **Energy-flow table.** Who gets energy from whom, by cluster: ground, meat, bites, gifts. Count real links. This gives trophic levels and finds a second-order predator.
3. **Coexistence where it happens.** Do clusters live side by side, or just in different places? Ask the overlap at 16-64 squares. Regional-only variety isn't depth.
4. **Rare-type advantage.** Per-head birth rate of each cluster against its local share, across patches of the same world. Falling means it's stabilised. It's cheap, from a single snapshot series. The proper test (Chesson's) is thinning a type to 1% on a saved world and seeing if it recovers.
5. **Change over time.** Turnover of the cluster mix, plus novelty against an archive of all past clusters. Stable, steady novelty and growing are different. Compare with the shadow (Bedau). Rising but random turnover is noise.
6. **Time-shift fights.** Past against present against future cells in small arenas, saved worlds only. If current always wins, escalation. If each era wins only against its own era's partners, Red Queen. Same design as the pond sediment work. Expensive, so finalists only.
7. **Cycles in a tournament.** Matrix of who beats whom. Split it into a strength ladder and cycles. [abstract](https://ar5iv.labs.arxiv.org/html/1901.08106), [abstract](https://arxiv.org/abs/2004.09468) These come from zero-sum games, so treat as loose inspiration.
8. **Group payoff curve.** For every target, how many attacked at once, against a shuffled control, and each attacker's gain by group size. This one directly detects hunting. A rise past 1 is what no current world can show.
9. **Scent that matters.** Follow rate of scent not self-emitted, against shuffled fields. Then the test from BRIEF: knock out scent on a saved world and compare each cluster's growth.
10. **Trails and travel.** Track re-use (share of movement on the top 5% of squares), long thin scent shapes, and centre-of-mass travel of clusters over 10,000 ticks. Directly addresses "sitting in place".
11. **Colony sizes against terrain.** Spread of size (`log10_sd`), and how much of it terrain sun explains. Colonies free of terrain are new behaviour, colonies tied to it are geography.
12. **Repertoire per cell.** Within-life switching of budget against between-cell differences. Tells a jack of all trades from specialists. Cost is one more moving statistic.

### C8. What's uncertain and where the literature is weak

- **Everything in C2-C5 is untested on our engine.** My count of limiting factors is a judgement. Levin is an equilibrium result, and our world isn't at equilibrium.
- **Levin, Tilman and Chesson come from simple equations.** They bound what can persist, not what a neural brain will discover. Raising a ceiling doesn't mean anything will reach it.
- **Open-endedness has no accepted test.** Geb's pass depends on a changed normalisation, and growth is only logarithmic. Most "open-ended" titles (JaxLife, Flow-Lenia) claim a step, not an arrival.
- **Theory with mixed field support:** R\* (disputed reading), Tilman 1994 (depends on steep trade-off), intransitivity (rare in one plant study), Lanchester square (rejected in fire ants), handicap (disputed), Cambrian causes (still debated).
- **Coevolution's headline result** has a matching rule built in, on an instruction-based genome. Whether a brain-set coat and key behaves the same with neural brains is the main open bet of this note. It may just settle into broad keys.
- **Raiding may need things no setting gives:** a nest, real relatedness, colony sizes in the thousands. Ants did it over millions of years. Getting trails may be out of reach even if every incentive is right, so count aggregation at prey, or scent changing cell behaviour, as success on the way.
- **Screening has its own bias.** Picking the world by a diversity score is an outer search on physics, not on cells. That's allowed, but it will find measurement artefacts, so finalists need knockouts, seeds and videos before anyone believes them.
- **Our in-house experiments** are thin. Biomes, weather, waste, roads, skills all say "no verdict"; armour, mass and pack mostly small worlds and short runs. None rules an idea out.

**Not reached.** MODES full text (blocked). Chromaria 2014 full text (only the abstract and the 2016 follow-up). Hardin 1960, Levin 1970, Tilman 1994, Hutchinson 1961, Holland's own Echo writing, Plowes and Adams, Mitri 2009, Stephens 1989, Taylor 1992 and Agrawal and Lively 2002: abstracts, citing papers or nothing, as marked. Fetching some PMC pages hit a robot check, so those carry `[abstract]` from search results. Pages I read through a summarising fetch tool (Harvard, JaxLife, EcoSim, ant robots) may have small errors in numbers.
