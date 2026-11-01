# Neural evolution at scale: what we know, and where to go next

Written 2026-10-31 for the discussion before building it. Nothing here is implemented yet except the throughput measurements. Sources are in [neural-evolution-academic.md](notes/neural-evolution-academic.md), [neural-evolution-scene.md](notes/neural-evolution-scene.md) and [neural-nets-on-gpu.md](notes/neural-nets-on-gpu.md).

## Your diploma project against Fraser's sim

Your 2022 project ([repo](https://github.com/ErikFrankling/Simulation-of-the-evolutionary-process)) is the same idea at a thousandth of the scale, and it already ran into the problems this field is still arguing about.

| | Your diploma project (2022) | Fraser Price (2026, closed source, his claims) |
|---|---|---|
| Stack | Unity, C#, CPU | Custom CUDA on 4 RTX Pro 6000 |
| Population | Hundreds (400 prey, 200 predators to start) | 1 million, with 10 million as his target |
| World | Continuous 125 by 125, plants on a grid | Particles in layered fields |
| Brain | NEAT: topology grows from nothing, about 12 inputs and 2 outputs | Fixed RNN per particle, up to about 10k parameters |
| Senses | 6 ray casts | Local fields and gradients |
| Actions | Turn and move | Energy split across eat, divide, thrust, signal, bond, fight and more |
| Species and roles | Hand-coded: plant, prey, predator | None. Anything like predation has to emerge |
| Selection | Energy, starvation, predation, age | Energy only |
| Fitness function | None | None |
| Bodies | One agent is one body | Bodies form from bonded particles |

**What is the same:** no objective, energy as the only currency, mutation on reproduction. That is the part that makes it evolution and not training.

**What your results already show:**
- Predators evolved a real strategy, spinning to sweep their ray and charging when they saw prey.
- Raising move cost stopped that strategy evolving, because staying still was safer than learning to aim. Small cost changes decide what evolves at all.
- Population swings ended in extinction, and thin survivors lost what the population had learned. The field calls this a major failure mode. The Harvard work finds larger worlds make behaviour more reliable.
- You found the network "led to very simple strategies" dominating. Stanley's ceiling argument is the same point: the sandbox limits what can evolve, not only the search.

**The biggest real change from then:** your organisms were given roles and bodies, so nothing above "a better predator" was possible. Fraser's only get physics. If anything, that is the idea to take from him, more than the scale.

## Is WebGPU good enough for neural nets?

**Yes for this project, for now.** I would not switch. The measurements behind that are in [the report](../report/).

- WebGPU itself costs almost nothing. wgpu matched hand-written Vulkan at large sizes, Chrome matched native, and Firefox was 4% to 17% behind.
- Kernel design matters far more than API. Rewriting the same RNN with 4x4 weight blocks, without changing API or GPU, made it over 10 times faster (from about 147 to about 1,585 steps/s at 1 million agents with 128 parameters).
- **CUDA is not an option on your card.** ROCm is, officially, but nothing suggests it would beat this: independent per-agent weights are limited by memory, and no API changes that.
- The real limits are hardware: 20 GB of VRAM and about 800 GB/s of memory bandwidth (AMD's figure).

### What fits, measured

| Population | Parameters each | Weights | Steps per second, brains only |
|---|---|---|---|
| 1 million | 128 | 8-bit | 1,585 |
| 1 million | 384 | 8-bit | 458 |
| 1 million | 1,280 | 8-bit | 131 |
| 65,536 | 4,608 | 8-bit | 502 |
| 65,536 | 17,408 | 8-bit | 119 |

The rough ceiling is 140 to 210 billion multiply-adds per second when weights stream from memory. With 65,536 agents at 128 parameters, which stay mostly in cache, it reached about 650 billion. That gap suggests sharing genomes between clones would pay off a lot, and most of a population is near-clones anyway.

1 million agents at 10k parameters would be 10 GB at 8-bit, so it fits, but I would extrapolate about 15 steps/s, not measured. Fraser claims 250 to 500 on his far bigger machine. Roughly 1,000 to 4,000 parameters at a few hundred thousand agents looks like the practical range here, and the simulation around the brains will cost more on top.

**Where WebGPU would hurt:** 2 GB per buffer through wgpu on Linux, though Chrome offered 4 GB (the sharding here already works around it), no 64-bit integers in the browser, and no ready-made batched matrix libraries. Gradient training would also be painful, but Fraser's result and Stanley's argument both suggest not starting there.

**Keeping browser support costs little:** only the 8 storage buffer default and some workgroup limits. Use native-only features only if a measurement shows they are worth it.

### Your Modal credits

$150 buys about 49 hours of an RTX Pro 6000, 38 of an H100 or 136 of an A10, from [Modal's price list](https://modal.com/pricing) as of today. As you said, that is not much, and I would hold it.

Long runs on your own GPU cost nothing except desktop stutter. The credits make sense later for one thing: a run too big for 20 GB, once a small version is already doing something interesting. Modal does not promise Vulkan in its containers, so that would need a short test first.

## What the field has actually found

**Nobody has shown open-ended evolution in a strong sense.** Bounded and finite versions have support. No system in this review shows indefinite, ecosystem-level innovation, and it is not something a finite run can prove.

Closest work to what you want:
- [EcoEvoJax](https://arxiv.org/abs/2302.09334): RNN agents on a grid, births and deaths from energy. Closest academic design. Published with code and seeds.
- [Harvard ecology](https://arxiv.org/abs/2510.18221): over 60,000 live agents with their own MLPs on one H100. Long-range foraging and predation emerged. Bigger worlds helped more than bigger starting populations.
- [JaxLife](https://arxiv.org/abs/2409.00853): agriculture and communication-like behaviour from LSTM agents.
- [ALIEN](https://github.com/chrxh/alien): bonded-cell creatures with genomes. Runs in CUDA, or on AMD through HIP or SCALE. Worth reading for its bonding and construction rules.
- [Flow-Lenia](https://arxiv.org/abs/2506.08569): no individual brains, but mass conservation and genes that live inside the matter.

**Things that repeatedly go wrong:** a burst of diversity followed by one fast-expanding winner (Chan 2023), extinction from boom and bust (your diploma, Life Engine), a sandbox that runs out of new things to do (Stanley), and things that look alive but measure as nothing. Fraser says he can't yet measure what he sees above the genome level.

**Things with evidence behind them:** conserve energy or mass so growth can't be free, vary the world in space so no strategy covers everything, keep interactions local so cost stays linear, keep the genome inside the matter so copying is physics. DISHTINY found novelty, complexity and adaptation can each move independently, so one "interestingness" score will mislead you.

## What I would build first

Plan for discussion, smallest useful version first:

1. **Start small on purpose.** 100,000 agents, about 256 parameters, 8-bit weights. That is well inside the measured range, so runs are quick and many.
2. **World:** a few diffusing fields on a grid, as Physarum already does here. Energy from patchy, renewing sources. Reuse this repo's field and scatter code.
3. **Agents:** sense local fields, split energy across move, harvest, divide and emit. Leave bonding until a single-cell ecology is stable.
4. **Measure from day one:** births and deaths by cause, lineage IDs, genome hashes, and actual use of each action. Keep controls with random and frozen genomes. Without these you can't tell evolution from pretty noise.
5. **First real question:** one connected world or islands with some migration, at the same total compute? It tests the refuge idea directly, and it speaks to the extinctions in your diploma.

## Decisions

Decided by Erik on 2026-10-31:

- **Fixed RNN, not NEAT.** We focus on what scales well on a GPU.
- **No fitness function.** Energy alone decides who survives and how evolution goes.

The toy in [toy/](toy/) tests the world rules on a small scale. [life/](life/) is the larger CPU test of Erik's goal: one kind of cell, no designed types, with predation, cooperation and species left to emerge.

Still open:

- **Natural selection only, or an outer search over world settings?** I would start with natural selection only. Sakana-style search over physics is a good later step, since physics decides so much.
- **Bodies from bonding?** It is the most striking thing in Fraser's videos and the hardest to run on a GPU. I would do it third, after a stable ecology.
- **What counts as success?** Fraser's bar is continued novelty from one fixed config. Picking the measure before running will save arguing with pretty videos later.
