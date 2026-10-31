# Neural-particle evolution, 2024–2026

Read-only sweep of X, GitHub, arXiv, YouTube, and project sites (Oct 2026). Tags: **said** = author’s claim, **shown** = a number or result in a paper, repo, or docs, **inferred** = my reading. No third party has replicated the closed sims.

## Edge projects

### Fraser Price (@fraserpricee) — closed CUDA particle RNNs

- https://x.com/fraserpricee (threads 10–28 Oct 2026). No repo. He **said** he will open-source it later (10 Oct).
- **Model.** Particles in a fixed “heat bath” of layered fields (forces, scent, energy). Each senses local fields and gradients and chooses how much to deposit. No programmed cell walls, drives, or mitosis (**said**, 14 Oct).
- **Brain.** He **said** each particle is its own ~8-layer, ~10k-parameter RNN with basic obs/actions and an energy level (11 and 14 Oct). No diagram.
- **Scale.** He **said** 1e6 particles at ~250 steps/s wall (11 Oct) and 500+ steps/s (14 Oct), and that 1–10e6 nets of 1–10k params at 200+ steps/s is the hard target (“a pile of cuda for my hardware”). A 28 Oct reply compares 1e5 vs 4e5 on a larger grid. **Inferred:** 10e6 is the target, not a reported run. His 4× RTX Pro 6000 posts are about LLM tokens/s; he does not tie that box to this sim.
- **Open.** No.
- **Emerged (said, videos).** Persistent novelty over an 8 h fixed-config timelapse; clumps he reads as multicellular bodies, energetic “nuclei,” mitosis/fusion. He **said** he has quantified change at particle/genome level and that higher hierarchy is not yet measured. Backprop on survival “got a lot more boring.” Smaller interaction range looked static; raising the interaction ceiling looked more dynamic. Complexity ceiling “bounded by compute” (**said**).

### Binx (@BinxNet) — WGPU trail particles

- https://x.com/BinxNet/status/2111189025893593122 area, 16 Oct 2026. No repo linked in those posts.
- **Said:** custom Rust engine, WGPU/WGSL, egui. Each particle is a Fourier net that senses nearby trails and sets movement. Click-to-select a cohort into a new environment. Hazards exist. MLP and GRU options were under consideration, “no training yet.”
- Population, steps/s, and what actually evolved: not stated.

### Fluoddity — @OA_paperclips (aphid91), rendered by @BockJakob

- https://github.com/aphid91/Fluoddity · https://fluoddity.com · https://github.com/aphid91/Fluoddity3D
- **Shown (repo):** 2D GPU particles, ModernGL compute (OpenGL 4.3). No particle–particle collision; interaction is a decaying, diffusing trail with a flow vector (physarum-style, after Sage Jenson). A “rule” is a 10×8 matrix, 80 parameters, sum of sines. Desktop app is the real tool; the browser build is a demo. 3D (ModernGL + optional OptiX/CUDA) reuses the same brains by sampling a plane tangent to velocity (**said**, 2 Jun 2026).
- **Selection.** The user clicks a lineage to keep. That is artificial selection, not an unattended ecology.
- **Scale.** Architecture text **says** thousands to millions. One checked config constant is `600000 * WORLD_SIZE` (150k at the default 0.25). Steps/s not stated.
- **Emerged.** Flowing, branching, lightning-like patterns. **Inferred:** these are selected generative-art phenotypes, not measured open-ended evolution.
- BockJakob (**said**, 13–14 Oct 2026) renders Fluoddity density into a “cell” look. Fraser credited that look as inspiration, not the evolution loop.

### ALIEN — Christian Heinemann (@chrx_h)

- https://alien-project.org · https://github.com/chrxh/alien (BSD-3) · r/AlienProject · YouTube channel linked from the README
- **Shown (README + `resources/docs/neural-networks.md`):** 2D particle soft bodies and fluids. Creatures are bonded cells (15 types: sensor, muscle, attacker, injector, digestor, memory, communicator, …). Every cell has one network: 16 inputs (8 signal channels, 4 memory, energy, attacked, age, velocity) → 12 outputs (8 signals + 4 gated memory). One weighted layer plus an activation, not a deep net. Memory is the recurrence. Cells cycle every 6 steps. Genomes build offspring cell by cell. Energy is conserved between cells, free particles, and an optional pool. Mutations include weights and mutation rate. No external fitness function; selection is energy, predation, and space (**said** in README). Parameters can vary in space (layers, radiation) to make habitats.
- **Hardware.** Physics in CUDA; Vulkan render (needs Vulkan 1.3); AMD via HIP or SCALE. NVIDIA: CC ≥ 7.5. Headless `alien-cli` and Docker `chrxh/alien:nightly`. “Millions of particles in real time” (**said**). Steps/s and creature counts are not in the README.
- **Emerged.** ALIFE 2024 Virtual Creatures, first prize, video *Emerging Ecosystems* (**shown**: the award). Site **says** one grazer lineage ran 38e6 steps. README describes diverging lineages and coexisting populations; those ecosystem claims are the project’s, with video, not an independent metric paper.

### Microcosmos — Mark Tensen, Ciaran Regan (Sakana / Tsukuba), Bert Wang-Chak Chan, Mizuki Oka, Kenneth O. Stanley, Grisha Szep

- https://alife.institute/en/blog/microcosmos-release/ · https://arxiv.org/abs/2607.02954 (3 Jul 2026; ALIFE 2026)
- **Shown:** elastic filament chains in a 2D viscous fluid (Lattice Boltzmann + immersed boundary). JAX, end-to-end differentiable, plus a WebGPU demo. Cost is local, so they report linear scaling to 5e5 particles on a fixed grid (pairwise sims go quadratic). Controllers are a shared CPPN evolved with NEAT, or quality-diversity search, not one big net per particle.
- **Emerged (their experiments).** Hand-built worms, tadpoles, rays, cilia, jellyfish match low-Reynolds expectations, including Purcell’s scallop theorem (time-reversible strokes fail as viscosity rises). Gradient descent folds a 1000-node filament into MNIST-like digits. NEAT + an energy field produces swimming and chemotaxis. QD finds undulation, turning, multi-part motion, flipper paddling. They **say** birth, death, and multi-agent competition are supported but only begun. No steps/s in the blog.

### Sakana — David Ha (@hardmaru), @SakanaAILabs

These are gradient or search systems, not Darwinian particle RNNs.

- **ASAL** (Dec 2024): Akarsh Kumar, Chris Lu, Louis Kirsch, Yujin Tang, Stanley, Phillip Isola, Ha. https://github.com/SakanaAI/asal · https://arxiv.org/abs/2412.17799. CLIP/DINO score videos of Boids, Particle Life, Life-like CA, Lenia, and NCA. Three modes: match a text target, temporal novelty, illumination. JAX, CUDA install called out. **Shown:** new Lenia and Boids forms; some CA ranked as open-ended as Life (Life itself landed in their top 5% on that score). The score is a foundation-model embedding, not a biological metric.
- **Petri Dish NCA** (Nov 2025, ALIFE 2025): Ivy Zhang, Sebastian Risi, Luke Darlow. https://sakana.ai/pd-nca/ · https://github.com/SakanaAI/petri-dish-nca. Several small NCAs share a grid and keep training by gradient descent on growth. **Said:** cycles, territorial defense, spontaneous cooperation. They argue backprop may scale better than pure evolution.
- **Digital Ecosystems** (Apr 2026): Darlow. https://sakana.ai/digital-ecosystem/ · https://github.com/SakanaAI/digital-ecosystem. Browser. A few CNNs, each seeing 3×3, attack/defend while learning online. **Said:** learning holds the ecosystem together under live edits (walls, erasures, 40+ parameters).

### Lenia people

- **Bert Chan, GECCO 2023**, arXiv:2304.05639. Large JAX Lenia worlds. Implicit reproduction and selection. **Shown pattern:** a creative phase, then domination by fast-expanding patterns. He lists environment design, mass conservation, and energy limits as what was missing.
- **Flow Lenia** (Plantec, Hamon, Etcheverry, Oudeyer, Moulin-Frier, Chan). ALIFE 2023 best paper; journal version arXiv:2506.08569 (2025). Mass-conserving continuous CA; update-rule parameters live in the mass, so species can mix. JAX. **Shown:** 255 μs ± 3.11 μs per step on a Tesla T4, 1 channel, 10 kernels, 128×128. A 1024² clip starts with 144 creatures. Not a per-creature neural net.
- **Particle Lenia** (Mordvintsev, Niklasson, Randazzo, Dec 2022): https://google-research.github.io/self-organising-systems/particle-lenia/. Lagrangian cousin of Flow Lenia. Energy-based, not evolved brains.
- **Neural Particle Automata** (Kim, Pajouheshgar, Süsstrunk, Wenzel Jakob, Park), SIGGRAPH 2026, arXiv:2601.16096. https://selforg-npa.github.io/. Shared MLP (about 11k–38k params) on CUDA SPH particles; trained by gradient descent for morphogenesis and textures. 4k–16k particles in the paper’s table. WebGL2 demo. Adjacent: one learned rule, not a population of evolving nets.
- **Biomaker CA**, ALIFE 2024 (https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00814). JAX plant-like NCA biomes: seeds grow, reproduce, and can be meta-evolved. Open-sourced via the paper’s link.

### The Bibites — Léo Caussan (@TheBibites)

- https://www.thebibites.com · https://thebibites.itch.io/the-bibites · Steam app 2736860 · https://www.youtube.com/@TheBibites · r/TheBibites
- **Model.** 2D energy, physics, digestion, organs. Brains start empty and grow by a NEAT-like real-time algorithm (synapses, hidden neurons, activation functions). Fitness is survival and reproduction, not a scored generation. 0.6.0 (Oct 2024) added organs, speciation, lineage tree/mesh. Unity. He **said** (29 May 2024) the GPU was unused. July 2026 Patreon: a rewrite called BIOME (nodes, connections, and modules that can grow) on Unity DOTS/ECS, because the old GA was “archaic.”
- **Scale (said).** Early video: ~200 bibites before his old PC fell over (https://www.youtube.com/watch?v=myJ7YOZGkv0). Patreon: the pre-rewrite sim hit ~1 fps and ≤0.1× realtime around 10k bibites and 50k pellets. Default clock later split from rendering at 30 ticks/s so speed-up does not change physics.
- **Emerged (his videos, not a paper).** Herds, predators (rare), speciation and migration over 100 h and 1000 h runs, then arguments inside the sim about what counts as a species (“classification crisis” in the 100 h video). Biomes and evolving plants are a stated direction; “Biomes” in his posts is this feature and the BIOME algorithm, not a separate lab.
- Source of the game is not the product. Free on itch; Steam adds workshop/achievements.

### Others on the same question

- **Genesis Engine** (teknium1): https://github.com/teknium1/genesis-engine. Browser WebGL2. Mass conservation, embedded genome, metabolism, novelty drive. Written as a response to “why virtual life stalls.”
- **Evo Particle Life** (bendavidsteel): https://github.com/bendavidsteel/evoparticlelife. JAX / jax-md. Attraction rules mutate in spatial patches. Example API uses 4000 particles. Not neural controllers.
- **Weinreich AI** (1 Feb 2026): https://www.youtube.com/watch?v=92DVR51GX48. **Said:** 3000 creatures, raycast vision, evolving deep nets and body traits; herbivores, omnivores, predators. No code or steps/s in the description. Treat the behaviors as narration.
- **sundellviz, “REAL Evolution”** (23 Jul 2025): https://www.youtube.com/watch?v=kmyO5h-6G9U. Browser demos, including prey/predator nets copied on eating, with no explicit objective: https://sundellviz.github.io/dataviz/evolution_neural/

## Still-cited earlier work

- **Life Engine, Max Robinson (@max_romana, Emergent Garden).** https://thelifeengine.net · https://github.com/MaxRobinsonTheGreat/LifeEngine · https://www.youtube.com/watch?v=4XEklaH9k6k (2020). Grid organisms of colored cells (mouth, producer, mover, killer, armor, eye). The “brain” is a gene table: for each seen cell type, approach, flee, or ignore. He **said** (2021, “How the Life Engine works”) a neural net would be too slow and too opaque. Community NN fork he linked: https://bboettcher3.github.io/LifeEngine/. Famous run **shows** zigzag vs figure-8 vs spike-tail, then total extinction. CPU, browser. Open.
- **biosim4, David Randall Miller.** https://github.com/davidrmiller/biosim4 · https://www.youtube.com/watch?v=N3tRFayqVtk (2020, ~4.5e6 views). Grid, C++. Default ini **shows** 128×128, population 3000, 300 steps/generation, genomes of 24 connections, ≤5 internal neurons, plus fixed sensors and actions. Selection can be a challenge at generation end (`chooseParentsByFitness`). **Shown in the video:** after 1200 generations a common brain used a handful of neurons (forward density → move). CPU threads. Open, MIT.
- **Primer (Justin Helps).** Playlist https://www.youtube.com/playlist?list=PLKortajF2dPBWMIS6KF4RLtQiG6KQrTdB. Blender + Python. Genes are traits (speed, size, sense), not nets. Selection is “eat and get back.” Interactive port by MinuteLabs: https://labs.minutelabs.io/evolution-simulator. Teaching model of selection, not open-ended novelty.
- **Species: Artificial Life, Real Evolution (Quasar / James Schumacher).** Steam 774541, https://speciesdevblog.wordpress.com. 3D creatures, biomes, genes, nursery for artificial selection. Not neural particles. Community posts **say** public updates stalled after about 2022. Useful as the “biomes + speciation” game people still name.
- **Growing NCA** (Mordvintsev et al., 2020) is the parent of the Sakana and NPA line: one convolutional rule, trained, then frozen. https://distill.pub/2020/growing-ca/

## People who are commenting, not shipping a sim

- **@carloAI** (12 Oct 2026, https://x.com/carloAI/status/2109762423737680057): an Interface explainer of Fraser’s project. Not an independent sim.
- **Matt Mazur (@mhmazur)** wrote up Kenneth Stanley’s ARC Prize 2026 talk (https://x.com/mhmazur/status/2114714032850473187) and called Fraser’s clip lifelike. He is not running this class of sim in the posts found.
- **Stanley** is a coauthor on ASAL and on Microcosmos, and the measurement talk below is his.

## Arguments and lessons

**What goes quiet.** Chan 2023: diversity, then one fast expander. Fraser: backprop on “survive” looked boring; low interaction range looked static; other projects “collapse into something boring pretty fast” (**said**). Life Engine’s flagship video ends in extinction. Fluoddity stays interesting only while a person keeps picking. Stanley (via Mazur): activity statistics can look strong in a toy that never gets biologically rich; a sandbox can exhaust itself and the ceiling is the sandbox.

**What they say keeps it moving.** No external objective, with selection coming from energy and competition (Fraser; ALIEN README; Bibites’ real-time NEAT). Mass conservation plus genes that sit inside the matter (Flow Lenia; Chan’s 2023 wishlist; Genesis). Spatial niches: ALIEN radiation layers, Bibites jungle maps, Digital Ecosystems walls. Evolvable modular structure (Bibites BIOME) so the brain’s inputs are not a fixed list. Stanley: collect stepping stones inside one ongoing run; an objective hides the stones (the “objective paradox” from *Why Greatness Cannot Be Planned*). He also **said** a benchmark for open-endedness will get gamed, and that interestingness is a judgment, not a counter.

**Disagreement on learning.** Fraser’s backprop trial and Stanley’s objective warning pull against Sakana’s PD-NCA bet that online gradient descent is the scalable path. Both are arguments with demos, not a shared metric. Fraser **said** the field does not agree on a definition; his own is non-static novelty, major transitions, and heredity-driven hierarchy above chance.

**Scaling tips that showed up more than once.** Keep the inner loop local (grid fields, filaments, SPH hash grids), or pairwise cost wins. Put the genome in the substrate so copying is the physics. Conserve mass or energy so “grow forever” is not the optimum. Vary the world in space so one strategy cannot cover it. Separate sim ticks from rendering (Bibites) before you chase a GPU port. Fraser’s practical note: fusing a custom CUDA kernel to *his* GPUs is what made 1e6 × 10k feasible, and that is why it is not portable yet.

## GPU API, as people actually used it

- **CUDA:** Fraser (closed, hardware-specific kernels), ALIEN physics, NPA’s SPH. Chosen where each agent is a real net or a bonded body and the author will write the kernel.
- **Vulkan:** ALIEN’s renderer, interop with CUDA, ray tracing. AMD path is HIP or SCALE, not a Vulkan compute rewrite. Bibites documents `-force-vulkan` only as a Linux display workaround.
- **JAX:** Chan’s large Lenia, Flow Lenia, Microcosmos, ASAL, Biomaker CA, evo-particle-life. Used when the same code must do evolution and gradients. Microcosmos then added a WebGPU viewer.
- **WebGPU / WebGL2:** shareable demos (Microcosmos, Genesis, NPA demo, Fluoddity browser). The heavy Fluoddity app is OpenGL 4.3 compute, with OptiX only for 3D path tracing.
- **WGPU/WGSL:** Binx’s Rust engine. Portable GPU without a CUDA toolkit; he had not swapped in GRU/MLP yet.
- **Unity CPU:** Bibites. Author **said** the GPU sat idle through mid-2024; the 2026 bet is ECS threads, not a GPU brain.

## Found vs inferred

**Found.** Fraser’s 1e6 × ~8×10k RNN, 250 and 500+ steps/s, closed CUDA, and the boring-if-trained remark are his posts. ALIEN’s 16→12 cell net, 6-step cycle, CUDA+Vulkan+HIP/SCALE, and the 2024 prize are in the repo. Microcosmos’s filament/LBM/JAX design, NEAT and QD behaviors, and linear scaling to 5e5 are in the blog and abstract. Flow Lenia’s T4 timing is in the 2025 paper. Bibites brain growth, CPU-bound history, and BIOME rewrite are his videos and Patreon. Fluoddity’s 80-parameter rule and user selection are in the repo.

**Inferred.** Fraser has not published a run at 10e6; the 4× RTX Pro 6000 box is a different project; “multicellular” is his reading of the video. Binx has not shown a sustained ecology. Fluoddity’s pretty frames are not evidence of natural selection. Sakana’s “open-ended” CA scores and PD-NCA cooperation are model- and author-judged. Species ALRE is the older biomes game, not part of the 2024–26 GPU wave. No Reddit thread in this sweep independently measured any of the closed sims.
