# Cellular automata people actually run

Ranking is an inference from LifeWiki attention, Golly/Ready/Visions of Chaos coverage, YouTube view counts, and 2025–26 posts. Rules, numbers, and hardware below are from the linked pages. No single vote exists.

Neighbourhood shorthand: Moore = 8-cell square (or its range-r extension). “Bits” means two states.

## 1. Twenty systems

1. **Conway’s Life** — the reference culture (gliders, guns, universal computation). B3/S23: birth on 3, survive on 2–3. Moore r=1, bits. https://en.wikipedia.org/wiki/Life-like_cellular_automaton https://mathworld.wolfram.com/GameofLife.html
2. **Lenia** — continuous “creatures”; Chan’s catalogue and a Rainmaker clip (1.3M views) are why hobbyists care. Float state in [0,1], kernel radius R (often 13), growth bell, Euler step dt=1/T. https://ar5iv.labs.arxiv.org/html/1812.05433 https://x.com/Rainmaker1973/status/1649839179189157890
3. **Gray–Scott** — mitosis, coral, worms. Two floats, reaction plus a 3×3 Laplacian, not a neighbour-count CA. https://www.karlsims.com/rd.html
4. **MNCA (Slackermanz)** — solitons, flocks, crystals from several big neighbourhoods. Bits or floats; each neighbourhood has its own threshold bands, applied in order. GPU is the practical target. https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata/
5. **SmoothLife** — organic gliders in any direction. Floats; inner disc plus outer annulus; smooth birth/survival. https://arxiv.org/abs/1111.1567 https://www.youtube.com/watch?v=KJe9H6qS82I
6. **Cyclic CA** — spirals (“demons”) from noise. Integer states 0..N−1. A cell steps toward a neighbour’s next state if the count passes a threshold. https://en.wikipedia.org/wiki/Cyclic_cellular_automaton
7. **Hodgepodge** — BZ-like spirals. Integers 0..n (healthy / infected / ill). Healthy uses two infection weights; infected climbs by a neighbourhood average plus a speed; ill resets. https://people.sc.fsu.edu/~jburkardt/classes/ysp_2003/dewdney_hodgepodge.pdf
8. **Brian’s Brain** — refractory “neurons”, many c ships. Generations B2/S/3: fire on 2 firing neighbours, then one dying state. Moore r=1, 3 states. https://en.wikipedia.org/wiki/Life-like_cellular_automaton
9. **Bugs / Bosco (Larger than Life)** — blobby life at range 5; studied, Turing-complete via a 2023 rosary. Bits, big Moore square. Strings disagree by one (see §4). https://conwaylife.com/wiki/OCA:Bosco%27s_Rule
10. **Neural CA** — trained pattern growth and the worms in Emergent Garden (~213k views on the explainer). Floats, often 16 channels, 3×3 perception, small dense net. https://distill.pub/2020/growing-ca/ https://www.youtube.com/watch?v=3H79ZcBuw4M
11. **Flow Lenia** — mass-conserving Lenia; parameters can sit in the field so species mix. Floats, flow field, not a stencil count. https://arxiv.org/abs/2506.08569 https://sites.google.com/view/flowlenia/
12. **Wireworld** — circuits you can watch. 4 states (empty, conductor, electron head, tail). Head→tail, tail→conductor, conductor→head if 1–2 heads. Moore r=1. https://conwaylife.com/wiki/List_of_universal_cellular_automata
13. **Rule 30** — the iconic chaotic triangle; people treat it as a picture and a PRNG. 1D, r=1, bits. https://world-of-mathematics.com/blog/cellular-automata/
14. **Rule 110** — the 1D rule people cite because Cook showed it is Turing-complete. Same stencil as Rule 30. https://world-of-mathematics.com/blog/cellular-automata/
15. **HighLife** — Life plus a replicator (B36/S23). Moore r=1, bits. https://conwaylife.com/wiki/List_of_universal_cellular_automata
16. **Day & Night** — self-complementary and still rich (B3678/S34678). Moore r=1, bits. Same list.
17. **Star Wars** — generations B2/S345/4; soups full of fast ships. 4 states, Moore r=1. Same list.
18. **Greenberg–Hastings** — the small excitable-media CA (target and spiral waves). Usually 3 integers: resting, excited, refractory; excite if enough excited neighbours. https://www.dpi.inpe.br/gilberto/cursos/papers/Fisch1991.pdf
19. **Particle Lenia** — ecology of moving blobs (a Feb 2026 clip is the recent “this looks alive” post). Agents in the plane, radial kernels, no grid. https://google-research.github.io/self-organising-systems/particle-lenia/ https://x.com/mathelirium/status/2020525160684478471
20. **Critters** — reversible “bouncing” life. Bits, Margolus 2×2 blocks, partitions alternate; the LifeViewer form conserves population. https://conwaylife.com/wiki/OCA:Critters

## 2. Scale and speed

These are cell-updates, or compressed futures, not a stored trillion-cell grid. I did not find a sourced GPU holding 10^12 resident cells. At 1 bit that grid is 125 GB (arithmetic).

- **Hashlife, not a big array.** Rokicki’s write-up: the breeder to 10^12 generations, population 1,302,083,334,180,208,337,404, in under a second, by quadtree memoization. https://jacobfilipp.com/DrDobbs/articles/DDJ/2006/0604/0604b/0604b.html
- **Golly Hashlife:** generation 6,366,548,773,467,669,985,195,496,000 of Rendell’s Turing machine, under 30 s on an Intel Core Duo 2 GHz. https://en.wikipedia.org/wiki/Hashlife
- **Marek Fiser, CUDA, GTX 580 (1.5 GB, 512 cores), bit per cell:** he reports ~2.8×10^13 cell evaluations/s and 480× a serial Xeon E5530. That is throughput, not universe size. https://marekfiser.com/projects/conways-game-of-life-on-gpu-using-cuda
- **Fujita et al., GTX TITAN X:** 1024 steps of 16384×16384 in 0.163 s (about 1.7×10^12 cell-steps/s; I divided). 357× one core of a Xeon X7460. Multi-step updates kept in registers, warp shuffles. https://www.computer.org/csdl/proceedings-article/candar/2015/9797a011/12OmNyprnwy
- **bryanoliveira, RTX 3080 + Ryzen 7 3700X:** 13500×13500 (182.25 million) cells, up to 729 gen/s with rendering, ~8.5 GB VRAM as the limit. https://github.com/bryanoliveira/cellular-automata
- **A40 bandwidth wall (2025–26):** int8 65536×65536 (~4 GB) is quoted at a 11.5 ms/step DRAM floor (696 GB/s) and 22.5 ms in Triton. A later multi-step kernel reports 5.4 ms for 8 steps (0.68 ms/step) by using shared memory. https://sudonull.com/fr/gpu-acceleration-of-the-game-of-life-cuda-triton-benchmarks https://www.boristhebrave.com/2025/09/21/more-accelerated-game-of-life/
- Dense Life is DRAM-bound: one bit in and out per cell per step. Hashlife wins on repetitive patterns and is the wrong tool for a live soup. Inference from those two write-ups, not a new benchmark.

## 3. Gems, including late chatter

- **MNCA** is the hobby favourite Softology pushed. Jason Rampe put Slackermanz’s shaders into Visions of Chaos (he wrote of 11,000+ shaders; later sets use 32 or 52 floats). https://softologyblog.wordpress.com/2021/02/14/even-more-explorations-with-multiple-neighborhoods-cellular-automata/ https://softology.com.au/voc.htm
- **Flow Lenia** (2025 Artificial Life / arXiv) is the paper people point at for evolution inside a CA: mass stays put, rule weights can advect and mutate. https://direct.mit.edu/artl/article/31/2/228/130572/Flow-Lenia-Emergent-Evolutionary-Dynamics-in-Mass
- **SmoothLifeL** is still the clip (Tim Hutton, ~312k views). Ready is the explorer. https://github.com/GollyGang/ready
- **Primordia** is Chan’s named step between Life and Lenia (hard thresholds, smooth kernel). https://colab.research.google.com/github/OpenLenia/Lenia-Tutorial/blob/main/Tutorial_From_Conway_to_Lenia_%28w_o_results%29.ipynb
- **Aquarium** (pattern 5N7KKM) is the multi-kernel soliton later work starts from. https://arxiv.org/html/2406.04235v1
- **Physarum** (Jones-style agents on a trail field) is the other “it looks alive” system. Emergent Garden (292k subs) sends viewers to Slackermanz and to neural patterns, not to Life. https://www.youtube.com/@EmergentGarden https://www.youtube.com/watch?v=NPlAEnGX4Dg
- **2025–26 hobby posts:** Rule Safari ranked EightLife (B3/S238) then Life, then HoneyLife, IronLife, Pedestrian Life, LowDeath, HighLife, on a mechanical screen, not a poll. https://r-hatim.github.io/rule-safari/ A TikTok-style browser, rulehunt.org, came out of ALIFE 2025. https://www.reddit.com/r/cellular_automata/comments/1obigk4/a_tiktoklike_interface_for_finding_interesting_2d/ CruiserOne’s “Evolution” (N=17, 240×135, predecessor eats successor) and a 16-state mass-conserving Golly rule drew the recent praise. https://www.reddit.com/r/cellular_automata/comments/1pcs7f4/evolution_from_chaos_to_order/ https://www.reddit.com/r/cellular_automata/comments/1rlokcd/16_state_mass_conserving_ca
- Also standard in Golly and still pretty: Seeds (B2/S), Life without Death (B3/S012345678), and the MCell-era life-like rules Maze, Coral, Diamoeba. Rulestrings for the first two are on the universal-CA list. https://conwaylife.com/wiki/List_of_universal_cellular_automata
- Langton’s ant is an agent (turn, flip, move), loved for the highway, not a lattice rule. I am not stuffing it into the twenty.

## 4. Seeds, palettes, parameters

**Lenia.** Public check of Chan’s catalogue: Orbium unicaudatus, R=13, T=10 (dt=0.1), β=[1], μ=0.15, σ=0.015, core index 1. Mass about 73.7, speed about 0.62 cells/step. Load the RLE; a blob is not this animal. Random patches do birth other species (Chan). A third-party dump of the same file lists Orbium bicaudatus σ=0.014, Gyrorbium μ=0.156 σ=0.0224, Scutium μ=0.290 σ=0.045. Kernel-core names (Gaussian vs polynomial) disagree across write-ups; trust `animals.json`. https://autonomaniac.github.io/lenia-field-guide/ https://github.com/Chakazul/Lenia/blob/master/Python/animals.json https://raw.githubusercontent.com/kosmickroma/somnivex/main/nca/lenia.py

**Particle Lenia.** `mu_k=4, sigma_k=1, w_k=0.022, mu_g=0.6, sigma_g=0.15, c_rep=1`. https://google-research.github.io/self-organising-systems/particle-lenia/

**Gray–Scott (Sims).** DA=1, DB=0.5, dt=1. Laplacian 3×3: centre −1, edge 0.2, corner 0.05. Start A=1, B=0, paint a little B=1. He gives f=0.055, k=0.062 as typical; mitosis f=0.0367, k=0.0649; coral f=0.0545, k=0.062. Interesting band on his map: k≈0.045–0.07, f≈0.01–0.1. Picture: A white, B black, optional emboss. https://www.karlsims.com/rd.html

**SmoothLifeL (Hutton’s render notes).** Birth 0.257–0.336, survival 0.365–0.549, αn=0.028, αm=0.147, dt=0.05, outer radius 20, radius ratio 3, 512² torus, state [0,1] drawn black→white. His config line also contains `10.0`, so confirm whether the code wants radius 10 or 20. A common non-L preset is birth 0.278–0.365, survival 0.267–0.445, inner 7, outer 21. https://www.youtube.com/watch?v=KJe9H6qS82I https://github.com/duckythescientist/SmoothLife/blob/master/smoothlife.py

**MNCA, hand-built, two neighbourhoods (ring and a filled disc).** Averages, later rules overwrite. Rule A: n0 sets 1 on [0.210,0.220], else 0 on [0.350,0.500], [0.750,0.850], [0.120,0.150]; n1 sets 0 on [0.100,0.280] and 1 on [0.430,0.550]. Rule B shifts those edges (0.185–0.200, 0.343–0.580, 0.150–0.180, 0.150–0.280, 0.445–0.680). Shaders: https://www.shadertoy.com/view/7ll3R7 https://www.shadertoy.com/view/sll3R7 Continuous cousin: https://www.shadertoy.com/view/7ts3D7 Softology’s early integer rule: die if sum0∈[0,17] or sum2∈[9,21] or sum3∈[78,89] or sum3>108; live if sum0∈[40,42] or sum1∈[10,13]. Neighbourhood shapes are the figures on that post, not restated as counts here. https://softologyblog.wordpress.com/2018/03/09/multiple-neighborhoods-cellular-automata/

**Cyclic.** Threshold 1 and N in the teens, random states, Moore or von Neumann: demons, then the grid locks to period N (or N+1 if N is odd on von Neumann). Reiter’s pictures are the period-16 case. CruiserOne’s pretty 2025 soup is N=17 on 240×135. Equal hue steps are the usual paint; that mapping is a convention, not part of the rule (inference). https://en.wikipedia.org/wiki/Cyclic_cellular_automaton

**Hodgepodge.** Dewdney’s Gerhardt–Schuster sample: n=100, k1=2, k2=3, von Neumann, 20×20. Reiter’s Moore (3×3) sets that look like the classic frames: (a,b,c,n) = (2, 3, 0.17, 100), and the four he prints (2, 6, 0.16, 100), (4, 1, 0.2, 100), (3, 2, 0.25, 100), (3, 2, 0.35, 100). https://webbox.lafayette.edu/~reiterc/mvq/hodgepodge/withj_hodgepodge.pdf

**Bugs strings, unresolved off-by-one.** LifeWiki: `R5,C2,S33-57,B34-45,NM`. Golly help: `R5,C0,M1,S34..58,B34..45,NM`. Slackermanz, 11×11 including centre (121 cells): die on 0–33 and 58–121, force on for 34–45, else keep — i.e. birth 34–45, survive 34–57. https://golly.sourceforge.io/Help/Algorithms/Larger_than_Life.html

**Life soups.** R-pentomino, acorn, and the Gosper gun are the seeds everyone already knows; Golly’s pattern folder is the practical source. I am not recopying RLE here. https://golly.sourceforge.net/

**Neural CA training setup (Mordvintsev et al.).** 16 floats, RGBA in the first channels, alive if alpha>0.1, perception 3×3, update dense-128 ReLU then dense-16, zero-init so it starts as the identity. https://distill.pub/2020/growing-ca/
