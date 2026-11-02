# Brief for one world experiment

Read this whole file before doing anything. Erik's goal, in his words: "emergent structure, strategies, life, evolution, not designed life that is boring". He wants different genomes and species to have a reason to form, living in different ways, "be that passive, scared, predatory, many such things, not artificially introduced, but make it possible for different ways of life to be viable".

You own one idea for making that happen. Implement it as physics in the shaders, test it from small to big, and report honestly whether it did anything.

## The rule that matters most

Add physics, never roles. A rule must be something every cell is subject to or can choose to use, with costs. Not allowed: cell types, castes, organs, predator or prey flags, scores, fitness functions, anything that tells a cell who its relatives are, and anything a cell gives off or shows because of its genes rather than because its own brain chose to. Erik rejected a genome-derived odour for this reason: "let's not decide for them". If you are unsure whether a rule is physics or a role, ask whether a cell's brain could evolve to use it or ignore it. Colour by genome exists only for humans watching.

Good physics usually means a trade-off no cell can escape, so that no single way of living beats all the others.

## The engine

`crates/auto-life-gpu/`. One cell per square, each with its own RNN of 864 8-bit weights and 8 memory units. Each tick a cell senses, thinks, then splits its energy across rest, move, ground, meat, grow, emit, heat, bite, give, enzyme, armour, clean, plus four free slots.

Everything you need to change is plain text. **You do not need to run cargo, and must not.** Every key in `shaders/defaults.json` becomes a WGSL constant: a number `foo_bar` becomes `FOO_BAR: f32`, and a switch becomes `FOO_BAR: bool` and `FOO_BAR_F` (1.0 or 0.0). A key ending in `_decay` also gives `STEM_DECAY_KEEP`.

| File | What it does |
|---|---|
| `shaders/act.wgsl` | One cell's tick: sense, think, spend, interact, birth, move |
| `shaders/sweep.wgsl` | Field physics: diffusion, decay, sun, rot. Runs twice a tick |
| `shaders/births.wgsl` | Writes a child's genome: crossover, mutation |
| `shaders/rain.wgsl` | Occasional random newcomers |
| `shaders/terrain.wgsl` | Builds the world. The word per square holds sun (16 bits), season phase (8), rock (bit 24), and from bit 28 the 4 bits of the room when biomes or walls are on |
| `shaders/view.wgsl` | Colours for the video |

Ten rules already exist, all off by default. `research/life-gpu/RULES.md` lists each one and the layout. Spare capacity reserved for you, so no Rust changes are needed, except that a new input or output must be added to `input_live` or `output_live` in `config.rs`:

- Per-cell state: none left. A new rule needs a bigger cell record, which means old saves won't load.
- Free budget actions: turn on `x3_on`. `x0` is mineral uptake, and `o6.y` and `o6.z` are read by mineral and birth rules. In `act.wgsl`, `s3` holds their share, and you work out their energy as `s3 * spend_here * scale`. They come from output row `o6`.
- Free inputs: `i12.w`, `i13.w`.
- Free fields: `fc.w`, with `fcw_flow` and `fcw_decay`.
- Counters: `ctr[C_X0]` to `ctr[C_X0 + 15]`, shown in the log only if you read them out, so prefer the existing measures.

Limits on what a dispatch may touch, or results go wrong silently: cells on the same dispatch are 4 squares apart. A cell may **write** only its own square and squares 1 away, and **read** up to 2 away, from the square where its tick began. `seen` is a read-only snapshot and can be read at any distance. WGSL has no closures, and function-local arrays are about ten times slower, so spell things out with vec4s. Keep the energy ledger honest: if energy leaves a cell or the ground and doesn't go elsewhere, call `lose(x)`. Energy must never appear from nothing.

## Running

```
/tmp/life-stable/life   real GPU, through a queue. Blocks until your run has finished, and shows its output
/tmp/life-stable/lifesw software driver, no GPU. Slow, for checking your shaders compile and behave
```

Run from your worktree and always pass your shaders:

```
LP_NUM_THREADS=2 /tmp/life-stable/lifesw --shaders crates/auto-life-gpu/shaders --size 128 --ticks 2000 --log-every 1000 --out /tmp/x.jsonl
/tmp/life-stable/life --shaders crates/auto-life-gpu/shaders --size 512 --ticks 60000 --log-every 10000 --seed 1 --out research/life-gpu/exp/NAME/seed1.jsonl
```

`--set key=value` changes a setting for one run. Other flags: `--seed`, `--max-minutes`, `--video path.mp4 --frame-every N --zoom 0.1 --seconds-per-shot 8 --view split`, `--save`, `--load`. `--zoom` is cells per pixel for the automatic camera that flies to the most diverse and predatory place.

Wrong WGSL panics with the compiler's message and a line number in the generated source.

## Rules of this machine. Breaking these freezes Erik's desktop

- This GPU also draws Erik's 4K screens, and other projects are using it. Every run goes through one queue, which takes its turn on the GPU and runs several jobs per turn. A wait of 30 to 60 minutes is normal, and giving up early wastes the experiment, so keep waiting. Queue your whole batch at once, one `agent-job` for each run, so they all go in the same turn. Never kill another process and never touch `/tmp/*-gpu.lock`.
- Go small to big, always: 128 on the software driver until it compiles and runs, then a short 256 on the GPU, then real results at 512, and 1024 only once 512 is doing something worth seeing. Do not go above 1024. Results at 128 are only a check that the code runs, since a world that small can't hold many ways of living, so don't base a verdict on them.
- Each run is cut off at 4 minutes of GPU time, which at 512 is roughly 250,000 ticks, ten times anything earlier in this project. Use `--ticks 300000 --log-every 25000`. Worlds differ a lot by seed, so use seeds 1, 2 and 3 where you can. To go on longer, `--save` the world and `--load` it in a new run. Aim for about 20 minutes of GPU time in total, and at most 6 runs in the queue at once. Never pass `--force`, and don't raise `--duty` or `--target-ms`.
- Software-driver runs: keep them short, and always set `LP_NUM_THREADS=2`.
- Start long runs with `agent-job run NAME -- CMD` so they survive, and wait with `agent-job attach NAME`. Do not use `&`.
- Do not run cargo, nix builds, or anything heavy. Don't commit videos. Put them in `/home/erikf/auto/videos/life-gpu/NAME/`.

## What to measure

Each log line is JSON. Compare with the baseline in `/home/erikf/auto/research/life-gpu/exp/base/` at the same size and seed. It may not be there yet, since the GPU is shared. If it isn't, run your own control with your rule switched off, which you should do anyway for anything you claim.

| Field | Meaning |
|---|---|
| `cells`, `density`, `births`, `births_mixed`, `deaths` | Population. `deaths` has starved, bitten, burned, old |
| `effective_species`, `species_at` | Genome groups at threshold 0.4, and at several. Genomes drift fast, so below about 0.3 this only finds recent family. Numbers that hold steady from 0.3 to 0.7 suggest real species |
| `genome_gap` | Real weight differences between random pairs |
| `species_explain_behaviour` | Share of how cells live that species explains, counting groups of at least 1% and corrected for how many there are. Near 0: species are only labels. High: species really live differently. **This is the number closest to Erik's goal**, and it needs at least two sizeable groups to be above 0 |
| `species`, `lifestyles` | Tables of groups by genome, and by behaviour ignoring genes: `lives_on` (ground, meat, bite, gift), `budget`, `moving`, `heads_to_bigger` (negative means fleeing bigger bodies), `heads_to_smaller` (positive means chasing), `heads_to_food`, `follows_scent`, `neighbours`, `age`, `spread`, `bites_on_similar` |
| `from_other_cells_hist` | How many sampled cells get each tenth of their energy from eating others. Two humps means predators and grazers |
| `founder_lines` | How many original lineages survive |

What counts as success: several lifestyles that coexist for a long time, differ in genome, and last across seeds. Say so plainly if your idea does nothing, makes things worse, or only looks good in a video. A null result is useful. Use at least 2 seeds before claiming anything, and run a control that differs only in your rule.

Beware: a cell choosing a behaviour isn't the same as it mattering. Test what matters by turning your rule or the behaviour off (`--load` a saved world and `--set something_on=false`) and seeing what happens to the population.

## What to deliver

In your worktree, on your branch:

- `research/life-gpu/exp/NAME/README.md`: under 40 lines. The rule in one paragraph, the settings, a table of results against baseline, what you saw, your verdict. Numbers only from your logs. Plain words.
- `research/life-gpu/exp/NAME/*.jsonl`: your logs.
- Your shader and `defaults.json` changes. New rules must be behind a switch or setting in `defaults.json` that defaults to **off**, so all experiments can be merged and combined later.
- Commit it all. Don't push, don't merge, and don't touch other experiments' folders.

Your final message: 5 lines at most. The verdict, the numbers behind it, the branch and commit, the video path, and anything that went wrong.
