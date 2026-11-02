# Armour

Every brain can spend x0 to build hardness in c.x0 (0–1). Hardness fades 0.5% per acting tick, divides bite and heat damage by 1+3h, divides movement probability by 1+3h, and costs 0.03h upkeep per tick even at rest. Construction adds 1.5 times effective x0 work. Children inherit the brain but zero hardness: free physical inheritance would evade construction costs. Hardness is a dissipative capability, not stored energy; building is already deducted in `spent` and `lose(spent-a1.x)`, extra upkeep is deducted with `lose(pay)`, heat loss and bite transfers retain their original ledger paths. Fading and newborn reset remove no energy because hardness contains none.

Defaults: `armour_on=false`; enabling it enables x0 without requiring `x0_on`. All other defaults unchanged. CPU runs use `LP_NUM_THREADS=2`. GPU 256 probe used default pacing; subsequent runs request target-ms=2 and duty=0.4. `run.sh` reproduces paired 512 worlds and a saved-world ablation, max 3 minutes per evolution run and 2 per replay. No cargo, force, or 1024 runs.

Software driver, 128 square worlds, tick 20,000 (same seed and settings except armour switch):

| Seed | Armour | Cells | Effective species | Species explain behaviour | Bite diet | Moving |
|---|---|---:|---:|---:|---:|---:|
| 1 | off | 4264 | 1.105 | .014 | .003 | .128 |
| 1 | on | 3937 | 1.206 | .015 | .019 | .168 |
| 2 | off | 5439 | 1.193 | .016 | .013 | .035 |
| 2 | on | 0 | — | — | — | — |

At tick 2,000 armour gave effective species 15.651 vs 12.797 (seed 1), but 2.581 vs 13.842 (seed 2); the apparent early diversity did not last. At 20,000 almost all surviving seed-1 cells fall into one effective genome group, so there is no evidence for stable genetically distinct armoured, running and hunting lifestyles. Inspect the species_at, lifestyles, diet histograms and species tables in the JSONL rather than treating the group count alone as success.

`snapshot.pl` reads actual hardness and behaviour from saved worlds. Seed 1 at 20,000: 3677 live low-hardness cells (<.2), 176 medium (.2–.6), 65 high (>=.6). Low vs high mean movement is .1763 vs .0177; bite diet .0128 vs .1268; hardness .0232 vs .8676. Thus armour is used, but the hard cells include bite feeders rather than demonstrating a distinct passive refuge. Counts from snapshot exclude nonpositive-energy cells awaiting cleanup; report population can differ.

Seed-1 saved-world ablation, 5,000 further ticks: keeping armour ends at 3117 cells / 1.476 effective species; disabling it ends at 3076 / 2.290. This does not establish a coexistence benefit. Disabling armour also disables its action slot and removes costs and protection together; it does not isolate resistance alone. Predation survives in seed 1 (bite diet .019 vs control .003), so armour is not an observed free win ending predation; seed-2 extinction prevents a general claim.

Validation: `perl crates/auto-core/tests/armour.pl` passes 23 scalar CPU-reference/source checks (not a full GPU state parity test). Software 128 runs compile and execute the actual shaders. Off-switch 40-tick software report matches the original shader after excluding timings and sorting tied species entries. GPU 256 reaches 2,000 ticks, 3135 cells / 33.764 effective species; longest submission 61.3 ms despite default pacing, motivating lower pacing afterward. Video: `/home/erikf/auto/videos/life-gpu/armour/software-seed1.mp4` (software driver, split genome/diet; not coloured by hardness).

Verdict: negative on the tested small worlds. Armour has meaningful costs and changes behaviour, but has not produced durable genetically distinct lifestyles. Larger GPU runs remain subject to the shared GPU queue; do not infer cross-size robustness from software results.

Genetic sketch check (first up to 128 live slots in each hardness bin, descriptive and spatially biased): seed-1 mean gap low/high .3955, within low .3481, within high .3246. This modest difference is not evidence of separate species; the full population's species explain behaviour is only .015.

Limitation: the GPU 512 matrix never acquired the shared lock after about 30 minutes of waiting across probe/matrix jobs. Its pending job was stopped; no 512 results or GPU video are claimed. A cars assignment job repeatedly held the shared cars lock while a life process held the auto lock waiting for it. No other process or lock was touched. `run.sh` remains available for the unperformed larger-world comparison. Software video is supplemental, not evidence of success.
