# mass

Stored energy slows movement and makes successful steps cost more. With m=max(E,0)/6, the movement chance is the existing chance divided by (1+drag*m); an empty step costs 0.04*(1+m). A living lighter neighbour can be swapped into the mover's old square for an additional 0.08*(1+m+their_E/6). Every extra cost is deducted and passed to lose(); swapping transfers no energy. No roles, genome signals or new actions are added.

Defaults: mass_on=false; mass_scale=6, mass_drag=1, mass_step_cost=0.04, mass_push_cost=0.08. Enable with --set mass_on=true. Other physics stays at defaults; all logged runs enable ledger. The 256 sweep uses drag 0.3, 1 and 3. run.sh reproduces the GPU sweep after the initial probe.

All rows below are endpoints at tick 2,000. The 128 worlds use LP_NUM_THREADS=2 software Vulkan; 256 uses the desktop GPU. Rule-on/control pairs differ only in mass_on.

| Size | Seed | Rule / drag | Cells | Effective species | Species explain behaviour |
| --- | --- | --- | ---: | ---: | ---: |
| 128 | 1 | off | 268 | 12.797 | 0.285 |
| 128 | 1 | on / 1 | 316 | 7.362 | 0.467 |
| 128 | 2 | off | 285 | 13.842 | 0.386 |
| 128 | 2 | on / 1 | 284 | 3.374 | 0.151 |
| 256 | 1 | on / 0.3 | 3988 | 44.442 | 0.397 |
| 256 | 1 | on / 1 | 2130 | 61.240 | 0.417 |
| 256 | 1 | on / 3 | 2580 | 63.428 | 0.355 |

At 128 seed 1, a heavy genome group has mean energy 26.466 and movement 0.003; a lean group has 4.107 and 0.074. Their heads_to_bigger/smaller are 0.001/0.002 and -0.017/-0.004. This is a size/mobility difference, not a demonstrated chase/flee split. Reported groups remain mostly ground feeders. Species explain behaviour improves in seed 1 but falls in seed 2; effective species falls in both. Seed 1 species_at ranges from 15.070 at 0.1 to 3.240 at 0.3; seed 2 from 3.751 to 2.528. Threshold dependence limits the species claim.

Verdict: the physical trade-off is present; these short runs do not demonstrate durable genetically distinct chasing/fleeing lifestyles. They cannot decide whether one body plan eventually wins. Larger, longer matched controls and saved-world rule-off interventions remain necessary.

Validation: 128 software compile/run probes (64 and 512 ticks), all three 256 GPU probes, two software seed pairs, and two standalone CPU reference tests passed. No cargo ran. Added GPU-versus-CPU swap/occupancy/snapshot tests and a mass-enabled whole-world ledger test are unrun under the no-cargo instruction. New dissipation is accounted for by construction; runtime ledger balance is not exposed by these JSON logs.

A pushed cell keeps its stamp, energy, voice and voluntary movement flags. If it had not acted and is swapped into an already-processed square, its turn can be delayed until next tick; the tests allow that ordering. Writes stay at the original square and front square, including occ, both pos entries, and both seen_next entries. No displaced cell is counted as voluntarily chasing.

Video: /home/erikf/auto/videos/life-gpu/mass/early128.mp4 (6.667 seconds; early software run, not evidence of long-term coexistence). video128.jsonl repeats seed1-true-128's endpoint. No video is committed.

The 512 GPU comparison was queued after the successful 256 ladder, but its first job never started within a sixty-minute wait (03:56–04:57 local time). Stopped the mass-sweep agent-job at the cap; no 512 results or intervention results exist. The reproducible run.sh retains the intended two-seed, 30,000-tick paired protocol. No 1024 run was justified.
