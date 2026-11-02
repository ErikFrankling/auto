# pack: wounds reward concentrated attacks

Every body resists bites while intact. Paid bite work increases its `x2` wound state (capped at 1); later bites scale from 0.12 to 1 times baseline damage. Wounds retain 70% per own tick and newborns start intact. No identity, role, signal, or grouping rule is added. A persistent lone attacker can also exploit wounds: rapid concentrated attacks have an advantage, but packs are not required.

Settings: `pack_on=false` by default; enabled runs use `pack_on=true`, `pack_intact=0.12`, `pack_wound_keep=0.7`, `pack_wound_rate=12`. Other defaults unchanged. No leakage added. Existing action spending debits and calls `lose()` for bite work. Wounds hold no energy. Target extraction still divides into `keep`, `spill`, and `lose(take-keep-spill)` with unchanged 0.5 bite yield. Fully wounded damage cannot exceed baseline.

Matched controls, software driver, size 128, tick 6000 (shared baseline logs were absent):

| Seed | Rule | Cells | Effective species | Species explain behaviour | Bite energy share | Bites on similar | Neighbours |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | off | 1862 | 10.288 | 0.168 | 0.043 | 0.289 | 2.743 |
| 1 | on | 3067 | 1.172 | 0.021 | 0.042 | 0.488 | 3.080 |
| 2 | off | 2887 | 1.488 | 0.080 | 0.030 | 0.290 | 4.969 |
| 2 | on | 2566 | 1.194 | 0.015 | 0.023 | 0.457 | 4.404 |

Numbers are matching-tick samples from `sw10k-{control,pack}-seed{1,2}.jsonl`, not time averages. Runs aimed at 10000 ticks with a 0.5-minute cap; seed 2 control stopped at 6250, hence comparison at 6000. Earlier 1000-tick pairs are also saved. At 6000, the most bite-dependent lifestyle has neighbours 3.727→4.730 (seed 1), 2.816→4.612 (seed 2), but bite energy share falls 0.321→0.172 and 0.133→0.079. Increased crowding does not establish packs. World bites-on-similar rises in both seeds: no evidence of better kin sparing. Logs do not count attackers per victim, so target concentration is unmeasured.

Verdict: worse diversity at these settings in both small-world seeds; no demonstrated evolved pack hunting. Effective species approaches one while behaviour separation declines. This is near monoculture, but bite energy remains low, so calling it a cannibal monoculture would be unsupported. No long-term or larger-world success claim, saved-world ablation, or video.

Validation: 128 software shader runs and two standalone CPU reference tests (`crates/auto-core/tests/pack.rs`) pass: cooperation, decay, cap, ledger, switch off. These are reference-rule checks, not full GPU/CPU world parity. No Cargo or application builds. The 256 GPU pilot completed 2000 ticks (3780 cells; longest final-interval submission 9.0 ms); it is not a matched control.

Reproduce: `LP_NUM_THREADS=2 /tmp/life-stable/lifesw --shaders crates/auto-life-gpu/shaders --size 128 --ticks 10000 --log-every 2000 --max-minutes 0.5 --seed N --set pack_on=BOOL --out PATH`.
`agent-job run NAME -- bash research/life-gpu/exp/pack/run-pairs.sh` runs matched size-512 controls and treatment for two seeds sequentially, capped at 4 minutes each (16 minutes active GPU total).

GPU limitation: the 256 pilot waited about 15 minutes before running. The 512 batch then waited another 15 minutes on the shared GPU locks and was cancelled at the queue cap before any world started. No 512 results exist; all jobs from this experiment have stopped. No lock files or other jobs were touched. Video: none.
