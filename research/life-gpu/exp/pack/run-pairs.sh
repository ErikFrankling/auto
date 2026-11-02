#!/usr/bin/env bash
set -euo pipefail
# Run only after the 128 software and 256 GPU pilots succeed.
# Four sequential runs: <=16 minutes active GPU time in total.
for seed in 1 2; do
  for rule in false true; do
    label=control
    if [ "$rule" = true ]; then label=pack; fi
    /tmp/life-stable/life --shaders crates/auto-life-gpu/shaders \
      --size 512 --ticks 60000 --log-every 5000 --seed "$seed" \
      --max-minutes 4 --set "pack_on=$rule" --set ledger=true \
      --out "research/life-gpu/exp/pack/${label}-seed${seed}.jsonl"
  done
done
