#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
shaders=crates/auto-life-gpu/shaders
out=research/life-gpu/exp/armour
mkdir -p /home/erikf/auto/videos/life-gpu/armour
for seed in 1 2; do
  for mode in control armour; do
    on=false
    extra=()
    if [[ $mode == armour ]]; then
      on=true
      extra=(--save "/tmp/armour-seed${seed}.world")
    fi
    /tmp/life-stable/life --shaders "$shaders" --size 512 --target-ms 2 --duty 0.4 --ticks 20000 --log-every 5000 --seed "$seed" --max-minutes 3 --set "armour_on=$on" --set ledger=true --out "$out/${mode}${seed}.jsonl" "${extra[@]}"
  done
done
for on in true false; do
  extra=()
  if [[ $on == true ]]; then
    extra=(--video /home/erikf/auto/videos/life-gpu/armour/seed1.mp4 --width 960 --height 540 --frame-every 40 --zoom 0.1 --seconds-per-shot 8 --view split)
  fi
  /tmp/life-stable/life --shaders "$shaders" --size 512 --target-ms 2 --duty 0.4 --load /tmp/armour-seed1.world --ticks 5000 --log-every 1000 --seed 1 --max-minutes 2 --set "armour_on=$on" --set ledger=true --out "$out/replay-${on}.jsonl" "${extra[@]}"
done
