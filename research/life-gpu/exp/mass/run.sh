#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../../.."
shaders=crates/auto-life-gpu/shaders
out=research/life-gpu/exp/mass
# The first 256 probe must complete successfully before starting this sweep.
for drag in 0.3 3; do
  /tmp/life-stable/life --shaders "$shaders" --size 256 --ticks 2000 --log-every 1000 --max-minutes 1 --set mass_on=true --set mass_drag="$drag" --set ledger=true --out "$out/drag${drag}-256.jsonl"
done
for seed in 1 2; do
  for on in false true; do
    video_args=()
    if [[ "$seed:$on" == "1:true" ]]; then
      mkdir -p /home/erikf/auto/videos/life-gpu/mass
      video_args=(--video /home/erikf/auto/videos/life-gpu/mass/seed1.mp4 --frame-every 100 --width 1280 --height 720 --zoom 0.1 --seconds-per-shot 8 --view split)
    fi
    /tmp/life-stable/life --shaders "$shaders" --size 512 --seed "$seed" --ticks 30000 --log-every 5000 --max-minutes 3 --set mass_on="$on" --set ledger=true --out "$out/seed${seed}-${on}-512.jsonl" "${video_args[@]}"
  done
done
