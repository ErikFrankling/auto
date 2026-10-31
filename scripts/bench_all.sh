#!/usr/bin/env bash
# Every benchmark behind the report. Run from the repo root in the dev shell:
#   nix develop -c scripts/bench_all.sh
# Results append to results/*.jsonl. Delete them first for a clean run.
# The largest grids need --force and hold the GPU for 50-110 ms per step, so the
# desktop stutters while they run. Do those when nobody is using the machine.
set -u
B="./target/release/auto --adapter 7900"
V="./target/release/auto-vk --adapter 7900"
GPU=results/gpu.jsonl
API=results/api.jsonl

# Which Life kernel is fastest, and at which size.
SIZES=1024x1024,4096x4096,16384x16384,32768x32768,65536x65536
$B bench life.naive --sizes 1024x1024,4096x4096,16384x16384 --tag kernels --seconds 2 --out $GPU
$B bench life.bits32.256x4 --sizes $SIZES --tag kernels --seconds 2 --out $GPU
$B bench life.bits64 --sizes 1024x1024,4096x4096,16384x16384,32768x32768 --tag kernels --seconds 2 --out $GPU
$B bench life.ghost.t8.w30.r256 --sizes $SIZES --tag kernels --seconds 2 --out $GPU
for t in 1 2 4 8 12 16; do
  $B bench life.column.t$t.g256.k3 --sizes 1024x1024,16384x16384,65536x65536 --tag blocking --seconds 2 --out $GPU
done

# Life at every scale: a thousand cells to the VRAM ceiling.
LADDER=64x64,128x128,256x256,512x512,1024x1024,2048x2048,4096x4096,8192x8192,16384x16384,32768x32768,65536x65536,131072x122880
$B bench life --sizes $LADDER --tag ladder --seconds 2 --warmup 16 --force --out $GPU
$B bench life.sharded2 --sizes 185344x185344 --tag ladder --seconds 4 --warmup 2 --force --out $GPU
if [ "$(cat /sys/class/drm/card*/device/mem_info_vram_used | tail -1)" -lt 6000000000 ]; then
  $B bench life.sharded3 --sizes 226464x227584 --tag ladder --seconds 4 --warmup 2 --force --out $GPU
fi

# The API question: the same kernels through each layer.
for k in life.bits32.256x1 life.bits32.256x4 life.column.t16.g256.k3; do
  for s in 256x256 1024x1024 4096x4096 16384x16384 65536x65536; do
    $B bench $k --sizes $s --tag api-wgpu-safe --seconds 2 --out $API
    $B --trusted bench $k --sizes $s --tag api-wgpu-trusted --seconds 2 --out $API
    $V $k --sizes $s --seconds 2 --out $API > /dev/null
  done
done

# Every automaton, across sizes, on the big GPU.
for a in brain wireworld; do
  $B bench $a --sizes 1024x1024,4096x4096,16384x16384,32768x32768,65536x65536 --tag automata --seconds 2 --out $GPU
done
$B bench rule30 --sizes 1024x2,65536x2,1048576x2,16777216x2 --tag automata --seconds 2 --out $GPU
$B bench rule110 --sizes 1024x2,65536x2,1048576x2,16777216x2 --tag automata --seconds 2 --out $GPU
for a in cyclic hodgepodge ants forest sandpile sand; do
  $B bench $a --sizes 1024x1024,4096x4096,16384x16384 --tag automata --seconds 2 --out $GPU
done
$B bench grayscott --sizes 1024x1024,4096x4096,8192x8192,16384x16384 --tag automata --seconds 2 --out $GPU
for a in lenia smoothlife mnca; do
  $B bench $a --sizes 512x512,1024x1024,2048x2048,4096x4096,8192x8192 --tag automata --seconds 2 --warmup 4 --out $GPU
done
$B bench physarum --sizes 1024x1024,4096x4096,8192x8192 --tag automata --seconds 2 --out $GPU

# What a brain costs: one RNN per agent, each with its own weights.
for h in 8 16 32 64 128; do
  for n in 65536 1048576; do
    $B bench brains.h$h.b32 --sizes ${n}x1 --tag brains --seconds 2 --warmup 4 --out results/brains.jsonl
    $B bench brains.h$h.b8 --sizes ${n}x1 --tag brains --seconds 2 --warmup 4 --out results/brains.jsonl
  done
done

# The same brain written for the GPU: 4x4 weight blocks, no scratch copy of the state.
for h in 8 16 32 64 128; do
  for b in 32 8; do
    $B bench brainsb.h$h.b$b --sizes 4096x1,65536x1,1048576x1 --tag brains --seconds 1 --warmup 2 --out results/brains.jsonl
  done
done

# The CPU.
./target/release/auto cpu --sizes 1024x1024,4096x4096,16384x16384,65536x16384 --seconds 2 --hashlife --threads 4 --out results/cpu.jsonl

# Everything on the CPU through the same WGSL, using Mesa's llvmpipe.
for a in life brain cyclic grayscott lenia physarum; do
  ./target/release/auto --adapter llvmpipe bench $a --sizes 1024x1024 --tag llvmpipe --seconds 2 --warmup 2 --out results/llvmpipe.jsonl
done

# The integrated GPU, for scale.
./target/release/auto --adapter RAPHAEL bench life --sizes 1024x1024,4096x4096 --tag igpu --seconds 2 --out results/igpu.jsonl

# The same kernels in a real browser. Build first: (cd crates/auto-web && trunk build --release --public-url ./)
BROWSER=firefox node scripts/browser_bench.mjs crates/auto-web/dist results/browser.jsonl \
  life.bits32.256x1:256x256 life.bits32.256x1:1024x1024 life.bits32.256x1:4096x4096 life.bits32.256x1:16384x16384 \
  life.bits32.256x4:256x256 life.bits32.256x4:1024x1024 life.bits32.256x4:4096x4096 life.bits32.256x4:16384x16384 \
  life.column.t16.g256.k3:256x256 life.column.t16.g256.k3:1024x1024 life.column.t16.g256.k3:4096x4096 life.column.t16.g256.k3:16384x16384 \
  brain:1024x1024 cyclic:1024x1024 grayscott:1024x1024 lenia:1024x1024 physarum:1024x1024
BROWSER=chrome-agent node scripts/browser_bench.mjs crates/auto-web/dist results/browser.jsonl \
  life.bits32.256x1:256x256 life.bits32.256x1:1024x1024 life.bits32.256x1:4096x4096 life.bits32.256x1:16384x16384 \
  life.column.t16.g256.k3:256x256 life.column.t16.g256.k3:1024x1024 life.column.t16.g256.k3:4096x4096 life.column.t16.g256.k3:16384x16384 \
  brain:1024x1024 cyclic:1024x1024 grayscott:1024x1024 lenia:1024x1024 physarum:1024x1024
