#!/usr/bin/env python3
"""Screens physics for the soft-cell engine: many settings, many seeds, measured world by world.

    soft_screen.py queue NAME COUNT [--ticks 200000] [--minutes 18] [--controls 2] [--no-stick 1]
    soft_screen.py analyse              measures finished batches, once
    soft_screen.py watch                the same, forever
    soft_screen.py rank [N]             worlds most like bodies, and each set against plain physics

A batch is one 1024-wide map holding 16 walled worlds 256 wide, with one physics and one seed per
world. Each runs in two stages: the long run with dumps through it, then 600 ticks with a dump every
10, since how long cells stay together can only be read from close dumps.

Runs go through the GPU queue's held list. Env: SOFT_BIN, SOFT_ANALYSE, SOFT_SHADERS, SOFT_RUNS.
"""
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("SOFT_OUT", ROOT / "research/life-soft/screen"))
RUNS = Path(os.environ.get("SOFT_RUNS", "/tmp/lg/soft"))
SPOOL = Path("/tmp/life-spool")
BIN = os.environ.get("SOFT_BIN", "/tmp/lg/soft-bin/soft")
ANALYSE = os.environ.get("SOFT_ANALYSE", "/tmp/lg/soft-bin/soft-analyse")
SHADERS = os.environ.get("SOFT_SHADERS", "/tmp/lg/soft-bin/shaders")
SIZE, TILE = 1024, 256
WORLDS = (SIZE // TILE) ** 2
ENV = "export LD_LIBRARY_PATH=/nix/store/la62431p5qf1ggr3rdzql9dksv5fhicc-vulkan-loader-1.4.363.0/lib\nexport PATH=/run/current-system/sw/bin:$PATH\nexport LIFE_DUTY=0.9\n"

# Wide on purpose. The middle of each range is near the default, unless the default looked like a limit.
RANGES = {
    "sun": (0.008, 0.25, True),
    "stick_max": (0.05, 4.0, True),
    "stick_fee": (0.00002, 0.003, True),
    "gap": (0.05, 0.6, True),
    "glue_sharp": (1.0, 12.0, True),
    "k_push": (0.2, 4.0, True),
    "drag": (0.5, 6.0, True),
    "thrust_rate": (15.0, 300.0, True),
    "push_power": (20.0, 800.0, True),
    "locked": (0.0, 0.98, False),
    "enzyme_flow": (0.02, 0.24, True),
    "enzyme_decay": (0.002, 0.05, True),
    "heat_harm": (0.01, 1.5, True),
    "heat_rate": (0.2, 6.0, True),
    "heat_scale": (2.0, 30.0, True),
    "bite_power": (2.0, 200.0, True),
    "bite_yield": (0.1, 0.9, False),
    "give_power": (1.0, 60.0, True),
    "give_yield": (0.5, 1.0, False),
    "meat_yield": (0.2, 0.95, False),
    "meat_rot": (0.0002, 0.02, True),
    "upkeep": (0.001, 0.03, True),
    "mut_rate": (0.02, 0.4, True),
    "mut_size": (0.03, 0.3, True),
    "max_energy": (15.0, 120.0, True),
    "womb_cap": (5.0, 40.0, True),
}
# Each set changes this many settings, so a rule's effect isn't always mixed with everything else.
ALWAYS = ["sun", "stick_max", "thrust_rate"]


def draw(rng, key):
    lo, hi, log = RANGES[key]
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
    return float(f"{v:.4g}")


def sample(rng):
    keys = ALWAYS + rng.sample([k for k in RANGES if k not in ALWAYS], rng.randint(2, 8))
    return {k: draw(rng, k) for k in keys}


def flags(physics):
    def text(v):
        return "true" if v is True else "false" if v is False else str(v)
    return " ".join(f"--set {k}={text(v)}" for k, v in physics.items())


def queue(name, count, ticks, minutes, controls, no_stick, order):
    RUNS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(sum(map(ord, name)) * 7919)
    (SPOOL / "held").mkdir(exist_ok=True)
    total = controls + no_stick + count
    for i in range(total):
        run = f"{name}-{i:04d}"
        d = RUNS / run
        d.mkdir(parents=True, exist_ok=True)
        seed = rng.randrange(1, 2**31)
        if i < controls:
            physics, kind = {}, "plain"
        elif i < controls + no_stick:
            physics, kind = {"stick_on": False}, "no stick"
        else:
            physics, kind = sample(rng), "mix"
        (d / "spec.json").write_text(json.dumps({"run": run, "kind": kind, "seed": seed, "size": SIZE, "tile": TILE, "ticks": ticks, "physics": physics}))
        sets = flags({**physics, "batch_tile": TILE})
        job = SPOOL / "held" / f"0-b-{order}-{run}"
        job.write_text(
            f"cd {d}\n{ENV}export LIFE_MAX_MINUTES={minutes + 2}\n"
            f"{BIN} --shaders {SHADERS} --size {SIZE} --seed {seed} {sets} --ticks {ticks} --log-every {ticks // 8}"
            f" --max-minutes {minutes} --out log.jsonl --dump-every {ticks // 16} --dump-dir dumps --png end.png --save world.bin || exit $?\n"
            f"{BIN} --shaders {SHADERS} --load world.bin --ticks 600 --log-every 600 --dump-every 10 --dump-dir fine --out /dev/null\n"
        )
        (SPOOL / "logs" / job.name).write_text("")
    print(f"queued {total} batches of {WORLDS} worlds as {name}, {controls} plain and {no_stick} without sticking")


def world_summaries(path):
    try:
        data = json.load(open(path))
    except (OSError, ValueError):
        return {}
    worlds = data.get("worlds") if isinstance(data, dict) else None
    if worlds is None:
        return {0: data}
    return {w.get("world", i): w for i, w in enumerate(worlds)}


def crop_maps(d, tile_png_dir):
    end = d / "end.png"
    if not end.exists():
        return
    tile_png_dir.mkdir(parents=True, exist_ok=True)
    per = SIZE // TILE
    for i in range(WORLDS):
        x, y = i % per, i // per
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(end), "-vf", f"crop=iw/{per}:ih/{per}:iw*{x}/{per}:ih*{y}/{per}", "-q:v", "3", str(tile_png_dir / f"t{i:03d}.jpg")],
            check=False,
        )


def analyse_once():
    OUT.mkdir(parents=True, exist_ok=True)
    results = OUT / "results.jsonl"
    seen = set()
    if results.exists():
        for line in open(results):
            seen.add(json.loads(line)["batch"])
    done = 0
    if not RUNS.exists():
        return 0
    for d in sorted(RUNS.iterdir()):
        spec_file = d / "spec.json"
        if not spec_file.exists() or d.name in seen:
            continue
        marks = list((SPOOL / "done").glob(f"0-b-*-{d.name}"))
        if not marks:
            continue
        spec = json.loads(spec_file.read_text())
        code = marks[0].read_text().strip()
        over, close = d / "over.json", d / "close.json"
        if (d / "dumps").exists():
            subprocess.run(["nice", "-n", "15", ANALYSE, str(d / "dumps"), "--out", str(over)], capture_output=True, check=False)
        if (d / "fine").exists():
            subprocess.run(["nice", "-n", "15", ANALYSE, str(d / "fine"), "--out", str(close)], capture_output=True, check=False)
        a, b = world_summaries(over), world_summaries(close)
        log = []
        if (d / "log.jsonl").exists():
            log = [json.loads(line) for line in open(d / "log.jsonl") if line.startswith("{")]
        with open(results, "a") as out:
            for i in range(WORLDS):
                out.write(json.dumps({
                    "run": f"{d.name}-t{i:03d}", "batch": d.name, "index": i, "kind": spec["kind"], "exit": code,
                    "seed": spec["seed"], "physics": spec["physics"], "ticks": spec["ticks"],
                    "reached": (log[-1].get("tick") if log else None), "over": a.get(i), "close": b.get(i),
                }) + "\n")
        crop_maps(d, d / "maps")
        shutil.rmtree(d / "dumps", ignore_errors=True)
        shutil.rmtree(d / "fine", ignore_errors=True)
        done += 1
        print("measured", d.name, flush=True)
    return done


def get(r, *path, default=None):
    v = r
    for k in path:
        v = v.get(k) if isinstance(v, dict) else None
    return default if v is None else v


MEASURES = {
    "cells": lambda r: get(r, "close", "cells"),
    "in bodies of 3+": lambda r: get(r, "close", "bodies", "share_in_3_plus"),
    "in bodies of 10+": lambda r: get(r, "close", "bodies", "share_in_10_plus"),
    "largest body": lambda r: get(r, "close", "bodies", "largest"),
    "bodies last vs touching": lambda r: ratio(get(r, "close", "membership", "half_life_ticks"), get(r, "close", "crowds_by_touching_only", "half_life_ticks")),
    "kin within vs nearby": lambda r: ratio(get(r, "over", "relatedness", "gap_between_nearby_bodies"), get(r, "over", "relatedness", "gap_within_bodies")),
    "size is inherited": lambda r: get(r, "over", "size_at_fragmenting", "parent_to_offspring_slope"),
    "body generations": lambda r: get(r, "over", "deepest_body_generation"),
    "position predicts budget": lambda r: ratio(get(r, "close", "position", "r2_by_contact_count"), get(r, "close", "position", "r2_shuffled_max")),
}


def ratio(a, b):
    return a / b if a is not None and b else None


def rank(top):
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    live = [r for r in rows if (MEASURES["cells"](r) or 0) >= 200]
    print(f"{len(rows)} worlds, {len(live)} alive")
    for m, f in MEASURES.items():
        vals = sorted((v, r["run"]) for r in live if isinstance(v := f(r), (int, float)) and not math.isnan(v))
        if not vals:
            continue
        print(f"\n{m}: median {vals[len(vals) // 2][0]:.3f}")
        for v, run in reversed(vals[-top:]):
            print(f"   {v:9.3f} {run}")
    print("\nSets, median over their live worlds:")
    batches = {}
    for r in rows:
        batches.setdefault(r["batch"], []).append(r)
    keys = list(MEASURES)
    print("batch        alive " + " ".join(k[:9].rjust(9) for k in keys))
    for name, rs in sorted(batches.items()):
        lv = [r for r in rs if (MEASURES["cells"](r) or 0) >= 200]
        cols = []
        for k in keys:
            v = sorted(x for r in lv if isinstance(x := MEASURES[k](r), (int, float)) and not math.isnan(x))
            cols.append(f"{v[len(v) // 2]:9.2f}" if v else " " * 9)
        print(f"{name:12} {len(lv):3d}/{len(rs)} " + " ".join(cols) + "  " + rs[0]["kind"])


def main():
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    def opt(name, default):
        return type(default)(a[a.index(name) + 1]) if name in a else default
    if a[0] == "queue":
        queue(a[1], int(a[2]), opt("--ticks", 200000), opt("--minutes", 18), opt("--controls", 2), opt("--no-stick", 1), opt("--order", "a5"))
    elif a[0] == "analyse":
        print(analyse_once(), "measured")
    elif a[0] == "watch":
        while True:
            if not analyse_once():
                time.sleep(30)
    elif a[0] == "rank":
        rank(int(a[1]) if len(a) > 1 else 5)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
