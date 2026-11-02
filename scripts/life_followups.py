#!/usr/bin/env python3
"""Queues the follow-ups to the first screen, in the order they should run.

Knockouts continue a saved batch for 30,000 ticks with biting off. Long runs continue a batch
to 600,000 ticks, and their +30,000 sample is what a knockout is compared with. The clean set
turns on one rule at a time with its default settings, since the first screen changed many
settings at once and so can't say what any one rule does.
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import life_screen as L

SPOOL = Path("/tmp/life-spool/held")
SHADERS = "/tmp/lg/screen/shaders-b0"
SCREEN = Path("/tmp/lg/screen")
LONG = Path("/tmp/lg/long")
BIN = "/tmp/life-v3/direct"


def write_job(order, name, minutes, cmd):
    SPOOL.mkdir(exist_ok=True)
    job = SPOOL / f"0-b-{order}-{name}"
    job.write_text(f"cd /tmp\nexport LIFE_MAX_MINUTES={minutes}\nexec {cmd}\n")
    (Path("/tmp/life-spool/logs") / job.name).write_text("")


def continuation(batch, order, name, ticks, every, extra, minutes):
    spec = json.loads((SCREEN / batch / "spec.json").read_text())
    d = LONG / name
    d.mkdir(parents=True, exist_ok=True)
    physics = {**spec["physics"], "batch_tile": spec["tile"], "trail_stats": True, **extra}
    sets = L.set_flags(physics)
    worlds = (spec["size"] // spec["tile"]) ** 2
    (d / "spec.json").write_text(json.dumps({**spec, "run": name, "ticks": ticks, "from": batch, "physics": {**spec["physics"], **extra}}))
    cmd = (
        f"{BIN} --shaders {SHADERS} --size {spec['size']} --seed {spec['seed']} {sets} --load {SCREEN / batch / 'world.bin'}"
        f" --ticks {ticks} --log-every {every} --max-minutes {minutes} --out {d}/log.jsonl"
        f" --dump-every {every} --dump-dir {d}/dumps --dump-cells {worlds * 800} --save {d}/world.bin"
    )
    write_job(order, name, minutes, cmd)


CLEAN = {
    "mass": {"mass_on": True},
    "armour": {"armour_on": True},
    "pack": {"pack_on": True},
    "waste": {"waste_on": True},
    "plants": {"plants_on": True},
    "sun-drift": {"weather_sun_on": True},
    "fire": {"weather_fire_on": True},
    "skills": {"skill_on": True},
    "biomes": {"biomes_on": True},
    "walls": {"walls_on": True},
    "blooms": {"bloom_share": 0.5},
    "roads": {"roads_on": True},
    "gang": {"gang_on": True, "max_energy": 80},
    "coat": {"coat_on": True},
    "mineral-fast": {"mineral_on": True, "rain_burst_ticks": 12000, "mineral_flow": 0.2},
    "birth": {"birth_choice": True},
    "ageing": {"age_metabolic": True},
    "compat": {"compat": 0.4},
    "control-a": {},
    "control-b": {},
}


def clean(order):
    for i, (rule, physics) in enumerate(CLEAN.items()):
        run = f"b1-{i:04d}"
        d = SCREEN / run
        d.mkdir(parents=True, exist_ok=True)
        seed = 7000 + i * 131
        tile, size, ticks = 256, 1024, 120000
        (d / "spec.json").write_text(json.dumps({"run": run, "rule": rule, "seed": seed, "size": size, "tile": tile, "ticks": ticks, "physics": physics}))
        sets = L.set_flags({**physics, "batch_tile": tile, "trail_stats": True})
        cmd = (
            f"{BIN} --shaders {SHADERS} --size {size} --seed {seed} {sets} --ticks {ticks} --log-every {ticks // 8} --max-minutes 25"
            f" --out {d}/log.jsonl --dump-every {ticks // 8} --dump-dir {d}/dumps --dump-cells {16 * 800} --save {d}/world.bin"
        )
        write_job(order, run, 25, cmd)


def main():
    for i, batch in enumerate(["b0-0001", "b0-0006", "b0-0000", "b0-0015"]):
        continuation(batch, f"a{i}", f"ko-{batch}", 30000, 30000, {"bite_on": False}, 8)
    continuation("b0-0001", "b0", "long-b0-0001", 480000, 30000, {}, 90)
    continuation("b0-0006", "b1", "long-b0-0006", 480000, 30000, {}, 90)
    clean("c")
    continuation("b0-0000", "d0", "long-b0-0000", 480000, 30000, {}, 90)
    continuation("b0-0015", "d1", "long-b0-0015", 480000, 30000, {}, 90)
    print("\n".join(sorted(p.name for p in SPOOL.iterdir())))


if __name__ == "__main__":
    main()
