#!/usr/bin/env python3
"""The screen: many small worlds, each with its own seed and physics, measured the same way.

  life_screen.py queue NAME COUNT [--size 256] [--ticks 120000] [--controls N] [--priority PREFIX] [--rule-chance 0.3]
  life_screen.py batch NAME COUNT [--size 1024] [--tile 256] [--ticks 120000] [--minutes 20] [--controls N] [--priority PREFIX] [--rule-chance 0.3]
  life_screen.py analyse     keeps measuring finished runs until none are left to wait for
  life_screen.py rank        prints the worlds that stand out

Runs are small on purpose, so hundreds fit. What holds up here gets scaled up.
A batch puts many separate worlds on one map, walled off from each other, with one set of
physics and each world its own land. The GPU does far more per cell on a big map, so this
is what makes hundreds of worlds affordable.
Every physics setting is sampled around its default, and each rule in research/life-gpu/RULES.md
is on by chance, with its settings sampled too. Nothing here feeds into a world.
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
RUNS = Path("/tmp/lg/screen")
OUT = ROOT / "research/life-gpu/screen"
SPOOL = Path("/tmp/life-spool")
BIN = os.environ.get("LIFE_BIN", "/tmp/life-batch/direct")
ANALYSE = os.environ.get("LIFE_ANALYSE", "/tmp/life-batch/life-analyse")
HISTORY = os.environ.get("LIFE_HISTORY", "/tmp/life-batch/life-history")

# Settings worth varying, as (low, high, log scale). Switches are chosen by chance.
RANGES = {
    "sun": (0.02, 0.12, True),
    "ground_cap": (1, 16, True),
    "season_ticks": (1000, 40000, True),
    "terrain_contrast": (0.8, 4, False),
    "bite_power": (5, 80, True),
    "bite_yield": (0.25, 0.8, False),
    "meat_yield": (0.4, 0.95, False),
    "meat_rot": (0.0005, 0.01, True),
    "commitment": (0.003, 0.15, True),
    "spend": (0.08, 0.5, True),
    "upkeep": (0.005, 0.04, True),
    "upkeep_e": (0.0001, 0.002, True),
    "rest_saving": (0.5, 0.97, False),
    "hazard": (0.00005, 0.002, True),
    "max_energy": (15, 120, True),
    "womb": (2, 12, True),
    "mut_rate": (0.003, 0.15, True),
    "mut_size": (0.03, 0.3, True),
    "move_rate": (3, 20, True),
    "emit_power": (10, 150, True),
    "noise_power": (0.2, 4, True),
    "heat_power": (4, 30, True),
    "give_power": (2, 20, True),
    "share_rate": (0.02, 0.4, True),
    "p0_decay": (0.001, 0.02, True),
    # Below here, only for rules, and sampled only when their rule is on. Ranges come from
    # each experiment's README, and most of them are guesses that have not been tested.
    "compat": (0.25, 0.8, False),
    "mass_drag": (0.3, 3, True),
    "mass_step_cost": (0.01, 0.1, True),
    "mass_push_cost": (0.02, 0.2, True),
    "armour_build": (0.5, 3, True),
    "armour_fade": (0.002, 0.02, True),
    "armour_resist": (1, 6, True),
    "armour_drag": (1, 6, True),
    "armour_upkeep": (0.01, 0.06, True),
    "pack_intact": (0.05, 0.5, True),
    "pack_wound_keep": (0.5, 0.9, False),
    "pack_wound_rate": (4, 24, True),
    "waste_harm": (0.01, 0.08, True),
    "waste_deposit": (0.05, 0.2, True),
    "waste_holds": (0.1, 0.4, False),
    "waste_decay": (0.001, 0.01, True),
    "waste_flow": (0.005, 0.1, True),
    "plants_peak": (1.5, 3, False),
    "weather_drift": (0.003, 0.03, True),
    "weather_fire_rate": (1e-9, 1e-8, True),
    "weather_fire_rest": (500, 4000, True),
    "mate_pay": (0.3, 0.7, False),
    "skill_naive": (0.6, 0.95, False),
    "skill_curve": (1, 4, False),
    "skill_inherit": (0, 1, False),
    "wall_pass": (4, 64, True),
    "wall_thick": (2, 8, False),
    "bloom_share": (0.3, 0.7, False),
    "bloom_fade": (0.005, 0.02, True),
    "track_decay": (0.0005, 0.01, True),
    "track_flow": (0.001, 0.02, True),
    "road_ease": (1, 6, False),
}

# The rules. Each is (switches set when it's on, settings sampled from RANGES, switches
# chosen by chance). Rules that hold only a number start from zero, so that's their switch.
RULES = {
    "mass": ({"mass_on": True}, ["mass_drag", "mass_step_cost", "mass_push_cost"], []),
    "armour": ({"armour_on": True}, ["armour_build", "armour_fade", "armour_resist", "armour_drag", "armour_upkeep"], []),
    "pack": ({"pack_on": True}, ["pack_intact", "pack_wound_keep", "pack_wound_rate"], []),
    "waste": ({"waste_on": True}, ["waste_harm", "waste_deposit", "waste_decay", "waste_flow"], ["waste_clean"]),
    "plants": ({"plants_on": True}, ["plants_peak"], []),
    "sun_drift": ({"weather_sun_on": True}, ["weather_drift"], []),
    "fire": ({"weather_fire_on": True}, ["weather_fire_rate", "weather_fire_rest"], []),
    "mate_pay": ({}, ["mate_pay"], []),
    "compat": ({}, ["compat"], []),
    "skills": ({"skill_on": True}, ["skill_naive", "skill_curve", "skill_inherit"], ["skill_sense"]),
    "biomes": ({"biomes_on": True}, [], ["biome_feel"]),
    "walls": ({"walls_on": True}, ["wall_pass", "wall_thick"], []),
    "blooms": ({}, ["bloom_share", "bloom_fade"], []),
    "roads": ({"roads_on": True}, ["track_decay", "track_flow", "road_ease"], []),
}
# Settings that belong to a rule, so a world without that rule doesn't vary them.
RULE_ONLY = {k for _, keys, _ in RULES.values() for k in keys} | {"waste_holds"}


def draw(rng, key):
    lo, hi, log = RANGES[key]
    v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
    return float(f"{v:.4g}")


def sample_rules(rng, chance, size):
    """Turns each rule on by chance, with its settings. Lengths and rates are given for a world
    512 wide, the size the experiments ran at, and scale here to keep the same meaning."""
    out = {}
    for rule in sorted(RULES):
        if rng.random() >= chance:
            continue
        switches, keys, optional = RULES[rule]
        out.update(switches)
        for k in keys:
            out[k] = draw(rng, k)
        for k in optional:
            out[k] = rng.random() < 0.5
    if out.get("waste_clean"):
        out["waste_holds"] = draw(rng, "waste_holds")
    if "weather_fire_rate" in out:
        out["weather_fire_rate"] = float(f"{out['weather_fire_rate'] * (512 / size) ** 2:.4g}")
    if out.get("walls_on"):
        for k in ("wall_pass", "wall_thick"):
            out[k] = float(f"{max(out[k] * size / 512, 1):.4g}")
    return out


def sample(rng, how_many, rule_chance=0.3, size=256):
    """A handful of changes from default, so a result can still be traced to something, and
    any rules that came up."""
    keys = rng.sample(sorted(k for k in RANGES if k not in RULE_ONLY), how_many)
    out = {k: draw(rng, k) for k in keys}
    out.update(sample_rules(rng, rule_chance, size))
    return out


def set_flags(physics):
    """As --set flags. Switches need lower case, or the binary refuses them."""
    return " ".join(f"--set {k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in physics.items())


def queue(name, count, size, ticks, controls, priority, rule_chance=0.3):
    RUNS.mkdir(parents=True, exist_ok=True)
    shaders = RUNS / f"shaders-{name}"
    if not shaders.exists():
        shutil.copytree(ROOT / "crates/auto-life-gpu/shaders", shaders)
    rng = random.Random(hash(name) & 0xFFFFFFF)
    for i in range(count + controls):
        run = f"{name}-{i:04d}"
        d = RUNS / run
        d.mkdir(parents=True, exist_ok=True)
        seed = rng.randrange(1, 2**31)
        physics = {} if i < controls else sample(rng, rng.randint(1, 6), rule_chance, size)
        (d / "spec.json").write_text(json.dumps({"run": run, "seed": seed, "size": size, "ticks": ticks, "physics": physics}))
        sets = set_flags(physics)
        # An exit from a world that dies out still counts: dead worlds are a result.
        cmd = (
            f"{BIN} --shaders {shaders} --size {size} --seed {seed} {sets} --ticks {ticks} --log-every {ticks // 8}"
            f" --out {d}/log.jsonl --dump-every {ticks // 8} --dump-dir {d}/dumps --save {d}/world.bin"
        )
        job = f"{priority}-{run}"
        tmp = SPOOL / "jobs" / f".{job}"
        tmp.write_text(f"cd /tmp\nexport LIFE_MAX_MINUTES=3\nexec {cmd}\n")
        tmp.rename(SPOOL / "jobs" / job)
        (SPOOL / "logs" / f"{job}.log").write_text("")
    print(f"queued {count + controls} runs as {name}")


def batch(name, count, size, tile, ticks, minutes, controls, priority, rule_chance=0.3):
    RUNS.mkdir(parents=True, exist_ok=True)
    shaders = RUNS / f"shaders-{name}"
    if not shaders.exists():
        shutil.copytree(ROOT / "crates/auto-life-gpu/shaders", shaders)
    rng = random.Random(sum(map(ord, name)) * 7919)
    worlds = (size // tile) ** 2
    for i in range(count + controls):
        run = f"{name}-{i:04d}"
        d = RUNS / run
        d.mkdir(parents=True, exist_ok=True)
        seed = rng.randrange(1, 2**31)
        physics = {} if i < controls else sample(rng, rng.randint(1, 6), rule_chance, tile)
        (d / "spec.json").write_text(json.dumps({"run": run, "seed": seed, "size": size, "tile": tile, "ticks": ticks, "physics": physics}))
        sets = set_flags({**physics, "batch_tile": tile})
        # Enough cells per world to tell its strategies apart. Dumps are capped by what they read.
        cmd = (
            f"{BIN} --shaders {shaders} --size {size} --seed {seed} {sets} --ticks {ticks} --log-every {ticks // 8} --max-minutes {minutes}"
            f" --out {d}/log.jsonl --dump-every {ticks // 8} --dump-dir {d}/dumps --dump-cells {worlds * 800} --save {d}/world.bin"
        )
        job = f"{priority}-{run}"
        tmp = SPOOL / "jobs" / f".{job}"
        tmp.write_text(f"cd /tmp\nexport LIFE_MAX_MINUTES={minutes}\nexec {cmd}\n")
        tmp.rename(SPOOL / "jobs" / job)
        (SPOOL / "logs" / f"{job}.log").write_text("")
    print(f"queued {count + controls} batches of {worlds} worlds as {name}")


def last_log(path):
    head, last = None, None
    try:
        for line in open(path):
            r = json.loads(line)
            if "config" in r:
                head = r
            elif "tick" in r:
                last = r
    except (OSError, ValueError):
        pass
    return head, last


def features(d):
    spec = json.loads((d / "spec.json").read_text())
    _, l = last_log(d / "log.jsonl")
    out = dict(spec)
    if not l or not l.get("world"):
        out["dead"] = True
        return out
    w = l["world"]
    total = sum(l["deaths"].values()) or 1
    out.update(
        tick=l["tick"], cells=l["cells"], density=l["density"], tps=l.get("ticks_per_second"),
        from_others=w["lives_on"]["meat"] + w["lives_on"]["bite"], bite=w["lives_on"]["bite"], gift=w["lives_on"]["gift"],
        bitten_deaths=l["deaths"]["bitten"] / total, burned_deaths=l["deaths"]["burned"] / total,
        moving=w["moving"], neighbours=w["neighbours"], age=w["age"], generation=w["generation"],
        mixed=l["births_mixed"] / max(l["births"], 1),
        species=l.get("effective_species"), species_at=l.get("species_at"),
        explain=l.get("species_explain_behaviour"), explain_scaled=l.get("species_explain_scaled"),
        budget={k: w["budget"][k] for k in ("rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme", "armour", "clean")},
        hardness=w.get("hardness"), wound=w.get("wound"), practice=w.get("practice"),
        follows_scent=w["follows_scent"], hist=l.get("from_other_cells_hist"), founders=l.get("founder_lines"),
        no_slot=l.get("births_no_slot"),
    )
    a = d / "analyse.json"
    if a.exists():
        x = json.loads(a.read_text())
        b = x["brains"]
        h = b["whole_population"]["headings"]
        out["brain"] = dict(
            k=b["strategies_k"], silhouette=b["strategies_silhouette"], dims=b["effective_dims"], reactivity=b["reactivity"],
            memory=b["memory_effect"], gap=b["behaviour_gap_between_cells"],
            top=[s["share"] for s in b["strategies"]],
            food=h["food"]["mean"], bigger=h["bigger body"]["mean"], smaller=h["smaller body"]["mean"],
            scent=max((h[f"scent {i}"]["mean"] for i in range(4)), key=abs),
        )
        e = x["economy"]
        out["land"] = dict(colonies=e["colonies"], overlap=e["colony_rich_overlap"])
    h = d / "history.json"
    if h.exists():
        x = json.loads(h.read_text())
        shares = [s["share_over_time"] for s in x["strategies"]]
        out["time"] = dict(
            k=x["strategies_k"], silhouette=x["strategies_silhouette"],
            labels=[s["label"] for s in x["strategies"] if s["peak"] >= 0.05],
            species_seen=len(x["species"]), events=len(x["events"]),
            drift=x["behaviour_drift"], overlap=x["map_overlap"],
            lasting=sum(1 for s in x["strategies"] if sum(1 for v in s["share_over_time"] if v >= 0.05) >= 3),
            turnover=sum(abs(a - b) for row in shares for a, b in zip(row, row[1:])) / max(len(shares[0]) - 1, 1) / 2 if shares else 0,
            new_species=x["new_species"], crashes=sum(1 for e in x["events"] if e["kind"] == "population crash"),
        )
    return out


def time_features(x):
    shares = [s["share_over_time"] for s in x["strategies"]]
    return dict(
        k=x["strategies_k"], silhouette=x["strategies_silhouette"],
        labels=[s["label"] for s in x["strategies"] if s["peak"] >= 0.05],
        species_seen=len(x["species"]), events=len(x["events"]),
        drift=x["behaviour_drift"], overlap=x["map_overlap"],
        lasting=sum(1 for s in x["strategies"] if sum(1 for v in s["share_over_time"] if v >= 0.05) >= 3),
        turnover=sum(abs(a - b) for row in shares for a, b in zip(row, row[1:])) / max(len(shares[0]) - 1, 1) / 2 if shares else 0,
        new_species=x["new_species"], crashes=sum(1 for e in x["events"] if e["kind"] == "population crash"),
        species_alive=x["species_alive"],
    )


def brain_features(x):
    b = x["brains"]
    h = b["whole_population"]["headings"]
    e = x["economy"]
    return dict(
        brain=dict(
            k=b["strategies_k"], silhouette=b["strategies_silhouette"], dims=b["effective_dims"], reactivity=b["reactivity"],
            memory=b["memory_effect"], gap=b["behaviour_gap_between_cells"],
            top=[s["share"] for s in b["strategies"]],
            food=h["food"]["mean"], bigger=h["bigger body"]["mean"], smaller=h["smaller body"]["mean"],
            scent=max((h[f"scent {i}"]["mean"] for i in range(4)), key=abs),
        ),
        land=dict(colonies=e["colonies"], overlap=e["colony_rich_overlap"]),
    )


def batch_rows(d, spec):
    """One row per world in a batch. Tiles that came through too thin are rows too."""
    code = None
    for f in (SPOOL / "done").glob(f"*-{d.name}"):
        code = f.read_text().strip()
    head = None
    last = None
    try:
        for line in open(d / "log.jsonl"):
            r = json.loads(line)
            if "tiles" in r:
                head = r["tiles"]
            elif "tick" in r:
                last = r
    except (OSError, ValueError):
        pass
    hist = json.loads((d / "history.json").read_text())["tiles"] if (d / "history.json").exists() else []
    ana = json.loads((d / "analyse.json").read_text())["tiles"] if (d / "analyse.json").exists() else []
    rows = []
    for i in range((spec["size"] // spec["tile"]) ** 2):
        out = {k: spec[k] for k in ("seed", "size", "tile", "ticks", "physics")}
        out.update(run=f"{d.name}-t{i:03d}", batch=d.name, index=i, exit=code, tick=last["tick"] if last else 0, tps=last.get("ticks_per_second") if last else None)
        if head and i < len(head):
            out.update(open=head[i]["open"], sun=head[i]["sun"])
        h = hist[i] if i < len(hist) else None
        if h:
            out["alive"] = h["alive"]
            out["cells"] = h["alive"][-1] if h["alive"] else 0
            if h.get("dead"):
                out["dead"] = True
            else:
                half = len(h["ticks"]) // 2
                late = lambda k: sum(h[k][half:]) / max(len(h[k][half:]), 1)
                out.update(
                    time=time_features(h), from_others=late("from_others"), bite=late("bite"), gift=late("gift"),
                    moving=late("moving"), rest=late("rest"), explain=late("species_explain"),
                    species=sum(h["species_alive"][half:]) / max(len(h["species_alive"][half:]), 1),
                    generation=h["generation"][-1], age=h["age"][-1],
                )
        else:
            out["dead"] = True
        a = ana[i] if i < len(ana) else None
        if a and not a.get("dead"):
            out.update(brain_features(a))
        rows.append(out)
    return rows


def analyse_once():
    done = 0
    results = OUT / "results.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    if results.exists():
        for line in open(results):
            row = json.loads(line)
            seen.add(row.get("batch") or row["run"])
    for d in sorted(RUNS.iterdir()):
        if not (d / "spec.json").exists() or d.name in seen:
            continue
        finished = (SPOOL / "done" / f"0-s-{d.name}").exists() or any((SPOOL / "done").glob(f"*-{d.name}"))
        if not finished:
            continue
        world = d / "world.bin"
        spec = json.loads((d / "spec.json").read_text())
        if spec.get("tile"):
            more = ["--tile", str(spec["tile"])]
            if world.exists():
                subprocess.run(["nice", "-n", "15", ANALYSE, d.name, str(world), str(d / "analyse.json")] + more, capture_output=True)
            if (d / "dumps").exists():
                subprocess.run(["nice", "-n", "15", HISTORY, d.name, str(d / "dumps"), str(d / "history.json")] + more, capture_output=True)
            with open(results, "a") as out:
                for row in batch_rows(d, spec):
                    out.write(json.dumps(row) + "\n")
            # Worlds stay, so a good one can be taken out and scaled up. Dumps have served their purpose.
            shutil.rmtree(d / "dumps", ignore_errors=True)
            done += 1
            continue
        if world.exists():
            subprocess.run(["nice", "-n", "15", ANALYSE, d.name, str(world), str(d / "analyse.json")], capture_output=True)
        if (d / "dumps").exists():
            subprocess.run(["nice", "-n", "15", HISTORY, d.name, str(d / "dumps"), str(d / "history.json")], capture_output=True)
        f = features(d)
        with open(results, "a") as out:
            out.write(json.dumps(f) + "\n")
        # Keep the measurements, and free the big files. A world that stands out can be rerun from its seed.
        if world.exists():
            world.unlink()
        shutil.rmtree(d / "dumps", ignore_errors=True)
        done += 1
    return done


def rank():
    rows = [json.loads(l) for l in open(OUT / "results.jsonl")]
    # A world too thin to have an ecology can't show one, whatever its numbers say.
    live = [r for r in rows if not r.get("dead") and r.get("brain") and r.get("cells", 0) >= 200]
    dead = sum(1 for r in rows if r.get("dead"))
    print(f"{len(rows)} worlds, {len(live)} with enough life to measure, {dead} dead or thin")

    def entropy(r):
        top = r["brain"]["top"]
        return math.exp(-sum(p * math.log(p) for p in top if p > 0))

    def late_new(r):
        new = (r.get("time") or {}).get("new_species") or []
        return sum(new[len(new) // 2:])

    # No single score: each of these is a different way for a world to be interesting.
    measures = {
        "behaviour clusters in use": entropy,
        "dimensions of behaviour": lambda r: r["brain"]["dims"],
        "species explain behaviour": lambda r: r.get("explain") or 0,
        "strategies that last": lambda r: (r.get("time") or {}).get("lasting") or 0,
        "strategy shares keep shifting": lambda r: (r.get("time") or {}).get("turnover") or 0,
        "new species late in the run": late_new,
        "species at once": lambda r: r.get("species") or 0,
        "lives on other cells": lambda r: r.get("from_others") or 0,
        "gifts": lambda r: r.get("gift") or 0,
        "steers by scent": lambda r: abs(r["brain"]["scent"]),
        "chases smaller bodies": lambda r: r["brain"]["smaller"],
        "flees bigger bodies": lambda r: -r["brain"]["bigger"],
    }
    def line(r):
        b, t = r["brain"], r.get("time") or {}
        return f"{r['run']:18s} cells {r.get('cells', 0):6d} species {r.get('species') or 0:4.1f} k {b['k']:2d} dims {b['dims']:.1f} explain {r.get('explain') or 0:.2f} others {(r.get('from_others') or 0) * 100:3.0f}% move {(r.get('moving') or 0) * 100:3.0f}% {','.join(t.get('labels') or [])[:60]}"
    picked = {}
    for name, f in measures.items():
        order = sorted(live, key=f, reverse=True)[:5]
        print(f"\n{name}")
        for r in order:
            print(f"{f(r):8.3f}  {line(r)}")
            picked[r["run"]] = picked.get(r["run"], 0) + 1
    print("\ntop on several measures")
    by = {r["run"]: r for r in live}
    for run, n in sorted(picked.items(), key=lambda kv: -kv[1])[:15]:
        print(f"{n:3d}  {line(by[run])} | {by[run]['physics']}")


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "queue":
        opt = lambda n, d: type(d)(a[a.index(n) + 1]) if n in a else d
        queue(a[1], int(a[2]), opt("--size", 256), opt("--ticks", 120000), opt("--controls", 0), opt("--priority", "0-s"), opt("--rule-chance", 0.3))
    elif a[0] == "batch":
        opt = lambda n, d: type(d)(a[a.index(n) + 1]) if n in a else d
        batch(a[1], int(a[2]), opt("--size", 1024), opt("--tile", 256), opt("--ticks", 120000), opt("--minutes", 20), opt("--controls", 0), opt("--priority", "0-b"), opt("--rule-chance", 0.3))
    elif a[0] == "analyse":
        idle = 0
        while idle < 2880:
            n = analyse_once()
            idle = 0 if n else idle + 1
            if n:
                print("measured", n, flush=True)
            time.sleep(30)
    elif a[0] == "rank":
        rank()


if __name__ == "__main__":
    main()
