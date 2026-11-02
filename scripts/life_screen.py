#!/usr/bin/env python3
"""The screen: many small worlds, each with its own seed and physics, measured the same way.

  life_screen.py queue NAME COUNT [--size 256] [--ticks 120000] [--controls N] [--priority PREFIX]
  life_screen.py analyse     keeps measuring finished runs until none are left to wait for
  life_screen.py rank        prints the worlds that stand out

Runs are small on purpose, so hundreds fit. What holds up here gets scaled up.
Every physics setting is sampled around its default. Nothing here feeds into a world.
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
BIN = "/tmp/life-stable/direct"
ANALYSE = "/tmp/life-target/release/life-analyse"
HISTORY = "/tmp/life-target/release/life-history"

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
    "compat": (0, 0, False),
}


def sample(rng, how_many):
    """A handful of changes from default, so a result can still be traced to something."""
    keys = rng.sample(sorted(k for k in RANGES if k != "compat"), how_many)
    out = {}
    for k in keys:
        lo, hi, log = RANGES[k]
        v = math.exp(rng.uniform(math.log(lo), math.log(hi))) if log else rng.uniform(lo, hi)
        out[k] = float(f"{v:.4g}")
    if rng.random() < 0.25:
        out["compat"] = float(f"{rng.uniform(0.25, 0.8):.3g}")
    return out


def queue(name, count, size, ticks, controls, priority):
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
        physics = {} if i < controls else sample(rng, rng.randint(1, 6))
        (d / "spec.json").write_text(json.dumps({"run": run, "seed": seed, "size": size, "ticks": ticks, "physics": physics}))
        sets = " ".join(f"--set {k}={v}" for k, v in physics.items())
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
        budget={k: w["budget"][k] for k in ("rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme")},
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


def analyse_once():
    done = 0
    results = OUT / "results.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    seen = set()
    if results.exists():
        for line in open(results):
            seen.add(json.loads(line)["run"])
    for d in sorted(RUNS.iterdir()):
        if not (d / "spec.json").exists() or d.name in seen:
            continue
        finished = (SPOOL / "done" / f"0-s-{d.name}").exists() or any((SPOOL / "done").glob(f"*-{d.name}"))
        if not finished:
            continue
        world = d / "world.bin"
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
    live = [r for r in rows if not r.get("dead") and r.get("brain")]
    def score(r):
        b, t = r["brain"], r.get("time") or {}
        top = b["top"]
        entropy = -sum(p * math.log(p) for p in top if p > 0)
        return (
            math.exp(entropy)
            + 0.4 * b["dims"]
            + 3 * max(r.get("explain") or 0, 0)
            + 0.5 * (t.get("lasting") or 0)
            + 4 * (t.get("turnover") or 0)
            + 0.6 * min(r.get("species") or 1, 8)
        )
    live.sort(key=score, reverse=True)
    print(f"{len(rows)} runs, {len(live)} measured, {sum(1 for r in rows if r.get('dead'))} died out")
    for r in live[:25]:
        b, t = r["brain"], r.get("time") or {}
        print(f"{score(r):5.1f} {r['run']:12s} cells {r['cells']:6d} species {r.get('species') or 0:4.1f} k {b['k']} dims {b['dims']:.1f} explain {r.get('explain') or 0:.2f} others {r['from_others'] * 100:3.0f}% move {r['moving'] * 100:3.0f}% {','.join(t.get('labels') or [])[:60]} | {r['physics']}")


def main():
    a = sys.argv[1:]
    if not a:
        print(__doc__)
    elif a[0] == "queue":
        opt = lambda n, d: type(d)(a[a.index(n) + 1]) if n in a else d
        queue(a[1], int(a[2]), opt("--size", 256), opt("--ticks", 120000), opt("--controls", 0), opt("--priority", "0-s"))
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
