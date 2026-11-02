#!/usr/bin/env python3
"""Short readable cards for screened worlds, and each physics against the unmodified worlds.

    life_world_cards.py cards BATCH     one card per world in a batch, as text
    life_world_cards.py compare         each physics' 16 seeds against every control world

Reads research/life-gpu/screen/results.jsonl and the analysis files next to each run.
A single good world proves little, because control worlds already differ a lot, so
`compare` tests whole batches of seeds, with a correction for how many tests it ran.
"""
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("LIFE_SCREEN_OUT", ROOT / "research/life-gpu/screen"))
RUNS = Path(os.environ.get("LIFE_SCREEN_RUNS", "/tmp/lg/screen"))
MIN_CELLS = 200


def load_rows():
    rows = [json.loads(line) for line in open(OUT / "results.jsonl")]
    tiles = {}
    for r in rows:
        if r["batch"] not in tiles:
            tiles[r["batch"]] = tiles_of(RUNS / r["batch"] / "analyse.json")
        r["flows"] = ((tiles[r["batch"]].get(r["index"]) or {}).get("economy") or {}).get("flows")
    return rows


def flow_share(row, *kinds):
    """The world's energy income that came through these routes, at the last sample."""
    f = row.get("flows")
    if not f or sum(f.values()) <= 0:
        return None
    return sum(f.get(k, 0.0) for k in kinds) / sum(f.values())


def last(values):
    return values[-1] if isinstance(values, list) and values else (values if not isinstance(values, list) else None)


def time_of(row):
    return row.get("time") or {}


def total(values):
    return sum(values) if isinstance(values, list) else values


# What's compared across physics. Each reads one world's row and may return None.
MEASURES = {
    "cells": lambda r: r.get("cells"),
    "lasting strategies over 3%": lambda r: time_of(r).get("persistent_3pct"),
    "lasting strategies over 1%": lambda r: time_of(r).get("persistent_1pct"),
    "strategies over 3% at the end": lambda r: last(time_of(r).get("distinct_3pct")),
    "species at the end": lambda r: r.get("species"),
    # With one species there's nothing to explain, so it isn't a score of 0.
    "species explain behaviour": lambda r: r.get("explain") if (r.get("species") or 0) >= 2 else None,
    "new species": lambda r: total(time_of(r).get("new_species")),
    "strategy turnover": lambda r: time_of(r).get("turnover"),
    "energy from other cells": lambda r: flow_share(r, "bite", "meat"),
    "energy from biting": lambda r: flow_share(r, "bite"),
    "average cell's food from other cells": lambda r: last(r.get("from_others")),
    "gifts": lambda r: last(r.get("gift")),
    "bites": lambda r: last(r.get("bite")),
    "moving": lambda r: last(r.get("moving")),
    "resting": lambda r: last(r.get("rest")),
    "behaviour dimensions": lambda r: (r.get("brain") or {}).get("dims"),
    "reactivity": lambda r: (r.get("brain") or {}).get("reactivity"),
    "memory": lambda r: (r.get("brain") or {}).get("memory"),
    "traffic on lines": lambda r: (r.get("trails") or {}).get("lines"),
}


def alive(rows):
    return [r for r in rows if (r.get("cells") or 0) >= MIN_CELLS]


def values(rows, measure):
    got = (MEASURES[measure](r) for r in rows)
    return [float(v) for v in got if isinstance(v, (int, float)) and not math.isnan(v)]


def median(xs):
    s = sorted(xs)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2 if n else float("nan")


def rank_sum_p(a, b):
    """Two-sided Mann-Whitney U by the normal approximation, with ties corrected."""
    if len(a) < 4 or len(b) < 4:
        return 1.0, 0.5
    joined = sorted([(x, 0) for x in a] + [(x, 1) for x in b])
    n = len(joined)
    ranks = [0.0] * n
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and joined[j + 1][0] == joined[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    na, nb = len(a), len(b)
    u = sum(r for r, (_, g) in zip(ranks, joined) if g == 0) - na * (na + 1) / 2
    var = na * nb / 12 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0:
        return 1.0, 0.5
    z = (u - na * nb / 2) / math.sqrt(var)
    return math.erfc(abs(z) / math.sqrt(2)), u / (na * nb)


def switches(physics):
    on = [k.removesuffix("_on") for k, v in physics.items() if v is True and k.endswith(("_on", "_choice", "_metabolic", "_feel", "_sense"))]
    return ", ".join(sorted(on)) or "unmodified"


def compare():
    rows = load_rows()
    batches = {}
    for r in rows:
        batches.setdefault(r["batch"], []).append(r)
    control = [r for r in rows if not r.get("physics")]
    control_alive = alive(control)
    print(f"{len(rows)} worlds in {len(batches)} physics. {len(control_alive)} of {len(control)} control worlds alive.\n")
    print(f"{'measure':34} {'control median':>14} {'lowest':>8} {'highest':>8}")
    for m in MEASURES:
        v = values(control_alive, m)
        if v:
            print(f"{m:34} {median(v):14.3f} {min(v):8.3f} {max(v):8.3f}")
    tested = []
    survival = []
    for name, rs in sorted(batches.items()):
        if not rs[0].get("physics"):
            continue
        living = alive(rs)
        survival.append((name, len(living), len(rs), switches(rs[0]["physics"])))
        for m in MEASURES:
            a, b = values(living, m), values(control_alive, m)
            p, effect = rank_sum_p(a, b)
            tested.append((p, name, m, median(a) if a else float("nan"), median(b) if b else float("nan"), effect, len(a)))
    print("\nSurvival by physics (worlds with at least", MIN_CELLS, "cells):")
    for name, n, of, rules in survival:
        print(f"  {name}  {n:2d}/{of}  {rules}")
    # Benjamini-Hochberg: how many of the smallest p values stand after allowing for every test.
    tested.sort()
    m = len(tested)
    keep = 0
    for i, t in enumerate(tested):
        if t[0] <= 0.05 * (i + 1) / max(m, 1):
            keep = i + 1
    print(f"\n{m} tests, {keep} stand at a 5% false discovery rate:")
    for p, name, measure, med, ctrl, effect, n in tested[:keep]:
        rules = switches(batches[name][0]["physics"])
        sign = "higher" if effect > 0.5 else "lower"
        print(f"  {name} {measure}: {med:.3f} vs {ctrl:.3f}, {sign}, wins {effect:.0%} of pairings, p={p:.1e}, n={n}  [{rules}]")


def percent(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def shares(xs):
    return " ".join(f"{100 * x:.0f}" for x in (xs or []))


def card(row, analyse, history):
    t = time_of(row)
    lines = [f"## {row['run']}  cells {row.get('cells')}  open squares {row.get('open')}  mean sun {row.get('sun', 0):.3f}"]
    lines.append(f"cells over time: {' '.join(str(x) for x in row.get('alive') or [])}")
    if (row.get("cells") or 0) < MIN_CELLS:
        lines.append("died out or nearly so")
        return "\n".join(lines)
    lines.append(
        f"species at end, mean of four genome cut-offs {row.get('species')}, species explain behaviour {row.get('explain')}, "
        f"new species over time {t.get('new_species')}, seen {t.get('species_seen')}"
    )
    lines.append(
        f"fine clusters, not the list below: lasting >3%: {t.get('persistent_3pct')}, >1%: {t.get('persistent_1pct')}, "
        f"over 3% through time: {t.get('distinct_3pct')}; turnover of the list below {t.get('turnover')}"
    )
    h = history or {}
    by_flow = flow_share(row, "bite", "meat")
    by_bite = flow_share(row, "bite")
    if by_flow is not None:
        lines.append(f"world's energy at the end: {percent(by_flow)} from other cells, {percent(by_bite)} of it all from biting")
    lines.append(
        f"over time, average per cell: from other cells {shares(h.get('from_others'))} | gifts {shares(h.get('gift'))} | "
        f"bites {shares(h.get('bite'))} | moving {shares(h.get('moving'))} | resting {shares(h.get('rest'))} (%)"
    )
    b = row.get("brain") or {}
    lines.append(
        f"brains: dims {b.get('dims')}, reactivity {b.get('reactivity')}, memory {b.get('memory')}"
    )
    if history:
        lines.append(f"events: {len(history.get('events') or [])}; colony travel {json.dumps(history.get('colony_travel'))}")
        lines.append("strategies over the whole run (share % at each of the 8 samples):")
        for s in history.get("strategies") or []:
            budget = sorted((s.get("budget") or {}).items(), key=lambda kv: -kv[1])[:4]
            living = s.get("lives_on") or {}
            food = " ".join(f"{k} {percent(v)}" for k, v in living.items() if v and v > 0.01)
            lines.append(
                f"  - {s.get('label')}: [{shares(s.get('share_over_time'))}] peak {percent(s.get('peak'))}, "
                f"moving {percent(s.get('moving'))}, neighbours {s.get('neighbours', 0):.1f}, "
                f"energy {s.get('energy', 0):.1f}, age {s.get('age', 0):.0f}; eats {food}; "
                f"spends on {', '.join(k for k, _ in budget)}"
            )
    if analyse:
        e = analyse.get("economy") or {}
        c = e.get("colonies") or {}
        lines.append(
            f"colonies: {c.get('count')}, median size {c.get('p50')}, largest {c.get('max')}; "
            f"flows {json.dumps(e.get('flows'))[:110]}; trails {json.dumps(analyse.get('trails'))[:110]}"
        )
        for s in (analyse.get("brains") or {}).get("strategies") or []:
            head = s.get("headings") or {}
            strong = sorted(head.items(), key=lambda kv: -abs(kv[1].get("mean", 0)))[:5]
            turned = ", ".join(f"{k} {v.get('mean', 0):+.2f}" for k, v in strong)
            lines.append(f"  brain group energy {s.get('energy', 0):.1f} age {s.get('age', 0):.0f}: steers by {turned}")
    return "\n".join(lines)


def tiles_of(path):
    try:
        return {t.get("tile"): t for t in json.load(open(path))["tiles"]}
    except (OSError, ValueError, KeyError):
        return {}


def cards(batch):
    rows = [r for r in load_rows() if r["batch"] == batch]
    if not rows:
        sys.exit(f"nothing for {batch}")
    run_dir = RUNS / batch
    analyse, history = tiles_of(run_dir / "analyse.json"), tiles_of(run_dir / "history.json")
    print(f"# {batch}: {switches(rows[0].get('physics') or {})}")
    print(f"physics: {json.dumps(rows[0].get('physics') or {})}")
    print("whole map only, not per world. group_advantage under 1 means a bite in company pays less than a bite alone,")
    print("extras under extras_control means cells don't pile on the same target more than on a body they could have picked.")
    print(f"group hunting {json.dumps(rows[0].get('group_hunting'))}")
    print(f"birth limit {json.dumps(rows[0].get('birth_limit'))}, follows others {rows[0].get('follows_others')}")
    print(f"coat {json.dumps(rows[0].get('coat'))}, mineral {json.dumps(rows[0].get('mineral'))}\n")
    for r in sorted(rows, key=lambda r: r["index"]):
        print(card(r, analyse.get(r["index"]), history.get(r["index"])))
        print()


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if what == "cards":
        cards(sys.argv[2])
    elif what == "compare":
        compare()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
