#!/usr/bin/env python3
"""Reads saved worlds and measures how cells live, one cell at a time.

  python3 scripts/analyse_life_worlds.py NAME world.bin [NAME world.bin ...]

Writes research/life-gpu/analysis/NAME.json and a raw image of who lives where.
Only the Python standard library, so it runs anywhere. Nothing here feeds back into a world.
"""
import json
import math
import random
import struct
import subprocess
import sys
from array import array
from collections import Counter, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research/life-gpu/analysis"
MEDIA = ROOT / "videos/life-gpu/maps"
NONE = 0xFFFFFFFF
FFMPEG = "ffmpeg"

KINDS = ["hunter", "scavenger", "rester", "settler", "wanderer"]
# Colours for humans only, one per way of living.
KIND_RGB = {
    "hunter": (229, 72, 63),
    "scavenger": (201, 133, 0),
    "rester": (57, 135, 229),
    "settler": (25, 158, 112),
    "wanderer": (144, 133, 233),
    "young": (70, 70, 80),
}


def kind_of(c):
    """A plain label from what a cell eats and how it moves. These labels are ours, for reading
    results. Cells are never told them."""
    d = c["diet"]
    total = sum(d)
    if total <= 1e-6 or c["age"] < 50:
        return "young"
    bite, meat = d[2] / total, d[1] / total
    if bite + meat >= 0.4:
        return "hunter" if bite >= meat else "scavenger"
    if c["rest"] >= 0.5 and c["moving"] < 0.05:
        return "rester"
    return "settler" if c["moving"] < 0.1 else "wanderer"


def sections(head):
    size, slots, hr = head["size"], head["slots"], head["hidden_rows"]
    blocks = hr * (13 + hr + 6)
    shard = head.get("shard_slots") or (2**31) // (blocks * 16)
    squares = size * size
    ring = 1
    while ring < slots:
        ring *= 2
    out = [
        ("occ", squares * 4), ("seen", squares * 4), ("seen_next", squares * 4), ("terrain", squares * 4),
        ("fa", squares * 16), ("fb", squares * 16), ("fc", squares * 16),
        ("cell", slots * 64), ("stat", slots * 112), ("sketch", slots * 16), ("mem", slots * hr * 16),
        ("genome", min(slots, shard) * blocks * 16),
    ]
    if slots > shard:
        out.append(("genome1", (slots - shard) * blocks * 16))
    out += [("ring", ring * 4), ("jobs", slots * 16), ("ctr", 128), ("ctl", 32)]
    return out


def analyse(name, path, max_cells=120000):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        at = 8 + n
        where = {}
        for sec, size in sections(head):
            where[sec] = (at, size)
            at += size

        def read(sec, kind, offset=0, count=None):
            start, size = where[sec]
            f.seek(start + offset)
            a = array(kind)
            a.frombytes(f.read(size - offset if count is None else count))
            return a

        size, slots = head["size"], head["slots"]
        occ = read("occ", "I")
        terrain = read("terrain", "I")
        cell_f, cell_u = read("cell", "f"), read("cell", "I")
        stat = read("stat", "f")
        start, sk_size = where["sketch"]
        f.seek(start)
        sketch_bytes = f.read(sk_size)
        fb = read("fb", "f")
        fa = read("fa", "f")

    rock = 0
    live = []
    for p, o in enumerate(occ):
        if o == NONE:
            rock += 1
        elif o:
            live.append((p, o - 1))
    open_squares = size * size - rock
    random.seed(1)
    picked = live if len(live) <= max_cells else random.sample(live, max_cells)

    cells = []
    by_square = {}
    for p, s in picked:
        c = dict(
            p=p, s=s, e=cell_f[s * 16], womb=cell_f[s * 16 + 1], age=cell_f[s * 16 + 2],
            voice=(cell_f[s * 16 + 8], cell_f[s * 16 + 9]), flags=cell_u[s * 16 + 11], clade=cell_u[s * 16 + 12],
            gen=cell_u[s * 16 + 13],
        )
        st = stat[s * 28 : s * 28 + 28]
        c["diet"] = list(st[0:4])
        c["budget"] = list(st[4:16])
        c["rest"] = st[4]
        c["chase"] = list(st[16:20])
        c["moving"] = st[19]
        c["follow"] = list(st[20:24])
        c["misc"] = list(st[24:28])
        c["kind"] = kind_of(c)
        c["sk"] = struct.unpack_from("<16b", sketch_bytes, s * 16)
        cells.append(c)
        by_square[p] = c
    # Who a square's cell is, whether or not it was sampled, comes from occ.
    mask = size - 1
    log = size.bit_length() - 1
    dirs = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]

    def gap(a, b):
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a["sk"], b["sk"])) / 16) / 127 * 4

    # Neighbours: who lives next to whom, how alike they are, whether they face the same way.
    next_to = {k: Counter() for k in KINDS + ["young"]}
    near_gap = {k: [] for k in next_to}
    align = {k: [0.0, 0] for k in next_to}
    for c in cells:
        x, y = c["p"] & mask, c["p"] >> log
        d = c["flags"] & 7
        for dx, dy in dirs:
            q = ((y + dy) & mask) << log | ((x + dx) & mask)
            other = by_square.get(q)
            if other is None:
                continue
            next_to[c["kind"]][other["kind"]] += 1
            if len(near_gap[c["kind"]]) < 20000:
                near_gap[c["kind"]].append(gap(c, other))
            d2 = other["flags"] & 7
            align[c["kind"]][0] += math.cos((d - d2) * math.pi / 4)
            align[c["kind"]][1] += 1

    # Colonies: groups of touching cells, on the real grid.
    seen = bytearray(size * size)
    sizes = []
    if size <= 2048:
        for p, o in enumerate(occ):
            if not o or o == NONE or seen[p]:
                continue
            count = 0
            todo = deque([p])
            seen[p] = 1
            while todo:
                q = todo.popleft()
                count += 1
                x, y = q & mask, q >> log
                for dx, dy in dirs:
                    r = ((y + dy) & mask) << log | ((x + dx) & mask)
                    if not seen[r] and occ[r] and occ[r] != NONE:
                        seen[r] = 1
                        todo.append(r)
            sizes.append(count)
    sizes.sort(reverse=True)
    total_in = sum(sizes) or 1
    bins = [(1, 1), (2, 4), (5, 16), (17, 64), (65, 256), (257, 1024), (1025, 4096), (4097, 10**9)]
    colony = [{"from": a, "to": b, "cells": sum(v for v in sizes if a <= v <= b) / total_in} for a, b in bins]

    groups = {}
    for k in KINDS + ["young"]:
        members = [c for c in cells if c["kind"] == k]
        m = len(members)
        if not m:
            groups[k] = {"share": 0.0}
            continue
        mean = lambda fn: sum(fn(c) for c in members) / m
        diet = [sum(c["diet"][i] for c in members) for i in range(4)]
        total = sum(diet) or 1
        vmean = (mean(lambda c: c["voice"][0]), mean(lambda c: c["voice"][1]))
        bk, bo = sum(c["misc"][1] for c in members), sum(c["misc"][2] for c in members)
        ages = sorted(c["age"] for c in members)
        energies = sorted(c["e"] for c in members)
        around = next_to[k]
        around_total = sum(around.values()) or 1
        groups[k] = {
            "share": m / len(cells),
            "count": m,
            "diet": [d / total for d in diet],
            "budget": [mean(lambda c, i=i: c["budget"][i]) for i in range(12)],
            "moving": mean(lambda c: c["moving"]),
            "to_bigger": mean(lambda c: c["chase"][0]),
            "to_smaller": mean(lambda c: c["chase"][1]),
            "to_food": mean(lambda c: c["chase"][2]),
            "follow": [mean(lambda c, i=i: c["follow"][i]) for i in range(4)],
            "neighbours": mean(lambda c: c["misc"][0] * 8),
            "age_median": ages[m // 2],
            "age_p90": ages[int(m * 0.9)],
            "energy_median": energies[m // 2],
            "generation": mean(lambda c: c["gen"]),
            "womb": mean(lambda c: c["womb"]),
            "mate_willing": mean(lambda c: 1.0 if c["flags"] & 8 else 0.0),
            "share_willing": mean(lambda c: 1.0 if c["flags"] & 16 else 0.0),
            "voice_mean": list(vmean),
            "voice_sd": [
                math.sqrt(max(mean(lambda c, i=i: (c["voice"][i] - vmean[i]) ** 2), 0)) for i in range(2)
            ],
            "bites_on_similar": bk / (bk + bo) if bk + bo > 1e-9 else None,
            "next_to": {o: around[o] / around_total for o in next_to},
            "gap_to_neighbours": sum(near_gap[k]) / len(near_gap[k]) if near_gap[k] else None,
            "alignment": align[k][0] / align[k][1] if align[k][1] else None,
        }

    # Genome map: the two main axes of difference, with how each sampled cell lives.
    pool = random.sample(cells, min(4000, len(cells)))
    mean_sk = [sum(c["sk"][i] for c in pool) / len(pool) for i in range(16)]
    cov = [[sum((c["sk"][a] - mean_sk[a]) * (c["sk"][b] - mean_sk[b]) for c in pool) / len(pool) for b in range(16)] for a in range(16)]
    axes = []
    for axis in range(2):
        v = [1.0 if i == axis else 0.1 for i in range(16)]
        for _ in range(40):
            w = [sum(cov[a][b] * v[b] for b in range(16)) for a in range(16)]
            n2 = math.sqrt(sum(x * x for x in w)) or 1
            v = [x / n2 for x in w]
        lam = sum(v[a] * sum(cov[a][b] * v[b] for b in range(16)) for a in range(16))
        for a in range(16):
            for b in range(16):
                cov[a][b] -= lam * v[a] * v[b]
        axes.append(v)
    points = []
    for c in random.sample(pool, min(1500, len(pool))):
        px = sum((c["sk"][i] - mean_sk[i]) * axes[0][i] for i in range(16)) / 127 * 4
        py = sum((c["sk"][i] - mean_sk[i]) * axes[1][i] for i in range(16)) / 127 * 4
        total = sum(c["diet"]) or 1
        points.append([round(px, 3), round(py, 3), c["kind"], round((c["diet"][1] + c["diet"][2]) / total, 3)])

    # How much scent is in the world, and is it near anyone.
    squares = size * size
    channel = [sum(fb[i * 4 + k] for i in range(squares)) / max(open_squares, 1) for k in range(4)]
    meat = sum(fa[i * 4 + 1] for i in range(squares)) / max(open_squares, 1)
    ground = sum(fa[i * 4] for i in range(squares)) / max(open_squares, 1)

    # Which founders' lines are alive, from the sample.
    clades = Counter(c["clade"] for c in cells)

    # Maps: who lives where, one pixel per square.
    img = bytearray(squares * 3)
    for p in range(squares):
        o = occ[p]
        if o == NONE:
            rgb = (58, 60, 70)
        elif o:
            c = by_square.get(p)
            rgb = KIND_RGB[c["kind"]] if c else (110, 110, 110)
        else:
            g = min(fa[p * 4] / 4.0, 1.0)
            m = min(fa[p * 4 + 1] / 3.0, 1.0)
            rgb = (int(10 + m * 60), int(12 + g * 30), 16)
        img[p * 3 : p * 3 + 3] = bytes(rgb)
    MEDIA.mkdir(parents=True, exist_ok=True)
    out_png = MEDIA / f"{name}-lifestyles.png"
    subprocess.run(
        [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size}x{size}", "-i", "-",
         "-frames:v", "1", "-update", "1", str(out_png)],
        input=bytes(img), check=True,
    )

    result = {
        "name": name, "tick": head["tick"], "size": size, "slots": slots, "alive": len(live), "open": open_squares,
        "sampled": len(cells), "density": len(live) / max(open_squares, 1),
        "groups": groups, "colony": colony, "largest_colony": sizes[:8],
        "points": points, "scent_mean": channel, "meat_mean": meat, "ground_mean": ground,
        "founder_lines": len(clades), "top_founder_share": clades.most_common(1)[0][1] / len(cells) if cells else 0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result))
    print(name, "alive", len(live), {k: round(g.get("share", 0), 3) for k, g in groups.items()}, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        analyse(args[i], args[i + 1])
