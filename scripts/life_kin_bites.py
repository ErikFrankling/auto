#!/usr/bin/env python3
"""Do biters choose? For each world in saved batches: the share of bite energy taken from close
kin, against the share of close kin among the neighbours of the cells doing the biting. If the
two match, cells bite whoever is next to them, and nothing has learned to tell kin from strangers.

    life_kin_bites.py BATCH_DIR...   prints one JSON line per world
"""
import array
import json
import math
import random
import struct
import sys
from pathlib import Path

TAU = 0.4  # report_tau in defaults.json: what the shader counts as close kin
NONE = 0xFFFFFFFF
STAT_WORDS = 36
MAX_BITERS = 3000


def load(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    head = json.loads(f.read(n))
    size, slots = head["size"], head["slots"]
    squares = size * size
    at = 8 + n
    where = {}
    for name, bytes_ in [("occ", squares * 4), ("seen", squares * 4), ("seen_next", squares * 4), ("terrain", squares * 4), ("fa", squares * 16),
                         ("fb", squares * 16), ("fc", squares * 16), ("fd", squares * 16), ("cell", slots * 96), ("stat", slots * 144), ("sketch", slots * 16)]:
        where[name] = at
        at += bytes_
    def read(name, code, bytes_):
        f.seek(where[name])
        a = array.array(code)
        a.frombytes(f.read(bytes_))
        return a
    return size, read("occ", "I", squares * 4), read("stat", "f", slots * 144), read("sketch", "b", slots * 16)


def gap(sketch, a, b):
    a, b = a * 16, b * 16
    total = sum(((sketch[a + k] - sketch[b + k]) / 127.0) ** 2 for k in range(16))
    return math.sqrt(total / 16.0) * 4.0


def world(size, occ, stat, sketch, index, tile, rng):
    per = size // tile
    ox, oy = index % per * tile, index // per * tile
    biters = []
    cells = 0
    for y in range(oy, oy + tile):
        for x in range(ox, ox + tile):
            v = occ[y * size + x]
            if v == 0 or v == NONE:
                continue
            cells += 1
            s = v - 1
            kin, other = stat[s * STAT_WORDS + 25], stat[s * STAT_WORDS + 26]
            if kin + other > 1e-5:
                biters.append((x, y, s, kin, other))
    if len(biters) > MAX_BITERS:
        biters = rng.sample(biters, MAX_BITERS)
    took_kin = took = chance = 0.0
    for x, y, s, kin, other in biters:
        near = close = 0
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                q = occ[((y + dy) % size) * size + (x + dx) % size]
                if q == 0 or q == NONE:
                    continue
                near += 1
                close += gap(sketch, s, q - 1) < TAU
        if near:
            w = kin + other
            took_kin += kin
            took += w
            chance += w * close / near
    if took <= 0:
        return {"tile": index, "cells": cells, "biters": len(biters)}
    return {"tile": index, "cells": cells, "biters": len(biters), "on_kin": took_kin / took, "kin_near": chance / took}


def main():
    rng = random.Random(1)
    for d in map(Path, sys.argv[1:]):
        spec = json.loads((d / "spec.json").read_text())
        save = d / "world.bin"
        if not save.exists():
            continue
        size, occ, stat, sketch = load(save)
        tile = spec.get("tile") or size
        for i in range((size // tile) ** 2):
            print(json.dumps({"run": f"{d.name}-t{i:03d}", "batch": d.name, **world(size, occ, stat, sketch, i, tile, rng)}), flush=True)


if __name__ == "__main__":
    main()
