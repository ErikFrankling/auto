#!/usr/bin/env python3
"""Reads a saved world run with mate_track=true: are wide hybrids worse, and do mates choose?
x2 on a cell: >=1000 mixed birth, near*1000 + mate gap*100, where near-1 is the gap*100 to a random
neighbour at birth. <0 clone. 0 newcomer. x3: children so far, as either parent."""
import json, struct, sys
import numpy as np

CELL = np.dtype([(n, "<f4") for n in "e womb age hit x0 x1 x2 x3 vx vy".split()]
                + [(n, "<u4") for n in "pos flags clade gen rgb stamp".split()])
BINS = [0, .1, .2, .3, .45, .6, .8, 9]

def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        N, slots, hr = head["size"], head["slots"], head["hidden_rows"]
        blocks = hr * (13 + hr + 6)
        sq = N * N
        ring = 1
        while ring < slots: ring *= 2
        sizes = dict(occ=sq*4, seen=sq*4, seen_next=sq*4, terrain=sq*4, fa=sq*16, fb=sq*16, fc=sq*16,
                     cell=slots*64, stat=slots*112, sketch=slots*16, mem=slots*hr*16,
                     genome=slots*blocks*16, ring=ring*4, jobs=slots*16, ctr=128, ctl=32)
        data = {k: f.read(v) for k, v in sizes.items()}
    return head, data

def main(path):
    head, d = load(path)
    N = head["size"]
    occ = np.frombuffer(d["occ"], "<u4")
    cell = np.frombuffer(d["cell"], CELL)
    sk = np.frombuffer(d["sketch"], "<i1").reshape(-1, 16) / 127.0 * 4.0
    live = np.nonzero((cell["e"] > 0) & (cell["pos"] != 0xFFFFFFFF))[0]
    live = live[occ[cell["pos"][live]] == live + 1]
    c = cell[live]
    out = {"tick": head["tick"], "cells": len(c), "willing": round(float(((c["flags"] & 8) != 0).mean()), 3)}
    if not len(c): return print(json.dumps(out))

    x2, age, kids = c["x2"], c["age"], c["x3"]
    mixed = x2 >= 1000
    clone = x2 < 0
    near = np.floor(x2 / 1000) - 1
    mg = (x2 - np.floor(x2 / 1000) * 1000) / 100
    near = near / 100
    y, o = np.quantile(age, [1/3, 2/3])
    out["age_thirds"] = [float(y), float(o)]
    out["alive_mixed"] = round(float(mixed.mean()), 3)
    out["alive_clone"] = round(float(clone.mean()), 3)
    # Do mates differ less than a neighbour picked with no choice at all?
    has = mixed & (near >= 0)
    if has.sum() > 50:
        out["mate_gap_mean"] = round(float(mg[has].mean()), 3)
        out["near_gap_mean"] = round(float(near[has].mean()), 3)
        diff = mg[has] - near[has]
        out["mate_minus_near"] = round(float(diff.mean()), 4)
        out["mate_minus_near_se"] = round(float(diff.std() / np.sqrt(len(diff))), 4)
        out["n_mixed"] = int(has.sum())
    cl = clone & (x2 < -1)
    if cl.sum() > 50: out["clone_near_gap"] = round(float(((-x2[cl] - 2) / 100).mean()), 3)

    # Hybrid survival: alive old against alive young, by how far apart the parents were.
    # Children per 1000 ticks of life is the other half of fitness.
    rows = []
    def row(name, m):
        ny = int((m & (age < y)).sum()); no = int((m & (age >= o)).sum())
        a = m & (age >= 50)
        rows.append(dict(parents=name, young=ny, old=no,
                         old_per_young=round(no / ny, 3) if ny else None,
                         se=round(no / ny * np.sqrt(1/max(no,1) + 1/ny), 3) if ny else None,
                         kids_per_1000=round(float(kids[a].sum() / max(age[a].sum(), 1) * 1000), 2) if a.sum() else None,
                         energy=round(float(c["e"][m].mean()), 2) if m.sum() else None))
    row("clone", clone)
    for a, b in zip(BINS, BINS[1:]):
        row(f"{a}-{b}", mixed & (mg >= a) & (mg < b))
    out["by_parent_gap"] = rows

    # Is there any signal to choose by? Neighbouring pairs: voice against genome gap.
    pos = c["pos"].astype(np.int64)
    x, yy = pos & (N - 1), pos >> int(np.log2(N))
    gaps, vd = [], []
    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        q = ((yy + dy) % N) << int(np.log2(N)) | ((x + dx) % N)
        j = occ[q]
        ok = (j != 0) & (j != 0xFFFFFFFF)
        a = live[ok]; b = j[ok].astype(np.int64) - 1
        keep = cell["e"][b] > 0
        a, b = a[keep], b[keep]
        gaps.append(np.sqrt(((sk[a] - sk[b]) ** 2).mean(1)))
        vd.append(np.hypot(cell["vx"][a] - cell["vx"][b], cell["vy"][a] - cell["vy"][b]))
    if gaps:
        gaps, vd = np.concatenate(gaps), np.concatenate(vd)
        if len(gaps) > 100:
            out["neighbour_pairs"] = len(gaps)
            out["neighbour_gap_q"] = [round(float(v), 3) for v in np.quantile(gaps, [.1, .5, .9])]
            out["voice_gap_corr"] = round(float(np.corrcoef(gaps, vd)[0, 1]), 3)
            far = gaps > .45
            if far.sum() > 50 and (~far).sum() > 50:
                out["voice_gap_kin"] = round(float(vd[~far].mean()), 3)
                out["voice_gap_far"] = round(float(vd[far].mean()), 3)
                out["far_neighbour_share"] = round(float(far.mean()), 3)
    # Do unlike neighbours avoid sex? Willingness by whether the nearest unlike is around.
    print(json.dumps(out))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p)
        main(p)
