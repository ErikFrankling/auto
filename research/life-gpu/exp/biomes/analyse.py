#!/usr/bin/env python3
"""Reads saved worlds: where do genomes and ways of living sit, room by room?

Asks whether a cell's kind of place (cold, hot, swamp, barren) explains its genome
and behaviour better than its room does, and how many cells look foreign where they live.
    analyse.py [--ledger] [--png out.png] world.save ...
"""
import json, struct, sys
import numpy as np

KINDS = ["cold", "hot", "swamp", "barren"]


def load(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    head = json.loads(f.read(n))
    w, s, hr = head["size"], head["slots"], head["hidden_rows"]
    sq = w * w
    blocks = hr * (13 + hr + 6)
    def take(bytes_, dtype):
        return np.frombuffer(f.read(bytes_), dtype=dtype).copy()
    occ = take(sq * 4, np.uint32)
    f.seek(sq * 8, 1)
    terrain = take(sq * 4, np.uint32)
    fa = take(sq * 16, np.float32).reshape(-1, 4)
    f.seek(sq * 16, 1)
    fc = take(sq * 16, np.float32).reshape(-1, 4)
    raw_cell = take(s * 64, np.uint8)
    cell_f = raw_cell.view(np.float32).reshape(-1, 16)
    cell_u = raw_cell.view(np.uint32).reshape(-1, 16)
    stat = take(s * 112, np.float32).reshape(-1, 28)
    sketch = take(s * 16, np.int8).reshape(-1, 16).astype(np.float32) / 127.0 * 4.0
    rest = f.read()
    return dict(head=head, w=w, occ=occ, terrain=terrain, fa=fa, fc=fc, cf=cell_f, cu=cell_u,
                stat=stat, sketch=sketch, tail=rest)


def eta2(x, labels):
    """Share of the variance of x (cells x dims) lying between the labelled groups, corrected for how many groups."""
    keep = labels >= 0
    x, labels = x[keep], labels[keep]
    n = len(x)
    if n < 10:
        return 0.0
    grand = x.mean(0)
    total = ((x - grand) ** 2).sum()
    between, used = 0.0, 0
    for l in np.unique(labels):
        m = labels == l
        if m.sum() < max(5, n // 200):
            continue
        used += 1
        between += m.sum() * ((x[m].mean(0) - grand) ** 2).sum()
    if used < 2 or total <= 0:
        return 0.0
    r = between / total
    return float(1.0 - (1.0 - r) * (n - 1) / (n - used))


def analyse(path):
    d = load(path)
    w = d["w"]
    cf, cu = d["cf"], d["cu"]
    alive = (cf[:, 0] > 0) & (cu[:, 10] != 0xFFFFFFFF) & (cf[:, 2] >= 1)
    pos = cu[alive, 10]
    room = (d["terrain"][pos] >> 28) & 15
    kind = ((room & 3) + (room >> 2)) & 3
    sk = d["sketch"][alive]
    st = d["stat"][alive]
    diet = st[:, :4] / np.maximum(st[:, :4].sum(1, keepdims=True), 1e-6)
    budget = st[:, 4:16]
    chase = st[:, 16:20]
    be = np.hstack([diet, budget, chase])
    open_sq = (d["terrain"] & (1 << 24)) == 0
    sq_room = (d["terrain"] >> 28) & 15
    sq_kind = ((sq_room & 3) + (sq_room >> 2)) & 3
    sun = (d["terrain"] & 0xFFFF) / 65535.0 * 8.0 * d["head"]["sun_k"] * 0.55
    out = dict(path=path.split("/")[-1], tick=d["head"]["tick"], seed=d["head"]["seed"], cells=int(alive.sum()),
               open=int(open_sq.sum()), total_sun=round(float(sun[open_sq].sum()), 1))
    per = {}
    for k in range(4):
        m = kind == k
        o = open_sq & (sq_kind == k)
        per[KINDS[k]] = dict(
            cells=int(m.sum()),
            density=round(float(m.sum()) / max(int(o.sum()), 1), 3),
            sun=round(float(sun[o].mean()) if o.any() else 0, 3),
            energy=round(float(cf[alive, 0][m].mean()) if m.any() else 0, 2),
            age=round(float(cf[alive, 2][m].mean()) if m.any() else 0),
            gen=round(float(cu[alive, 13][m].mean()) if m.any() else 0),
            diet=[round(float(v), 2) for v in (diet[m].mean(0) if m.any() else np.zeros(4))],
            rest=round(float(budget[m, 0].mean()) if m.any() else 0, 3),
            move=round(float(budget[m, 1].mean()) if m.any() else 0, 3),
            bite=round(float(budget[m, 7].mean()) if m.any() else 0, 3),
            moving=round(float(chase[m, 3].mean()) if m.any() else 0, 3),
        )
    out["kinds"] = per
    # Variance explained, by kind of place and by room. Room includes kind, so room minus kind is drift between rooms.
    out["genome_by_kind"] = round(eta2(sk, kind.astype(int)), 3)
    out["genome_by_room"] = round(eta2(sk, room.astype(int)), 3)
    out["behaviour_by_kind"] = round(eta2(be, kind.astype(int)), 3)
    out["behaviour_by_room"] = round(eta2(be, room.astype(int)), 3)
    # Room centroids. Same-kind room pairs against different-kind pairs. Same kind is never adjacent.
    rooms = [r for r in range(16) if (room == r).sum() >= 30]
    cen_g = {r: sk[room == r].mean(0) for r in rooms}
    cen_b = {r: be[room == r].mean(0) for r in rooms}
    def rk(r): return ((r & 3) + (r >> 2)) & 3
    def adjacent(a, b):
        ax, ay, bx, by = a & 3, a >> 2, b & 3, b >> 2
        dx = min((ax - bx) % 4, (bx - ax) % 4)
        dy = min((ay - by) % 4, (by - ay) % 4)
        return dx + dy == 1
    same_g, diff_g, same_b, diff_b = [], [], [], []
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if adjacent(a, b):
                continue
            dg = float(np.linalg.norm(cen_g[a] - cen_g[b]))
            db = float(np.linalg.norm(cen_b[a] - cen_b[b]))
            (same_g if rk(a) == rk(b) else diff_g).append(dg)
            (same_b if rk(a) == rk(b) else diff_b).append(db)
    m = lambda v: round(float(np.mean(v)), 3) if v else None
    out["gap_rooms_same_kind_genome"] = m(same_g)
    out["gap_rooms_diff_kind_genome"] = m(diff_g)
    out["gap_rooms_same_kind_behaviour"] = m(same_b)
    out["gap_rooms_diff_kind_behaviour"] = m(diff_b)
    # Foreigners: share of cells whose genome sits nearest some other room's average.
    if len(rooms) >= 2:
        keys = np.array(rooms)
        cen = np.stack([cen_g[r] for r in rooms])
        idx = np.random.default_rng(1).choice(len(sk), min(len(sk), 20000), replace=False)
        dist = ((sk[idx, None, :] - cen[None, :, :]) ** 2).sum(2)
        near = keys[dist.argmin(1)]
        known = np.isin(room[idx], keys)
        out["foreign_looking"] = round(float((near[known] != room[idx][known]).mean()), 3)
        near_kind = ((near & 3) + (near >> 2)) & 3
        rkind = ((room[idx] & 3) + (room[idx] >> 2)) & 3
        out["foreign_kind_looking"] = round(float((near_kind[known] != rkind[known]).mean()), 3)
    return out


def png(path, out):
    from PIL import Image
    d = load(path)
    w = d["w"]
    colours = np.array([[70, 120, 200], [210, 90, 40], [60, 150, 90], [170, 150, 110]], dtype=float)
    room = (d["terrain"] >> 28) & 15
    kind = ((room & 3) + (room >> 2)) & 3
    img = colours[kind] * (0.45 + 0.55 * np.minimum(1.0, (d["terrain"] & 0xFFFF) / 65535.0 * 8.0)[:, None])
    img[(d["terrain"] & (1 << 24)) != 0] = [20, 20, 25]
    Image.fromarray(img.reshape(w, w, 3).astype(np.uint8)).save(out)


def components(path):
    """How many connected regions of open squares, and the biggest one's share."""
    d = load(path)
    w = d["w"]
    closed = ((d["terrain"] & (1 << 24)) != 0).reshape(w, w)
    seen = closed.copy()
    sizes = []
    for start in range(w * w):
        y, x = divmod(start, w)
        if seen[y, x]:
            continue
        stack = [(y, x)]
        seen[y, x] = True
        n = 0
        while stack:
            a, b = stack.pop()
            n += 1
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    p, q = (a + da) % w, (b + db) % w
                    if not seen[p, q]:
                        seen[p, q] = True
                        stack.append((p, q))
        sizes.append(n)
    sizes.sort(reverse=True)
    return {"regions": len(sizes), "biggest_share": round(sizes[0] / sum(sizes), 3), "top": sizes[:6]}


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--png":
        png(args[2], args[1])
    elif args and args[0] == "--regions":
        print(json.dumps(components(args[1])))
    else:
        for p in args:
            print(json.dumps(analyse(p)))
