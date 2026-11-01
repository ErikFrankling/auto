#!/usr/bin/env python3
"""Builds report/life/index.html from the run logs and the saved-world analysis.

Every number here comes from a log in research/life-gpu/exp or a file in
research/life-gpu/analysis. Re-run it as results arrive; missing data shows as pending.
Phone first: charts are drawn about phone width and scale up.
"""
import html
import json
import math
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "research/life-gpu/exp"
ANALYSIS = ROOT / "research/life-gpu/analysis"
OUT = ROOT / "report/life"
TREES = Path.home() / ".t3/worktrees/auto"

# Where each experiment's logs are until its branch is merged.
EXPERIMENTS = [
    ("mass", "Fat or lean", "Stored energy is mass. Heavy cells bite harder, but moving gets less likely and costs more. Moving into a lighter cell can push it aside.",
     [EXP / "mass-512"], False),
    ("armour", "Armour", "A cell can spend energy on hardness, which blocks bites and heat but slows it down and costs upkeep.",
     [EXP / "armour-512"], False),
    ("pack", "Pack hunting", "A bite opens a wound that makes the next bites on that body stronger, so attacking together pays. A lone biter gets little from a healthy cell.",
     [EXP / "pack-512"], False),
    ("waste", "Waste", "Eating leaves poison in the ground, so a dense colony slowly harms itself and has to move or thin out.",
     [TREES / "t3-c8cb10e9/research/life-gpu/exp/waste"], True),
    ("plants", "Living ground", "Food grows fastest at middling density and barely at all when grazed bare, so overgrazing costs the future.",
     [TREES / "t3-3867c537/research/life-gpu/exp/plants"], True),
    ("weather", "Weather", "Rich sun drifts across the world, and rare fires burn through dense food and anyone on it.",
     [TREES / "t3-b009dca5/research/life-gpu/exp/weather"], True),
    ("mating", "Mating", "Tests what splits species: a real payoff for sex, a genome limit on who can mate, and a lower mutation rate.",
     [TREES / "t3-d9ed4e5e/research/life-gpu/exp/mating"], True),
    ("skills", "Practice", "Doing something makes a cell better at it, at the cost of others, so generalists pay a price.",
     [TREES / "t3-b0bf4842/research/life-gpu/exp/skills"], True),
    ("biomes", "Biomes", "Physics differs by place: thin cold land, rich hot land, swamp, barren stretches and strong barriers.",
     [TREES / "t3-509c6643/research/life-gpu/exp/biomes"], True),
    ("roads", "Roads and signals", "Rich blooms that fade fast, and paths that make walking cheaper, to give scent a reason to exist.",
     [TREES / "t3-31a264e4/research/life-gpu/exp/roads"], True),
]

KIND_NAMES = {"hunter": "Hunters", "scavenger": "Scavengers", "rester": "Resters", "settler": "Settlers", "wanderer": "Wanderers"}
KIND_COLOUR = {"hunter": "#e5483f", "rester": "#3987e5", "scavenger": "#c98500", "wanderer": "#9085e9", "settler": "#199e70"}
SEED_COLOUR = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
ACTIONS = ["rest", "move", "ground", "meat", "grow", "emit", "heat", "bite", "give", "enzyme"]
DIET = ["ground", "meat", "bite", "gift"]
DIET_COLOUR = {"ground": "#3cc878", "meat": "#f0aa28", "bite": "#eb463c", "gift": "#5a8cff"}


def esc(s):
    return html.escape(str(s), quote=True)


def load_log(path):
    head, rows = None, []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None, []
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "config" in r:
            head = r
        elif "tick" in r:
            rows.append(r)
    return head, rows


def pct(v, digits=0):
    return "–" if v is None else f"{v * 100:.{digits}f}%"


def other_share(r):
    w = r.get("world")
    return None if not w else w["lives_on"]["meat"] + w["lives_on"]["bite"]


def bitten_share(r):
    d = r["deaths"]
    total = d["bitten"] + d["starved"] + d["burned"] + d["old"]
    return d["bitten"] / total if total else None


def mixed_share(r):
    return r["births_mixed"] / r["births"] if r.get("births") else None


# ---- Charts ----------------------------------------------------------------

W = 400


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def chart(title, sub, body, items=None):
    return f'<figure class="chart"><div class="chart-title">{esc(title)}</div><div class="chart-sub">{esc(sub)}</div>{legend(items) if items else ""}{body}</figure>'


def nice_ticks(hi, count=4):
    if hi <= 0:
        return [0, 1]
    step = 10 ** math.floor(math.log10(hi / count))
    for m in (1, 2, 2.5, 5, 10):
        if hi / (step * m) <= count:
            step *= m
            break
    n = int(math.ceil(hi / step - 1e-9))
    return [step * k for k in range(n + 1)]


def fmt_num(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:g}M"
    if a >= 1e3:
        return f"{v / 1e3:g}k"
    if a >= 10 or a == 0:
        return f"{v:g}"
    return f"{v:.2g}"


def lines(title, sub, series, xlab, yfmt=fmt_num, h=210, ymax=None, xfmt=fmt_num):
    """series: (name, colour, [(x, y)])."""
    pts = [p for _, _, s in series for p in s if p[1] is not None]
    if not pts:
        return f'<p class="sub">No data yet: {esc(title)}</p>'
    L, R, T, B = 46, 10, 8, 34
    x0, x1 = min(p[0] for p in pts), max(p[0] for p in pts)
    x1 = x1 if x1 > x0 else x0 + 1
    y1 = ymax if ymax else max(p[1] for p in pts) * 1.05 or 1
    yt = nice_ticks(y1)
    y1 = yt[-1]
    fx = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    fy = lambda v: h - B - v / y1 * (h - T - B)
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><text x="{L - 6}" y="{fy(t) + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    for t in nice_ticks(x1, 4):
        if t >= x0:
            o.append(f'<text x="{fx(t):.1f}" y="{h - B + 15}" text-anchor="middle">{esc(xfmt(t))}</text>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{h - 4}" text-anchor="middle">{esc(xlab)}</text>')
    for name, colour, s in series:
        s = sorted((p for p in s if p[1] is not None), key=lambda p: p[0])
        d = " ".join(f"{'M' if k == 0 else 'L'}{fx(p[0]):.1f} {fy(min(p[1], y1)):.1f}" for k, p in enumerate(s))
        o.append(f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="2" stroke-linejoin="round"/>')
        for p in s:
            o.append(f'<circle class="mark" cx="{fx(p[0]):.1f}" cy="{fy(min(p[1], y1)):.1f}" r="7" fill="transparent" data-tip="{esc(name)}, {esc(xfmt(p[0]))}: {esc(yfmt(p[1]))}"/>')
    o.append("</svg>")
    return chart(title, sub, "".join(o), [(n, c) for n, c, _ in series] if len(series) > 1 else None)


def stacks(title, sub, rows, parts, h_row=26, label_w=108):
    """rows: (label, {part: share}). parts: (key, colour, name)."""
    h = len(rows) * h_row + 6
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}">']
    for i, (label, vals) in enumerate(rows):
        y = i * h_row + 3
        o.append(f'<text class="lbl" x="{label_w - 6}" y="{y + 15}" text-anchor="end">{esc(label)}</text>')
        x = label_w
        total = sum(max(vals.get(k, 0) or 0, 0) for k, _, _ in parts) or 1
        for key, colour, name in parts:
            v = max(vals.get(key, 0) or 0, 0) / total
            w = v * (W - label_w - 6)
            if w > 0.4:
                o.append(f'<rect class="mark" x="{x:.1f}" y="{y}" width="{max(w - 1.5, 0.5):.1f}" height="{h_row - 8}" rx="2" fill="{colour}" data-tip="{esc(label)}: {esc(name)} {v * 100:.1f}%"/>')
            x += w
    o.append("</svg>")
    return chart(title, sub, "".join(o), [(n, c) for _, c, n in parts])


def strip(title, sub, rows, xmax, fmt, band_row=0, label_w=128, h_row=25):
    """rows: (label, [values]). A band shows the range of the first row."""
    h = len(rows) * h_row + 24
    fx = lambda v: label_w + min(v, xmax) / xmax * (W - label_w - 12)
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}">']
    base = [v for v in rows[band_row][1] if v is not None] if rows else []
    if base:
        o.append(f'<rect x="{fx(min(base)):.1f}" y="2" width="{max(fx(max(base)) - fx(min(base)), 2):.1f}" height="{len(rows) * h_row}" rx="3" fill="#3987e5" opacity=".14"/>')
    for t in nice_ticks(xmax, 4):
        o.append(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="2" y2="{len(rows) * h_row + 2}"/><text x="{fx(t):.1f}" y="{h - 6}" text-anchor="middle">{esc(fmt(t))}</text>')
    for i, (label, vals) in enumerate(rows):
        y = i * h_row + h_row / 2 + 2
        o.append(f'<text class="lbl" x="{label_w - 8}" y="{y + 4}" text-anchor="end">{esc(label)}</text>')
        for k, v in enumerate(vals):
            if v is None:
                continue
            o.append(f'<circle class="mark" cx="{fx(v):.1f}" cy="{y}" r="5.5" fill="#3987e5" stroke="#111" stroke-width="1.5" data-tip="{esc(label)}, run {k + 1}: {esc(fmt(v))}"/>')
    o.append("</svg>")
    return chart(title, sub, "".join(o))


def bars_v(title, sub, values, labels, colour="#3987e5", h=170, tipfmt=None):
    L, R, T, B = 40, 8, 8, 34
    top = max(values) or 1
    yt = nice_ticks(top)
    top = yt[-1]
    bw = (W - L - R) / len(values)
    fy = lambda v: h - B - v / top * (h - T - B)
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><text x="{L - 6}" y="{fy(t) + 4:.1f}" text-anchor="end">{esc(fmt_num(t))}</text>')
    for i, v in enumerate(values):
        x = L + i * bw
        tip = tipfmt(i, v) if tipfmt else f"{labels[i]}: {v:g}"
        o.append(f'<rect class="mark" x="{x + 1:.1f}" y="{fy(v):.1f}" width="{bw - 2:.1f}" height="{max(h - B - fy(v), 0):.1f}" rx="2" fill="{colour}" data-tip="{esc(tip)}"/>')
        o.append(f'<text x="{x + bw / 2:.1f}" y="{h - B + 15}" text-anchor="middle">{esc(labels[i])}</text>')
    o.append("</svg>")
    return chart(title, sub, "".join(o))


def scatter(title, sub, points):
    """points: x, y, kind, share from others."""
    pts = [p for p in points if p[2] in KIND_COLOUR]
    if not pts:
        return ""
    h = 300
    L, R, T, B = 8, 8, 8, 8
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    fx = lambda v: L + (v - x0) / ((x1 - x0) or 1) * (W - L - R)
    fy = lambda v: h - B - (v - y0) / ((y1 - y0) or 1) * (h - T - B)
    order = ["wanderer", "settler", "rester", "scavenger", "hunter"]
    o = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="{esc(title)}"><rect x="0" y="0" width="{W}" height="{h}" rx="6" fill="#1b1b1d"/>']
    for k in order:
        for p in pts:
            if p[2] == k:
                o.append(f'<circle cx="{fx(p[0]):.1f}" cy="{fy(p[1]):.1f}" r="2.4" fill="{KIND_COLOUR[k]}" opacity=".85"/>')
    o.append("</svg>")
    return chart(title, sub, "".join(o), [(KIND_NAMES[k], KIND_COLOUR[k]) for k in order])


def table(headers, rows, cls=""):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c if str(c).startswith('<') else esc(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def video(path, caption):
    if not (OUT / path).exists():
        return ""
    return f'<figure class="media"><video controls playsinline preload="metadata" src="{esc(path)}"></video><figcaption>{esc(caption)}</figcaption></figure>'


def image(path, caption):
    if not (OUT / path).exists():
        return ""
    return f'<figure class="media"><a href="{esc(path)}"><img loading="lazy" src="{esc(path)}" alt="{esc(caption)}"></a><figcaption>{esc(caption)}</figcaption></figure>'


# ---- Data ------------------------------------------------------------------


def baseline():
    v2 = sorted((EXP / "base-v2").glob("*.jsonl")) if (EXP / "base-v2").exists() else []
    good = [p for p in v2 if load_log(p)[1]]
    use = good if len(good) >= 3 else sorted((EXP / "base").glob("*.jsonl"))
    worlds = []
    for p in use:
        head, rows = load_log(p)
        if rows:
            worlds.append((p.stem.replace("512-seed", "Seed "), head, rows))
    return worlds, len(good) >= 3


def analyses():
    out = {}
    for n in (1, 2, 3):
        for prefix in ("v2", "old"):
            f = ANALYSIS / f"{prefix}-seed{n}.json"
            if f.exists():
                out[n] = json.loads(f.read_text())
                break
    return out


def experiment_logs(dirs):
    found = []
    for d in dirs:
        if not Path(d).exists():
            continue
        for p in sorted(Path(d).glob("*.jsonl")):
            head, rows = load_log(p)
            if head and rows and head.get("size", 0) >= 512:
                found.append((p.stem, head, rows))
    return found


def changed_settings(head, base_cfg):
    if not head or not base_cfg:
        return ""
    cfg = head.get("config", {})
    bits = []
    for k, v in cfg.items():
        if base_cfg.get(k, "absent") != v and k not in ("report_tau", "ledger") and v not in (False, 0, 0.0):
            bits.append(f"{k}={v}")
    return ", ".join(bits)


# ---- Sections --------------------------------------------------------------


def section_scale():
    rows = []
    for size, name in ((1024, "1024"), (2048, "2048"), (4096, "4096"), (8192, "8192")):
        _, r = load_log(EXP / f"giant/{name}.jsonl")
        if r:
            last = r[-1]
            rows.append((size, last["cells"], sum(x["ticks_per_second"] for x in r) / len(r), sum(x["gpu_ms_per_tick"] for x in r) / len(r), last["world"]["neighbours"], other_share(last), last["world"]["moving"]))
    _, base = load_log(sorted((EXP / "base").glob("*seed1*"))[0]) if list((EXP / "base").glob("*seed1*")) else (None, [])
    if base:
        last = base[-1]
        rows.insert(0, (512, last["cells"], sum(x["ticks_per_second"] for x in base) / len(base), sum(x["gpu_ms_per_tick"] for x in base) / len(base), last["world"]["neighbours"], other_share(last), last["world"]["moving"]))
    t = table(
        ["World", "Squares", "Cells", "Ticks/s", "GPU ms per tick", "Neighbours", "From other cells"],
        [[f"{s}²", fmt_num(s * s), fmt_num(c), f"{tps:.0f}", f"{ms:.1f}", f"{nb:.1f}", pct(o)] for s, c, tps, ms, nb, o, _ in rows],
    )
    cpu = []
    try:
        for line in (ROOT / "research/life/results/4-world-256.jsonl").read_text().splitlines():
            r = json.loads(line)
            if "cells" in r and "seconds" in r:
                cpu.append(r)
    except OSError:
        pass
    note = ""
    if len(cpu) >= 2:
        speed = (cpu[-1]["tick"] - cpu[0]["tick"]) / max(cpu[-1]["seconds"] - cpu[0]["seconds"], 1e-6)
        note = f"The CPU version reached {fmt_num(cpu[-1]['cells'])} cells at about {speed:.0f} ticks per second on 4 threads. "
    chart_ = lines("GPU time per tick", "Grows with squares and cells. At 4096² it's still under one 60 Hz frame.",
                   [("World", "#3987e5", [(c, ms) for _, c, _, ms, _, _, _ in rows])], "cells", yfmt=lambda v: f"{v:g} ms") if rows else ""
    return f"""<h2 id="scale">How big it goes</h2>
<p>{note}The GPU version is the same idea on a 7900 XT that also draws the desktop. Work goes to the card in short pieces, and each run takes turns with other GPU jobs.</p>
{t}{chart_}
<p>The world's physics isn't exactly the same at every size. A 512² world has about 4 season regions, and 4096² has 256, so a big world always has somewhere in good season. And 8192² is a warning: the memory limit at the time forced the sun down 80% so 2.6 million cells would fit, and cells ended up with 0.6 neighbours on average. Predators vanished. Size alone doesn't make a world richer: density matters too.</p>"""


def section_time(worlds):
    charts = []
    def series(fn):
        return [(n, SEED_COLOUR[i], [(r["tick"], fn(r)) for r in rows]) for i, (n, _, rows) in enumerate(worlds)]
    p = lambda v: f"{v * 100:.0f}%"
    charts.append(lines("Population", "Cells alive. Some worlds fill up, others are held down by predators.", series(lambda r: r["cells"]), "tick"))
    charts.append(lines("Energy from eating other cells", "Meat and biting together.", series(other_share), "tick", yfmt=p))
    charts.append(lines("Deaths from being bitten", "Share of all deaths between log points.", series(bitten_share), "tick", yfmt=p))
    charts.append(lines("Time spent moving", "How often a move actually happens.", series(lambda r: r["world"]["moving"] if r.get("world") else None), "tick", yfmt=p))
    charts.append(lines("Neighbours", "Out of 8 squares around a cell. High means packed colonies.", series(lambda r: r["world"]["neighbours"] if r.get("world") else None), "tick"))
    charts.append(lines("Species", "Effective number of genome groups at threshold 0.4.", series(lambda r: r.get("effective_species")), "tick"))
    charts.append(lines("Mixed births", "Share of births with two parents.", series(mixed_share), "tick", yfmt=p))
    charts.append(lines("Genome distance", "Typical weight difference between two random cells. Random genomes start at 0.57.", series(lambda r: r.get("genome_gap", {}).get("p50")), "tick"))
    return f"""<h2 id="time">How worlds change over time</h2>
<p>Same rules and same settings, with only the random seed different. These worlds don't converge on one outcome.</p>{''.join(charts)}"""


def section_diet(worlds, giant_last):
    out = []
    labels = [f"{k * 10}" for k in range(10)]
    for n, _, rows in worlds:
        hist = rows[-1].get("from_other_cells_hist")
        if hist:
            top = sum(hist[8:]) / (sum(hist) or 1)
            out.append(bars_v(f"{n}: where cells get energy", f"Sampled cells by % of energy from other cells, tick {fmt_num(rows[-1]['tick'])}. {top * 100:.0f}% get over 80%.", hist, labels))
    if giant_last and giant_last.get("from_other_cells_hist"):
        hist = giant_last["from_other_cells_hist"]
        top = sum(hist[8:]) / (sum(hist) or 1)
        out.append(bars_v("The 4096² world: where cells get energy", f"Carnivores carried over to the giant world. {top * 100:.0f}% get over 80%.", hist, labels, colour="#d95926"))
    return "".join(out)


def section_lifestyles(ana):
    if not ana:
        return ""
    out = ['<h2 id="lifestyles">Ways of living</h2>', """<p>Every cell is the same kind of thing. To read the results I sort them by what they do. These labels are mine, cells never see them, and the cut-offs are simple:</p>
<ul><li><b>Hunters</b>: at least 40% of energy from other cells, more from biting than from meat.</li>
<li><b>Scavengers</b>: the same 40%, more from meat than from biting.</li>
<li><b>Resters</b>: spend at least half their budget resting and barely move.</li>
<li><b>Settlers</b>: mostly graze and move under 10% of the time.</li>
<li><b>Wanderers</b>: mostly graze and move more than that.</li></ul>
<p>Cells under age 50 aren't counted, because they haven't done enough to label.</p>"""]
    for n in sorted(ana):
        a = ana[n]
        maps = image(f"media/map-seed{n}.jpg", f"Seed {n}, who lives where: red hunters, amber scavengers, blue resters, green settlers, purple wanderers, dim grey cells under age 50. Tap to enlarge.")
        groups = a["groups"]
        kinds = [k for k in KIND_NAMES if groups.get(k, {}).get("count", 0) >= 20]
        counted = sum(groups[k]["share"] for k in kinds) or 1
        out.append(f"<h3>Seed {n}: {a['alive']:,} cells at tick {a['tick']:,}</h3>{maps}")
        rows = []
        for k in kinds:
            g = groups[k]
            kin = g["bites_on_similar"]
            rows.append([
                f'<span class="dot" style="background:{KIND_COLOUR[k]}"></span>{KIND_NAMES[k]}',
                f"{g['share'] / counted * 100:.0f}%", f"{g['diet'][0] * 100:.0f}", f"{g['diet'][1] * 100:.0f}", f"{g['diet'][2] * 100:.0f}",
                f"{g['moving'] * 100:.0f}%", f"{g['neighbours']:.1f}", f"{g['age_median']:.0f}", f"{g['energy_median']:.1f}",
            ])
        out.append(table(["", "Share", "% ground", "% meat", "% bite", "Moving", "Neighbours", "Age", "Energy"], rows))
        budget_rows = [(KIND_NAMES[k], {ACTIONS[i]: groups[k]["budget"][i] for i in range(10)}) for k in kinds]
        palette = [("rest", "#5a5a60", "rest"), ("move", "#c8c8c8", "move"), ("ground", "#3cc878", "ground"), ("meat", "#f0aa28", "meat"), ("grow", "#9085e9", "grow"), ("bite", "#eb463c", "bite"), ("emit", "#3987e5", "scent"), ("heat", "#ff8ac2", "heat"), ("give", "#39c5cf", "give"), ("enzyme", "#a3a321", "enzyme")]
        out.append(stacks(f"Seed {n}: how each group spends its energy", "Share of the spending budget, averaged over time.", budget_rows, palette))
        order = KIND_NAMES.keys()
        head = ["Next to →"] + [KIND_NAMES[k] for k in order]
        grid = []
        for k in kinds:
            cells = []
            for o in order:
                v = groups[k]["next_to"].get(o, 0)
                cells.append(f'<span class="heat" style="background:rgba(57,135,229,{min(v * 1.4, 0.85):.2f})">{v * 100:.0f}%</span>')
            grid.append([KIND_NAMES[k]] + cells)
        out.append(f'<div class="chart-title">Seed {n}: who lives next to whom</div>' + table(head, grid, "matrix"))
        out.append(scatter(f"Seed {n}: genome map", "Each dot is a cell, placed by the two main axes of genome difference and coloured by how it lives. Separate clouds of one colour would mean a species with its own way of life. Mixed clouds mean behaviour isn't tied to a genome group.", a["points"]))
    return "".join(out)


def status_budget(v):
    return "Barely used" if v < 0.01 else "Used"


def section_behaviours(worlds, ana, removal):
    """One row per behaviour, across the seeds, with the numbers behind it."""
    last = [rows[-1] for _, _, rows in worlds]
    names = [n for n, _, _ in worlds]
    def each(fn, fmt=lambda v: f"{v:.2f}"):
        out = []
        for r in last:
            try:
                v = fn(r)
            except (KeyError, TypeError):
                v = None
            out.append("–" if v is None else fmt(v))
        return out
    p0 = lambda v: f"{v * 100:.0f}%"
    p1 = lambda v: f"{v * 100:.1f}%"
    def from_ana(fn, fmt=lambda v: f"{v:.2f}"):
        out = []
        for i in range(len(worlds)):
            a = ana.get(i + 1)
            try:
                v = fn(a) if a else None
            except (KeyError, TypeError):
                v = None
            out.append("–" if v is None else fmt(v))
        return out
    def hunters_vs_settlers(a, field):
        g = a["groups"]
        return abs(g["hunter"][field][0] - g["settler"][field][0])
    def big_colonies(a):
        return sum(c["cells"] for c in a["colony"] if c["from"] >= 257)
    def align(a):
        g = a["groups"]
        vals = [(v["alignment"], v["count"]) for v in g.values() if v.get("alignment") is not None and v.get("count")]
        return sum(x * c for x, c in vals) / sum(c for _, c in vals)
    def willing(a, field):
        g = a["groups"]
        vals = [(v[field], v["count"]) for v in g.values() if v.get("count")]
        return sum(x * c for x, c in vals) / sum(c for _, c in vals)
    rows = [
        ("Grazing", "Energy from ground food", each(lambda r: r["world"]["lives_on"]["ground"], p0), "The base of every world"),
        ("Biting", "Energy from biting", each(lambda r: r["world"]["lives_on"]["bite"], p0), "A real way to live, in some worlds"),
        ("Being bitten", "Share of deaths", each(bitten_share, p0), "Up to the main cause of death"),
        ("Scavenging", "Energy from meat", each(lambda r: r["world"]["lives_on"]["meat"], p0), "Corpses feed real groups"),
        ("Cannibalism", "Bites on similar genomes", each(lambda r: r["world"]["bites_on_similar"], p0), "High in some worlds, so relatives aren't spared by default"),
        ("Moving", "Share of ticks", each(lambda r: r["world"]["moving"], p0), "Splits nomads from settlers"),
        ("Following food", "Heading vs food gradient, +1 is straight at it", each(lambda r: r["world"]["heads_to_food"]), "Strong in all but the still worlds"),
        ("Avoiding bodies", "Heading vs bodies, negative is away", each(lambda r: r["world"]["heads_to_bigger"]), "Cells steer away from others"),
        ("Fleeing or chasing", "Heading towards smaller minus towards bigger", each(lambda r: r["world"]["heads_to_smaller"] - r["world"]["heads_to_bigger"]), "Near zero, so size doesn't change the response. Not seen yet"),
        ("Resting", "Budget share", each(lambda r: r["world"]["budget"]["rest"], p0), "Common, sometimes the main habit"),
        ("Colonies", "Cells in touching groups over 256", from_ana(big_colonies, p0), "Real, with sharp borders"),
        ("Living close", "Neighbours out of 8", each(lambda r: r["world"]["neighbours"], lambda v: f"{v:.1f}"), ""),
        ("Heading together", "Neighbours' heading alignment, 1 is identical", from_ana(align), "Mild. Swarms that travel aren't shown"),
        ("Energy sharing", "Cells willing to share", from_ana(lambda a: willing(a, "share_willing"), p0), "See the removal test"),
        ("Gifts", "Budget share", each(lambda r: r["world"]["budget"]["give"], p1), "Barely used"),
        ("Heat", "Budget share", each(lambda r: r["world"]["budget"]["heat"], p1), "Barely used, so nobody cooks prey the way Fraser's cells do"),
        ("Enzyme", "Budget share", each(lambda r: r["world"]["budget"]["enzyme"], p1), "Barely used"),
        ("Scent", "Budget share", each(lambda r: r["world"]["budget"]["emit"], p1), "Barely used"),
        ("Following scent", "Largest following of any channel", each(lambda r: max(r["world"]["follows_scent"], key=abs)), "Weak. No trails yet"),
        ("Touch signals", "Hunters' signal minus settlers'", from_ana(lambda a: hunters_vs_settlers(a, "voice_mean")), "Where this is high, a group shows a different signal"),
        ("Mating", "Mixed births", each(mixed_share, p0), "Varies a lot"),
        ("Long life", "Mean age", each(lambda r: r["world"]["age"], lambda v: f"{v:.0f}"), ""),
    ]
    t = table(["Behaviour", "Measure"] + names + ["So far"], [[a, b] + c + [d] for a, b, c, d in rows])
    return f"""<h2 id="behaviours">Every behaviour, measured</h2>
<p>Everything I can measure at the end of each baseline run. A number in the budget means what cells choose to spend on. It doesn't prove that it matters, so the removal test below checks that.</p>{t}"""


def section_voice(ana):
    if not ana:
        return ""
    out = []
    for n in sorted(ana):
        g = ana[n]["groups"]
        kinds = [k for k in KIND_NAMES if g.get(k, {}).get("count", 0) >= 20]
        rows = []
        for k in kinds:
            v = g[k]
            kin = v["bites_on_similar"]
            rows.append([
                f'<span class="dot" style="background:{KIND_COLOUR[k]}"></span>{KIND_NAMES[k]}',
                f"{v['voice_mean'][0]:+.2f}", f"{v['voice_mean'][1]:+.2f}", f"{v['share_willing'] * 100:.0f}%", f"{v['mate_willing'] * 100:.0f}%",
                "–" if kin is None else f"{kin * 100:.0f}%", f"{v['to_bigger']:+.2f}", f"{v['to_smaller']:+.2f}", f"{(v['alignment'] or 0):.2f}",
            ])
        out.append(f"<h3>Seed {n}</h3>" + table(["", "Signal 1", "Signal 2", "Will share", "Will mate", "Bites on similar", "Heads to bigger", "Heads to smaller", "Heading together"], rows))
    return f"""<h2 id="signals">Signals, sharing and targets, group by group</h2>
<p>Signals are two numbers each cell chooses to show its neighbours, with no built-in meaning. Differences between groups could come from real signalling or from one lineage happening to carry a value, and these numbers alone can't tell those apart.</p>{''.join(out)}"""


def section_removal():
    d = EXP / "removal"
    files = list(d.glob("*.jsonl")) if d.exists() else []
    intro = """<h2 id="removal">What they actually rely on</h2>
<p>A cell using something doesn't prove it matters. So each evolved world is copied, one ability is switched off, and the population is compared with an untouched copy over the next 30,000 ticks.</p>"""
    if not files:
        return intro + '<p class="pending">Queued for the GPU, and will appear here when it runs.</p>'
    seeds = sorted({f.stem.split("-")[0] for f in files})
    names = {"pheromones_on": "Scent", "voice_on": "Touch signals", "bite_on": "Biting", "heat_on": "Heat", "give_on": "Gifts", "enzyme_on": "Enzyme", "share_on": "Energy sharing", "sex_on": "Mating", "memory_on": "Memory", "noise_on": "Hearing movement", "vision_steps": "Seeing beyond touch"}
    means = {}
    for f in files:
        seed, what = f.stem.split("-", 1)
        _, rows = load_log(f)
        if rows:
            means[(seed, what)] = sum(r["cells"] for r in rows) / len(rows)
    rows = []
    for what, label in names.items():
        row = [label]
        for s in seeds:
            base, v = means.get((s, "none")), means.get((s, what))
            row.append("–" if not base or v is None else f"{(v / base - 1) * 100:+.0f}%")
        rows.append(row)
    return intro + table(["Removed"] + [s.replace("seed", "Seed ") for s in seeds], rows) + "<p>Each cell compares one run with one control, so a few percent is noise. Big negative numbers mean the world relied on that ability. Big positive numbers mean it was a burden on the population as a whole.</p>"


def section_species(worlds, ana):
    def species_series(rows, name, colour):
        at = rows[-1].get("species_at") or {}
        return (name, colour, [(float(k), v) for k, v in at.items()])
    series = [species_series(rows, n, SEED_COLOUR[i]) for i, (n, _, rows) in enumerate(worlds)]
    _, g = load_log(EXP / "giant/4096.jsonl")
    if g:
        series.append(species_series(g, "4096²", SEED_COLOUR[3]))
    capped = [(n, c, [(x, min(y, 20)) for x, y in s]) for n, c, s in series]
    c1 = lines("How many species, depending on how strict you are", "Genome groups at each threshold, capped at 20. A flat stretch means groups that stay separate however you draw the line, which is what real species look like.", capped, "genome distance threshold", ymax=20, xfmt=lambda v: f"{v:g}")
    return f"""<h2 id="species">Species</h2>
<p>Species here means genome groups: cells joined if their weights are close, with no rule creating them. Genomes drift fast, about 0.03 of weight difference per generation, so a low threshold only finds recent family. That's why 0.4 is the reference.</p>{c1}"""


def section_experiments(worlds):
    base_cfg = worlds[0][1].get("config") if worlds and worlds[0][1] else None
    def row_vals(fn):
        return [fn(rows[-1]) for _, _, rows in worlds]
    strip_a = [("Baseline", row_vals(other_share))]
    strip_b = [("Baseline", row_vals(lambda r: r.get("species_explain_behaviour")))]
    cards = []
    for key, name, idea, dirs, running in EXPERIMENTS:
        logs = experiment_logs(dirs)
        on = [(s, h, r) for s, h, r in logs if not any(w in s for w in ("off", "control", "tune", "sweep", "check", "probe", "smoke", "pilot"))]
        if on:
            strip_a.append((name, [other_share(r[-1]) for _, _, r in on]))
            strip_b.append((name, [r[-1].get("species_explain_behaviour") for _, _, r in on]))
        rows = []
        for s, h, r in logs:
            l = r[-1]
            rows.append([s.replace("_", " "), fmt_num(l["tick"]), fmt_num(l["cells"]), f"{l.get('effective_species', 0):.1f}", f"{l.get('species_explain_behaviour', 0):.2f}", pct(other_share(l)), pct(bitten_share(l)), pct(l["world"]["moving"]) if l.get("world") else "–", f"{l['world']['neighbours']:.1f}" if l.get("world") else "–"])
        status = "Still running" if running else "Agent finished, rerun at 512²"
        body = table(["Run", "Tick", "Cells", "Species", "Live differently", "From others", "Bitten", "Moving", "Neighbours"], rows) if rows else '<p class="pending">No GPU results at 512² yet.</p>'
        vids = sorted((ROOT / f"videos/life-gpu/{key}").glob("*-phone.mp4")) if (ROOT / f"videos/life-gpu/{key}").exists() else []
        cards.append(f'<details><summary><b>{esc(name)}</b> <span class="pill">{status}</span> {len(logs)} runs</summary><p>{esc(idea)}</p>{body}</details>')
    a = strip("Energy from eating other cells", "Each dot is one world at its last log. The band is the range across baseline.", strip_a, 0.5, lambda v: f"{v * 100:.0f}%")
    b = strip("How differently species live", "Species explain this share of behaviour, counting groups of at least 1%, corrected for group count. 0 means they're only labels.", strip_b, 0.4, lambda v: f"{v:.2f}")
    return f"""<h2 id="experiments">The 10 experiments</h2>
<p>Each one adds a single piece of physics, never a role, and compares against baseline at 512² on the same seeds. Runs differ in length and only some are finished, so read these as leads. Two or three seeds can't credit a rule, because worlds differ this much by chance.</p>{a}{b}{''.join(cards)}"""


CSS = """
:root{color-scheme:dark}
html,body{margin:0;background:#111113;color:#ececee;font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif}
main{max-width:760px;margin:0 auto;padding:18px 14px 80px}
h1{font-size:26px;line-height:1.2;margin:6px 0 6px}h2{font-size:21px;margin:38px 0 8px;line-height:1.25}h3{font-size:16px;margin:24px 0 6px}
p,li{color:#d4d4d8}a{color:#6db0ff}.sub{color:#a1a1aa;font-size:14px}.pending{color:#f0c060}
nav{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0}nav a{background:#222226;border-radius:14px;padding:3px 10px;font-size:13px;text-decoration:none;color:#d4d4d8}
.heroes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;margin:14px 0}.hero{background:#1c1c1f;border-radius:10px;padding:10px 12px}.hero b{display:block;font-size:21px;line-height:1.2}.hero span{font-size:12.5px;color:#a1a1aa}
figure{margin:18px 0}.chart svg{width:100%;height:auto;display:block;max-width:560px}.chart-title{font-weight:600;font-size:14.5px}.chart-sub{color:#a1a1aa;font-size:12.5px;margin-bottom:4px}
svg text{fill:#a1a1aa;font-size:11px}svg text.lbl{fill:#ececee;font-size:11.5px}.grid{stroke:#2c2c30;stroke-width:1}
.legend{display:flex;flex-wrap:wrap;gap:2px 12px;font-size:12px;color:#c4c4c8;margin:2px 0 4px}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.media video,.media img{width:100%;display:block;border-radius:8px;background:#000}figcaption{font-size:13px;color:#a1a1aa;margin-top:5px}
.scroll{overflow-x:auto;margin:10px 0;-webkit-overflow-scrolling:touch}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;min-width:100%}th,td{padding:5px 8px;text-align:left;border-bottom:1px solid #27272b;white-space:nowrap}th{color:#a1a1aa;font-weight:500}td:last-child{white-space:normal;min-width:140px}table.matrix td:last-child{min-width:0}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}.heat{display:inline-block;padding:1px 6px;border-radius:4px}
details{background:#1a1a1d;border-radius:8px;padding:8px 12px;margin:8px 0}summary{cursor:pointer}.pill{font-size:11.5px;background:#2b2b30;border-radius:10px;padding:1px 8px;color:#c4c4c8}
#tip{position:fixed;pointer-events:none;background:#ececee;color:#111;font-size:12.5px;padding:4px 8px;border-radius:6px;opacity:0;max-width:260px;z-index:9}
"""

JS = """
const tip=document.getElementById('tip');
function show(e){const m=e.target.closest&&e.target.closest('[data-tip]');if(!m){tip.style.opacity=0;return;}
 const p=e.touches?e.touches[0]:e;tip.textContent=m.dataset.tip;tip.style.opacity=1;const w=tip.offsetWidth;
 tip.style.left=Math.max(6,Math.min(p.clientX+10,innerWidth-w-6))+'px';tip.style.top=Math.max(6,p.clientY-38)+'px';}
document.addEventListener('mousemove',show);document.addEventListener('touchstart',show,{passive:true});
"""


def separated_species(ana):
    """Worlds where one way of living sits in its own genome cloud, split along the main axis."""
    found = []
    for n in sorted(ana):
        pts = [p for p in ana[n]["points"] if p[2] in KIND_NAMES]
        if len(pts) < 100:
            continue
        lo, hi = min(p[0] for p in pts), max(p[0] for p in pts)
        cut = (lo + hi) / 2
        for _ in range(10):
            a = [p[0] for p in pts if p[0] < cut]
            b = [p[0] for p in pts if p[0] >= cut]
            if not a or not b:
                break
            cut = (sum(a) / len(a) + sum(b) / len(b)) / 2
        left = [p for p in pts if p[0] < cut]
        right = [p for p in pts if p[0] >= cut]
        if not left or not right:
            continue
        for side, other in ((left, right), (right, left)):
            for k in KIND_NAMES:
                in_side = sum(1 for p in side if p[2] == k)
                if in_side >= 30 and in_side / len(side) > 0.8:
                    found.append((n, k))
    return found


def findings(worlds, ana, is_v2):
    last = [(n, rows[-1]) for n, _, rows in worlds]
    predator = [(n, r) for n, r in last if (other_share(r) or 0) >= 0.15]
    grazer = [(n, r) for n, r in last if (other_share(r) or 0) < 0.15]
    items = []
    if predator:
        b = max(bitten_share(r) or 0 for _, r in predator)
        items.append(f"<b>Predators evolve with nothing telling them to.</b> In {len(predator)} of {len(last)} baseline worlds cells get at least 15% of their energy from other cells, and being bitten causes up to {b * 100:.0f}% of deaths. No cell type, score or genome signal is involved.")
    if grazer:
        items.append(f"<b>The same rules can also stay boring.</b> {len(grazer)} of {len(last)} worlds had almost no predation and were close to one grazing lineage. Which way a world goes looks like chance, so one run proves little.")
    a = ana.get(3) or ana.get(1)
    if a:
        g = a["groups"]
        kinds = [k for k in KIND_NAMES if g.get(k, {}).get("count", 0) >= 20]
        counted = sum(g[k]["share"] for k in kinds) or 1
        parts = ", ".join(f"{g[k]['share'] / counted * 100:.0f}% {KIND_NAMES[k].lower()}" for k in kinds)
        items.append(f"<b>Different ways of life live side by side.</b> One world holds {parts}, and the maps show they occupy different places.")
    for n, k in separated_species(ana)[:2]:
        items.append(f"<b>A species with its own way of life.</b> In seed {n}, a whole genome cloud is almost entirely {KIND_NAMES[k].lower()}, set apart from the other groups. See the genome map below.")
    items.append("<b>Colonies hold territory.</b> Different genomes keep separate ground with sharp borders, and you can see it directly in the genome-coloured views.")
    items.append("<b>Scared hasn't evolved.</b> Cells steer away from bodies in general, but treat bigger and smaller ones about the same, so there's no real fleeing or chasing yet. Hunters mostly sit and bite what's next to them.")
    items.append("<b>Scent, gifts and heat are barely used.</b> Each takes under about 1% of spending in baseline, and scent following is weak. Giving cells a feature doesn't mean evolution picks it up.")
    items.append("<b>Nothing here beats baseline clearly yet.</b> Several experiments are still waiting for the GPU, so no rule can be called a winner.")
    return "<ul>" + "".join(f"<li>{i}</li>" for i in items) + "</ul>"


def main():
    worlds, is_v2 = baseline()
    ana = analyses()
    _, giant = load_log(EXP / "giant/4096.jsonl")
    giant_last = giant[-1] if giant else None
    updated = datetime.now().strftime("%-d %B %Y, %H:%M")
    finished = sum(1 for _, _, _, d, _ in EXPERIMENTS if experiment_logs(d))
    videos = "".join([
        video("media/world4096.mp4", "4096² world, 2.9 million cells. Left: colour is genome. Right: diet, with green for ground, red for meat and biting. The camera flies to the most varied and predatory places."),
        video("media/world2048.mp4", "2048² world, 950,000 cells, seeded from two evolved worlds that meet at the borders."),
        video("media/close-world4096.mp4", "Close-up in the 4096² world. Each square is one cell."),
        video("media/close-seed3.mp4", "Close-up, seed 3. Watch individual cells move, bite and divide."),
        video("media/close-seed1.mp4", "Close-up, seed 1."),
        video("media/world8192.mp4", "8192², 67 million squares. Too sparse: predators vanish. Useful as a warning."),
    ])
    stills = "".join([
        image("media/zoom4096.jpg", "Zoomed into 4096²: colonies with different genomes hold territory (left), and hunters and scavengers sit among them (right)."),
        image("media/world2048.jpg", "The whole 2048² world, coloured by genome."),
        image("media/world4096.jpg", "The whole 4096² world, coloured by genome."),
    ])
    full_links = [(f"media/{n}-full.png", label) for n, label in (("world2048", "2048², 10 MB"), ("world4096", "4096², 37 MB")) if (OUT / f"media/{n}-full.png").exists()]
    full = ('<p class="sub">Full pictures at one pixel per square, to zoom into: ' + ", ".join(f'<a href="{esc(a)}">{esc(b)}</a>' for a, b in full_links) + ".</p>") if full_links else ""
    heroes = ""
    if giant_last:
        heroes = f"""<div class="heroes">
<div class="hero"><b>{giant_last['cells'] / 1e6:.1f}M</b><span>cells on 16.8M squares, 4096² world</span></div>
<div class="hero"><b>{sum(r['ticks_per_second'] for r in giant) / len(giant):.0f}</b><span>ticks per second at that size</span></div>
<div class="hero"><b>{max(bitten_share(r[-1]) or 0 for _, _, r in worlds) * 100:.0f}%</b><span>of deaths from biting, in the most predatory baseline world</span></div>
<div class="hero"><b>{finished} of 10</b><span>experiments have results at 512² so far</span></div></div>"""
    baseline_note = "" if is_v2 else '<p class="pending">These baseline numbers predate a fix to how cell behaviour is measured. Fixed runs are queued and will replace them automatically. Physics isn\'t affected.</p>'
    body = f"""
<h1>Thermodynamic life at GPU scale</h1>
<p class="sub">Updated {esc(updated)}. Work in progress: experiments are still running and this page fills in as they finish.</p>
{heroes}
<nav><a href="#short">Short version</a><a href="#video">Videos</a><a href="#rules">Rules</a><a href="#lifestyles">Ways of living</a><a href="#behaviours">Every behaviour</a><a href="#signals">Signals</a><a href="#removal">Removal test</a><a href="#species">Species</a><a href="#time">Over time</a><a href="#experiments">Experiments</a><a href="#scale">Scale</a><a href="#limits">Limits</a></nav>
<h2 id="short">The short version</h2>{findings(worlds, ana, is_v2)}{baseline_note}
<h2 id="video">Watch it</h2>{videos}{stills}{full}
<h2 id="rules">The rules</h2>
<p>One kind of cell, one per square. Each has its own recurrent neural network with 672 weights and 8 memory units, and an energy store. Each tick it senses, thinks, and splits its energy across actions. There's no fitness function: energy alone decides who survives, and a cell that fills its womb divides. Nothing here assigns a role.</p>
{table(["Physics", "Rule"], [
    ["Sun", "Grows ground food. Patchy at several scales, with seasons. Rock ridges form regions joined by gaps."],
    ["Death", "A dead cell's body and energy become meat, which rots back into ground food."],
    ["Biting", "Pulls energy from the cell in front, harder for stronger cells. Only open sides can be reached, so being inside a clump protects you. It loses energy: biters keep half and the rest spills as meat."],
    ["Heat", "Lands on the four squares around the emitter and burns whoever is there."],
    ["Enzyme", "Lets anyone nearby eat ground food up to 3x faster, so cheating is possible."],
    ["Scent", "Four channels, from long trails to short shouts, with no built-in meaning. A cell only emits what its brain chooses."],
    ["Touch signals", "Two numbers a cell chooses to show its neighbours."],
    ["Sight", "Bodies up to 4 squares away in 8 directions: how close, bigger or smaller, and what signals they show."],
    ["Hearing", "Moving makes noise, louder for heavier bodies, so stillness hides you."],
    ["Sharing", "Two touching cells that both agree level out their energy."],
    ["Birth", "A clone, or a mix with any willing neighbour. Whole memory units come from one parent or the other. Mutations change about 10% of weights."],
    ["Commitment", "Spending a little on something barely works, so dabbling is wasteful."],
    ["Rain", "A random cell lands now and then, so the world can't die for good."],
])}
<p><b>Removed because they decide for the cells:</b> an odour derived from the genome, a built-in sense of how related a neighbour is, and a genome check on who can mate. Cells must work out who is who from what others choose to show. Colour by genome is only for us watching: nearby colours mean similar weights, using the two or three main axes along which the population's genomes differ.</p>
{section_lifestyles(ana)}
<h2 id="diet">Two humps: grazers and carnivores</h2>
<p>If predators were just grazers who sometimes bite, this would be one hump. A second hump at the far right means cells that live almost entirely on other cells.</p>
{section_diet(worlds, giant_last)}
{section_behaviours(worlds, ana, None)}
{section_voice(ana)}
{section_removal()}
{section_species(worlds, ana)}
{section_time(worlds)}
{section_experiments(worlds)}
{section_scale()}
<h2 id="limits">Limits</h2>
<ul>
<li><b>Few runs.</b> Three baseline seeds, and two or three for most experiments. Worlds differ a lot by chance, so treat everything here as leads.</li>
<li><b>Giant worlds are young.</b> The 4096² world ran about 16,000 ticks from copied tiles, so the regular pattern you can see comes from that. It isn't long enough for much new evolution.</li>
<li><b>512² isn't the same physics as 4096².</b> Small worlds have fewer season regions and smaller landscape features, so what's true at one size may not carry over.</li>
<li><b>Labels are mine.</b> Hunter, settler and the rest are cut-offs for reading results, and different cut-offs would shift the shares.</li>
<li><b>Species depend on a threshold.</b> The chart shows how much. Some worlds hold steady across thresholds, and others don't.</li>
<li><b>The GPU is shared.</b> Experiments wait behind other projects, so this page updates slowly.</li>
<li><b>Checked:</b> 13 tests cover the energy ledger, grid consistency, repeatable runs and each rule against plain arithmetic. An independent review of the engine found no concurrency or energy bugs. It did find measurement problems, which are fixed, and the population limit that shaped the 4096² and 8192² runs.</li>
</ul>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thermodynamic life at GPU scale</title><link rel="icon" href="data:,"><style>{CSS}</style></head><body><main>{body}</main><div id="tip"></div><script>{JS}</script></body></html>')
    print("wrote", OUT / "index.html", os.path.getsize(OUT / "index.html") // 1024, "KB")


if __name__ == "__main__":
    main()
