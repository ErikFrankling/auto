#!/usr/bin/env python3
"""Builds report/index.html and one standalone HTML per chart from results/*.jsonl.

Every number in the report comes from the results files or from PUBLISHED below,
which holds other people's numbers with their sources.
"""
import html
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report"

# Other people's dense Life numbers, in cell updates per second. See
# research/notes/fastest-life.md for how far each can be trusted.
PUBLISHED = [
    ("NVIDIA A40, CUDA (Binary Banter, 2023)", 11.720e12, "https://binary-banter.github.io/game-of-life/"),
    ("Radeon 7900 XTX, OpenCL (Binary Banter, 2023)", 8.939e12, "https://binary-banter.github.io/game-of-life/"),
    ("GTX Titan X, CUDA (Fujita et al., 2015)", 1.69e12, "https://tk-www.elcom.nitech.ac.jp/WTCS2015/WTCS2015_Proceedings.pdf"),
    ("RTX 3080, Taichi (Hu et al., 2021)", 0.511e12, "https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf"),
    ("i9-11900K, threaded SIMD (Binary Banter, 2023)", 0.296e12, "https://binary-banter.github.io/game-of-life/"),
    ("Xeon, Golly QuickLife (Hu et al., 2021)", 0.00546e12, "https://yuanming.taichi.graphics/publication/2021-quantaichi/quantaichi.pdf"),
]

CSS = """
.viz-root { color-scheme: light; --surface-1:#fcfcfb; --surface-2:#f2f1ed; --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#7a7972; --grid:#e4e3de; --axis:#b5b4ad;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4; --s6:#008300; --s7:#6250d6; }
@media (prefers-color-scheme: dark) { .viz-root { color-scheme: dark; --surface-1:#1a1a19; --surface-2:#242422; --text-primary:#f0efec; --text-secondary:#c3c2b7; --text-muted:#94938a; --grid:#333330; --axis:#55544f;
  --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181; --s6:#008300; --s7:#9085e9; } }
html, body { margin:0; }
.viz-root { background:var(--surface-1); color:var(--text-primary); font:15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1020px; margin: 0 auto; padding: 32px 20px 96px; }
h1 { font-size: 30px; line-height:1.2; margin:0 0 6px; } h2 { font-size:22px; margin:44px 0 8px; } h3 { font-size:16px; margin:22px 0 4px; }
p { margin: 8px 0; max-width: 74ch; } a { color: var(--s1); } .sub { color:var(--text-secondary); } code { font: 13px ui-monospace, monospace; background:var(--surface-2); padding:1px 4px; border-radius:4px; }
.heroes { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; margin:18px 0; }
.hero { background:var(--surface-2); border-radius:10px; padding:12px 14px; } .hero b { display:block; font-size:25px; line-height:1.2; font-variant-numeric:tabular-nums; } .hero span { color:var(--text-secondary); font-size:13px; }
.videos { display:grid; grid-template-columns:repeat(auto-fill,minmax(310px,1fr)); gap:14px; margin-top:14px; }
.videos figure { margin:0; } video { width:100%; border-radius:8px; background:#000; display:block; } figcaption { font-size:13px; color:var(--text-secondary); margin-top:4px; }
figure.chart { margin:18px 0; } .chart-title { font-weight:600; } .chart-sub { color:var(--text-secondary); font-size:13px; margin-bottom:6px; }
svg text { fill:var(--text-secondary); font: 12px ui-sans-serif, system-ui, sans-serif; } svg .lbl { fill:var(--text-primary); }
.legend { display:flex; flex-wrap:wrap; gap:4px 14px; font-size:13px; color:var(--text-secondary); margin:4px 0; } .legend i { display:inline-block; width:12px; height:3px; border-radius:2px; margin-right:5px; vertical-align:middle; }
table { border-collapse:collapse; font-size:13.5px; margin:10px 0; width:100%; font-variant-numeric:tabular-nums; } th { text-align:left; color:var(--text-secondary); font-weight:600; border-bottom:1px solid var(--axis); padding:4px 8px 4px 0; } td { border-bottom:1px solid var(--grid); padding:4px 8px 4px 0; vertical-align:top; } td.n, th.n { text-align:right; }
details { margin:8px 0; } summary { cursor:pointer; color:var(--text-secondary); font-size:13px; }
#tip { position:fixed; pointer-events:none; background:var(--text-primary); color:var(--surface-1); font-size:12.5px; padding:4px 8px; border-radius:5px; opacity:0; transition:opacity .08s; z-index:9; white-space:nowrap; }
.mark { cursor:default; }
.note { border-left:3px solid var(--axis); padding:2px 0 2px 12px; color:var(--text-secondary); margin:12px 0; max-width:74ch; }
"""

TIP_JS = """
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const m=e.target.closest&&e.target.closest('[data-tip]');
 if(!m){tip.style.opacity=0;return;} tip.textContent=m.dataset.tip; tip.style.opacity=1;
 const w=tip.offsetWidth; tip.style.left=Math.min(e.clientX+12,innerWidth-w-8)+'px'; tip.style.top=(e.clientY+14)+'px';});
"""


def load(name):
    path = RESULTS / name
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def si(v, unit=""):
    for f, s in [(1e12, " T"), (1e9, " G"), (1e6, " M"), (1e3, " k")]:
        if abs(v) >= f:
            return f"{v / f:.3g}{s}{unit}"
    return f"{v:.3g} {unit}".strip()


def cells_name(n):
    for f, s in [(1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")]:
        if n >= f * 0.95:
            return f"{n / f:.3g} {s}"
    return f"{n:.0f}"


def esc(s):
    return html.escape(str(s), quote=True)


def log_ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0**k for k in range(a, b + 1)]


def line_chart(title, sub, series, xlabel, ylabel, x_fmt=si, y_fmt=si, logx=True, logy=True, w=960, h=420, id_="", direct=True):
    """series: list of (name, [(x, y, tooltip)]). Colour slots follow list order."""
    pts = [p for _, s in series for p in s]
    if not pts:
        return f'<p class="sub">No data yet for: {esc(title)}</p>'
    L, R, T, B = 66, 16 + (250 if direct else 0), 12, 44
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    if logx:
        x0, x1 = min(xs) / 1.3, max(xs) * 1.3
        fx = lambda v: L + (math.log10(v) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * (w - L - R)
        xt = [t for t in log_ticks(x0, x1) if x0 <= t <= x1]
    else:
        x0, x1 = min(xs), max(xs)
        pad = (x1 - x0) * 0.04 or 1
        x0 -= pad
        x1 += pad
        fx = lambda v: L + (v - x0) / (x1 - x0) * (w - L - R)
        xt = sorted(set(xs))
    if logy:
        y0, y1 = min(ys) / 1.5, max(ys) * 1.5
        fy = lambda v: h - B - (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (h - T - B)
        yt = [t for t in log_ticks(y0, y1) if y0 <= t <= y1]
    else:
        y0, y1 = 0, max(ys) * 1.1
        fy = lambda v: h - B - (v - y0) / (y1 - y0) * (h - T - B)
        step = 10 ** math.floor(math.log10(y1))
        if y1 / step < 3:
            step /= 2
        yt = [step * k for k in range(0, int(y1 / step) + 1)]
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}">']
    for t in yt:
        y = fy(t)
        o.append(f'<line x1="{L}" x2="{w - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--grid)"/><text x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{esc(y_fmt(t))}</text>')
    for t in xt:
        x = fx(t)
        o.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{h - B}" y2="{h - B + 4}" stroke="var(--axis)"/><text x="{x:.1f}" y="{h - B + 18}" text-anchor="middle">{esc(x_fmt(t))}</text>')
    o.append(f'<line x1="{L}" x2="{w - R}" y1="{h - B}" y2="{h - B}" stroke="var(--axis)"/>')
    o.append(f'<text x="{(L + w - R) / 2}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>')
    o.append(f'<text x="14" y="{(T + h - B) / 2}" text-anchor="middle" transform="rotate(-90 14 {(T + h - B) / 2})">{esc(ylabel)}</text>')
    labels = []
    for i, (name, s) in enumerate(series):
        c = f"var(--s{i + 1})"
        s = sorted(s, key=lambda p: p[0])
        d = " ".join(f"{'M' if k == 0 else 'L'}{fx(p[0]):.1f} {fy(p[1]):.1f}" for k, p in enumerate(s))
        o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for p in s:
            o.append(f'<circle class="mark" cx="{fx(p[0]):.1f}" cy="{fy(p[1]):.1f}" r="4" fill="{c}" stroke="var(--surface-1)" stroke-width="2" data-tip="{esc(name)}: {esc(p[2])}"/>')
        if direct and s:
            labels.append([fy(s[-1][1]) + 4, w - R + 10, name, c])
    labels.sort()
    for k in range(1, len(labels)):
        if labels[k][0] - labels[k - 1][0] < 15:
            labels[k][0] = labels[k - 1][0] + 15
    for y, x, name, c in labels:
        o.append(f'<text class="lbl" x="{x:.1f}" y="{y:.1f}">{esc(name)}</text>')
    o.append("</svg>")
    legend = "".join(f'<span><i style="background:var(--s{i + 1})"></i>{esc(n)}</span>' for i, (n, _) in enumerate(series))
    return chart_wrap(title, sub, f'<div class="legend">{legend}</div>' + "".join(o), id_)


def bar_chart(title, sub, rows, value_fmt=si, log=True, w=960, row_h=27, id_="", label_w=320):
    """rows: list of (label, value, tooltip, slot). Horizontal bars, thin, value at the end."""
    if not rows:
        return f'<p class="sub">No data yet for: {esc(title)}</p>'
    L, R = label_w, 90
    vals = [r[1] for r in rows]
    if log:
        lo, hi = min(vals) / 2, max(vals) * 1.1
        fx = lambda v: (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (w - L - R)
    else:
        hi = max(vals)
        fx = lambda v: v / hi * (w - L - R)
    h = row_h * len(rows) + 6
    o = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(title)}">']
    for i, (label, v, tip, slot) in enumerate(rows):
        y = i * row_h + 4
        bw = max(fx(v), 2)
        o.append(f'<text class="lbl" x="{L - 10}" y="{y + 14}" text-anchor="end">{esc(label)}</text>')
        o.append(f'<rect class="mark" x="{L}" y="{y + 3}" width="{bw:.1f}" height="14" rx="4" fill="var(--s{slot})" data-tip="{esc(tip)}"/>')
        o.append(f'<text x="{L + bw + 8:.1f}" y="{y + 14}">{esc(value_fmt(v))}</text>')
    o.append("</svg>")
    return chart_wrap(title, sub, "".join(o), id_)


CHARTS = {}


def chart_wrap(title, sub, body, id_):
    block = f'<figure class="chart"><div class="chart-title">{esc(title)}</div><div class="chart-sub">{esc(sub)}</div>{body}</figure>'
    if id_:
        CHARTS[id_] = block
    return block


def page(title, body):
    return f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{CSS}</style></head><body class="viz-root"><main>{body}</main><div id="tip"></div><script>{TIP_JS}</script></body></html>'


def table(headers, rows, numeric=()):
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def best(rows):
    """Keep the latest measurement per (id, tag, adapter, size), so a rerun replaces an old number."""
    seen = {}
    for r in rows:
        seen[(r.get("id"), r.get("tag"), r.get("adapter"), r.get("width"), r.get("height"), r.get("checks"))] = r
    return list(seen.values())


def main():
    gpu = best(load("gpu.jsonl"))
    api = best(load("api.jsonl"))
    brains = best(load("brains.jsonl"))
    cpu_raw = load("cpu.jsonl")
    cpu = [r for r in cpu_raw if "cells_per_second" in r]
    hash_rows = [r for r in cpu_raw if "pattern" in r]
    browser = load("browser.jsonl")
    llvm = load("llvmpipe.jsonl")
    igpu = load("igpu.jsonl")
    videos = json.loads((ROOT / "videos" / "index.json").read_text()) if (ROOT / "videos" / "index.json").exists() else []

    tag = lambda t: [r for r in gpu if r.get("tag") == t]
    # Kernels
    names = {
        "life.naive": "One u32 per cell (naive)",
        "life.bits32.256x4": "32 cells per word",
        "life.bits64": "64 cells per word",
        "life.ghost.t8.w30.r256": "First blocking design",
        "life.column.t16.g256.k3": "Column blocking (winner)",
    }
    kernel_series = []
    rows = tag("kernels") + tag("blocking")
    for k, n in names.items():
        s = [(r["cells"], r["cells_per_second"], f'{r["width"]}x{r["height"]}: {si(r["cells_per_second"], "cell/s")}') for r in rows if r["id"] == k]
        kernel_series.append((n, s))
    cpu4 = [r for r in cpu if r["id"] == "cpu.avx2" and r["threads"] == 4]
    kernel_series.append(("CPU, AVX2, 4 threads", [(r["cells"], r["cells_per_second"], f'{r["width"]}x{r["height"]}: {si(r["cells_per_second"], "cell/s")}') for r in cpu4]))
    c_kernels = line_chart(
        "Conway's Life: which kernel is fastest, and at what size",
        "Cell updates per second on the RX 7900 XT. Higher is better. Log scales.",
        kernel_series, "Cells in the grid", "Cell updates per second", id_="kernels",
    )

    # Ladder
    ladder = sorted(tag("ladder"), key=lambda r: r["cells"])
    c_gens = line_chart(
        "Conway's Life: generations per second at every grid size",
        "With the winning kernel. Each step right is a bigger world; the curve is a straight line once throughput is flat.",
        [("Generations per second", [(r["cells"], r["generations_per_second"], f'{cells_name(r["cells"])} cells: {r["generations_per_second"]:,.1f} generations/s') for r in ladder])],
        "Cells in the grid", "Generations per second", id_="generations", direct=False,
    )
    c_flat = line_chart(
        "Conway's Life: throughput stays flat up to the VRAM limit",
        "Same runs as above, as cell updates per second. Small grids lose to dispatch overhead, big ones stay near 5 trillion.",
        [("Cell updates per second", [(r["cells"], r["cells_per_second"], f'{cells_name(r["cells"])} cells: {si(r["cells_per_second"], "cell/s")}') for r in ladder])],
        "Cells in the grid", "Cell updates per second", id_="throughput", direct=False,
    )
    # Blocking
    blocking = defaultdict(list)
    for r in tag("blocking"):
        t = int(r["id"].split(".t")[1].split(".")[0])
        blocking[r["cells"]].append((t, r["cells_per_second"], f't={t}: {si(r["cells_per_second"], "cell/s")}'))
    c_blocking = line_chart(
        "How many generations to run per trip to VRAM",
        "Temporal blocking at three grid sizes. More generations per dispatch means less memory traffic, until compute becomes the limit.",
        [(f"{cells_name(c)} cells", v) for c, v in sorted(blocking.items())],
        "Generations per dispatch", "Cell updates per second", x_fmt=lambda v: f"{v:.0f}", logx=False, logy=False, id_="blocking",
    )

    # Comparison with published work
    top = max((r["cells_per_second"] for r in ladder), default=0)
    cmp_rows = [(n, v, f"{si(v, 'cell/s')}, published", 7) for n, v, _ in PUBLISHED]
    if top:
        cmp_rows.append(("This repo: RX 7900 XT, wgpu", top, f"{si(top, 'cell/s')}, measured here", 1))
    if browser:
        chrome = [r for r in browser if "life" in r["id"] and "Chrome" in r.get("browser", "") and not r.get("browser_adapter", {}).get("fallback")]
        if chrome:
            fb = max(r["cells_per_second"] for r in chrome)
            cmp_rows.append(("This repo: same GPU, in Chrome", fb, f"{si(fb, 'cell/s')}, measured here", 1))
    if cpu:
        cb = max(r["cells_per_second"] for r in cpu)
        cmp_rows.append(("This repo: 7800X3D, 4 of 16 threads", cb, f"{si(cb, 'cell/s')}, measured here", 3))
    cmp_rows.sort(key=lambda r: -r[1])
    c_compare = bar_chart(
        "Conway's Life compared with published results",
        "Cell updates per second. Published numbers are the authors' own, on their hardware and grid sizes. Log scale.",
        cmp_rows, id_="compare",
    )

    # API
    api_tables = ""
    by = defaultdict(dict)
    for r in api:
        layer = {"api-wgpu-safe": "wgpu, bounds checks on", "api-wgpu-trusted": "wgpu, checks off", "raw-vulkan": "raw Vulkan"}.get(r.get("tag"))
        if layer:
            by[(r["id"], r["width"], r["height"])][layer] = r
    for r in browser:
        if r.get("browser_adapter", {}).get("fallback"):
            continue
        by[(r["id"], r["width"], r["height"])]["Firefox WebGPU" if "Firefox" in r.get("browser", "") else "Chrome WebGPU"] = r
    layers = ["raw Vulkan", "wgpu, checks off", "wgpu, bounds checks on", "Firefox WebGPU", "Chrome WebGPU"]
    api_series = defaultdict(list)
    rows_out = []
    for (kid, w, h), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        base = d.get("raw Vulkan")
        if not base:
            continue
        cells_row = [f"<code>{esc(kid.replace('life.', ''))}</code>", f"{w}x{h}"]
        for l in layers:
            if l in d:
                v = d[l]["cells_per_second"]
                pct = (v / base["cells_per_second"] - 1) * 100
                cells_row.append(f"{si(v)}<br><span class='sub'>{pct:+.1f}%</span>" if l != "raw Vulkan" else si(v))
                if l != "raw Vulkan":
                    api_series[(kid, l)].append((w * h, max(v / base["cells_per_second"] * 100, 0.1), f"{w}x{h}: {pct:+.1f}% against raw Vulkan"))
            else:
                cells_row.append("")
        rows_out.append(cells_row)
    api_table = table(["Kernel", "Grid"] + layers, rows_out, numeric=(2, 3, 4, 5, 6))
    series_api = [(f"{l}, {'plain' if 'bits32' in k else 'blocking'}", v) for (k, l), v in sorted(api_series.items(), key=lambda kv: (kv[0][0], layers.index(kv[0][1]))) if "256x4" not in k and "checks off" not in l]
    c_api = line_chart(
        "What the layers cost, against raw Vulkan",
        "100% means the same speed as hand-written Vulkan running the same shader. Small grids run many tiny dispatches, so per-dispatch overhead shows there.",
        series_api[:6], "Cells in the grid", "% of raw Vulkan speed", y_fmt=lambda v: f"{v:.0f}%", logy=False, id_="api",
    )

    # Automata
    labels = {
        "life": "Conway's Life", "brain": "Brian's Brain", "wireworld": "Wireworld", "rule30": "Rule 30", "rule110": "Rule 110",
        "cyclic": "Cyclic CA", "hodgepodge": "Hodgepodge", "ants": "Langton's ants", "forest": "Forest fire", "sandpile": "Sandpile",
        "sand": "Falling sand", "grayscott": "Gray-Scott", "lenia": "Lenia (FFT)", "smoothlife": "SmoothLife (FFT)", "mnca": "MNCA (FFT)", "physarum": "Physarum (agents)",
    }
    auto_rows = tag("automata")
    life_rows = [r for r in ladder if r["id"].startswith("life.column")]
    auto_rows = auto_rows + [dict(r, id="life") for r in life_rows]
    bars = []
    table_rows = []
    for k, label in labels.items():
        rs = [r for r in auto_rows if r["id"] == k]
        if not rs:
            continue
        b = max(rs, key=lambda r: r["cells_per_second"])
        slot = 1 if k in ("life", "brain", "wireworld", "rule30", "rule110") else 2 if k in ("lenia", "smoothlife", "mnca") else 3 if k == "physarum" else 4
        bars.append((label, b["cells_per_second"], f'{label} at {b["width"]}x{b["height"]}: {si(b["cells_per_second"], "cell/s")}', slot))
        big = max(rs, key=lambda r: r["cells"])
        mem = big["memory_bytes"] * 8 / big["cells"] if k not in ("rule30", "rule110") else 1
        one_m = min(rs, key=lambda r: abs(math.log(max(r["cells"], 1) / 1e6)))
        table_rows.append([label, f"{mem:.0f}", f'{one_m["width"]}x{one_m["height"]}', f'{one_m["generations_per_second"]:,.0f}',
                           f'{big["width"]}x{big["height"]}', f'{big["generations_per_second"]:,.1f}', si(b["cells_per_second"])])
    bars.sort(key=lambda r: -r[1])
    c_auto = bar_chart(
        "Every automaton at its fastest",
        "Peak cell updates per second, or agent updates for Physarum. Bit-sliced in blue, one word per cell in yellow, FFT in orange, agents in green.",
        bars, id_="automata", label_w=170,
    )
    # Colours: fix the legend meaning by mapping slots above.
    auto_series = []
    for k in ["brain", "cyclic", "grayscott", "lenia", "physarum"]:
        rs = [r for r in auto_rows if r["id"] == k]
        auto_series.append((labels[k], [(r["cells"], r["generations_per_second"], f'{r["width"]}x{r["height"]}: {r["generations_per_second"]:,.1f} ticks/s') for r in rs]))
    c_auto_scale = line_chart(
        "Ticks per second as the world grows",
        "One automaton from each family. The gap between them is the cost of richer cells: bits, then words, floats, FFTs, agents.",
        auto_series, "Cells in the grid", "Ticks per second", id_="automata_scale",
    )

    # Brains
    brain_series = defaultdict(list)
    brain_rows = []
    for r in brains:
        h = int(r["id"].split(".h")[1].split(".")[0])
        bits = int(r["id"].split(".b")[1])
        blocked = r["id"].startswith("brainsb")
        params = h * h + 8 * h
        agents = r["width"]
        steps = r["steps_per_second"]
        if agents >= 65536:
            brain_series[(blocked, bits, agents)].append((params, steps, f'{params} params, {agents:,} agents: {steps:,.1f} steps/s, {si(r["cells_per_second"] * params)} multiply-adds/s'))
        brain_rows.append((blocked, params, bits, agents, steps, r["memory_bytes"], r["cells_per_second"] * params))
    keys = [(True, 8, 1048576), (True, 32, 1048576), (True, 8, 65536), (False, 32, 1048576)]
    c_brains = line_chart(
        "What a brain costs: one RNN per agent, all with different weights",
        "Steps per second for the whole population on the RX 7900 XT, in WGSL that also runs in a browser. No simulation around it, so this is the ceiling.",
        [(f"{'1M' if a > 1e6 else '65k'} agents, {'f32' if b == 32 else '8-bit'}{'' if bl else ', naive kernel'}", sorted(brain_series[(bl, b, a)])) for bl, b, a in keys if brain_series.get((bl, b, a))],
        "Parameters per brain", "Population steps per second", id_="brains",
    )
    brain_table = table(
        ["Kernel", "Params", "Weights", "Agents", "Steps/s", "Memory", "Multiply-adds/s"],
        [["blocked" if bl else "naive", f"{p:,}", "f32" if b == 32 else "8-bit", f"{a:,}", f"{s_:,.1f}", si(m, "B"), si(mac)] for bl, p, b, a, s_, m, mac in sorted(brain_rows, key=lambda x: (not x[0], x[1], x[2], x[3]))],
        numeric=(1, 3, 4, 5, 6),
    )

    # CPU
    cpu_names = {"cpu.bytes": "One byte per cell", "cpu.swar64": "64-bit scalar", "cpu.avx2": "AVX2", "cpu.avx512": "AVX-512"}
    cpu_s = []
    for k, n in cpu_names.items():
        for t in sorted({r["threads"] for r in cpu}):
            rs = [r for r in cpu if r["id"] == k and r["threads"] == t and r["cells"] > 100]
            if rs and k in ("cpu.avx2", "cpu.avx512", "cpu.swar64"):
                cpu_s.append((f"{n}, {t} thread{'s' if t > 1 else ''}", [(r["cells"], r["cells_per_second"], f'{r["width"]}x{r["height"]}: {si(r["cells_per_second"], "cell/s")}') for r in rs]))
    c_cpu = line_chart(
        "Conway's Life on the Ryzen 7 7800X3D",
        "The agent sandbox caps this at 4 of 16 threads, so the machine can do more than shown. Once the grid leaves cache, memory sets the limit.",
        cpu_s[:6], "Cells in the grid", "Cell updates per second", id_="cpu",
    )
    hash_table = table(
        ["Pattern", "Generation", "Universe side", "Live cells", "Time", "Nodes"],
        [[esc(r["pattern"].replace("_", " ")), r["generation"], f'2^{r["universe_side_log2"]}', f'{int(r["population"]):,}', f'{r["seconds"] * 1000:.1f} ms', f'{r["nodes"]:,}'] for r in hash_rows],
        numeric=(3, 4, 5),
    )

    vid_html = "".join(
        f'<figure><video controls loop muted preload="metadata" src="{esc(v["web"])}" poster="{esc(v.get("poster",""))}"></video>'
        f'<figcaption><b>{esc(v["name"])}</b>. {esc(v["caption"])}</figcaption></figure>'
        for v in videos
    )

    def life_at(target):
        if not ladder:
            return None
        return min(ladder, key=lambda r: abs(math.log(r["cells"] / target)))

    hero = []
    for target, word in [(1e6, "a million"), (1e9, "a billion")]:
        r = life_at(target)
        if r:
            hero.append(f'<div class="hero"><b>{r["generations_per_second"]:,.0f} /s</b><span>generations of Life on {word} cells ({r["width"]}x{r["height"]})</span></div>')
    if ladder:
        r = ladder[-1]
        hero.append(f'<div class="hero"><b>{r["generations_per_second"]:,.0f} /s</b><span>generations on {cells_name(r["cells"])} cells, the biggest grid measured</span></div>')
        hero.append(f'<div class="hero"><b>{si(top, "cell/s")}</b><span>peak cell updates per second</span></div>')
    rv = [r for r in api if r["id"] == "life.bits32.256x1" and r.get("tag") == "api-wgpu-safe" and r["width"] == 16384]
    for name in ("Chrome", "Firefox"):
        bv = [r for r in browser if r["id"] == "life.bits32.256x1" and r["width"] == 16384 and not r.get("browser_adapter", {}).get("fallback") and name in r.get("browser", "")]
        if bv and rv:
            diff = (bv[-1]["cells_per_second"] / rv[0]["cells_per_second"] - 1) * 100
            hero.append(f'<div class="hero"><b>{diff:+.1f}%</b><span>speed in {name} against native wgpu, plain kernel at 16384x16384</span></div>')

    prose = (ROOT / "report" / "prose.html").read_text() if (ROOT / "report" / "prose.html").exists() else ""
    sections = {
        "HEROES": f'<div class="heroes">{"".join(hero)}</div>',
        "VIDEOS": f'<div class="videos">{vid_html}</div>',
        "C_KERNELS": c_kernels, "C_GENS": c_gens, "C_FLAT": c_flat, "C_BLOCKING": c_blocking, "C_COMPARE": c_compare,
        "API_TABLE": api_table, "C_API": c_api, "C_AUTO": c_auto, "C_AUTO_SCALE": c_auto_scale,
        "AUTO_TABLE": table(["Automaton", "Bits per cell", "Near 1M: grid", "Ticks/s", "Largest: grid", "Ticks/s", "Peak updates/s"], table_rows, numeric=(1, 3, 5, 6)),
        "C_BRAINS": c_brains, "BRAIN_TABLE": brain_table, "C_CPU": c_cpu, "HASH_TABLE": hash_table,
        "LLVM": table(["Automaton", "Grid", "Ticks/s", "Updates/s"], [[r["id"], f'{r["width"]}x{r["height"]}', f'{r["generations_per_second"]:,.1f}', si(r["cells_per_second"])] for r in llvm], numeric=(2, 3)),
        "IGPU": table(["Automaton", "Grid", "Ticks/s", "Updates/s"], [[r["id"], f'{r["width"]}x{r["height"]}', f'{r["generations_per_second"]:,.1f}', si(r["cells_per_second"])] for r in igpu], numeric=(2, 3)),
    }
    for key, val in sections.items():
        prose = prose.replace("{{" + key + "}}", val)
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(page("auto: very fast cellular automata", prose))
    (OUT / "charts").mkdir(exist_ok=True)
    for key, block in CHARTS.items():
        (OUT / "charts" / f"{key}.html").write_text(page(key, block))
    print(f"wrote {OUT / 'index.html'}, {len(CHARTS)} charts, {len(gpu)} GPU results")


if __name__ == "__main__":
    sys.exit(main())
