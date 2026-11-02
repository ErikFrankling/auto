#!/usr/bin/env python3
"""Builds report/life/screen.html, the results of the artificial-life screen, and its thumbnails.

Standard library only, apart from ffmpeg for the JPEGs. Every number on the page is worked out
here from the data at build time, so a rerun picks up whatever has landed and says what hasn't.
The two exceptions, measured once from saved worlds and too slow to redo here, are in RIM below.

    source /home/erikf/auto/target/nix-env-*.sh; python3 scripts/make_screen_report.py

Where it reads from:
    LIFE_RESEARCH      research/life-gpu with the live results (default: /home/erikf/auto's, else this checkout's)
    LIFE_SCREEN_RUNS   per-world detail of the screen (default /tmp/lg/screen)
    LIFE_LONG_RUNS     per-world detail of long runs and knockouts (default /tmp/lg/long)
    LIFE_MAPS          rendered maps (default /tmp/lg/maps); batches with a saved world and no maps get rendered
"""
import html
import json
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_live = Path("/home/erikf/auto/research/life-gpu")
LIVE = Path(os.environ.get("LIFE_RESEARCH", _live if (_live / "screen/results.jsonl").exists() else ROOT / "research/life-gpu"))
os.environ.setdefault("LIFE_SCREEN_OUT", str(LIVE / "screen"))
sys.path.insert(0, str(ROOT / "scripts"))
import life_world_cards as cards  # noqa: E402

RUNS = cards.RUNS
LONG_RUNS = Path(os.environ.get("LIFE_LONG_RUNS", "/tmp/lg/long"))
MAPS = Path(os.environ.get("LIFE_MAPS", "/tmp/lg/maps"))
REPORT_DATA = Path("/tmp/lg/report-data")
ANALYSE = Path("/home/erikf/auto/target/release/life-analyse")
OUT = ROOT / "report/life"
MEDIA = OUT / "media/screen"
DESIGN_URL = "https://github.com/ErikFrankling/auto/blob/main/research/life-soft/DESIGN.md"
MEDIA_BUDGET = 60_000_000
CLOSE_KIN_MIN_BITERS = 300
SEASON_DEFAULT = 8000
SAMPLE_TICKS = 30000

# Mineral at the rim against inside, from saved worlds, per set. Medians per world from reading
# fd.y in world.bin (outer 8 squares against the rest). Rechecked 2026-11-02.
RIM = {"b0-0004": 11, "b0-0009": 19, "b0-0011": 15, "b0-0013": 35, "b0-0020": 33}
# Open squares touching natural rock against other inner squares, in the two worlds measured.
ROCK = {"b0-0009-t001": 0.194 / 0.018, "b0-0020-t010": 0.033 / 0.005}

# Checked with the dataviz validator on the page colour: blue and orange pass every test, and the neutral
# clears the separation floors. It fails only the chroma floor, on purpose, since it means "neither".
BLUE, ORANGE, NEUTRAL = "#3987e5", "#d95926", "#9a9da8"


def esc(s):
    return html.escape(str(s), quote=True)


# ---- Formatting ------------------------------------------------------------


def num(v):
    return "n/a" if v is None else f"{v:,.0f}"


def short_num(v):
    a = abs(v)
    if a >= 1000:
        return f"{v / 1000:.1f}k".replace(".0k", "k")
    return f"{v:g}"


def pc(v, digits=0):
    return "n/a" if v is None else f"{100 * v:.{digits}f}%"


def pc_fine(v):
    return pc(v, 1) if v is not None and v < 0.1 else pc(v)


def trim(v, digits=1):
    if v is None:
        return "n/a"
    s = f"{v:.{digits}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def signed(v, digits=2):
    return f"{v:+.{digits}f}".replace("-", "−")


def words_number(n):
    names = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
    return names[n] if 0 <= n < len(names) else str(n)


def cap(s):
    return s[0].upper() + s[1:] if s else s


def world_tag(run):
    """b0-0001-t000 reads better with a space on small labels."""
    return re.sub(r"-t(\d+)$", r" t\1", run)


# ---- Reading the data ------------------------------------------------------


def read_jsonl(path):
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def load():
    # Read line by line rather than with cards.load_rows, since a batch may be landing right now.
    rows = read_jsonl(LIVE / "screen/results.jsonl")
    long_rows = read_jsonl(LIVE / "long/results.jsonl")
    tiles = {}
    for r in rows + long_rows:
        if r["batch"] not in tiles:
            base = LONG_RUNS if r["batch"].startswith(("long-", "ko-")) else RUNS
            tiles[r["batch"]] = cards.tiles_of(base / r["batch"] / "analyse.json")
        r["flows"] = ((tiles[r["batch"]].get(r["index"]) or {}).get("economy") or {}).get("flows")
    kin = read_jsonl(LIVE / "screen/kin.jsonl")
    kin_long = []
    for f in sorted(REPORT_DATA.glob("kin-long-*.jsonl")):
        kin_long += read_jsonl(f)
    reads = {p.stem: p.read_text() for p in sorted((LIVE / "screen/reads").glob("*.md"))}
    return rows, long_rows, kin, kin_long, reads


class Lazy:
    """History files are big, so read them only when a sentence needs one."""

    def __init__(self):
        self.cache = {}

    def history(self, batch, index):
        if batch not in self.cache:
            base = LONG_RUNS if batch.startswith(("long-", "ko-")) else RUNS
            self.cache[batch] = cards.tiles_of(base / batch / "history.json")
        return self.cache[batch].get(index) or {}


HISTORY = Lazy()


def batch_state(directory, done):
    if done:
        return "done"
    return "running" if (directory / "log.jsonl").exists() else "waiting"


def clean_specs():
    """The one-rule-at-a-time set: its spec files exist before its results do."""
    specs = {}
    for p in sorted(RUNS.glob("b1-*/spec.json")):
        try:
            specs[p.parent.name] = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return specs


def m(row, key):
    return cards.MEASURES[key](row)


def by_batch(rows):
    out = {}
    for r in rows:
        out.setdefault(r["batch"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: r["index"])
    return out


# ---- Statistics ------------------------------------------------------------


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2
        i = j + 1
    return out


def corr(x, y):
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx * sy else float("nan")


def rank_corr(x, y):
    return corr(ranks(x), ranks(y))


def med(xs):
    xs = [v for v in xs if v is not None]
    return statistics.median(xs) if xs else None


def compare_sets(sets, control_alive):
    """Each set against the controls on every measure, then Benjamini-Hochberg over all of those tests."""
    tested = []
    for name, rs in sets.items():
        living = cards.alive(rs)
        for measure in cards.MEASURES:
            a, b = cards.values(living, measure), cards.values(control_alive, measure)
            p, effect = cards.rank_sum_p(a, b)
            tested.append((p, name, measure, effect, len(a)))
    tested.sort(key=lambda t: t[0])
    n = len(tested)
    keep = 0
    for i, t in enumerate(tested):
        if t[0] <= 0.05 * (i + 1) / max(n, 1):
            keep = i + 1
    standing = {(t[1], t[2]): ("higher" if t[3] > 0.5 else "lower", t[0]) for t in tested[:keep]}
    return standing, n, keep


# ---- Plain words for rules -------------------------------------------------

RULE_WORDS = [
    ("mass_on", "heavy cells move slower and can shove lighter ones"),
    ("armour_on", "cells can build armour against bites"),
    ("pack_on", "bites do little until a body is wounded"),
    ("gang_on", "biters on one target add their force"),
    ("gang_sense", "cells sense bites nearby"),
    ("coat_on", "a bite needs a key that matches the target's coat"),
    ("coat_sense", "cells can see coats"),
    ("waste_on", "eating leaves harmful waste"),
    ("plants_on", "food grows best at middling density"),
    ("bloom_share", "some sun lands as rich piles"),
    ("mineral_on", "births need mineral from the ground"),
    ("weather_sun_on", "the sun map drifts"),
    ("weather_fire_on", "fires burn dense food"),
    ("skill_on", "practice makes a cell better at one food"),
    ("skill_sense", "cells can feel their practice"),
    ("biomes_on", "16 rooms of four climates"),
    ("biome_feel", "cells feel their room"),
    ("walls_on", "rock walls between rooms"),
    ("roads_on", "walking wears easier paths"),
    ("birth_choice", "cells choose when to give birth"),
    ("mate_pay", "a mate shares the cost of a birth"),
    ("compat", "genomes too far apart can't mix"),
    ("age_metabolic", "cells that spend less live longer"),
]
RULE_PARTS = re.compile(
    r"^(mass_|armour_|pack_|gang_|coat_|waste_|plants_|bloom_|mineral_|rain_burst|weather_|skill_|biomes_|biome_|walls_|wall_|"
    r"roads_|road_|track_|birth_|womb_cap|mate_pay|compat|age_)"
)


def rule_words(physics):
    on = [text for key, text in RULE_WORDS if physics.get(key)]
    other = [k for k in physics if not RULE_PARTS.match(k)]
    return on, other


# ---- Reads -----------------------------------------------------------------


def inline_md(text, batch, known):
    s = esc(text)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)

    def world_link(match):
        run = f"{batch}-{match.group(0)}"
        return f'<button type="button" class="wlink" data-run="{run}">{match.group(0)}</button>' if run in known else match.group(0)

    return re.sub(r"\bt\d{3}\b", world_link, s)


def read_html(text, batch, known):
    out, para = [], []

    def flush():
        if para:
            out.append("<p>" + inline_md(" ".join(para), batch, known) + "</p>")
            para.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            head = line.lstrip("# ").strip()
            if not head.startswith(batch):
                out.append(f"<h4>{inline_md(head, batch, known)}</h4>")
            continue
        if re.match(r"^(\d+\.|-)\s", line):
            flush()
            out.append('<p class="item">' + inline_md(line, batch, known) + "</p>")
            continue
        para.append(line)
    flush()
    return "".join(out)


def first_candidate(text):
    """World and first sentence of the first candidate in a read, for sets with no written note yet."""
    part = re.split(r"candidates", text, flags=re.I)
    if len(part) < 2:
        return None
    match = re.search(r"\bt(\d{3})\b\.?\*{0,2}\s*(.+)", part[1])
    if not match:
        return None
    body = re.sub(r"\*\*|^(Seen|Saw):\s*", "", match.group(2).strip())
    sentence = re.split(r"(?<=[.!?])\s+", body.lstrip(" :.*"))[0]
    return (int(match.group(1)), cap(sentence)) if len(sentence) >= 40 else None


# ---- Maps and thumbnails ---------------------------------------------------


def render_maps(batch):
    """Maps for a finished batch that has none yet. Only writes under MAPS."""
    d = MAPS / batch
    if list(d.glob("t*.png")):
        return
    is_long = batch.startswith(("long-", "ko-"))
    world = (LONG_RUNS if is_long else RUNS) / batch / "world.bin"
    if not world.exists():
        return
    print(f"rendering maps for {batch}")
    d.mkdir(parents=True, exist_ok=True)
    try:
        if is_long:
            subprocess.run(["nice", "-n", "15", str(ANALYSE), batch, str(world), "/dev/null", "--tile", "256", "--maps", str(d), "--maps-only"],
                           check=True, stderr=subprocess.DEVNULL)
            for ppm in d.glob("t*.ppm"):
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(ppm), "-vf", "scale=iw*2:ih*2:flags=neighbor", str(ppm.with_suffix(".png"))], check=True)
                ppm.unlink()
        else:
            subprocess.run(["/tmp/lg/maps.sh", batch], check=True)
    except (OSError, subprocess.CalledProcessError) as why:
        print(f"  no maps for {batch}: {why}", file=sys.stderr)


def source_map(run):
    batch, tile = run.rsplit("-t", 1)
    return MAPS / batch / f"t{tile}.png"


def jpeg(src, dst, filters, q, pix):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if filters:
        cmd += ["-vf", filters]
    cmd += ["-q:v", str(q), "-pix_fmt", pix, str(dst)]
    subprocess.run(cmd, check=True)


def build_media(runs):
    """A native-size JPEG for cards and a doubled one for the enlarged view, under the size budget."""
    MEDIA.mkdir(parents=True, exist_ok=True)
    manifest_path = MEDIA / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError):
        manifest = {}
    have = {r for r in runs if source_map(r).exists()}
    # The maps are doubled pixel for pixel, so 4:2:0 at double size keeps each cell's colour. At
    # native size it would smear single-pixel cells, so thumbnails stay 4:4:4.
    for q_large in (4, 6, 8, 11, 15):
        for run in sorted(have):
            src = source_map(run)
            thumb, large = MEDIA / f"{run}.jpg", MEDIA / f"{run}-l.jpg"
            stamp = src.stat().st_mtime
            seen = manifest.get(run, {})
            if not thumb.exists() or seen.get("src") != stamp:
                jpeg(src, thumb, "scale=iw/2:ih/2:flags=neighbor", 7, "yuvj444p")
            if not large.exists() or seen.get("src") != stamp or seen.get("q") != q_large:
                jpeg(src, large, "", q_large, "yuvj420p")
            manifest[run] = {"src": stamp, "q": q_large}
        total = sum((MEDIA / f"{r}.jpg").stat().st_size + (MEDIA / f"{r}-l.jpg").stat().st_size for r in have)
        if total <= MEDIA_BUDGET:
            break
        print(f"media at {total / 1e6:.0f} MB with quality {q_large}, stepping down")
    # Anything left from an earlier build is out of date.
    for p in MEDIA.glob("*.jpg"):
        if p.name.removesuffix(".jpg").removesuffix("-l") not in have:
            p.unlink()
    manifest = {k: v for k, v in manifest.items() if k in have}
    manifest_path.write_text(json.dumps(manifest))
    print(f"media: {len(have)} worlds, {total / 1e6:.1f} MB, enlarged quality {q_large}")
    return have, total


# ---- Chart pieces ----------------------------------------------------------


def nice_ticks(hi, count=4):
    if hi <= 0:
        return [0, 1]
    step = 10 ** math.floor(math.log10(hi / count))
    for mult in (1, 2, 2.5, 5, 10):
        if hi / (step * mult) <= count:
            step *= mult
            break
    n = int(math.ceil(hi / step - 1e-9))
    return [step * k for k in range(n + 1)]


def swarm(xs, radius, limit):
    """Vertical offsets so that dots at nearby x don't sit on top of each other, squeezed to fit."""
    placed, out = [], [0.0] * len(xs)
    gap = radius * 2 + 0.6
    for i in sorted(range(len(xs)), key=lambda k: xs[k]):
        x = xs[i]
        for level in range(0, 200):
            dy = ((level + 1) // 2) * gap * (1 if level % 2 else -1)
            if all((x - px) ** 2 + (dy - py) ** 2 >= gap * gap for px, py in placed):
                break
        placed.append((x, dy))
        out[i] = dy
    biggest = max((abs(v) for v in out), default=0)
    if biggest > limit:
        out = [v * limit / biggest for v in out]
    return out


def outliers(values, most=3):
    """Those beyond 1.5 times the middle half, furthest first. If none are, the highest one."""
    s = sorted(v for _, v in values)
    n = len(s)
    q1, q3 = s[n // 4], s[(3 * n) // 4]
    fence = 1.5 * (q3 - q1)
    picked = [(k, v) for k, v in values if v > q3 + fence or v < q1 - fence]
    picked.sort(key=lambda kv: -abs(kv[1] - statistics.median(s)))
    return picked[:most] or [max(values, key=lambda kv: kv[1])]


def strip_chart(title, note, stat, points, fmt):
    """points: (run, value). One dot per world on a single number line."""
    W, H, L, R = 440, 158, 14, 14
    top_labels, mid, axis = 34, 88, 134
    points = [(run, v) for run, v in points if v is not None]
    ticks = nice_ticks(max(v for _, v in points) or 1)
    hi = ticks[-1]
    fx = lambda v: L + v / hi * (W - L - R)
    xs = [fx(v) for _, v in points]
    dys = swarm(xs, 4.5, axis - mid - 8)
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}: {esc(stat)}">']
    o.append(f'<line class="base" x1="{L}" x2="{W - R}" y1="{axis}" y2="{axis}"/>')
    for t in ticks:
        o.append(f'<line class="tick" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{axis}" y2="{axis + 5}"/><text class="ax" x="{fx(t):.1f}" y="{axis + 19}" text-anchor="middle">{esc(fmt(t))}</text>')
    middle = statistics.median(v for _, v in points)
    o.append(f'<line class="median" x1="{fx(middle):.1f}" x2="{fx(middle):.1f}" y1="{mid - 38}" y2="{axis}"/>')
    for (run, v), x, dy in zip(points, xs, dys):
        o.append(f'<circle class="pt" cx="{x:.1f}" cy="{mid + dy:.1f}" r="4.5" fill="{BLUE}" data-run="{esc(run)}" data-tip="{esc(fmt(v))}|{esc(run)}"/>')
    # Labels above, on a second row if two would touch, with a thin line down to the dot.
    used = [[], []]
    centre = {run: (x, mid + dy) for (run, _), x, dy in zip(points, xs, dys)}
    for run, v in sorted(outliers(points), key=lambda kv: fx(kv[1])):
        x, y = centre[run]
        width = 92
        left = min(max(x - width / 2, L), W - R - width)
        row = 0 if all(abs(left - u) > width + 4 for u in used[0]) else 1
        used[row].append(left)
        ly = 14 + row * 14
        o.append(f'<line class="leader" x1="{x:.1f}" x2="{x:.1f}" y1="{ly + 3}" y2="{y - 5:.1f}"/><text class="lab" x="{left + width / 2:.1f}" y="{ly}" text-anchor="middle">{esc(world_tag(run))}</text>')
    o.append("</svg>")
    return f'<figure class="chart"><figcaption><b>{esc(title)}</b><span>{esc(note)}</span><span class="stat">{esc(stat)}</span></figcaption>{"".join(o)}</figure>'


def legend(items):
    return '<div class="legend">' + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items) + "</div>"


def scatter_kin(screen, long_points):
    W, H, L, R, T, B = 440, 440, 62, 12, 12, 44
    fx = lambda v: L + v * (W - L - R)
    fy = lambda v: H - B - v * (H - T - B)
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Share of bite energy taken from close kin against share of close kin next to the biter, one dot per world">']
    for t in (0, 0.25, 0.5, 0.75, 1):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{T}" y2="{H - B}"/>')
        o.append(f'<text class="ax" x="{L - 7}" y="{fy(t) + 4:.1f}" text-anchor="end">{t * 100:.0f}%</text><text class="ax" x="{fx(t):.1f}" y="{H - B + 18}" text-anchor="middle">{t * 100:.0f}%</text>')
    o.append(f'<line class="identity" x1="{fx(0):.1f}" y1="{fy(0):.1f}" x2="{fx(1):.1f}" y2="{fy(1):.1f}"/>')
    o.append(f'<line class="leader" x1="{fx(0.20):.1f}" y1="{fy(0.20):.1f}" x2="{fx(0.20):.1f}" y2="{fy(0.09):.1f}"/><text class="lab" x="{fx(0.20) + 5:.1f}" y="{fy(0.09) + 4:.1f}">y = x: bites land by who is near</text>')
    o.append(f'<text class="ax" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">Close kin among the biter\'s neighbours</text>')
    o.append(f'<text class="ax" transform="translate(12,{(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Bite energy taken from close kin</text>')
    for k in screen:
        tip = f'{k["on_kin"] * 100:.0f}% bitten, {k["kin_near"] * 100:.0f}% nearby|{k["run"]}, 120,000 ticks'
        o.append(f'<circle class="pt" cx="{fx(k["kin_near"]):.1f}" cy="{fy(k["on_kin"]):.1f}" r="4" fill="{BLUE}" data-run="{esc(k["run"])}" data-tip="{esc(tip)}"/>')
    for k in long_points:
        tip = f'{k["on_kin"] * 100:.0f}% bitten, {k["kin_near"] * 100:.0f}% nearby|{k["run"]}, 600,000 ticks'
        o.append(f'<circle class="pt" cx="{fx(k["kin_near"]):.1f}" cy="{fy(k["on_kin"]):.1f}" r="4" fill="{ORANGE}" data-run="{esc(k["run"])}" data-tip="{esc(tip)}"/>')
    far = sorted(screen, key=lambda k: -abs(k["on_kin"] - k["kin_near"]))[:2]
    for k in far:
        x, y = fx(k["kin_near"]), fy(k["on_kin"])
        if k["on_kin"] > k["kin_near"]:
            o.append(f'<text class="lab" x="{x + 7:.1f}" y="{y + 4:.1f}">{esc(world_tag(k["run"]))}</text>')
        else:
            # Below the line there's empty space, so the label goes there.
            o.append(f'<line class="leader" x1="{x:.1f}" y1="{y + 5:.1f}" x2="{x:.1f}" y2="{y + 84:.1f}"/><text class="lab" x="{x:.1f}" y="{y + 98:.1f}" text-anchor="{"end" if x > W * 0.8 else "middle"}">{esc(world_tag(k["run"]))}</text>')
    o.append("</svg>")
    return "".join(o)


def dumbbell(rows):
    """rows: dict with world, alone, off, biting. Sorted as given."""
    h_row, top = 24, 26
    W, L, R = 440, 90, 56
    H = top + len(rows) * h_row + 28
    hi = nice_ticks(max(max(r["alone"], r["off"]) for r in rows))
    fx = lambda v: L + v / hi[-1] * (W - L - R)
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Cells with biting off against left alone, one row per world">']
    o.append(f'<text class="ax" x="4" y="16">World, energy from biting</text><text class="ax" x="{W - 4}" y="16" text-anchor="end">Change</text>')
    for t in hi:
        o.append(f'<line class="grid" x1="{fx(t):.1f}" x2="{fx(t):.1f}" y1="{top}" y2="{top + len(rows) * h_row}"/><text class="ax" x="{fx(t):.1f}" y="{top + len(rows) * h_row + 18}" text-anchor="middle">{esc(short_num(t))}</text>')
    for i, r in enumerate(rows):
        y = top + i * h_row + h_row / 2
        change = r["off"] / r["alone"] - 1
        tip = f'{change * 100:+.0f}% with biting off|{r["run"]}: {r["alone"]:,} left alone, {r["off"]:,} with biting off'
        o.append(f'<text class="lab" x="4" y="{y + 4:.1f}">t{r["index"]:03d}</text><text class="ax" x="{L - 10}" y="{y + 4:.1f}" text-anchor="end">{pc(r["biting"])}</text>')
        o.append(f'<line class="link" x1="{fx(r["alone"]):.1f}" x2="{fx(r["off"]):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>')
        o.append(f'<circle class="pt" cx="{fx(r["alone"]):.1f}" cy="{y:.1f}" r="4.5" fill="{BLUE}" data-run="{esc(r["run"])}" data-tip="{esc(tip)}"/>')
        o.append(f'<circle class="pt" cx="{fx(r["off"]):.1f}" cy="{y:.1f}" r="4.5" fill="{ORANGE}" data-run="{esc(r["run"])}" data-tip="{esc(tip)}"/>')
        shown = "0%" if abs(change) < 0.005 else f"{change * 100:+.0f}%".replace("-", "\u2212")
        o.append(f'<text class="lab" x="{W - 4}" y="{y + 4:.1f}" text-anchor="end">{shown}</text>')
    o.append("</svg>")
    return "".join(o)


def slope_chart(title, stat, pairs, fmt):
    """pairs: (run, before, after, colour). Each line is one world."""
    W, H, L, R, T, B = 300, 214, 42, 20, 12, 30
    ticks = nice_ticks(max(max(a, b) for _, a, b, _ in pairs) or 1)
    fy = lambda v: H - B - v / ticks[-1] * (H - T - B)
    x0, x1 = L + 10, W - R
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)} at 120,000 and 600,000 ticks, one line per world: {esc(stat)}">']
    for t in ticks:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R + 6}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><text class="ax small" x="{L - 6}" y="{fy(t) + 4:.1f}" text-anchor="end">{esc(fmt(t))}</text>')
    o.append(f'<text class="ax small" x="{x0}" y="{H - 8}" text-anchor="middle">120k</text><text class="ax small" x="{x1}" y="{H - 8}" text-anchor="middle">600k</text>')
    order = sorted(pairs, key=lambda p: 0 if p[3] == NEUTRAL else 1)
    for run, a, b, colour in order:
        o.append(f'<line class="sl" data-run="{esc(run)}" x1="{x0}" y1="{fy(a):.1f}" x2="{x1}" y2="{fy(b):.1f}" stroke="{colour}"/>')
    for run, a, b, colour in order:
        tip = f"{fmt(a)} to {fmt(b)}|{run}, {title.lower()}"
        o.append(f'<circle class="pt" cx="{x0}" cy="{fy(a):.1f}" r="3.2" fill="{colour}" data-run="{esc(run)}" data-tip="{esc(tip)}"/>')
        o.append(f'<circle class="pt" cx="{x1}" cy="{fy(b):.1f}" r="3.2" fill="{colour}" data-run="{esc(run)}" data-tip="{esc(tip)}"/>')
    o.append("</svg>")
    return f'<figure class="chart"><figcaption><b>{esc(title)}</b><span class="stat">{esc(stat)}</span></figcaption>{"".join(o)}</figure>'


def time_order_chart(run, ticks, cells):
    W, H, L, R, T, B = 440, 230, 46, 12, 12, 44
    yt = nice_ticks(max(cells))
    fx = lambda v: L + (v - ticks[0]) / (ticks[-1] - ticks[0]) * (W - L - R)
    fy = lambda v: H - B - v / yt[-1] * (H - T - B)
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Cells in {esc(run)} in the order they were sampled">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><text class="ax" x="{L - 7}" y="{fy(t) + 4:.1f}" text-anchor="end">{esc(short_num(t))}</text>')
    for t in range(ticks[0] - ticks[0] % 100000 + 100000, ticks[-1] + 1, 100000):
        o.append(f'<text class="ax" x="{fx(t):.1f}" y="{H - B + 18}" text-anchor="middle">{t // 1000}k</text>')
    o.append(f'<text class="ax" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">Ticks</text>')
    d = " ".join(f"{'M' if i == 0 else 'L'}{fx(t):.1f} {fy(c):.1f}" for i, (t, c) in enumerate(zip(ticks, cells)))
    o.append(f'<path d="{d}" class="ln" stroke="{BLUE}"/>')
    for t, c in zip(ticks, cells):
        o.append(f'<circle class="pt" cx="{fx(t):.1f}" cy="{fy(c):.1f}" r="4" fill="{BLUE}" data-tip="{esc(f"{c:,} cells|{t:,} ticks")}"/>')
    o.append("</svg>")
    return "".join(o)


def season_chart(run, ticks, cells, season):
    W, H, L, R, T, B = 440, 230, 46, 12, 12, 44
    yt = nice_ticks(max(cells))
    phases = sorted({(t % season) / season for t in ticks})
    fx = lambda p: L + 40 + phases.index(p) / max(len(phases) - 1, 1) * (W - L - R - 80)
    fy = lambda v: H - B - v / yt[-1] * (H - T - B)
    o = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="The same samples of {esc(run)} sorted by point in the season">']
    for t in yt:
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(t):.1f}" y2="{fy(t):.1f}"/><text class="ax" x="{L - 7}" y="{fy(t) + 4:.1f}" text-anchor="end">{esc(short_num(t))}</text>')
    for p in phases:
        o.append(f'<text class="ax" x="{fx(p):.1f}" y="{H - B + 18}" text-anchor="middle">{int(p * season):,}</text>')
    o.append(f'<text class="ax" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">Ticks into the {season:,}-tick year</text>')
    seen = {}
    for t, c in zip(ticks, cells):
        p = (t % season) / season
        k = seen[p] = seen.get(p, -1) + 1
        o.append(f'<circle class="pt" cx="{fx(p) + (k - 1.5) * 10:.1f}" cy="{fy(c):.1f}" r="4.5" fill="{BLUE}" data-tip="{esc(f"{c:,} cells|{t:,} ticks")}"/>')
    o.append("</svg>")
    return "".join(o)


def table(headers, rows, cls=""):
    head = "".join(f"<th scope=\"col\">{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def twin(summary, inner):
    return f'<details class="twin"><summary>{esc(summary)}</summary>{inner}</details>'


# ---- Worlds as plates ------------------------------------------------------

VIEWS = [("0", "Genome"), ("1", "Diet"), ("2", "Movement"), ("3", "Sun"), ("4", "Steps"), ("5", "Last panel"), ("all", "All six")]


def view_control(default="0"):
    buttons = "".join(
        f'<button type="button" class="viewbtn" data-v="{v}" aria-pressed="{"true" if v == default else "false"}">{n}</button>' for v, n in VIEWS
    )
    return f'<div class="seg" role="group" aria-label="Which panel to show"><span class="seg-label">Show</span>{buttons}</div>'


def world_stats(row, label=None):
    return (
        f'{num(m(row, "cells"))} cells, {pc(m(row, "energy from biting"))} of energy from biting, '
        f'{pc(m(row, "resting"))} resting, {num(m(row, "lasting strategies over 3%"))} lasting {plural_strategy(m(row, "lasting strategies over 3%"))}'
    )


def plural_strategy(n):
    return "strategy" if n == 1 else "strategies"


class Plates:
    """Which worlds get thumbnails, and the facts the enlarged view shows for each."""

    def __init__(self):
        self.worlds = {}
        self.have = set()

    def add(self, run, title, caption):
        if run not in self.worlds:
            self.worlds[run] = {"title": title, "cap": caption}

    def exists(self, run):
        return run in self.have

    def tag(self, run, gallery, data="", alt=None):
        if run not in self.have:
            return f'<div class="plate empty"><span>Map not rendered yet</span></div>'
        alt = alt or f"Map of world {run}"
        return (
            f'<a class="plate" href="media/screen/{esc(run)}-l.jpg" data-world="{esc(run)}" data-gallery="{esc(gallery)}" {data} aria-label="Enlarge {esc(alt)}">'
            f'<img loading="lazy" decoding="async" src="media/screen/{esc(run)}.jpg" width="776" height="516" alt="{esc(alt)}"></a>'
        )

    def json(self):
        payload = {r: {"t": f"media/screen/{r}.jpg", "l": f"media/screen/{r}-l.jpg", **v} for r, v in self.worlds.items() if r in self.have}
        return json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")


# ---- The most unusual worlds, per the reads, with their numbers from the data ------------------


def strategy(h, label=None):
    found = [s for s in h.get("strategies") or [] if label is None or s.get("label") == label]
    return max(found, key=lambda s: s.get("peak", 0)) if found else None


def share_run(s):
    return " to ".join(f"{100 * v:.0f}%" for v in (s["share_over_time"][0], s["share_over_time"][-1]))


NOTES = {
    "b0-0006-t007": lambda r, h, c: (
        f'{pc(m(r, "energy from other cells"))} of its energy comes from other cells and {pc(m(r, "energy from biting"))} from biting, above every unmodified world '
        f'(highest {pc(c["others"])} and {pc(c["bite"])}). A sitting omnivore went from {share_run(strategy(h, "sitting omnivore"))} of cells.'
    ),
    "b0-0006-t004": lambda r, h, c: (
        f'Full: {num(r["cells"])} cells on {num(r["open"])} open squares, one colony of {num(r["land"]["colonies"]["max"])}, with sharp-edged family territories.'
    ),
    "b0-0004-t008": lambda r, h, c: (
        f'The strongest biter in its set: {pc(m(r, "energy from biting"))} of energy from biting, with the next world at {pc(c["next_bite"](r))}. '
        f'A roaming omnivore went from {share_run(strategy(h, "roaming omnivore"))} of cells.'
    ),
    "b0-0001-t000": lambda r, h, c: (
        f'Biting came back on its own after the founders stopped: a slow omnivore went from {share_run(strategy(h, "slow omnivore"))} of cells, and {pc(m(r, "energy from biting"))} of energy now comes from biting, the highest of the unmodified worlds at this point.'
    ),
    "b0-0000-t012": lambda r, h, c: (
        f'A roaming omnivore ends richer than the slow grazers around it ({trim(strategy(h, "roaming omnivore")["energy"])} energy against {" and ".join(trim(x["energy"]) for x in h["strategies"] if x["label"] == "slow grazer" and x["peak"] > 0.2)}). The reader says roamers elsewhere earn less than sitters.'
    ),
    "b0-0000-t015": lambda r, h, c: (
        f'One colony of {num(r["land"]["colonies"]["max"])} holds {pc(r["land"]["colonies"]["max"] / r["cells"])} of the {num(r["cells"])} cells.'
    ),
    "b0-0011-t004": lambda r, h, c: (
        f'Species predict behaviour almost perfectly ({r["explain"]:.2f}; highest unmodified world {c["explain"]:.2f}). But every group lives on {pc(cards.flow_share(r, "ground"))} ground food, so the reader suspects drift.'
    ),
    "b0-0015-t014": lambda r, h, c: (
        f'Caught mid-sweep: a slow omnivore goes {", ".join(f"{100 * v:.0f}%" for v in strategy(h, "slow omnivore")["share_over_time"][-3:])} of cells over the last three samples, with {pc(m(r, "energy from biting"))} of energy from biting.'
    ),
    "b0-0017-t014": lambda r, h, c: (
        f'{r["time"]["persistent_3pct"]} lasting strategies and turnover of {r["time"]["turnover"]:.2f}, where the highest unmodified world has {c["turnover"]:.2f}. Cells went {", ".join(num(v) for v in r["alive"][-3:])} over the last three samples, as sitters vanished. Die-off or relabelling is not settled.'
    ),
    "b0-0017-t003": lambda r, h, c: (
        f'A largest colony of {num(r["land"]["colonies"]["max"])} out of {num(r["cells"])} cells, made of touching single-colour blocks with sharp borders. The reader thinks it is a frozen mosaic, not a fight.'
    ),
    "b0-0010-t012": lambda r, h, c: (
        f'Fell from {num(r["alive"][-2])} to {num(r["alive"][-1])} cells in the last 15,000 ticks. The survivors are one blob at the edge, and the steps panel shows the world once spread wide.'
    ),
    "b0-0021-t015": lambda r, h, c: (
        f'Boom and crash: {num(r["alive"][0])} cells, up to {num(max(r["alive"]))}, down to {num(r["alive"][-1])}, while roamers took over from slow grazers.'
    ),
    "b0-0003-t012": lambda r, h, c: (
        f'Resting grazers went from {share_run(strategy(h, "resting grazer"))} of cells, on average {num(strategy(h, "resting grazer")["age"])} ticks old, in one mass of {num(r["land"]["colonies"]["max"])} cells of touching lawns.'
    ),
    "b0-0005-t000": lambda r, h, c: (
        f'{trim(r["species"], 2)} species but {r["time"]["persistent_3pct"]} lasting strategy: many families, one way of living, {pc(r["rest"])} resting.'
    ),
    "b0-0009-t004": lambda r, h, c: (
        f'{r["time"]["persistent_3pct"]} lasting strategies, and species predict behaviour at {r["explain"]:.2f} where its set has a median of {c["set_explain"](r):.2f}. But its families hold separate stretches of rim, so this may be isolation, not living side by side.'
    ),
    "b0-0012-t008": lambda r, h, c: (
        f'The only world in its set the reader saw keeping a mixed income: {pc(m(r, "energy from other cells"))} of energy from other cells, while resting reaches {pc(h["rest"][-1])} of spending.'
    ),
    "b0-0013-t004": lambda r, h, c: (
        f'Bites per cell rose from {pc(h["bite"][0])} to {pc(h["bite"][-1])} while the population stayed near {num(round(r["cells"], -3))} from 60,000 ticks on, so crowding does not explain it.'
    ),
    "b0-0020-t010": lambda r, h, c: (
        f'The only world in its set where species predict behaviour ({r["explain"]:.2f}). Its families sit in separate corners, so geography may explain it, not living side by side.'
    ),
}


# ---- Sections --------------------------------------------------------------


def group_of(after):
    """How a long run ended. Thresholds read off the data: the two groups don't overlap on these."""
    if (m(after, "resting") or 0) >= 0.40:
        return "lawn"
    if (m(after, "energy from biting") or 0) >= 0.13:
        return "scramble"
    return "neither"


GROUP_COLOUR = {"lawn": BLUE, "scramble": ORANGE, "neither": NEUTRAL}
GROUP_NAME = {"lawn": "Lawn", "scramble": "Scramble", "neither": "Neither"}


def detrended(y):
    n = len(y)
    xm, ym = (n - 1) / 2, sum(y) / n
    b = sum((i - xm) * (v - ym) for i, v in enumerate(y)) / sum((i - xm) ** 2 for i in range(n))
    return [v - (ym + b * (i - xm)) for i, v in enumerate(y)]


def season_share(alive, period=4):
    """How much of the swing around a world's slow trend the point in the season explains, and where it peaks."""
    d = detrended(alive)
    total = sum(v * v for v in d) or 1
    means = [statistics.mean(d[p::period]) for p in range(period)]
    return sum(len(d[p::period]) * means[p] ** 2 for p in range(period)) / total, max(range(period), key=lambda p: means[p])


STRIP_MEASURES = [
    ("cells", "Cells", "Living cells at the last sample.", lambda v: f"{v:,.0f}" if v < 1000 else short_num(v), lambda v: f"{v:,.0f}"),
    ("species at the end", "Species", "Groups of related cells, averaged over the second half of the run and four genome cut-offs.", trim, trim),
    ("lasting strategies over 3%", "Lasting strategies", "A strategy is a group of cells that behave alike. Lasting: at least 3% of cells for three samples in a row.", lambda v: f"{v:g}", lambda v: f"{v:g}"),
    ("energy from other cells", "Energy from other cells", "Share of the world's energy income from biting and eating corpses, at the last sample.", pc, pc),
    ("energy from biting", "Energy from biting", "The biting part of that share alone.", pc, pc_fine),
    ("resting", "Resting", "Share of a cell's spending that goes on resting, second half of the run.", pc, pc_fine),
    ("behaviour dimensions", "Behaviour dimensions", "Roughly how many independent ways the brains in a world differ.", trim, trim),
]


def section_range(controls, control_alive, plates):
    cells = [r["cells"] for r in control_alive]
    charts = []
    independent = []
    for key, title, note, tickfmt, fmt in STRIP_MEASURES:
        points = [(r["run"], m(r, key)) for r in control_alive]
        vals = [v for _, v in points if v is not None]
        rho = None
        if key != "cells":
            pairs = [(r["cells"], m(r, key)) for r in control_alive if m(r, key) is not None]
            rho = rank_corr([a for a, _ in pairs], [b for _, b in pairs])
            independent.append((title, rho))
        stat = f"Median {fmt(statistics.median(vals))}, {fmt(min(vals))} to {fmt(max(vals))}" + (f". Rank correlation with cells {signed(rho)}" if rho is not None else "")
        charts.append(strip_chart(title, note, stat, points, tickfmt))
    species_rho = dict(independent)["Species"]
    resting_rho = dict(independent)["Resting"]
    moving_rho = rank_corr(cells, [m(r, "moving") for r in control_alive])
    others = [t for t, rho in independent if t not in ("Species", "Resting")]
    highest = max(abs(rho) for t, rho in independent if t in others)
    rows = []
    for r in sorted(control_alive, key=lambda r: r["run"]):
        rows.append(
            [f'<button type="button" class="wlink" data-run="{esc(r["run"])}">{esc(r["run"])}</button>'] + [fmt(m(r, key)) for key, _, _, _, fmt in STRIP_MEASURES]
        )
    table_view = table(["World"] + [t for _, t, _, _, _ in STRIP_MEASURES], rows, "num")
    return f"""
<section id="range">
<h2>Same physics, very different worlds</h2>
<div class="prose">
<p>All {len(controls)} unmodified worlds lived, and they differ by seed alone. Each dot is one world. Three seeds could never have shown this range. Point at a dot to find that world in the other charts, and click it to open its map.</p>
<p>These measures barely depend on how many cells a world has, so they aren't just population in disguise: {esc(", ".join(t.lower() for t in others))} all have a rank correlation with cells of at most {highest:.2f} either way. Species do follow population ({signed(species_rho)}), and resting does a little ({signed(resting_rho)}). A measure left out for this reason is share of time moving, at {signed(moving_rho)}.</p>
</div>
<div class="chart-grid">{"".join(charts)}</div>
<p class="note">A sub-agent sorted these 32 by eye: 9 steady grazers, 8 drifting to rest, 7 part-time biters, 4 relays where the main strategy keeps being replaced, 4 hunting economies. That is a reading, not a measurement, and the middle cuts are arbitrary since biting runs in a continuum.</p>
{twin(f"All {len(control_alive)} worlds as a table", table_view)}
</section>
"""


def difference_scores(control_alive):
    """How far a world sits from the middle, averaged over the measures. 0 is typical, 0.5 is extreme on all of them."""
    n = len(control_alive)
    score = {r["run"]: 0.0 for r in control_alive}
    keys = [k for k, *_ in STRIP_MEASURES]
    for key in keys:
        vals = [(m(r, key) if m(r, key) is not None else 0.0) for r in control_alive]
        rk = ranks(vals)
        for r, v in zip(control_alive, rk):
            score[r["run"]] += abs(v / max(n - 1, 1) - 0.5) / len(keys)
    return score


PANEL_LEGEND = [
    ("0", "Genome", "Cells coloured by genome. The colours are chosen for this world alone, so don't compare them between worlds."),
    ("1", "Recent diet", "Red is biting, green is ground food, blue is meat and gifts."),
    ("2", "Movement", "Blue sits, yellow moves."),
    ("3", "Sun", "Where the food grows. Rock is dark in every panel."),
    ("4", "Steps", "Every step taken over the whole run, on a log scale."),
    ("5", "Last panel", "Tracks on the ground if the world has them, otherwise mineral, otherwise cells by energy (orange here)."),
]


def section_look(control_alive, scores, plates, unusual_html):
    legend_world = max(control_alive, key=lambda r: scores[r["run"]])
    legend_tiles = "".join(
        f'<figure data-view="{v}" class="legend-tile"><div class="plate static"><img loading="lazy" src="media/screen/{esc(legend_world["run"])}.jpg" alt="{esc(name)} panel of {esc(legend_world["run"])}"></div><figcaption><b>{esc(name)}</b><span>{esc(note)}</span></figcaption></figure>'
        for v, name, note in PANEL_LEGEND
    )
    order = sorted(control_alive, key=lambda r: -scores[r["run"]])
    tiles = []
    for r in order:
        run = r["run"]
        data = (
            f'data-cells="{m(r, "cells")}" data-bite="{m(r, "energy from biting") or 0:.4f}" data-rest="{m(r, "resting") or 0:.4f}" '
            f'data-strat="{m(r, "lasting strategies over 3%") or 0}" data-diff="{scores[run]:.4f}"'
        )
        tiles.append(
            f'<figure class="tile" {data}>{plates.tag(run, "base")}<figcaption><b>{esc(run)}</b>'
            f'<span>{num(m(r, "cells"))} cells</span><span>{pc(m(r, "energy from biting"))} biting</span><span>{pc(m(r, "resting"))} resting</span>'
            f'<span>{num(m(r, "lasting strategies over 3%"))} lasting</span></figcaption></figure>'
        )
    sort = "".join(
        f'<button type="button" class="sortbtn" data-k="{k}" aria-pressed="{"true" if k == "diff" else "false"}">{esc(n)}</button>'
        for k, n in [("diff", "Most different"), ("cells", "Cells"), ("bite", "Biting"), ("rest", "Resting"), ("strat", "Lasting strategies")]
    )
    return f"""
<section id="look">
<h2>Look at them</h2>
<div class="prose"><p>Every map is one world at the end of its run, 256 squares wide. Each map has six panels. Here they are for one world, {esc(legend_world["run"])}.</p></div>
<div class="panel-legend">{legend_tiles}</div>
<div class="viewroot" data-view="0">
<div class="controls">
{view_control("0")}
<div class="seg" role="group" aria-label="Sort the worlds"><span class="seg-label">Sort by</span>{sort}</div>
</div>
<p class="note">Most different means furthest from the middle, averaged over the seven measures above. Click a world to enlarge it and to step through the panels.</p>
<div class="wall" id="wall">{"".join(tiles)}</div>
</div>
{unusual_html}
</section>
"""


def section_unusual(rows_by_run, reads, control_alive, plates, candidates):
    c = {
        "others": max(m(r, "energy from other cells") or 0 for r in control_alive),
        "bite": max(m(r, "energy from biting") or 0 for r in control_alive),
        "explain": max((r.get("explain") or 0) for r in control_alive),
        "turnover": max(r["time"]["turnover"] for r in control_alive),
        "set_explain": lambda row: med([x.get("explain") for x in rows_by_run.values() if x["batch"] == row["batch"] and (x.get("cells") or 0) >= cards.MIN_CELLS and (x.get("species") or 0) >= 2]),
        "next_bite": lambda row: max(m(x, "energy from biting") or 0 for x in rows_by_run.values() if x["batch"] == row["batch"] and x["run"] != row["run"] and (x.get("cells") or 0) >= cards.MIN_CELLS),
    }
    cardsout = []
    for run in candidates:
        r = rows_by_run.get(run)
        if not r:
            continue
        note = candidates[run]
        if callable(note):
            try:
                note = note(r, HISTORY.history(r["batch"], r["index"]), c)
            except (KeyError, TypeError, IndexError, ZeroDivisionError) as why:
                print(f"  note for {run} skipped: {why}", file=sys.stderr)
                continue
        physics = r.get("physics") or {}
        on, _ = rule_words(physics)
        rules = "Unmodified physics" if not physics else cap("; ".join(on)) if on else "Rule settings changed"
        plates.worlds[run]["cap"] += " " + note
        cardsout.append(
            f'<figure class="odd">{plates.tag(run, "odd")}<figcaption><b>{esc(run)}</b><span class="rules">{esc(rules)}</span><span class="say">{esc(note)}</span>'
            f'<span class="nums">{esc(world_stats(r))}</span></figcaption></figure>'
        )
    return f"""
<div class="viewroot" data-view="0">
<h3>The most unusual worlds, from all sets</h3>
<div class="prose"><p>A sub-agent read each set of worlds and named the ones worth a closer look. The numbers below come from the data; the reasons are the reader's, and none has been followed up unless a section below says so.</p></div>
<div class="controls">{view_control("0")}</div>
<div class="odds">{"".join(cardsout)}</div>
</div>
"""


TABLE_COLUMNS = [
    ("cells", "Cells", lambda v: num(v)),
    ("species at the end", "Species", lambda v: trim(v)),
    ("lasting strategies over 3%", "Lasting strategies", lambda v: trim(v)),
    ("energy from other cells", "From other cells", lambda v: pc(v, 1)),
    ("energy from biting", "From biting", lambda v: pc(v, 1)),
    ("resting", "Resting", lambda v: pc(v, 1)),
    ("behaviour dimensions", "Dimensions", lambda v: trim(v)),
]


def set_row(name, rs, standing, reads, known, label_html, read_prefix=""):
    living = cards.alive(rs)
    cols = []
    head = [f'<td class="setname" data-l="Set">{label_html}</td>', f'<td class="alive" data-l="Alive of {len(rs)}">{len(living)}</td>']
    if not living:
        cols.append(f'<td class="none" colspan="{len(TABLE_COLUMNS)}" data-l="">None alive at the end</td>')
    else:
        for key, title, fmt in TABLE_COLUMNS:
            vals = cards.values(living, key)
            value = cards.median(vals) if vals else None
            mark = standing.get((name, key))
            cls, arrow = "", ""
            if mark:
                cls = " up" if mark[0] == "higher" else " down"
                arrow = "▲ " if mark[0] == "higher" else "▼ "
            cols.append(f'<td class="n{cls}" data-l="{esc(title)}">{arrow}{esc(fmt(value))}</td>')
    has_read = name in reads
    btn = f'<button type="button" class="readbtn" aria-expanded="false" aria-controls="read-{name}">Read</button>' if has_read else ""
    row = f'<tr>{"".join(head)}{"".join(cols)}<td class="action" data-l="">{btn}</td></tr>'
    if has_read:
        row += f'<tr class="readrow" id="read-{name}" hidden><td colspan="{len(TABLE_COLUMNS) + 3}"><div class="read">{read_html(reads[name], name, known)}</div></td></tr>'
    return row


def set_table(rows_html):
    heads = "".join(f'<th scope="col" class="n">{esc(t)}</th>' for _, t, _ in TABLE_COLUMNS)
    return f'<div class="settable"><table><thead><tr><th scope="col">Set</th><th scope="col" class="alive">Alive</th>{heads}<th scope="col"><span class="vh">The reader\'s notes</span></th></tr></thead><tbody>{"".join(rows_html)}</tbody></table></div>'


def section_sets(sets, specs, controls, control_alive, reads, plates):
    first = {k: v for k, v in sets.items() if not k.startswith("b1-") and v[0].get("physics")}
    clean = {k: v for k, v in sets.items() if k.startswith("b1-") and v[0].get("physics")}
    known = plates.have
    standing, n_tests, n_kept = compare_sets(first, control_alive)

    def label(name, rs):
        on, other = rule_words(rs[0].get("physics") or {})
        words = cap("; ".join(on)) if on else "No rule switched on"
        extra = f' <span class="more" title="{esc(", ".join(sorted(other)))}">Plus {len(other)} base {"setting" if len(other) == 1 else "settings"} changed.</span>' if other else ""
        return f'<span class="sid">{esc(name)}</span><span class="rw">{esc(words)}.{extra}</span>'

    base_batches = [b for b, v in sets.items() if not v[0].get("physics")]
    base_label = f'<span class="sid">Plain physics</span><span class="rw">{len(control_alive)} worlds pooled: {esc(", ".join(sorted(base_batches)))}. No extra rules.</span>'
    base = set_row("plain", controls, {}, {}, known, base_label)
    ordered = sorted(first.items(), key=lambda kv: (-len(cards.alive(kv[1])), kv[0]))
    table_rows = [base] + [set_row(k, v, standing, reads, known, label(k, v)) for k, v in ordered]
    for b in sorted(base_batches, reverse=True):
        if b in reads:
            table_rows.insert(1, set_row(b, sets[b], {}, reads, known, f'<span class="sid">{esc(b)}</span><span class="rw">Unmodified, one half of the pool above.</span>'))
    dead = [k for k, v in first.items() if not cards.alive(v)]
    nearly = [k for k, v in first.items() if 0 < len(cards.alive(v)) <= 2]
    lasting_up = sorted(k for k in first if standing.get((k, "lasting strategies over 3%"), ("",))[0] == "higher")
    extra = ""
    if lasting_up:
        k = lasting_up[0]
        living = cards.alive(first[k])
        ground = med([cards.flow_share(r, "ground") for r in living])
        one_species = sum(1 for r in living if (r.get("species") or 0) <= 1.0)
        cells_med = med([r["cells"] for r in living])
        extra = (
            f'<p><strong>The one exception on lasting strategies is {esc(k)}</strong>, with a median of {trim(cards.median(cards.values(living, "lasting strategies over 3%")))} against {trim(cards.median(cards.values(control_alive, "lasting strategies over 3%")))}, which does survive the correction. '
            f"We don't count it as a win. Only {len(living)} of its 16 worlds lived, with a median of {num(cells_med)} cells, so a 3% group is about {num(cells_med * 0.03)} cells. "
            f'{cap(words_number(one_species))} of the {len(living)} have a single species, and they get a median of {pc(ground)} of their energy from the ground. The reader judged the differences to be probably neutral drift.</p>'
        )
    alone = {}
    for k, rs in first.items():
        bites = (rs[0].get("group_hunting") or {}).get("bites") or []
        if sum(bites) > 0:
            alone[k] = bites[0] / sum(bites)
    gang_note = ""
    if "b0-0012" in alone:
        g = first["b0-0012"][0]["group_hunting"]
        gang_note = f'Where the gang rule is on it does pay per bite (in b0-0012, a bite in company earns {g["group_advantage"]:.1f} times a lone bite), but bites stay solo: {pc(alone["b0-0012"])} of bites there were alone. Across all sets with biting, {pc(min(alone.values()))} to {pc(max(alone.values()))} of bites were alone.'
    clean_part = ""
    done = [k for k in specs if k in sets]
    if clean:
        c_standing, c_tests, c_kept = compare_sets(clean, control_alive)

        def c_label(name, rs):
            spec = specs.get(name) or {}
            on, _ = rule_words(rs[0].get("physics") or {})
            return f'<span class="sid">{esc(spec.get("rule") or name)}</span><span class="rw">{esc(cap("; ".join(on)) if on else name)}. {esc(name)}.</span>'

        c_rows = [set_row("plain", controls, {}, {}, known, base_label)] + [set_row(k, v, c_standing, reads, known, c_label(k, v)) for k, v in sorted(clean.items())]
        clean_part = f"""
<h3>The clean set: one rule at a time</h3>
<div class="prose"><p>{len(done)} of {len(specs)} batches are in so far. Each set turns on a single rule with its other settings left as they are, so a difference here can be put down to that rule. Of {c_tests} tests on all 19 measures, {c_kept} {"stands" if c_kept == 1 else "stand"} at a 5% false discovery rate, counting measures this table doesn't show. The correction will change as more batches land, and there is no sub-agent's read of these yet.</p></div>
{set_table(c_rows)}
"""
    else:
        names = ", ".join(sorted((s.get("rule") or k) for k, s in specs.items() if s.get("physics")))
        clean_part = f"""
<h3>The clean set: one rule at a time</h3>
<div class="prose"><p><strong>Still running: {len(done)} of {len(specs)} batches are in.</strong> This set turns on one rule at a time ({esc(names)}), plus two more batches of unmodified worlds, so it can say what a single rule does. Its row-by-row table appears here when it lands.</p></div>
"""
    return f"""
<section id="sets">
<h2>Each set of physics against plain physics</h2>
<div class="prose">
<p>The first screen tried {len(first)} random mixes, 16 worlds each. A mix turns on several rules and changes many settings at once, so <strong>it can't say what any one rule does.</strong> It can only say whether anything in that part of the rule space looked better.</p>
<p>A world counts as alive with at least {cards.MIN_CELLS} cells at the end. All {len(control_alive)} plain worlds lived. Of the {len(first)} mixes, {len(dead)} lost all 16 worlds and {len(nearly)} more kept only 1 or 2. A rain of random cells can restart a dead world, by design, so some worlds that regrew are new starts. Most mixes made worlds duller: fewer species and less biting.</p>
{extra}
<p class="key"><span class="k up">▲</span> higher or <span class="k down">▼</span> lower than plain physics, in a difference that survives the false-discovery correction across all {n_tests} comparisons (each set, 19 measures, rank-sum test): {n_kept} do, some of them on measures this table doesn't show. Unmarked medians can't be told apart from plain physics. Medians are over the worlds still alive. Sets with fewer than 4 live worlds aren't tested.</p>
</div>
<h3>First screen: many settings at once</h3>
{set_table(table_rows)}
<div class="prose"><p class="note">No map a reader opened shows thin trails between places, and no set shows group hunting. {esc(gang_note)}</p></div>
{clean_part}
</section>
"""


def section_kin(kin, kin_long):
    usable = [k for k in kin if k.get("on_kin") is not None and k.get("kin_near") is not None and k.get("biters", 0) >= CLOSE_KIN_MIN_BITERS]
    if not usable:
        return ""
    r = corr([k["kin_near"] for k in usable], [k["on_kin"] for k in usable])
    diffs = [100 * (k["on_kin"] - k["kin_near"]) for k in usable]
    within = sum(abs(d) <= 10 for d in diffs)
    long_usable = [k for k in kin_long if k.get("on_kin") is not None and k.get("biters", 0) >= CLOSE_KIN_MIN_BITERS]
    long_r = corr([k["kin_near"] for k in long_usable], [k["on_kin"] for k in long_usable]) if len(long_usable) > 2 else None
    long_diffs = [100 * (k["on_kin"] - k["kin_near"]) for k in long_usable]
    long_text = (
        f" At 600,000 ticks, in {len(long_usable)} plain worlds, it's the same: correlation {long_r:.2f}, median gap {signed(statistics.median(long_diffs), 1)} points, within 10 points in {sum(abs(d) <= 10 for d in long_diffs)} of {len(long_usable)}."
        if long_usable
        else ""
    )
    rows = [
        [f'<button type="button" class="wlink" data-run="{esc(k["run"])}">{esc(k["run"])}</button>', pc(k["kin_near"], 1), pc(k["on_kin"], 1), signed(100 * (k["on_kin"] - k["kin_near"]), 1), num(k["biters"])]
        for k in sorted(usable, key=lambda k: -abs(k["on_kin"] - k["kin_near"]))
    ]
    return f"""
<section id="kin">
<h2>Biters don't choose</h2>
<div class="two">
<div class="prose">
<p>Telling friend from foe is what group behaviour would need, and nothing here has learned to do it. In {len(usable)} worlds with at least {CLOSE_KIN_MIN_BITERS} sampled biters, the share of bite energy taken from close kin matches the share of close kin next to the biter.</p>
<p>Correlation {r:.2f}. The median gap is {signed(statistics.median(diffs), 1)} points, the typical size of a gap either way is {statistics.median(abs(d) for d in diffs):.1f} points, and {within} of {len(usable)} worlds are within 10 points. Bites land on whoever is adjacent.{long_text}</p>
<p class="note">Close kin means a genome sketch less than 0.4 away. Every dot on the line would mean bites follow who is next to the biter and nothing else. The labelled worlds are the two furthest from it.</p>
{legend([("120,000 ticks", BLUE), ("600,000 ticks", ORANGE)])}
</div>
<figure class="chart wide">{scatter_kin(usable, long_usable)}</figure>
</div>
{twin(f"All {len(usable)} worlds as a table, furthest from the line first", table(["World", "Close kin nearby", "Bite energy from close kin", "Gap, points", "Biters sampled"], rows, "num"))}
</section>
"""


def knockout_pairs(long_batches, screen_rows):
    """Biting off against the same saved world left alone, at the same tick. Returns pairs by set, and sets with no partner yet."""
    found, waiting = {}, []
    for b in sorted(long_batches):
        if not b.startswith("ko-"):
            continue
        base = b[3:]
        partner = long_batches.get("long-" + base)
        if not partner:
            waiting.append(base)
            continue
        by_index = {r["index"]: r for r in partner}
        for ko in long_batches[b]:
            lr = by_index.get(ko["index"])
            start = screen_rows.get(f'{base}-t{ko["index"]:03d}')
            if not lr or not lr.get("alive") or not start:
                continue
            ticks = HISTORY.history(lr["batch"], lr["index"]).get("ticks") or []
            # The long run's sample at the knockout's last tick, so both are at the same point in the season.
            if ko["tick"] not in ticks:
                continue
            found.setdefault(base, []).append({
                "run": start["run"], "index": ko["index"], "batch": base, "alone": lr["alive"][ticks.index(ko["tick"])], "off": ko["cells"],
                "biting": m(start, "energy from biting") or 0.0, "plain": not start.get("physics"),
            })
    for pairs in found.values():
        pairs.sort(key=lambda p: -p["biting"])
    return found, waiting


def knockout_table(pairs):
    rows = [
        [f'<button type="button" class="wlink" data-run="{esc(p["run"])}">{esc(p["run"])}</button>', pc(p["biting"], 1), num(p["alone"]), num(p["off"]), f'{100 * (p["off"] / p["alone"] - 1):+.1f}%'.replace("-", "−")]
        for p in pairs
    ]
    return table(["World", "Energy from biting", "Left alone", "Biting off", "Change"], rows, "num")


def section_knockout(long_batches, screen_rows, plates):
    found, waiting = knockout_pairs(long_batches, screen_rows)
    if not found:
        return ""
    plain = [p for pairs in found.values() for p in pairs if p["plain"]]
    plain.sort(key=lambda p: -p["biting"])
    mixes = {k: v for k, v in found.items() if not v[0]["plain"]}
    wait_text = f" Knockouts for {esc(', '.join(waiting))} are done, but their long runs aren't, so there's nothing yet to compare them with." if waiting else ""
    main_part = ""
    if plain:
        ratios = [p["off"] / p["alone"] for p in plain]
        best = max(plain, key=lambda p: p["off"] / p["alone"])
        top = plain[0]
        lost = sum(1 for x in ratios if x < 0.95)
        main_part = f"""
<p>It barely changes anything. In {len(plain)} plain worlds, the median world has {pc(statistics.median(ratios) - 1)} more cells with biting off. {"None" if not lost else cap(words_number(lost))} of the {len(plain)} lost more than 5% (the worst lost {pc(1 - min(ratios), 1)}). One gained a lot: {esc(best["run"])}, {pc(best["off"] / best["alone"] - 1)} more without biting. The world with the most biting, {esc(top["run"])} at {pc(top["biting"])} of its energy, has {pc(top["off"] / top["alone"] - 1)} more cells without it.</p>
<p>So biting is mostly a cost to the population, even where it's a large part of the energy flow. This is one run of each, compared at one moment.{wait_text}</p>
"""
    mix_part = ""
    for name, pairs in sorted(mixes.items()):
        ratios = [p["off"] / p["alone"] for p in pairs]
        worst = min(pairs, key=lambda p: p["off"] / p["alone"])
        best = max(pairs, key=lambda p: p["off"] / p["alone"])
        mix_part += f"""
<details class="twin"><summary>The same test on a mix: {esc(name)}, {len(pairs)} worlds, median {signed(100 * (statistics.median(ratios) - 1), 0)}%</summary>
<div class="two"><div class="prose"><p>Here it isn't harmless in every world. The range runs from {esc(worst["run"])}, which has {pc(1 - worst["off"] / worst["alone"])} fewer cells with biting off, to {esc(best["run"])}, with {pc(best["off"] / best["alone"] - 1)} more. {sum(1 for x in ratios if x < 0.95)} of {len(pairs)} lost more than 5%. This set has other rules on, so it says nothing about plain physics.</p></div>
<figure class="chart wide">{dumbbell(pairs)}</figure></div>{knockout_table(pairs)}</details>"""
    return f"""
<section id="biting-off">
<h2>Biting off</h2>
<div class="two">
<div class="prose">
<p>We took saved worlds at 120,000 ticks and ran 30,000 ticks more, once with biting switched off and once left alone, starting from the same state. If cells were living off biting, turning it off should hurt.</p>
{main_part}
{legend([("Left alone", BLUE), ("Biting off", ORANGE)])}
<p class="note">Sorted by how much energy the world got from biting at 120,000 ticks.</p>
</div>
<figure class="chart wide">{dumbbell(plain) if plain else ""}</figure>
</div>
{twin("These numbers as a table", knockout_table(plain)) if plain else ""}
{mix_part}
</section>
"""


LONG_MEASURES = [
    ("cells", "Cells", short_num, lambda v: f"{v:,.0f}"),
    ("resting", "Resting", pc, pc),
    ("energy from biting", "Energy from biting", pc, pc),
    ("species at the end", "Species", trim, trim),
    ("strategies over 3% at the end", "Strategies over 3%, at the end", lambda v: f"{v:g}", lambda v: f"{v:g}"),
    ("behaviour dimensions", "Behaviour dimensions", trim, trim),
]


def long_pairs(long_batches, screen_rows):
    """(before, after) for every long run, split into plain physics and the rest by set."""
    plain, mixes = [], {}
    for b in sorted(long_batches):
        if not b.startswith("long-"):
            continue
        for a in long_batches[b]:
            before = screen_rows.get(f'{b[5:]}-t{a["index"]:03d}')
            if before:
                (mixes.setdefault(b[5:], []) if a.get("physics") else plain).append((before, a))
    return plain, mixes


def long_blocks(pairs, plates, gallery):
    """Slope charts, table and before-and-after maps for one group of long runs. Returns charts, table, maps, medians."""
    groups = {a["run"]: group_of(a) for _, a in pairs}
    charts, medians = [], {}
    for key, title, tick, fmt in LONG_MEASURES:
        ps = [(a["run"], m(b, key) or 0.0, m(a, key) or 0.0, GROUP_COLOUR[groups[a["run"]]]) for b, a in pairs]
        medians[key] = (med([p[1] for p in ps]), med([p[2] for p in ps]))
        charts.append(slope_chart(title, f"Median {fmt(medians[key][0])} to {fmt(medians[key][1])}", ps, tick))
    order = ["lawn", "scramble", "neither"]
    ranked = sorted(pairs, key=lambda p: (order.index(groups[p[1]["run"]]), p[1]["index"]))
    rows = []
    cards_html = []
    for b, a in ranked:
        name = groups[a["run"]]
        rows.append(
            [f'<button type="button" class="wlink" data-run="{esc(a["run"])}">{esc(a["run"])}</button>', GROUP_NAME[name]]
            + [f"{fmt(m(b, key) or 0)} to {fmt(m(a, key) or 0)}" for key, _, _, fmt in LONG_MEASURES]
        )
        cards_html.append(
            f'<figure class="pair"><figcaption><b>t{a["index"]:03d}</b> <span>{GROUP_NAME[name]}. Cells {num(b["cells"])} to {num(a["cells"])}, resting {pc(m(b, "resting"))} to {pc(m(a, "resting"))}, biting {pc(m(b, "energy from biting"))} to {pc(m(a, "energy from biting"))}.</span></figcaption>'
            f'<div class="pair-plates"><div>{plates.tag(b["run"], gallery)}<span class="when">{b["tick"]:,} ticks</span></div><div>{plates.tag(a["run"], gallery)}<span class="when">{a["tick"]:,} ticks</span></div></div></figure>'
        )
    tab = table(["World", "Ended as"] + [t for _, t, _, _ in LONG_MEASURES], rows, "num")
    return "".join(charts), tab, "".join(cards_html), medians, groups


def section_long(long_batches, screen_rows, plates, statuses):
    pairs, mixes = long_pairs(long_batches, screen_rows)
    if not pairs and not mixes:
        return '<section id="longer"><h2>Five times longer</h2><div class="prose"><p>No long run has finished yet.</p></div></section>'
    if not pairs:
        return '<section id="longer"><h2>Five times longer</h2><div class="prose"><p>No long run of plain physics has finished yet.</p></div></section>'
    charts, tab, cards_html, medians, groups = long_blocks(pairs, plates, "pairs")
    lasting_b = med([m(b, "lasting strategies over 3%") for b, _ in pairs])
    lasting_a = med([m(a, "lasting strategies over 3%") for _, a in pairs])

    def group_rows(name):
        return [(b, a) for b, a in pairs if groups[a["run"]] == name]

    def tnames(ps):
        return ", ".join(f't{a["index"]:03d}' for _, a in ps)

    lawns, scrambles, neither = group_rows("lawn"), group_rows("scramble"), group_rows("neither")

    def span(ps, key, fmt):
        vals = [m(a, key) or 0 for _, a in ps]
        return f"{fmt(min(vals))} to {fmt(max(vals))}"

    def growth(ps):
        vals = [a["cells"] / max(b["cells"], 1) for b, a in ps]
        return f"{min(vals):.1f} to {max(vals):.1f}"

    lawn_text = scramble_text = ""
    if lawns:
        least = min(lawns, key=lambda p: (m(p[1], "species at the end") or 0) / max(m(p[0], "species at the end") or 1, 0.1))
        lawn_text = (
            f'<p><strong>Lawns</strong> ({tnames(lawns)}). Resting ends at {span(lawns, "resting", pc)}, and cells multiply by {growth(lawns)}. Biting stays small ({span(lawns, "energy from biting", lambda v: pc(v, 1))}). '
            f'Median species go from {trim(med([m(b, "species at the end") for b, _ in lawns]))} to {trim(med([m(a, "species at the end") for _, a in lawns]))}, as sharp-edged single-colour patches, though t{least[1]["index"]:03d} only goes from {trim(m(least[0], "species at the end"))} to {trim(m(least[1], "species at the end"))}.</p>'
        )
    if scrambles:
        scramble_text = (
            f'<p><strong>Scrambles</strong> ({tnames(scrambles)}). Biting ends at {span(scrambles, "energy from biting", pc)} of energy and resting at {span(scrambles, "resting", lambda v: pc(v, 1))}. Median cells go from {num(med([b["cells"] for b, _ in scrambles]))} to {num(med([a["cells"] for _, a in scrambles]))}.</p>'
        )
    neither_text = f'<p class="note">The other {len(neither)} ({tnames(neither)}) did neither. Lawn here means at least 40% resting at the end, and scramble at least 13% of energy from biting. Both cut-offs are read off the data, where there is a clear gap.</p>' if neither else ""
    mix_html = ""
    for name, ps in sorted(mixes.items()):
        m_charts, m_tab, m_cards, m_med, m_groups = long_blocks(ps, plates, f"pairs-{name}")
        on, _ = rule_words(ps[0][1].get("physics") or {})
        lawn_n = sum(1 for v in m_groups.values() if v == "lawn")
        scr_n = sum(1 for v in m_groups.values() if v == "scramble")
        s_end, d_end = m_med["strategies over 3% at the end"], m_med["behaviour dimensions"]
        mix_html += f"""
<details class="twin"><summary>The same test on a mix: {esc(name)}, {len(ps)} worlds. {esc(cap("; ".join(on)))}.</summary>
<div class="prose"><p>This isn't plain physics, so it isn't part of the numbers above. It tells the same story: strategies over 3% at the end stay at a median of {s_end[0]:g} to {s_end[1]:g}, and behaviour dimensions go from {d_end[0]:.1f} to {d_end[1]:.1f}. By the same cut-offs, {words_number(lawn_n)} worlds end as lawns and {words_number(scr_n)} as scrambles. Its year is {int(float(ps[0][1]["physics"].get("season_ticks", SEASON_DEFAULT))):,} ticks, not {SEASON_DEFAULT:,}, so 120,000 and 600,000 ticks may fall at different points in the year here.</p></div>
<div class="chart-grid three">{m_charts}</div>
{twin("These numbers as a table", m_tab)}
<div class="viewroot" data-view="2"><div class="controls">{view_control("2")}</div><div class="pairs">{m_cards}</div></div>
</details>"""
    # The season alias, shown on the world where it first fooled us.
    first_b, first_a = next((p for p in pairs if p[1]["run"] == "long-b0-0001-t000"), pairs[0])
    season = int(float((first_a.get("physics") or {}).get("season_ticks", SEASON_DEFAULT)))
    hist = HISTORY.history(first_a["batch"], first_a["index"])
    ticks, alive = hist.get("ticks") or [], first_a.get("alive") or []
    season_html = ""
    if len(ticks) > 1 and len(ticks) == len(alive) and ticks[1] - ticks[0] == SAMPLE_TICKS and SAMPLE_TICKS * 4 % season == 0:
        shares = [(a["index"], *season_share(a["alive"])) for _, a in pairs if len(a.get("alive") or []) >= 8]
        strong = [x for x in shares if x[1] >= 0.75]
        weak = sorted((x for x in shares if x[1] < 0.75), key=lambda x: x[1])
        weak_text = f' The rest are {", ".join(f"t{i:03d} at {100 * x:.0f}%" for i, x, _ in weak)}.' if weak else ""
        peak_text = " Worlds peak at different points in the year." if len({x[2] for x in shares}) > 1 else ""
        same_phase = f" Both ends of the comparison above, {first_b['tick']:,} and {first_a['tick']:,} ticks, fall at the same point in the year, so the before and after isn't affected." if first_b["tick"] % season == first_a["tick"] % season else ""
        season_html = f"""
<h3 id="seasons">Population over time is not a cycle</h3>
<div class="prose">
<p>Cells in {esc(first_a["run"])} swing between about {num(round(min(alive), -2))} and {num(round(max(alive), -2))} with a period of 4 samples, which looks like a population cycle. It isn't one. The world has an {season:,}-tick year, and samples come every {SAMPLE_TICKS:,} ticks, which is 3.75 years. So each sample lands a quarter of a year earlier in the season than the last, and the same point comes round every 4 samples. Sort the samples by point in the year and the cycle collapses into seasons.</p>
<p>After removing each world's slow trend, the point in the year explains at least 75% of the swing in {len(strong)} of {len(shares)} plain worlds.{weak_text}{peak_text}{same_phase}</p>
</div>
<div class="chart-pair">
<figure class="chart"><figcaption><b>The samples in order</b><span>What looks like a cycle. It comes from when we looked.</span></figcaption>{time_order_chart(first_a["run"], ticks, alive)}</figure>
<figure class="chart"><figcaption><b>The same samples, by point in the year</b><span>{len(alive)} samples. The same point in the year gives about the same count.</span></figcaption>{season_chart(first_a["run"], ticks, alive, season)}</figure>
</div>
"""
    strategies_end, dims = medians["strategies over 3% at the end"], medians["behaviour dimensions"]
    waiting = [x for x in statuses if x[1] != "done"]
    waiting_text = f' Still to land: {esc(", ".join(f"{n} ({st})" for n, st in waiting))}.' if waiting else ""
    return f"""
<section id="longer">
<h2>Five times longer</h2>
<div class="prose">
<p>{len(pairs)} plain worlds continued from 120,000 to 600,000 ticks. More time doesn't help: strategies over 3% at the end stay at a median of {strategies_end[0]:g} to {strategies_end[1]:g}, and behaviour dimensions at {dims[0]:.1f} to {dims[1]:.1f}.{waiting_text}</p>
<p>The lasting count does rise, from {lasting_b:g} to {lasting_a:g}, but it isn't a fair comparison. Lasting means three samples in a row, which is 45,000 ticks in a short run and 90,000 here, and a run of 16 samples has more chances to find one than a run of 8.</p>
<p>What changes is that worlds settle in two directions.</p>
{lawn_text}
{scramble_text}
{neither_text}
{legend([("Lawn", BLUE), ("Scramble", ORANGE), ("Neither", NEUTRAL)])}
</div>
<div class="chart-grid three">{charts}</div>
{twin("These numbers as a table", tab)}
<h3>The same worlds, before and after</h3>
<div class="viewroot" data-view="2">
<div class="prose"><p>Movement is shown first, since lawns turn blue. Switch to genome to see the patches.</p></div>
<div class="controls">{view_control("2")}</div>
<div class="pairs">{cards_html}</div>
</div>
{mix_html}
{season_html}
</section>
"""


def section_false(rows, controls, long_batches):
    longs = [r for b, rs in long_batches.items() if b.startswith("long-") for r in rs if not r.get("physics")]
    cycle = ""
    if longs:
        first = next((r for r in longs if r["run"] == "long-b0-0001-t000"), sorted(longs, key=lambda r: (r["batch"], r["index"]))[0])
        alive = first["alive"]
        cycle = (
            f'<li><b>Population cycles.</b> {esc(first["run"])} and others seem to swing, about {num(round(min(alive), -2))} to {num(round(max(alive), -2))} cells, with a period of 4 samples. '
            f'Samples every 30,000 ticks alias against the built-in {SEASON_DEFAULT:,}-tick year: it\'s seasons. <a href="#seasons">Shown above</a>. Lesson: don\'t sample at a period that shares a low common multiple with the season.</li>'
        )
    worst = max(controls, key=lambda r: r.get("from_others") or 0)
    flows_share = cards.flow_share(worst, "bite", "meat")
    # Traffic against what it was thought to follow, on the controls.
    lines = [(r["trails"] or {}).get("lines") for r in controls]
    col_rho = rank_corr([r["land"]["colonies"]["count"] for r in controls], lines)
    cell_rho = rank_corr([r["cells"] for r in controls], lines)
    living = [r for r in cards.alive(rows) if (r.get("trails") or {}).get("lines") is not None and r.get("land")]
    no_colony = [r["trails"]["lines"] for r in living if not r["land"]["colonies"]["count"]]
    every = [r["trails"]["lines"] for r in living]
    lo, hi = min(RIM.values()), max(RIM.values())
    rock_lo, rock_hi = min(ROCK.values()), max(ROCK.values())
    return f"""
<section id="false">
<h2>Things that looked real and weren't</h2>
<ul class="falses">
{cycle}
<li><b>Life hugging the outer wall in mineral worlds.</b> Real, but an artefact of the mineral rule. Mineral piles against the outer rock at a median of {lo} to {hi} times the density inside the world, set by set across all the flow settings tried, and cells follow it. Next to natural rock it's about {rock_lo:.0f} to {rock_hi:.0f} times higher too, in the two worlds measured. So mineral worlds didn't test a second resource. It's likely carried there by cells and released where they die, but that mechanism isn't tested.</li>
<li><b>"Traffic on lines" as a sign of trails.</b> No map showed lines, so it's useless as a trail detector as built. It doesn't even track colonies: in plain worlds its rank correlation with colony count is {signed(col_rho)}. The {len(no_colony)} live worlds with no colonies have a median of {med(no_colony):.0f}, against {med(every):.0f} for all live worlds. Readers saw it rise wherever cells pace around, whether inside a colony or along a wall.</li>
<li><b>"Lives on other cells: {pc(worst.get("from_others"))}"</b> in our earlier rank tables is the average of per-cell ratios, in {esc(worst["run"])}. The world's energy flows say {pc(flows_share)} for the same world. Flows are what this page uses.</li>
</ul>
</section>
"""


def section_next(specs, sets):
    return f"""
<section id="next">
<h2>Why this can't make bodies, and what's next</h2>
<div class="prose">
<p>This engine allows one cell per square. Cells have no bond between them, and force doesn't cross a group: the most a cell can do is shove one neighbour. So there is no way for a group to hold together, move together or push. No choice of rule in this engine can change that, so running more of it won't produce bodies.</p>
<p>A new engine is being designed: soft discs in continuous space that can stick to each other, pull, push, and trade energy across contact. It keeps the same discipline. One kind of cell, no fitness function, no body IDs, energy conserved, and an off switch on every rule so each can be tested against a control. The bar is bodies that reproduce as units with no designed help.</p>
<p><a href="{esc(DESIGN_URL)}">Read the design</a>. It works out supply and demand before anything is built.</p>
</div>
</section>
"""


def section_answer(rows, controls, control_alive, first_sets, kin, long_batches, specs, sets, band):
    n_first = len(first_sets)
    lasting = cards.median(cards.values(control_alive, "lasting strategies over 3%"))
    standing, _, _ = compare_sets(first_sets, control_alive)
    higher = sorted(k for k in first_sets if standing.get((k, "lasting strategies over 3%"), ("",))[0] == "higher")
    dead = [k for k, v in first_sets.items() if not cards.alive(v)]
    usable = [k for k in kin if k.get("on_kin") is not None and k.get("biters", 0) >= CLOSE_KIN_MIN_BITERS]
    r = corr([k["kin_near"] for k in usable], [k["on_kin"] for k in usable])
    pairs, _ = long_pairs(long_batches, {x["run"]: x for x in rows})
    long_text = ""
    if pairs:
        long_text = (
            f'<p><strong>More time doesn\'t help.</strong> {len(pairs)} plain worlds run five times longer end with a median of {med([m(a, "strategies over 3% at the end") for _, a in pairs]):g} strategies over 3%, '
            f'the same as before, and {med([m(a, "behaviour dimensions") for _, a in pairs]):.1f} behaviour dimensions, also the same.</p>'
        )
    exception = ""
    if higher:
        k = higher[0]
        living = cards.alive(first_sets[k])
        exception = (
            f' One mix, {esc(k)}, scores {trim(cards.median(cards.values(living, "lasting strategies over 3%")))}, but with a median of {num(med([x["cells"] for x in living]))} cells nearly all one species on the same diet.'
        )
    band_html = "".join(
        f'<div class="plate static" data-view="0"><img src="media/screen/{esc(run)}.jpg" alt="Genome map of {esc(run)}" width="776" height="516"></div>' for run in band
    )
    return f"""
<header class="hero">
<div class="band" aria-hidden="true">{band_html}</div>
<div class="hero-text">
<h1>So far, no physics tried here beat plain physics at producing lasting different ways of living.</h1>
<div class="lead">
<p>{len(controls)} worlds ran on plain physics and {n_first} sets of 16 on mixes of extra rules. A strategy is a group of cells that behave alike, and it lasts if it holds at least 3% of cells for three samples in a row. Plain worlds keep a median of {lasting:g}. {cap(words_number(len(dead)))} of the mixes died out completely, and most of the others got duller.{exception}</p>
<p><strong>Biters don't choose whom they bite.</strong> Across {len(usable)} worlds, the bite energy taken from close kin matches how many close kin are next to the biter (correlation {r:.2f}).</p>
{long_text}
<p><strong>Bodies are physically impossible in this engine.</strong> One cell per square, no bonds, and force can't cross a group. So a new engine is being built.</p>
</div>
</div>
</header>
"""


def status_block(rows, controls, first_sets, long_rows, specs, sets, long_statuses, ko_batches, have, total_media, built):
    clean_done = sum(1 for k in specs if k in sets)
    clean_running = [k for k in specs if k not in sets and (RUNS / k / "log.jsonl").exists()]
    long_done = sum(1 for _, st in long_statuses if st == "done")
    parts = [
        f"{len(rows)} worlds: {len(controls)} unmodified and {len(first_sets)} sets of 16 in the first screen" + (f", and {sum(len(v) for k, v in sets.items() if k.startswith('b1-'))} from the clean set" if clean_done else "") + ".",
        f"{long_done} of {len(long_statuses)} long runs and {len(ko_batches)} knockouts.",
    ]
    running = []
    if clean_done < len(specs):
        running.append(f"the clean one-rule set ({clean_done} of {len(specs)} batches in" + (f", {esc(clean_running[0])} now" if clean_running else "") + ")")
    waiting = [n for n, st in long_statuses if st != "done"]
    if waiting:
        now = [n for n, st in long_statuses if st == "running"]
        running.append(f"{len(waiting)} long runs ({esc(', '.join(waiting))}" + (f"; {esc(now[0])} now" if now else "") + ")")
    run_text = "Still running: " + " and ".join(running) + ". This page picks them up when it's rebuilt." if running else "Nothing is still running."
    return f"""
<aside class="status" aria-label="What this page covers">
<p><b>Covers</b> {" ".join(parts)}</p>
<p><b>{esc(run_text.split(":")[0])}{":" if ":" in run_text else ""}</b>{esc(run_text.split(":", 1)[1]) if ":" in run_text else ""}</p>
<p class="built">Built {esc(built)}.</p>
</aside>
"""


NAV = [("range", "Same physics, different worlds"), ("look", "Look at them"), ("sets", "Physics against physics"), ("kin", "Biters don't choose"), ("biting-off", "Biting off"), ("longer", "Five times longer"), ("false", "Looked real and wasn't"), ("next", "What's next")]


# ---- Page ------------------------------------------------------------------

CSS = r"""
:root{--page:#12141e;--raised:#1a1d2a;--rule:#2d3140;--ink:#eceae4;--ink2:#b4b7c2;--ink3:#8c90a0;--sun:#ffdc5a;--up:rgba(57,135,229,.20);--down:rgba(217,89,38,.22);
--serif:Charter,"Bitstream Charter","Iowan Old Style","Palatino Linotype",Palatino,"Noto Serif","Source Serif Pro",Georgia,"DejaVu Serif",serif;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,"Noto Sans","Liberation Sans",sans-serif;color-scheme:dark}
*{box-sizing:border-box}
html{background:var(--page);-webkit-text-size-adjust:100%;scroll-behavior:smooth;scroll-padding-top:12px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
body{margin:0;background:var(--page);color:var(--ink);font:400 17px/1.6 var(--sans);overflow-x:hidden}
[hidden]{display:none!important}
a{color:var(--ink);text-decoration-color:var(--sun);text-underline-offset:3px}
a:hover{color:var(--sun)}
:focus-visible{outline:2px solid var(--sun);outline-offset:2px}
main,.hero-text,.status,.toc{max-width:1240px;margin-inline:auto;padding-inline:20px}
h1,h2,h3{font-family:var(--serif);font-weight:600;line-height:1.16;letter-spacing:-.005em}
h1{font-size:clamp(1.85rem,4.1vw,3.05rem);max-width:21em;margin:1.05em 0 .55em}
h2{font-size:clamp(1.6rem,3vw,2.2rem);margin:0 0 .6em}
h3{font-size:clamp(1.2rem,2vw,1.45rem);margin:2.4em 0 .7em}
section{padding-top:clamp(52px,8vw,92px)}
.prose{max-width:68ch}.prose p{margin:0 0 1em}
.lead{max-width:62ch;font-size:clamp(1.03rem,1.6vw,1.17rem)}.lead p{margin:0 0 .85em}
strong{font-weight:650}
.note{color:var(--ink2);font-size:.92rem;max-width:68ch}
.band{display:grid;grid-template-columns:repeat(8,1fr);gap:2px;background:var(--page)}
.status{color:var(--ink2);font-size:.93rem}
.status p{margin:.2em 0;max-width:80ch}.status b{color:var(--ink);font-weight:600}.status .built{color:var(--ink3);font-size:.85rem}
.toc{margin-top:18px}.toc ul{display:flex;flex-wrap:wrap;gap:4px 18px;list-style:none;margin:0;padding:12px 0 0;border-top:1px solid var(--rule);font-size:.93rem}
.toc a{color:var(--ink2);text-decoration:none}.toc a:hover{color:var(--sun)}
/* Plates: one image with six panels, showing one panel or all of them. */
.plate{display:block;position:relative;overflow:hidden;aspect-ratio:1;background:#000}
.plate img{position:absolute;max-width:none;height:auto;width:303.125%;left:0;top:0;display:block}
[data-view="1"] .plate img,.plate[data-view="1"] img{left:-101.5625%}
[data-view="2"] .plate img,.plate[data-view="2"] img{left:-203.125%}
[data-view="3"] .plate img,.plate[data-view="3"] img{top:-101.5625%}
[data-view="4"] .plate img,.plate[data-view="4"] img{left:-101.5625%;top:-101.5625%}
[data-view="5"] .plate img,.plate[data-view="5"] img{left:-203.125%;top:-101.5625%}
[data-view="all"] .plate{aspect-ratio:776/516}
[data-view="all"] .plate img{width:100%;left:0;top:0}
a.plate{cursor:zoom-in}a.plate:hover{outline:2px solid var(--sun);outline-offset:-2px}
.plate.empty{display:grid;place-items:center;color:var(--ink3);font-size:.85rem;text-align:center;padding:8px}
/* Legend and gallery */
.panel-legend{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin:18px 0 8px}
.legend-tile figcaption{font-size:.84rem;line-height:1.35;color:var(--ink2);padding-top:6px}
.legend-tile b{display:block;color:var(--ink);font-weight:600}.legend-tile figure,.legend-tile{margin:0}
.controls{display:flex;flex-wrap:wrap;gap:10px 28px;margin:26px 0 8px;position:sticky;top:0;z-index:5;background:var(--page);padding:8px 0}
.seg{display:flex;flex-wrap:wrap;align-items:center;gap:4px}
.seg-label{color:var(--ink3);font-size:.86rem;margin-right:6px}
.seg button{font:inherit;font-size:.86rem;color:var(--ink2);background:transparent;border:1px solid var(--rule);border-radius:4px;padding:5px 10px;min-height:32px;cursor:pointer}
.seg button:hover{color:var(--ink);border-color:var(--ink3)}
.seg button[aria-pressed="true"]{color:#12141e;background:var(--ink);border-color:var(--ink);font-weight:600}
.wall{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:18px 10px;margin-top:12px}
.tile{margin:0}.tile figcaption{display:grid;grid-template-columns:1fr 1fr;gap:0 8px;font-size:.8rem;line-height:1.4;color:var(--ink2);padding-top:6px}
.tile figcaption b{grid-column:1/-1;color:var(--ink);font-weight:600;font-size:.84rem}
[data-view="all"] .wall{grid-template-columns:repeat(auto-fill,minmax(300px,1fr))}
.odds{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:30px 18px;margin-top:12px}
.odd{margin:0}.odd figcaption{padding-top:8px;font-size:.92rem;line-height:1.45}
.odd b{display:block;font-weight:600}.odd .rules{display:block;color:var(--ink3);font-size:.82rem;line-height:1.35;margin:.1em 0 .4em}
.odd .say{display:block}.odd .nums{display:block;color:var(--ink2);font-size:.82rem;margin-top:.45em}
/* Charts */
.chart{margin:0;min-width:0}
.chart figcaption{padding:0 0 8px;font-size:.88rem;line-height:1.4;color:var(--ink2);min-height:0}
.chart figcaption b{display:block;color:var(--ink);font-weight:600;font-size:.98rem}
.chart figcaption span{display:block}.chart .stat{color:var(--ink);margin-top:2px}
.chart-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:30px 34px;margin:24px 0}
.chart-grid.three{grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr))}
.chart-pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,340px),1fr));gap:30px 34px;margin:20px 0}
.two{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);gap:30px 44px;align-items:start}
.wide{max-width:600px}
svg{display:block;width:100%;height:auto;overflow:visible;font-family:var(--sans)}
svg text{fill:var(--ink2);font-size:12.5px}
svg .ax{fill:#a6a9b6;font-variant-numeric:tabular-nums}svg .small{font-size:11px}
svg .lab{fill:var(--ink);font-size:12px}
svg .base{stroke:#5a5e6e;stroke-width:1}svg .tick{stroke:#5a5e6e}svg .grid{stroke:var(--rule);stroke-width:1}
svg .median{stroke:var(--ink2);stroke-width:1}svg .leader{stroke:var(--ink3);stroke-width:.8}
svg .identity{stroke:var(--ink2);stroke-width:1.2}svg .link{stroke:#5a5e6e;stroke-width:1.5}
svg .ln{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
svg .sl{stroke-width:1.6;stroke-linecap:round;opacity:.9}
svg .pt{stroke:var(--page);stroke-width:1.5}
svg .pt.on{stroke:var(--sun);stroke-width:2.4;r:6px}svg .sl.on{stroke-width:3;opacity:1}
svg.plot{touch-action:pan-y}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;font-size:.88rem;color:var(--ink2);margin:.5em 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:baseline}
#tip{position:fixed;z-index:50;pointer-events:none;background:#000;border:1px solid var(--rule);border-radius:4px;padding:5px 9px;font-size:.84rem;line-height:1.35;max-width:280px;color:var(--ink2)}
#tip b{display:block;color:var(--ink);font-weight:650}
/* Tables */
details.twin{margin:18px 0 0;font-size:.9rem;color:var(--ink2)}
details.twin summary{cursor:pointer;padding:6px 0}
.scroll{overflow-x:auto;max-height:440px;overflow-y:auto;border:1px solid var(--rule);margin-top:8px}
.scroll table{border-collapse:collapse;width:100%;font-size:.85rem}
.scroll th{position:sticky;top:0;background:var(--raised);color:var(--ink);text-align:left;font-weight:600;padding:6px 10px;white-space:nowrap}
.scroll td{padding:4px 10px;border-top:1px solid var(--rule);white-space:nowrap;font-variant-numeric:tabular-nums}
.wlink{font:inherit;color:inherit;background:none;border:0;padding:0;text-decoration:underline;text-decoration-color:var(--ink3);text-underline-offset:2px;cursor:pointer}
.wlink:hover{color:var(--sun)}
.settable table{border-collapse:collapse;width:100%;font-size:.88rem;line-height:1.3}
.settable th{font-weight:600;text-align:left;padding:8px 8px;color:var(--ink);border-bottom:1px solid #5a5e6e;vertical-align:bottom}
.settable td{padding:9px 8px;border-bottom:1px solid var(--rule);vertical-align:top}
.settable .n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.settable .alive{text-align:right;font-variant-numeric:tabular-nums}
.settable .up{background:var(--up);font-weight:650}.settable .down{background:var(--down);font-weight:650}
.settable .none{color:var(--ink3)}
.sid{display:block;font-weight:600;white-space:nowrap}.rw{display:block;color:var(--ink2);font-size:.82rem;max-width:46ch}
.more{color:var(--ink3)}
.key .k{display:inline-block;padding:0 5px;border-radius:2px}.k.up{background:var(--up)}.k.down{background:var(--down)}
.readbtn{font:inherit;font-size:.8rem;color:var(--ink2);background:none;border:1px solid var(--rule);border-radius:4px;padding:3px 8px;cursor:pointer;min-height:28px}
.readbtn:hover{color:var(--ink);border-color:var(--ink3)}.readbtn[aria-expanded="true"]{color:var(--page);background:var(--ink2)}
.readrow td{background:var(--raised)}
.read{max-width:80ch;font-size:.92rem;line-height:1.5;padding:6px 4px}.read p{margin:0 0 .7em}.read h4{margin:.8em 0 .2em;font-size:.95rem;font-weight:650}
.read .item{padding-left:1em}
/* Before and after */
.pairs{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,420px),1fr));gap:30px 26px;margin-top:12px}
.pair{margin:0}.pair figcaption{font-size:.86rem;color:var(--ink2);line-height:1.4;padding-bottom:6px;min-height:2.8em}
.pair figcaption b{color:var(--ink);font-weight:600}
.pair-plates{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.when{display:block;font-size:.78rem;color:var(--ink3);padding-top:3px}
[data-view="all"] .pairs{grid-template-columns:1fr}
.falses{list-style:none;margin:0;padding:0;max-width:72ch}.falses li{padding:0 0 1.1em;margin:0}.falses b{font-weight:650;display:block;font-family:var(--serif);font-size:1.12rem;margin-bottom:.15em}
footer{color:var(--ink3);font-size:.85rem;max-width:1240px;margin:70px auto 0;padding:20px 20px 40px;border-top:1px solid var(--rule)}
/* Enlarged view */
.lb{position:fixed;inset:0;z-index:40;background:#07080c;display:flex;flex-direction:column;overflow-y:auto}
.lb-bar{display:flex;align-items:center;gap:8px;padding:10px 14px;flex:none}
.lb-title{font-weight:600;margin-right:auto;font-size:.98rem;min-width:0}
.lb button{font:inherit;font-size:.88rem;color:var(--ink);background:var(--raised);border:1px solid var(--rule);border-radius:4px;padding:6px 12px;min-height:36px;cursor:pointer}
.lb button:hover{border-color:var(--ink3)}
.lb .seg button[aria-pressed="true"]{color:#12141e;background:var(--ink)}
.lb-stage{flex:1 1 auto;display:flex;justify-content:center;align-items:flex-start;padding:0 14px;min-height:0}
.lb-stage .plate{width:100%;max-width:calc(100dvh - 190px);min-width:min(100%,280px)}
.lb[data-view="all"] .lb-stage .plate{max-width:calc((100dvh - 190px) * 1.504)}
.lb-stage img{image-rendering:pixelated}
.lb .seg{justify-content:center;padding:10px 14px 2px;flex:none}
.lb-cap{max-width:80ch;margin:6px auto 16px;padding:0 14px;color:var(--ink2);font-size:.92rem;line-height:1.45;flex:none}
/* Small screens */
@media (max-width:980px){.two{grid-template-columns:1fr}.panel-legend{grid-template-columns:repeat(3,1fr)}.wide{max-width:560px}}
@media (max-width:760px){
 body{font-size:16px}
 .band{grid-template-columns:repeat(4,1fr)}.band .plate:nth-child(n+5){display:none}
 .wall{grid-template-columns:repeat(2,1fr);gap:16px 8px}
 [data-view="all"] .wall{grid-template-columns:1fr}
 .odds{grid-template-columns:1fr}
 .panel-legend{grid-template-columns:repeat(2,1fr)}
 .controls{gap:6px;position:static}
 .seg button{padding:5px 8px}
 .settable table,.settable tbody,.settable tr,.settable td{display:block;width:100%}
 .settable thead{display:none}
 .settable tr{display:grid;grid-template-columns:repeat(4,1fr);gap:6px 8px;padding:12px 0;border-bottom:1px solid var(--rule)}
 .settable tr.readrow{display:block;padding:0;border:0}
 .settable td{border:0;padding:0;text-align:left!important}
 .settable td.setname{grid-column:1/-1}
 .settable td.none{grid-column:1/-1}
 .settable td.action{grid-column:1/-1}
 .settable td.action:empty{display:none}
 .settable td[data-l]:not(.setname):not(.none):not(.action)::before{content:attr(data-l);display:block;color:var(--ink3);font-size:.72rem;line-height:1.2;white-space:normal;margin-bottom:1px}
 .settable .up,.settable .down{padding:2px 4px;margin:-2px -4px}
 .readrow td{padding:10px!important}
 .pair figcaption{min-height:0}
 svg text{font-size:14.5px}svg .ax{font-size:14px}svg .lab{font-size:14px}svg .small{font-size:11px}
 .lb-stage .plate{max-width:100%}
 .lb[data-view="all"] .lb-stage .plate{max-width:100%}
}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
"""

JS = r"""
(function(){
var worlds=JSON.parse(document.getElementById('worlds').textContent);
var tip=document.getElementById('tip');
function showTip(text,x,y){var p=text.split('|');tip.textContent='';var b=document.createElement('b');b.textContent=p[0];tip.appendChild(b);if(p[1]){tip.appendChild(document.createTextNode(p[1]));}tip.hidden=false;var r=tip.getBoundingClientRect();var L=Math.min(Math.max(8,x+14),innerWidth-r.width-8),T=y-r.height-12;if(T<8)T=y+18;tip.style.left=L+'px';tip.style.top=T+'px';}
function hideTip(){tip.hidden=true;}
function mark(run,on){if(!run)return;document.querySelectorAll('.pt[data-run="'+run+'"],.sl[data-run="'+run+'"]').forEach(function(n){n.classList.toggle('on',on);if(on&&n.parentNode&&n.tagName==='circle')n.parentNode.appendChild(n);});}
document.querySelectorAll('svg.plot').forEach(function(svg){
  var pts=[].slice.call(svg.querySelectorAll('.pt'));if(!pts.length)return;var cur=null;
  function nearest(ev){var ctm=svg.getScreenCTM();if(!ctm)return null;var best=null,bd=676;pts.forEach(function(p){var x=ctm.a*p.cx.baseVal.value+ctm.e,y=ctm.d*p.cy.baseVal.value+ctm.f,d=(x-ev.clientX)*(x-ev.clientX)+(y-ev.clientY)*(y-ev.clientY);if(d<bd){bd=d;best=p;}});return best;}
  svg.addEventListener('pointermove',function(ev){var p=nearest(ev);if(p!==cur){if(cur)mark(cur.getAttribute('data-run'),false);cur=p;if(p)mark(p.getAttribute('data-run'),true);}if(p){showTip(p.getAttribute('data-tip'),ev.clientX,ev.clientY);svg.style.cursor=worlds[p.getAttribute('data-run')]?'pointer':'default';}else{hideTip();svg.style.cursor='default';}});
  svg.addEventListener('pointerleave',function(){if(cur)mark(cur.getAttribute('data-run'),false);cur=null;hideTip();});
  svg.addEventListener('click',function(ev){var p=nearest(ev);if(p&&worlds[p.getAttribute('data-run')]){hideTip();openWorld(p.getAttribute('data-run'),null,null);}});
});
/* Which panel to show, per gallery */
document.querySelectorAll('.viewroot').forEach(function(root){root.querySelectorAll('.viewbtn').forEach(function(b){b.addEventListener('click',function(){root.setAttribute('data-view',b.getAttribute('data-v'));root.querySelectorAll('.viewbtn').forEach(function(o){o.setAttribute('aria-pressed',o===b?'true':'false');});});});});
/* Sorting the wall */
document.querySelectorAll('.sortbtn').forEach(function(b){b.addEventListener('click',function(){var k=b.getAttribute('data-k'),wall=document.getElementById('wall');var t=[].slice.call(wall.children);t.sort(function(x,y){return parseFloat(y.getAttribute('data-'+k))-parseFloat(x.getAttribute('data-'+k));});t.forEach(function(n){wall.appendChild(n);});document.querySelectorAll('.sortbtn').forEach(function(o){o.setAttribute('aria-pressed',o===b?'true':'false');});});});
/* Reads */
document.querySelectorAll('.readbtn').forEach(function(b){b.addEventListener('click',function(){var r=document.getElementById(b.getAttribute('aria-controls'));var open=r.hidden;r.hidden=!open;b.setAttribute('aria-expanded',open?'true':'false');b.textContent=open?'Close':'Read';});});
/* Enlarged view */
var lb=document.getElementById('lb'),lbPlate=document.getElementById('lb-plate'),lbImg=document.getElementById('lb-img'),lbTitle=document.getElementById('lb-title'),lbCap=document.getElementById('lb-cap');
var list=[],pos=0,lastFocus=null,loading=0;
function setView(v){lb.setAttribute('data-view',v);lb.querySelectorAll('.lbview').forEach(function(o){o.setAttribute('aria-pressed',o.getAttribute('data-v')===v?'true':'false');});}
lb.querySelectorAll('.lbview').forEach(function(b){b.addEventListener('click',function(){setView(b.getAttribute('data-v'));});});
function show(run){var w=worlds[run];if(!w)return;lbTitle.textContent=w.title;lbCap.textContent=w.cap;lbImg.alt='Map of '+run;lbImg.src=w.t;var token=++loading;var pre=new Image();pre.onload=function(){if(token===loading)lbImg.src=w.l;};pre.src=w.l;
  document.getElementById('lb-prev').hidden=document.getElementById('lb-next').hidden=list.length<2;}
function openWorld(run,gallery,view){var runs=[];if(gallery){document.querySelectorAll('a[data-world][data-gallery="'+gallery+'"]').forEach(function(a){runs.push(a.getAttribute('data-world'));});}
  if(!runs.length){document.querySelectorAll('a[data-world][data-gallery="base"]').forEach(function(a){runs.push(a.getAttribute('data-world'));});}
  if(runs.indexOf(run)<0)runs=[run];list=runs;pos=runs.indexOf(run);
  lastFocus=document.activeElement;lb.hidden=false;document.body.style.overflow='hidden';setView(view||(innerWidth<760?'0':'all'));show(run);document.getElementById('lb-close').focus();}
function step(d){if(!list.length)return;pos=(pos+d+list.length)%list.length;show(list[pos]);}
function closeLb(){lb.hidden=true;document.body.style.overflow='';lbImg.src='';if(lastFocus&&lastFocus.focus)lastFocus.focus();}
document.getElementById('lb-close').addEventListener('click',closeLb);
document.getElementById('lb-prev').addEventListener('click',function(){step(-1);});
document.getElementById('lb-next').addEventListener('click',function(){step(1);});
document.addEventListener('keydown',function(e){if(lb.hidden)return;if(e.key==='Escape')closeLb();else if(e.key==='ArrowLeft')step(-1);else if(e.key==='ArrowRight')step(1);});
var sx=null;lbPlate.addEventListener('touchstart',function(e){sx=e.touches[0].clientX;},{passive:true});lbPlate.addEventListener('touchend',function(e){if(sx===null)return;var dx=e.changedTouches[0].clientX-sx;sx=null;if(Math.abs(dx)>50)step(dx<0?1:-1);},{passive:true});
document.addEventListener('click',function(e){var a=e.target.closest&&e.target.closest('a[data-world]');if(a){e.preventDefault();var root=a.closest('.viewroot');var v=root?root.getAttribute('data-view'):null;openWorld(a.getAttribute('data-world'),a.getAttribute('data-gallery'),v);return;}
  var w=e.target.closest&&e.target.closest('.wlink');if(w&&worlds[w.getAttribute('data-run')]){openWorld(w.getAttribute('data-run'),null,null);}});
document.querySelectorAll('.wlink').forEach(function(w){if(!worlds[w.getAttribute('data-run')]){var s=document.createElement('span');s.textContent=w.textContent;w.parentNode.replaceChild(s,w);}});
})();
"""


def lightbox():
    tabs = "".join(f'<button type="button" class="lbview" data-v="{v}" aria-pressed="false">{n}</button>' for v, n in VIEWS)
    return f"""
<div id="lb" class="lb" hidden data-view="all" role="dialog" aria-modal="true" aria-label="Enlarged world map">
<div class="lb-bar"><span class="lb-title" id="lb-title"></span><button type="button" id="lb-prev">Previous</button><button type="button" id="lb-next">Next</button><button type="button" id="lb-close">Close</button></div>
<div class="lb-stage"><div class="plate" id="lb-plate"><img id="lb-img" alt=""></div></div>
<div class="seg" role="group" aria-label="Which panel to show">{tabs}</div>
<p class="lb-cap" id="lb-cap"></p>
</div>
"""


def main():
    rows, long_rows, kin, kin_long, reads = load()
    rows = [r for r in rows if "index" in r]
    sets = by_batch(rows)
    screen_rows = {r["run"]: r for r in rows}
    long_batches = by_batch(long_rows)
    controls = [r for r in rows if not r.get("physics")]
    control_alive = cards.alive(controls)
    first_sets = {k: v for k, v in sets.items() if not k.startswith("b1-") and v[0].get("physics")}
    specs = clean_specs()

    long_status = []
    for d in sorted(LONG_RUNS.glob("long-*")):
        long_status.append((d.name, batch_state(d, d.name in long_batches)))
    for b in long_batches:
        if b.startswith("long-") and b not in {n for n, _ in long_status}:
            long_status.append((b, "done"))
    ko_batches = [b for b in long_batches if b.startswith("ko-")]
    plain_long = {b for b, rs in long_batches.items() if b.startswith("long-") and not rs[0].get("physics")}
    kin_long = [k for k in kin_long if k.get("batch") in plain_long]

    # Which worlds get thumbnails.
    scores = difference_scores(control_alive)
    candidates = {run: fn for run, fn in NOTES.items() if run in screen_rows}
    for batch in sorted(reads):
        if batch in sets and not any(run.startswith(batch + "-") for run in candidates):
            found = first_candidate(reads[batch])
            if found:
                candidates[f"{batch}-t{found[0]:03d}"] = found[1]
    plates = Plates()
    wanted = []
    for r in control_alive:
        wanted.append(r["run"])
        plates.add(r["run"], f'{r["run"]}, unmodified physics', cap(world_stats(r)) + f'. Species {trim(m(r, "species at the end"))}, {trim(m(r, "behaviour dimensions"))} behaviour dimensions.')
    for run in candidates:
        r = screen_rows[run]
        on, _ = rule_words(r.get("physics") or {})
        wanted.append(run)
        plates.add(run, run, cap(world_stats(r)) + ".")
    for b, rs in long_batches.items():
        if b.startswith("long-"):
            for a in rs:
                before = screen_rows.get(f'{b[5:]}-t{a["index"]:03d}')
                if before:
                    wanted += [a["run"], before["run"]]
                    plates.add(a["run"], f'{a["run"]} at {a["tick"]:,} ticks', cap(world_stats(a)) + ".")
                    plates.add(before["run"], f'{before["run"]} at {before["tick"]:,} ticks', cap(world_stats(before)) + ".")
    wanted = list(dict.fromkeys(wanted))
    for batch in sorted({run.rsplit("-t", 1)[0] for run in wanted}):
        render_maps(batch)
    have, total_media = build_media(wanted)
    plates.have = have
    band = [r["run"] for r in sorted(control_alive, key=lambda r: -scores[r["run"]]) if r["run"] in have][:8]

    built = datetime.now().strftime("%-d %B %Y, %H:%M")
    unusual = section_unusual(screen_rows, reads, control_alive, plates, candidates)
    body = [
        section_answer(rows, controls, control_alive, first_sets, kin, long_batches, specs, sets, band),
        status_block(rows, controls, first_sets, long_rows, specs, sets, long_status, ko_batches, have, total_media, built),
        '<nav class="toc" aria-label="Sections"><ul>' + "".join(f'<li><a href="#{a}">{esc(n)}</a></li>' for a, n in NAV) + "</ul></nav>",
        "<main>",
        section_range(controls, control_alive, plates),
        section_look(control_alive, scores, plates, unusual),
        section_sets(sets, specs, controls, control_alive, reads, plates),
        section_kin(kin, kin_long),
        section_knockout(long_batches, screen_rows, plates),
        section_long(long_batches, screen_rows, plates, long_status),
        section_false(rows, controls, long_batches),
        section_next(specs, sets),
        "</main>",
        f'<footer>Built from {esc(LIVE.name)} results by scripts/make_screen_report.py. Every number except the mineral densities is worked out from the data at build time; the mineral ones were measured once from saved worlds.</footer>',
    ]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#12141e">
<title>What the artificial-life screen found</title>
<style>{CSS}</style></head>
<body>
{"".join(body)}
<div id="tip" hidden></div>
{lightbox()}
<script type="application/json" id="worlds">{plates.json()}</script>
<script>{JS}</script>
</body></html>
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen.html").write_text(page)
    print(f"wrote {OUT / 'screen.html'}: {len(page) / 1e3:.0f} kB page, {len(rows)} worlds, {len(have)} worlds with maps, {total_media / 1e6:.1f} MB media")


if __name__ == "__main__":
    main()
