#!/usr/bin/env python3
"""The hand calculations behind PREDICTIONS.md, with the real constants from defaults.json.
No simulation of life, no brains: only the arithmetic of the rules. Run it with no arguments."""
import json
import math
from pathlib import Path

D = json.loads((Path(__file__).resolve().parents[2] / "crates/auto-life-gpu/shaders/defaults.json").read_text())
SPEND, COMMIT, POWER, YIELD = D["spend"], D["commitment"], D["bite_power"], D["bite_yield"]
UPKEEP, UPKEEP_E = D["upkeep"], D["upkeep_e"]


def work(a):
    return a * a / (a + COMMIT)


def fight(n, e0, te0, blockers, rule, sharp=2.0, share=0.8, back=False, ticks=400):
    """n attackers spending `share` of their budget on biting one target with `blockers` passive
    neighbours. Returns (ticks, energy each attacker gained per tick, total each gained, target killed)."""
    a = SPEND * share
    w = work(a)
    es = [e0] * n
    te = te0
    gained = 0.0
    last = [False] * n
    for t in range(1, ticks + 1):
        next_last = [False] * n
        for i in range(n):
            if es[i] <= 0 or te <= 0:
                continue
            flagged = sum(1 for j in range(n) if j != i and last[j] and es[j] > 0)
            empties = 8 - blockers - n
            if rule == "old":
                sides = 1 + empties
                force = 2 * es[i] / (es[i] + te + 1e-3)
            else:
                sides = 1 + empties + flagged
                s = es[i] + sum(es[j] for j in range(n) if j != i and last[j] and es[j] > 0)
                force = 2 * s**sharp / (s**sharp + te**sharp + 1e-3)
            take = min(w * POWER * force * sides / 8, te)
            te -= take
            keep = take * YIELD
            net = keep - a - (UPKEEP + UPKEEP_E * es[i])
            es[i] += net
            es[i] = min(es[i], D["max_energy"] * 2)
            gained += net
            next_last[i] = take > 0
        if back and te > 0:
            live = [i for i in range(n) if es[i] > 0]
            if live:
                k = live[0]
                f = 2 * te**sharp / (te**sharp + es[k] ** sharp + 1e-3) if rule != "old" else 2 * te / (te + es[k] + 1e-3)
                open_ = 1.0 if n == 1 else 0.75
                hit = min(w * POWER * f * open_, es[k])
                es[k] -= hit
                gained -= hit
                te += hit * YIELD - a - (UPKEEP + UPKEEP_E * te)
        last = next_last
        if te <= 0 or not any(e > 0 for e in es):
            break
    return t, gained / n / t, gained / n, te <= 0


def table_gang():
    print("## Rule 1: per head, a fight to the end. Attackers start at 5 energy, spend 80% of budget on biting.")
    print("Each cell: ticks / net energy per head per tick. Total per head in brackets. K = target died.\n")
    for label, te, blockers in [
        ("fat alone, 40", 40, 0), ("very fat alone, 80", 80, 0), ("lean alone, 5", 5, 0),
        ("fat on a colony's edge (4 sides walled), 40", 40, 4), ("very fat on a colony's edge (4 walled), 80", 80, 4),
        ("fat inside a clump (6 walled), 40", 40, 6),
    ]:
        print(f"**{label}**\n")
        print("| Rule | " + " | ".join(f"N={n}" for n in (1, 2, 3, 4, 6, 8) if blockers + n <= 8) + " |")
        print("|---|" + "---|" * sum(1 for n in (1, 2, 3, 4, 6, 8) if blockers + n <= 8))
        for name, rule, sharp, back in [
            ("today", "old", 1, False), ("sum, sharp 1", "new", 1, False), ("sum, sharp 2", "new", 2, False),
            ("today, bites back", "old", 1, True), ("sum, sharp 2, bites back", "new", 2, True),
        ]:
            cells = []
            for n in (1, 2, 3, 4, 6, 8):
                if blockers + n > 8:
                    continue
                t, rate, total, killed = fight(n, 5.0, te, blockers, rule, sharp, back=back)
                cells.append(f"{t}{'K' if killed else ''} / {rate:+.2f} ({total:+.1f})")
            print(f"| {name} | " + " | ".join(cells) + " |")
        print()


def table_first_bite():
    print("## Rule 1: one tick's take, per head, by N. Attackers already biting, target with all other sides empty.\n")
    a = SPEND * 0.8
    w = work(a)
    print("Break-even keeps `a` = %.2f, so take must pass %.2f. Full take at force 1 is %.2f.\n" % (a, a / YIELD, w * POWER))
    print("| Attackers' energy each, target | sharp | " + " | ".join(f"N={n}" for n in (1, 2, 3, 4, 6, 8)) + " |")
    print("|---|---|" + "---|" * 6)
    for e, te in [(5, 5), (5, 40), (5, 80), (2, 40), (20, 40)]:
        for sharp in (1, 2):
            row = []
            for n in (1, 2, 3, 4, 6, 8):
                s = e * n
                f = 2 * s**sharp / (s**sharp + te**sharp)
                row.append(f"{w * POWER * f:.2f}")
            print(f"| {e}, {te} | {sharp} | " + " | ".join(row) + " |")
    print()


def fit(m, sharp, floor):
    return floor + (1 - floor) * m**sharp


def match(d):
    return 0.5 + 0.5 * math.cos(2 * math.pi * d)


def mean_fit(sharp, floor, steps=720):
    return sum(fit(match(i / steps), sharp, floor) for i in range(steps)) / steps


def table_coat():
    print("## Rule 2: what a bite is worth against coats\n")
    a = SPEND * 0.8
    w = work(a)
    full = w * POWER * YIELD
    print(f"Biter keeps {full:.2f} a tick at force 1, open 1, perfect match, and pays {a:.2f}. So it needs fit of at least {a / full:.3f} in the open, and {a / full * 8:.2f} through a clump's 1 open side in 8.\n")
    print("| sharp | half width in turns | floor | fit against evenly spread coats | best key on a monoculture | gain from tracking | open-air profit, mismatched | clump profit at match |")
    print("|---|---|---|---|---|---|---|---|")
    for sharp, floor in [(1, 0.05), (2, 0.05), (4, 0.05), (8, 0.05), (16, 0.05), (32, 0.05), (8, 0.0), (8, 0.15), (8, 0.3)]:
        half = math.acos(2 * 0.5 ** (1 / sharp) - 1) / (2 * math.pi)
        mean = mean_fit(sharp, floor)
        print(f"| {sharp} | {half:.3f} | {floor} | {mean:.3f} | 1 | {1 / mean:.1f}x | {full * floor - a:+.2f} | {full / 8 - a:+.2f} |")
    print()
    print("Kill the winner. Shares of two coats half a turn apart, sharp 8, floor 0.05. Biters track the common coat.\n")
    print("| Common coat's share | Biter's average fit, keyed on common | keyed on rare | Damage per body on common | on rare | Common coat suffers |")
    print("|---|---|---|---|---|---|")
    far = fit(match(0.5), 8, 0.05)
    for p in (0.5, 0.7, 0.9, 0.99):
        on_common = p * 1 + (1 - p) * far
        on_rare = (1 - p) * 1 + p * far
        print(f"| {p} | {on_common:.2f} | {on_rare:.2f} | 1 | {far:.2f} | {1 / far:.0f}x |")
    print()
    rate, cost = D.get("coat_rate", 0.002), D.get("coat_cost", 2.0)
    print(f"Moving coat: at {rate} turns a tick, half a turn takes {0.5 / rate:.0f} ticks and costs {0.5 * cost:.2f} energy, at most {rate * cost:.4f} a tick against upkeep {UPKEEP}.\n")


def table_mineral():
    print("## Rule 3: where mineral limits births\n")
    # Work needed so that move_chance = work * move_rate = 1 for one step a tick, and so that
    # uptake = work * mineral_rate = 1 clears a square in one tick. The budget can't do both in one tick.
    def energy_for(target):
        return (target + math.sqrt(target * target + 4 * target * COMMIT)) / 2
    step = energy_for(1 / D["move_rate"])
    up = energy_for(1 / D.get("mineral_rate", 6))
    a = step + up
    print(f"A step costs {step:.3f} energy and clearing a square {up:.3f}, {a:.3f} together, from a budget of {SPEND} a tick, so a walker gathers about every second tick.\n")
    print("Stock limit: every living cell holds `mineral_birth` in its body, so density can't pass start / birth.\n")
    print("Price of mineral for a walker, in energy per child, with free ground mineral g = start - density * birth left on a square, walking to a fresh square each tick. A child's energy price is 5.\n")
    print("| start / birth | density | g per square, at birth 1 | squares walked per child | energy per child's mineral | Which limits |")
    print("|---|---|---|---|---|---|")
    for cap in (0.15, 0.25, 0.5, 1.0):
        for dens_share in (0.3, 0.7, 0.9, 0.97):
            d = cap * dens_share
            g = cap - d
            squares = 1 / g
            price = squares * a
            which = "mineral" if price > 5 else "energy"
            print(f"| {cap} | {d:.3f} | {g:.3f} | {squares:.1f} | {price:.1f} | {which} |")
    print()
    print("Income from sun, net of upkeep, for a cell eating all it can. Ticks to fill a womb of 5, against ticks to gather 1 mineral sitting still.\n")
    flow = D.get("mineral_flow", 0.005)
    print(f"Sitting still, mineral arrives only by flow. Each square trades with 2 partners a tick, one per axis. A stripped square gets {flow} x g / 2 from each at equal richness, so {flow:.3f} x g a tick.\n")
    print("| Sun on this square | Net energy a tick | Ticks to fill a womb | Sitter's ticks per mineral, g=0.1 | g=0.3 | g=1 | Sitter is limited by |")
    print("|---|---|---|---|---|---|---|")
    for sun in (0.03, 0.05, 0.1, 0.2, 0.4, 1.0):
        net = sun - UPKEEP - UPKEEP_E * 10
        te = 5 / net if net > 0 else math.inf
        tm = [1 / (flow * g) for g in (0.1, 0.3, 1.0)]
        which = "mineral" if tm[1] > te else "energy"
        print(f"| {sun} | {net:.3f} | {te:.0f} | {tm[0]:.0f} | {tm[1]:.0f} | {tm[2]:.0f} | {which} at g=0.3 |")
    print()


def table_life():
    print("## Rule 4: child size and resting\n")
    print("| Child's energy | Womb to fill | Share reaching the child | Children per 100 energy | Ticks a child lives unfed | Force, lone, against a 5 energy neighbour |")
    print("|---|---|---|---|---|---|")
    for k in (0.5, 1, 2, 4, 8, 16, 32):
        womb = k + D["body"]
        unfed = k / (UPKEEP + UPKEEP_E * k)
        f = 2 * k**2 / (k**2 + 25)
        print(f"| {k} | {womb} | {k / womb:.2f} | {100 / womb:.0f} | {unfed:.0f} | {f:.2f} |")
    print()
    hazard, age_rest = D["hazard"], D.get("age_rest", 0.05)
    print(f"Old-age hazard is {hazard} times `age_rest + (1 - age_rest) * spent / spend`, age_rest {age_rest}.\n")
    print("| Rest share | Spending a tick | Upkeep a tick, at 10 energy | Hazard factor | Expected life, ticks | Today's life |")
    print("|---|---|---|---|---|---|")
    for rest in (0, 0.25, 0.5, 0.75, 0.9, 0.99):
        spent = SPEND * (1 - rest)
        up = (UPKEEP + UPKEEP_E * 10) * (1 - D["rest_saving"] * rest)
        factor = age_rest + (1 - age_rest) * (1 - rest)
        print(f"| {rest} | {spent:.3f} | {up:.4f} | {factor:.3f} | {1 / (hazard * factor):.0f} | {1 / hazard:.0f} |")
    print()


if __name__ == "__main__":
    table_first_bite()
    table_gang()
    table_coat()
    table_mineral()
    table_life()
