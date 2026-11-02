#!/usr/bin/env python3
"""Supply and demand for soft cells, worked out before any run. Answers two questions with the
numbers the first engine starts from: how big must a group be before staying together pays, and
how far does a group's heat reach? Steady state of a diffusing field on a torus, by iteration."""
import math

SIZE = 96


def field(cells, q, flow, decay, rounds=2500):
    """Steady field from sources of q per tick per cell, with neighbour exchange flow and loss decay."""
    f = [[0.0] * SIZE for _ in range(SIZE)]
    src = [[0.0] * SIZE for _ in range(SIZE)]
    for x, y in cells:
        src[y][x] += q
    for _ in range(rounds):
        g = [[0.0] * SIZE for _ in range(SIZE)]
        for y in range(SIZE):
            row, up, down = f[y], f[y - 1], f[(y + 1) % SIZE]
            for x in range(SIZE):
                c = row[x]
                g[y][x] = (c + flow * (row[x - 1] + row[(x + 1) % SIZE] + up[x] + down[x] - 4 * c)) * (1 - decay) + src[y][x]
        f = g
    return f


def disc(n):
    """About n cells at one per square, in a disc at the middle."""
    if n <= 1:
        return [(SIZE // 2, SIZE // 2)]
    r = math.sqrt(n / math.pi)
    m = SIZE // 2
    return [(x, y) for y in range(SIZE) for x in range(SIZE) if (x - m) ** 2 + (y - m) ** 2 <= r * r]


# The starting physics, per tick. Sun supply on a patch is what limits density: 1 cell per square
# needs supply >= upkeep, so bodies can only sit where SUN_RICH >= UPKEEP.
EAT_MAX = 0.20      # what a cell can take with full access to food under it
LOCKED = 0.90       # share of food a cell can't reach without enzyme
ENZYME_HALF = 1.0   # enzyme level giving half of the locked share
ENZYME_COST = 0.004 # energy per tick to keep secreting at the rate below
ENZYME_RATE = 0.05  # enzyme made per tick at that budget
ENZYME_FLOW, ENZYME_DECAY = 0.10, 0.02
UPKEEP = 0.012      # everything that isn't enzyme, per tick, for a cell of ordinary size
HEAT_RATE, HEAT_FLOW, HEAT_DECAY = 0.02, 0.15, 0.05
HEAT_HARM = 0.10    # energy lost per tick at heat 1 by a cell with no reserves
HEAT_SCALE = 8.0    # energy at which the harm from heat has halved

print("Does staying together pay? One cell per square, all secreting, food not limiting.")
print(f"{'cells':>6} {'radius':>7} {'enzyme mid':>11} {'edge':>7} {'access mid':>11} {'edge':>7} {'net mid':>9} {'edge':>8}")
for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    cells = disc(n)
    z = field(cells, ENZYME_RATE, ENZYME_FLOW, ENZYME_DECAY)
    m = SIZE // 2
    r = int(math.sqrt(len(cells) / math.pi))
    mid, edge = z[m][m], z[m][m + r]
    access = lambda c: (1 - LOCKED) + LOCKED * c / (c + ENZYME_HALF)
    net = lambda c: EAT_MAX * access(c) - UPKEEP - ENZYME_COST
    print(f"{len(cells):6d} {r:7d} {mid:11.2f} {edge:7.2f} {access(mid):11.2f} {access(edge):7.2f} {net(mid):9.3f} {net(edge):8.3f}")
free = EAT_MAX * (1 - LOCKED) - UPKEEP
print(f"A lone cell that makes no enzyme nets {free:+.3f} a tick, so it can live, slowly.")
print(f"Food sets density: a cell at full access could eat {EAT_MAX} but a body can't take more than")
print(f"the supply under it. It needs sun >= {UPKEEP + ENZYME_COST:.3f} per square at 1 cell per square, so bodies fit rich patches.")

print("\nHow far does a group's heat reach? Everyone in the group emitting.")
print(f"{'cells':>6} {'heat mid':>9} {'+2':>7} {'+5':>7} {'+10':>7}   harm to a thin cell (energy 2) at +2, +5, +10 / to a rich cell (energy 30) mid")
for n in (1, 16, 64, 256, 1024):
    cells = disc(n)
    h = field(cells, HEAT_RATE, HEAT_FLOW, HEAT_DECAY)
    m = SIZE // 2
    r = int(math.sqrt(len(cells) / math.pi))
    at = lambda d: h[m][(m + r + d) % SIZE]
    harm = lambda v, e: HEAT_HARM * v / (1 + e / HEAT_SCALE)
    print(f"{len(cells):6d} {h[m][m]:9.2f} {at(2):7.2f} {at(5):7.2f} {at(10):7.2f}   {harm(at(2), 2):.3f} {harm(at(5), 2):.3f} {harm(at(10), 2):.3f} / {harm(h[m][m], 30):.3f}")
print(f"A thin cell nets at most {free:+.3f} a tick from food, so it dies where harm exceeds that: the moat.")
