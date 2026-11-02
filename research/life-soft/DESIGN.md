# Soft life: design

Why this exists: the lattice engine (`crates/auto-life-gpu`) can't make bodies. One cell per square, no bond, and force can't cross a group, since `mass_on` only shoves one neighbour. After 384 screened worlds the result is sitting clonal patches, roamers between them, and biters that hit kin exactly as often as kin are next to them (`research/life-gpu/screen/kin.jsonl`, correlation 0.96). Fraser Price's simulation shows bodies, tissues and moats from particles that can bond, push, share and fight (`research/fraser-price/README.md`, `research/notes/multicellularity.md`). Nobody has published bodies that reproduce as units with no fitness function and no designed help, so that's the target and the bar.

## Rules we keep

- One kind of cell. No types, no parts, no species, no body IDs, no group replication, no family bonus, no fitness function. Energy alone decides.
- Anything a cell shows others is something its own brain chose. No genome-derived odour or tag.
- Everything costs energy, and every action sheds some as heat. Energy is conserved and checked by a ledger.
- Every rule below has an off switch or a setting that removes it, so its effect can be tested against a control.

## The cell

A soft disc in continuous 2D. Position `x, y`, heading, energy `E`, radius from energy (`r = r0 * sqrt(1 + E / E_size)`, capped so `r <= 0.9` squares), recurrent brain with its own 8-bit weights and memory rows. Reuse the brain generator and genome layout from `auto-life-gpu` where that's clean (`act.wgsl` `//@BRAIN`, `births.wgsl`), including the mutation and the fixed random projection used for colour.

Each cell spends a share of its energy per tick and its brain divides that across actions, as in the lattice engine. Upkeep is charged whatever it does. A fraction of all spending becomes heat.

### Forces, overdamped

`v = F / drag`, and the step is capped at 0.4 squares per tick. For each pair within range:

- **Crowding:** `k_push * overlap`, where overlap is `r_i + r_j - d`.
- **Adhesion:** a pull up to `gap` beyond contact, strength `stick_i * stick_j * match(glue_i, glue_j) * stick_max`. Both pay in proportion to contacts. `stick` is a budget action. `glue` is two brain outputs, and `match` is high when they're alike. There are no stored bonds: a pair pulled too far apart stops sticking, which gives fracture and fragmenting for free. Stored bonds may come later if bodies turn out to be only droplets.
- **Shove:** a `push` action adds repulsion on contacts in the front sector.
- **Thrust:** along the heading, for itself. It reaches the rest of a body only through adhesion.

Forces are equal and opposite exactly: compute them from the unordered pair in a fixed order (lower slot first), and accumulate in fixed point so the order within a bin can't change a result.

### Trades across contacts

Each side works out the same trade from the same old state, so nothing needs writing to another cell.

- **Bite:** drains a contact in the front sector. Some goes to the biter, the rest to heat. Scaled so that the most a cell can lose in one tick to all neighbours together is bounded by its energy.
- **Give:** passes energy to contacts in proportion to `match`, with a loss to heat.
- **Contact senses, by sector (front, left, back, right in the body's frame):** count, size gap, and sums of the neighbours' two voice outputs and `match`. Also pressure, the total of contact forces. Pressure and count tell a cell whether it's on the surface or inside, which is how position can steer what it does.

### Fields

On a grid of one value per square. Exchange with neighbours and decay, Jacobi on two buffers:

- **Food:** regrows toward the sun map (reuse the scale-free terrain from `terrain.wgsl`), with patches that can drift. Harvest is settled as supply and demand: cells add what they want to a per-square atomic in fixed point, then each receives `min(1, supply / demand)` of its wish.
- **Locked food:** a cell's reach is `(1 - locked) + locked * Z / (Z + enzyme_half)`, where `Z` is the enzyme field under it. `locked = 0` removes the rule.
- **Enzyme:** cells secrete it as a budget action.
- **Heat:** every action sheds it, and cells can also emit it on purpose. Harm per tick is `heat_harm * H / (1 + E / heat_scale)`, which hits thin cells hardest. `heat_harm = 0` removes it.
- **Meat:** a cell's body and energy when it dies. Food for whoever eats it.
- **Scents:** at least two channels, emitted by choice, with value and gradient as senses.

Senses for each field are the value and the gradient in the body's frame.

### Splitting

`grow` fills a womb. When the womb covers a body and the brain's split output says so, a free slot is taken from the ring (`atomicAdd`, as now), and the child appears at contact distance at an angle the brain chooses, with a share of energy and a copy of the parent's memory. A child that keeps its parent's outputs sticks to its parent with no recognition rule at all, which is how clonal bodies can start. Mutation rate is heritable.

New random cells rain in at a low rate, as now, so a world can't die for good.

## The tick

Everything reads the old state and writes only its own record, so no phases or checkerboards are needed:

1. Clear bins and insert each cell by `atomicAdd`. Bins are 2 squares wide with room for 8. A full bin drops the extra for one tick and counts it, and the count is reported.
2. Contact pass: scan the 3x3 bins, and accumulate force, trades and senses.
3. Field wishes and emissions: atomics per square.
4. Settle harvest, integrate position and energy, handle death and split.
5. Fields: deposit, exchange, decay, regrow.
6. Brains every `brain_every` ticks, staggered by slot, so the load per tick stays even. Default 4.

Atomics are integers in WGSL, so use fixed point. Keep each GPU submission short and paced the way `world.rs` does: split passes into row ranges, and never hold the GPU for over about 8 ms when the world can be split. See `AGENTS.md`.

Batches of walled-off worlds on one map must work, as in the lattice engine: rock squares repel, and walls are at least 5 squares thick.

## Supply and demand, worked out first

`predictions.py` runs the numbers for the starting physics, and its output is below. Rerun it if the defaults change.

```
 cells  radius  enzyme mid    edge  access mid    edge   net mid     edge
     1       0        0.20    0.20        0.25    0.25     0.034    0.034
     9       1        0.76    0.67        0.49    0.46     0.082    0.076
    21       2        1.18    0.90        0.59    0.53     0.101    0.089
    69       4        1.87    1.23        0.69    0.60     0.121    0.103
   253       8        2.38    1.53        0.73    0.64     0.131    0.113
   509      12        2.47    1.47        0.74    0.64     0.132    0.111
```

- **A lone cell lives, slowly.** With no enzyme it nets +0.008 a tick. Making enzyme alone raises that to +0.034. So secreting pays on its own and can evolve without any group.
- **Groups pay up to about 70 cells, then level off.** Income triples from 1 cell to 21 and nearly quadruples at 69, and past that little is gained, because enzyme only reaches about 2 squares (`sqrt(flow / decay)`). So this physics predicts a group size set by the enzyme's reach. To favour bigger bodies, the reach is what has to change.
- **Edges earn less than middles.** At 69 cells, 0.103 against 0.121, about 15%. That's a small reason for a body to round up, and a position difference that differentiation could use.
- **Density is capped by sun.** A body at one cell per square needs sun of at least 0.016 per square, so bodies should fit on rich patches and shrink on poor ones. Where sun is lower than that, only the rim of a group can live, so expect rings.
- **At these settings heat makes no moat.** Harm to a thin cell 2 squares from a group of 253 is 0.007, under its +0.008 income, and it falls fast beyond that. Heat has to be about three times stronger for moats, at which point a group's own thin cells suffer too. So heat should select for rich cells in the middle of a body. These are predictions to test, and the screen should vary `heat_rate` and `heat_harm` widely.

Also predicted: with `stick` off, none of this forms bodies, and diffusion alone gives only loose crowds.

## Measures

A body is a connected group under contact with adhesion. It's only real if it passes these, compared with the same run with adhesion off:

1. Membership half-life longer than the control's.
2. Ten or more tracked body generations, with relatedness higher within bodies than between.
3. Size at fragmentation is heritable: parent to offspring slope above zero.
4. Clonemates in a body keep different budgets, predicted by their position, beating a shuffle.
5. Competition: the same genomes with adhesion off, on saved worlds, compete at 50/50 and rare both ways, in 3 or more seeds.

Keep the lattice engine's habits: every measure gets a test on a case with a known answer, and every rule's effect is tested against controls before it's believed.

## Targets

On the RX 7900 XT, which shares its VRAM with about 11 GB of other programs: 256 wide with about 30,000 cells at 1,000+ ticks a second, 1024 wide with about 300,000 cells at 150+, and 2048 wide with a million cells at 40+. If a target is missed, say by how much and where the time goes.
