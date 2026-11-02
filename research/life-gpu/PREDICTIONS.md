# What the four new rules should do, written before any world ran

Written 2026-11-02, before running any world with these rules. Everything here is reasoning and arithmetic, not a result. If a run disagrees, the run wins, and the section at the end records what changed. The tables in the appendix come from `predictions_calc.py`, which uses the constants in `defaults.json`.

**The question for every rule.** Today's world has one currency, one supply, and about two ways to live (`research/notes/depth-of-strategy.md`, C1 and C2). A rule is only worth having if it creates a real choice: something a cell can do or skip, that pays in some situations and costs in others, where the best answer depends on what everyone else is doing.

**Two surprises from the arithmetic, which changed the design.**

1. The note's `force = 2S/(S+te)` doesn't do what we want. With a sharpness of 1, a lone cell of 5 energy on a passive fat cell of 40 still gets the best deal of all: +16.9 per fight against +7.9 for each of a pair. So force now has a setting, `gang_sharp`, and 2 is what gives the wanted shape: a lone 5 loses 5.1 and each of a pair gains 7.6.
2. Per head, a kill is a fixed prize, so the total each attacker gets always falls with N. What rises is the rate, and whether the kill can happen at all. So the signature to look for is a threshold and a rate, not a rising total, and free riding is built in.

## Rule 1. Fights where numbers pay (`gang_on`)

**The rule.** A neighbour of the target that really bit it last tick, read from `seen`, counts as an open side instead of a blocked one, and its energy joins the attacker's in `force = 2 S^g / (S^g + te^g)`. "Really bit" means it spent at least `gang_effort` of work, because a resting cell spends about 8% on everything and would otherwise count as an attacker for free. `seen` holds last tick, so a new attacker helps from its second tick.

**Supply and demand.** Supply: fat bodies, which no lone small cell can now open, so today's sit-and-bite stops working on them. Demand: every cell that can't afford to graze. The price of a fat body has changed from "whoever stands next to it" to "whoever brings enough friends".

**Who gains and who pays.** Small cells together gain access to food they couldn't touch. Fat cells pay, since wealth used to be safe from small cells and now has a price. Colony edges pay more than the inside: 4 walled sides against 6. Attackers pay the bite budget whether or not the kill comes, and pay again if the target bites back, which the free riders don't.

**The numbers** (appendix, with sharp 2, attackers of 5, budget 80% on biting).

| Target | Lone | Pair | 4 | 8 |
|---|---|---|---|---|
| Passive, lean, 5: net per head per tick | +2.28 | +1.03 | +0.52 | +0.15 |
| Passive, fat, 40 | -0.18, fails | +0.69 | +1.07 | +1.06 |
| Passive, very fat, 80 | -0.21, fails | -0.19, fails | +0.69 | +1.08 |
| Fat, 40, at a colony edge with 4 sides walled | -0.20, fails | -0.12, fails | +0.64 | not possible |

Lean prey is best hunted alone, with each extra biter earning less. Fat prey can't be hunted alone at all, and the rate rises to about 6 attackers, then flattens. This is the shape the note asks for. With today's rule the rate falls with every extra attacker on every target.

**A limit worth knowing.** A fat cell that fights back at 80% budget kills a 5-energy attacker every tick. It beats a group of 4 at 40 energy, needs a group of 6 to lose, and beats 8 at 80. A group has to be larger or fatter than the prey's owner can handle, and prey has to spend its own budget fighting. That is a real arms race, but a steep one, and it may be too steep (see below).

**What should evolve, if the theory is right.** Aggregation at fat prey, then more than one cell biting one target, more often than chance. Fat cells that keep biting back, or stay thinner. Colonies that keep their fat cells inside. Later perhaps use of noise (bites already make it) or scent to draw others, if prey is patchy.

**How it could go wrong.**
- Colonies eat themselves. Half of today's bites land on near relatives, and now all those neighbours can gang up. Density and diversity crash. `pack` went that way, in small runs.
- Nobody takes the first step. A lone biter on a fat target earns nothing, so the first two biters need another reason to be there. Lean prey and the fact that bites are cheap to try provide the stepping stone, and sharp 1 has a smoother path, so the screen samples 1 to 3.
- With `max_energy` at 40 there may be too few fat cells for any of this to pay. Recommended: `max_energy=80`.

**What would show it's wrong.** `group_hunting` in the log gives, for sampled bites, how many attackers the target had and each attacker's gain.
- The theory is wrong if, across seeds at 512, `gain_per_bite` doesn't rise from n=1 to n=3 while `gang_on`, or `extras` (other attackers per bite) stays within 20% of `extras_control`, the same count on a body next to the biter it did not pick.
- The rule is harmful if density falls by more than half against a control and `bites_on_similar` rises.

## Rule 2. Attack must match defence (`coat_on`)

**The rule.** A coat is a point on a circle, held in the cell, moved at most `coat_rate` turns a tick toward an output of the cell's own brain, at `coat_cost` energy per turn moved. A bite carries a key, another output, free and instant. Damage is multiplied by `floor + (1 - floor) * m^sharp`, with `m` being 1 at a perfect match and 0 half a turn off. Neither is sensed, unless `coat_sense` lets a cell feel the coat in front of it and its own. A child starts with its parent's coat, like its memory, and can move it.

**Supply and demand.** Supply: the bodies wearing each coat. Demand: biters choosing a key. The value of a key is how many bodies wear what it opens, so the supply of prey is set by what the prey chooses. Rarity is the good being bought, and nobody can own it.

**Who gains and who pays.** A rare coat is safe: at sharp 8 and floor 0.05, a body half a turn from the key takes 1/20th of the damage. A common coat feeds whoever matches it. A biter gets up to 4.2 times as much by tracking a monoculture as an unmatched key gets from spread coats. Bodies pay energy to move, at most 0.004 a tick against upkeep of 0.015, and half a turn takes 250 ticks, which is slow against a key that changes at once but fast against a generation.

**The numbers.** At force 1 in the open, a committed biter keeps 2.61 and pays 0.20, so it breaks even at a fit of 0.077. So floor must be below about 0.08, or mismatched bites pay anyway and the rule does nothing: at floor 0.05 a mismatched bite loses 0.07 a tick, and at floor 0.15 it earns 0.19. Through a clump's single open side it needs a fit of 0.61. A key on the common coat has an average fit of about the coat's share, so biters should switch at about 50%. A fat biter at force near 2 can break even on floor bites at 0.05, so fat cells care less about matching.

**What should evolve.** Biters whose key sits on the commonest local coat. Coat frequencies that rise and collapse. Big uniform colonies that get hit and shrink, which gives colony size a limit that depends on the neighbours. With `coat_sense`, keys that follow the coat in front (a master key, which is why it's a separate switch) and mimicry.

**How it could go wrong.**
- Without a sense, a brain can only learn a key across generations, and a stuck key means biters die out, which leaves the single-type monoculture of seed 2.
- A brain's output may be nearly constant, so coats and keys never move and "coat" is a slow-drifting label. Then this is only a tax on biting.
- With `coat_sense` on, every biter may just read the coat and set the key, which removes the rule's effect, leaving only the coat's cost.
- A coat is inherited by default. It could become a kin marker through `coat_sense`. That's allowed, since a brain chooses it, but it means cooperation would be read in the coat. Worlds at 256 may be too small for spirals (Reichenbach, note C5.2).

**What would show it's wrong.** `coat` in the log gives `effective` (the number of coats in use) and `key_match` against `key_chance`, which is what a key placed at random would get.
- The theory is wrong if `key_match / key_chance` stays near 1 where biters survive, or `effective` settles at 1 or at the number you get from drift with no cycle: history shows no rise and fall of common coats.
- It is harmful if `from_others` falls near 0 at floor 0.05 in all seeds.

## Rule 3. A second essential nutrient in a closed cycle (`mineral_on`)

**The rule.** Mineral lies on the ground in `fd.y`. Cells take it up with a budget action and hold it in `x0`. A birth takes `mineral_birth` from the parent, which goes into the child's body (`x2`) and stays bound there until the child dies, when body and store return to the ground. A bite moves the same share of the target's free store as of its energy. A gift can carry mineral instead of energy, if the brain says so. Nothing creates or destroys it, and amounts are whole grains of 1/4096, so the total is exact to the last bit, not just to rounding. Ground mineral drains slowly toward each square's richness, a separate noise map, with trades in pairs, so the world keeps its mineral lands for ever while the total stays fixed. Without that drain, mineral would ride in bodies to wherever bodies die, and the patches would be gone in a few lifetimes.

**Why bound in the body.** If a child's mineral could be spent again, one lineage could pass a single unit down for ever and mineral would never limit anything. Binding it means each living cell holds some, so a long-lived cell holds it back from everyone, and it gives a hard cap: density can't pass `mineral_start / mineral_birth`.

**Supply and demand.** Supply: a fixed stock, slow to move, in places that don't line up with sun (`mineral_anti` pushes them apart). Demand: every birth, at a fixed price per child, regardless of the child's size. For the first time no square is best for everything, so land rent splits, and a cell can be rich in one thing and poor in another.

**Who gains and who pays.** Sitters pay most. A stripped square gets only `mineral_flow * g` a tick, so with g = 0.3 a sitter waits 667 ticks per mineral, and it can fill a womb in 13 to 161 ticks on sun of 0.05 to 0.4. Sitting on rich sun stops paying. Walkers pay about 0.34 energy per square, so at g = 0.3 mineral adds 1.1 energy to a child's 5, which is cheap. Hunters gain, since a kill gives the free store. Old, resting cells hold mineral off the market, which competitors pay for.

**At what settings it limits births.** Two limits.
- Stock: mineral limits once density passes about `start / birth - 0.07 * birth`, where gathering costs more than the child's own 5 energy (appendix). The wall is steep: with `start / birth` at 0.25, gathering costs 2.0 energy at density 0.075, 4.6 at 0.175, and 13.7 at 0.225.
- Today's worlds run at density 0.18 to 0.54 on average, near 1 on rich land and under 0.07 on poor. So **0.25 is the setting where rich land is limited by mineral and poor land by energy**, which is the mix we want. Above about 0.8, mineral is plentiful and the rule does little, and Harpole's result says adding plenty lowers diversity. Below about 0.1, the world is thin and energy never limits.

**What should evolve.** Cells that walk between sun and mineral, bodies worth more than their energy, biting aimed at stores, use of `give` with mineral going to neighbours, and specialists in sun land and mineral land. Moving waves of scarcity where a boom locks up the local mineral. Long-lived cells that lock up mineral could be a new niche.

**How it could go wrong.**
- Everyone crowds the borders where both are decent, leaving one strategy again. `mineral_anti` narrows this but does not remove it.
- Everything is limited by mineral everywhere: cells hoard, density drops, and it is a thin copy of the old world.
- A flexible brain handles it all with a small budget share, so no specialists: a generalist with 1.1 energy of walking per child beats any specialist.
- Rain newcomers hold no mineral, so if ground mineral is scarce no newcomer can breed, and worlds that die out stay dead. Watch for dying worlds below 0.15.

**What would show it's wrong.** `birth_limit` in the log splits sampled cells into energy, mineral, both, room and waiting.
- It does nothing if the mineral share is under 5% in all seeds at 0.25.
- It's only a tax if mineral is over 80% and strategy counts don't rise over a control.
- The important check is in `life-analyse`: `land_table` should show occupancy and moving differ across sun and mineral together. `coexistence` tells if the specialists are only in different places, and if `overlap` falls while strategy count rises, it's geography and not depth.

## Rule 4. Life-history freedom (`birth_choice`, `age_metabolic`, `womb_cap`)

**The rule.** With `birth_choice`, an output decides when to give birth, and the child gets all that's in the womb, less the body, once it's over `birth_min`. The womb fills up to `womb_cap`. With `age_metabolic`, the old-age chance is `hazard * (age_rest + (1 - age_rest) * spent / spend)`, using what the cell really spent last tick.

**Supply and demand.** Supply: empty land, which appears after a crash or at a frontier, and safe rich land, which is nearly all taken. Demand: a cell's choice of many small children, which fill empty land fast, or few large ones, which can hold ground. Before this, nothing could buy size, and idle wealth sat at the cap.

**Who gains and who pays.** Each child costs one `body` of energy. So a child of 0.5 uses only a third of what's spent on it, and one of 16 uses 94%. Small ones are cheap and many: 67 per 100 energy against 6. But a 0.5 starves in 33 ticks unfed, and at sharp 2 has force 0.02 against a neighbour of 5. A child of 16 has 1.82. Mineral costs the same per child however big, so where mineral limits births, big children should win, and where energy limits, many small. That links rules 3 and 4 with no extra code. Resting pays for life: 90% rest gives 27,586 ticks of expected life against 4,000 for full activity, but no food comes in, and with mineral on it holds mineral off the market.

**What should evolve.** A spread of child sizes that's wider where it's been crashing, weeds on fresh land, big-child holders on rich land, and dormant cells waiting for a better season. Mainly, colony sizes should vary more.

**How it could go wrong.**
- All cells settle on one child size, so the spread is only noise.
- A lot of tiny children die, so nothing changes but waste, and Adler warns a many-species outcome needs a steep trade-off.
- Resting cells on rich land live for ever at the cost of nothing, freezing the map. At full rest, life is 67,000 ticks. The note's Chesson point applies too: dormancy without different responses to weather isn't a storage effect.

**What would show it's wrong.** `child_size_hist` in the log. It's a null if nearly all births land in one or two bins in all seeds, or if the mix of child sizes doesn't differ between strategies in the strategy tables (`child_energy`). For ageing, compare the mean age of resting and active strategies, and the map overlap (`map_overlap`) against a control. If overlap rises towards 1, the map has frozen and it's harmful.

## Depth, all together

None of this predicts trails or raids. The note found no case of those evolving from scratch, and I don't expect them here. A world that passes would show several of these at once, in different seeds: per-head gain rising with group size, persistent strategies above chance in the fine count, side-by-side coexistence, and a mix of nutrient limits. A single good number is not a pass.

## What changed after running

Nothing yet. This section is added to after small runs and doesn't replace what's above.

## Appendix: the calculations

Output of `predictions_calc.py`.

## Rule 1: one tick's take, per head, by N. Attackers already biting, target with all other sides empty.

Break-even keeps `a` = 0.20, so take must pass 0.40. Full take at force 1 is 5.22.

| Attackers' energy each, target | sharp | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|---|
| 5, 5 | 1 | 5.22 | 6.96 | 7.83 | 8.35 | 8.94 | 9.28 |
| 5, 5 | 2 | 5.22 | 8.35 | 9.39 | 9.82 | 10.15 | 10.27 |
| 5, 40 | 1 | 1.16 | 2.09 | 2.85 | 3.48 | 4.47 | 5.22 |
| 5, 40 | 2 | 0.16 | 0.61 | 1.29 | 2.09 | 3.76 | 5.22 |
| 5, 80 | 1 | 0.61 | 1.16 | 1.65 | 2.09 | 2.85 | 3.48 |
| 5, 80 | 2 | 0.04 | 0.16 | 0.35 | 0.61 | 1.29 | 2.09 |
| 2, 40 | 1 | 0.50 | 0.95 | 1.36 | 1.74 | 2.41 | 2.98 |
| 2, 40 | 2 | 0.03 | 0.10 | 0.23 | 0.40 | 0.86 | 1.44 |
| 20, 40 | 1 | 3.48 | 5.22 | 6.26 | 6.96 | 7.83 | 8.35 |
| 20, 40 | 2 | 2.09 | 5.22 | 7.22 | 8.35 | 9.39 | 9.82 |

## Rule 1: per head, a fight to the end. Attackers start at 5 energy, spend 80% of budget on biting.
Each cell: ticks / net energy per head per tick. Total per head in brackets. K = target died.

**fat alone, 40**

| Rule | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| today | 14K / +1.21 (+16.9) | 10K / +0.79 (+7.9) | 8K / +0.62 (+5.0) | 8K / +0.43 (+3.4) | 9K / +0.17 (+1.5) | 29K / -0.13 (-3.8) |
| sum, sharp 1 | 14K / +1.21 (+16.9) | 6K / +1.47 (+8.8) | 4K / +1.47 (+5.9) | 3K / +1.47 (+4.4) | 2K / +1.45 (+2.9) | 2K / +1.06 (+2.1) |
| sum, sharp 2 | 28 / -0.18 (-5.1) | 11K / +0.69 (+7.6) | 5K / +1.12 (+5.6) | 4K / +1.07 (+4.3) | 3K / +0.95 (+2.9) | 2K / +1.06 (+2.1) |
| today, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) | 6 / -0.83 (-5.0) | 8 / -0.63 (-5.0) |
| sum, sharp 2, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) | 4K / +0.18 (+0.7) | 2K / +0.81 (+1.6) |

**very fat alone, 80**

| Rule | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| today | 40K / +0.78 (+31.2) | 30K / +0.45 (+13.6) | 27K / +0.28 (+7.6) | 27K / +0.16 (+4.2) | 41 / -0.12 (-5.1) | 26 / -0.19 (-5.0) |
| sum, sharp 1 | 40K / +0.78 (+31.2) | 14K / +1.21 (+16.9) | 8K / +1.45 (+11.6) | 6K / +1.46 (+8.7) | 4K / +1.47 (+5.9) | 3K / +1.46 (+4.4) |
| sum, sharp 2 | 24 / -0.21 (-5.0) | 27 / -0.19 (-5.1) | 36K / +0.16 (+5.7) | 11K / +0.69 (+7.6) | 5K / +1.13 (+5.7) | 4K / +1.08 (+4.3) |
| today, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) | 6 / -0.83 (-5.0) | 8 / -0.62 (-5.0) |
| sum, sharp 2, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) | 6 / -0.83 (-5.0) | 8 / -0.62 (-5.0) |

**lean alone, 5**

| Rule | N=1 | N=2 | N=3 | N=4 | N=6 | N=8 |
|---|---|---|---|---|---|---|
| today | 1K / +2.28 (+2.3) | 1K / +1.03 (+1.0) | 1K / +0.69 (+0.7) | 1K / +0.52 (+0.5) | 1K / +0.31 (+0.3) | 1K / +0.12 (+0.1) |
| sum, sharp 1 | 1K / +2.28 (+2.3) | 1K / +1.03 (+1.0) | 1K / +0.69 (+0.7) | 1K / +0.52 (+0.5) | 1K / +0.31 (+0.3) | 1K / +0.12 (+0.1) |
| sum, sharp 2 | 1K / +2.28 (+2.3) | 1K / +1.03 (+1.0) | 1K / +0.69 (+0.7) | 1K / +0.52 (+0.5) | 1K / +0.31 (+0.3) | 1K / +0.15 (+0.1) |
| today, bites back | 1K / +2.28 (+2.3) | 1K / +1.03 (+1.0) | 1K / +0.69 (+0.7) | 1K / +0.52 (+0.5) | 1K / +0.31 (+0.3) | 1K / +0.12 (+0.1) |
| sum, sharp 2, bites back | 1K / +2.28 (+2.3) | 1K / +1.03 (+1.0) | 1K / +0.69 (+0.7) | 1K / +0.52 (+0.5) | 1K / +0.31 (+0.3) | 1K / +0.15 (+0.1) |

**fat on a colony's edge (4 sides walled), 40**

| Rule | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| today | 30K / +0.45 (+13.4) | 24K / +0.20 (+4.8) | 29K / +0.01 (+0.4) | 30 / -0.17 (-5.1) |
| sum, sharp 1 | 30K / +0.45 (+13.4) | 11K / +0.70 (+7.7) | 7K / +0.76 (+5.3) | 5K / +0.79 (+4.0) |
| sum, sharp 2 | 25 / -0.20 (-5.0) | 44 / -0.12 (-5.1) | 9K / +0.52 (+4.7) | 6K / +0.64 (+3.9) |
| today, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) |
| sum, sharp 2, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) |

**very fat on a colony's edge (4 walled), 80**

| Rule | N=1 | N=2 | N=3 | N=4 |
|---|---|---|---|---|
| today | 44 / -0.12 (-5.2) | 35 / -0.15 (-5.1) | 30 / -0.17 (-5.2) | 26 / -0.19 (-5.1) |
| sum, sharp 1 | 44 / -0.12 (-5.2) | 30K / +0.45 (+13.5) | 16K / +0.62 (+9.8) | 11K / +0.70 (+7.7) |
| sum, sharp 2 | 24 / -0.21 (-5.1) | 25 / -0.20 (-5.1) | 28 / -0.18 (-5.2) | 40 / -0.13 (-5.1) |
| today, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) |
| sum, sharp 2, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) | 3 / -1.67 (-5.0) | 4 / -1.25 (-5.0) |

**fat inside a clump (6 walled), 40**

| Rule | N=1 | N=2 |
|---|---|---|
| today | 43 / -0.12 (-5.0) | 30 / -0.17 (-5.2) |
| sum, sharp 1 | 43 / -0.12 (-5.0) | 22K / +0.24 (+5.2) |
| sum, sharp 2 | 24 / -0.21 (-5.0) | 27 / -0.19 (-5.1) |
| today, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) |
| sum, sharp 2, bites back | 1 / -5.00 (-5.0) | 2 / -2.50 (-5.0) |

## Rule 2: what a bite is worth against coats

Biter keeps 2.61 a tick at force 1, open 1, perfect match, and pays 0.20. So it needs fit of at least 0.077 in the open, and 0.61 through a clump's 1 open side in 8.

| sharp | half width in turns | floor | fit against evenly spread coats | best key on a monoculture | gain from tracking | open-air profit, mismatched | clump profit at match |
|---|---|---|---|---|---|---|---|
| 1 | 0.250 | 0.05 | 0.525 | 1 | 1.9x | -0.07 | +0.13 |
| 2 | 0.182 | 0.05 | 0.406 | 1 | 2.5x | -0.07 | +0.13 |
| 4 | 0.131 | 0.05 | 0.310 | 1 | 3.2x | -0.07 | +0.13 |
| 8 | 0.093 | 0.05 | 0.237 | 1 | 4.2x | -0.07 | +0.13 |
| 16 | 0.066 | 0.05 | 0.183 | 1 | 5.5x | -0.07 | +0.13 |
| 32 | 0.047 | 0.05 | 0.144 | 1 | 6.9x | -0.07 | +0.13 |
| 8 | 0.093 | 0.0 | 0.196 | 1 | 5.1x | -0.20 | +0.13 |
| 8 | 0.093 | 0.15 | 0.317 | 1 | 3.2x | +0.19 | +0.13 |
| 8 | 0.093 | 0.3 | 0.437 | 1 | 2.3x | +0.58 | +0.13 |

Kill the winner. Shares of two coats half a turn apart, sharp 8, floor 0.05. Biters track the common coat.

| Common coat's share | Biter's average fit, keyed on common | keyed on rare | Damage per body on common | on rare | Common coat suffers |
|---|---|---|---|---|---|
| 0.5 | 0.53 | 0.53 | 1 | 0.05 | 20x |
| 0.7 | 0.71 | 0.34 | 1 | 0.05 | 20x |
| 0.9 | 0.91 | 0.14 | 1 | 0.05 | 20x |
| 0.99 | 0.99 | 0.06 | 1 | 0.05 | 20x |

Moving coat: at 0.002 turns a tick, half a turn takes 250 ticks and costs 1.00 energy, at most 0.0040 a tick against upkeep 0.015.

## Rule 3: where mineral limits births

A step costs 0.150 energy and clearing a square 0.193, 0.343 together, from a budget of 0.25 a tick, so a walker gathers about every second tick.

Stock limit: every living cell holds `mineral_birth` in its body, so density can't pass start / birth.

Price of mineral for a walker, in energy per child, with free ground mineral g = start - density * birth left on a square, walking to a fresh square each tick. A child's energy price is 5.

| start / birth | density | g per square, at birth 1 | squares walked per child | energy per child's mineral | Which limits |
|---|---|---|---|---|---|
| 0.15 | 0.045 | 0.105 | 9.5 | 3.3 | energy |
| 0.15 | 0.105 | 0.045 | 22.2 | 7.6 | mineral |
| 0.15 | 0.135 | 0.015 | 66.7 | 22.8 | mineral |
| 0.15 | 0.145 | 0.005 | 222.2 | 76.1 | mineral |
| 0.25 | 0.075 | 0.175 | 5.7 | 2.0 | energy |
| 0.25 | 0.175 | 0.075 | 13.3 | 4.6 | energy |
| 0.25 | 0.225 | 0.025 | 40.0 | 13.7 | mineral |
| 0.25 | 0.242 | 0.008 | 133.3 | 45.7 | mineral |
| 0.5 | 0.150 | 0.350 | 2.9 | 1.0 | energy |
| 0.5 | 0.350 | 0.150 | 6.7 | 2.3 | energy |
| 0.5 | 0.450 | 0.050 | 20.0 | 6.9 | mineral |
| 0.5 | 0.485 | 0.015 | 66.7 | 22.8 | mineral |
| 1.0 | 0.300 | 0.700 | 1.4 | 0.5 | energy |
| 1.0 | 0.700 | 0.300 | 3.3 | 1.1 | energy |
| 1.0 | 0.900 | 0.100 | 10.0 | 3.4 | energy |
| 1.0 | 0.970 | 0.030 | 33.3 | 11.4 | mineral |

Income from sun, net of upkeep, for a cell eating all it can. Ticks to fill a womb of 5, against ticks to gather 1 mineral sitting still.

Sitting still, mineral arrives only by flow. Each square trades with 2 partners a tick, one per axis. A stripped square gets 0.005 x g / 2 from each at equal richness, so 0.005 x g a tick.

| Sun on this square | Net energy a tick | Ticks to fill a womb | Sitter's ticks per mineral, g=0.1 | g=0.3 | g=1 | Sitter is limited by |
|---|---|---|---|---|---|---|
| 0.03 | 0.011 | 455 | 2000 | 667 | 200 | mineral at g=0.3 |
| 0.05 | 0.031 | 161 | 2000 | 667 | 200 | mineral at g=0.3 |
| 0.1 | 0.081 | 62 | 2000 | 667 | 200 | mineral at g=0.3 |
| 0.2 | 0.181 | 28 | 2000 | 667 | 200 | mineral at g=0.3 |
| 0.4 | 0.381 | 13 | 2000 | 667 | 200 | mineral at g=0.3 |
| 1.0 | 0.981 | 5 | 2000 | 667 | 200 | mineral at g=0.3 |

## Rule 4: child size and resting

| Child's energy | Womb to fill | Share reaching the child | Children per 100 energy | Ticks a child lives unfed | Force, lone, against a 5 energy neighbour |
|---|---|---|---|---|---|
| 0.5 | 1.5 | 0.33 | 67 | 33 | 0.02 |
| 1 | 2 | 0.50 | 50 | 65 | 0.08 |
| 2 | 3 | 0.67 | 33 | 127 | 0.28 |
| 4 | 5 | 0.80 | 20 | 241 | 0.78 |
| 8 | 9 | 0.89 | 11 | 440 | 1.44 |
| 16 | 17 | 0.94 | 6 | 748 | 1.82 |
| 32 | 33 | 0.97 | 3 | 1151 | 1.95 |

Old-age hazard is 0.00025 times `age_rest + (1 - age_rest) * spent / spend`, age_rest 0.05.

| Rest share | Spending a tick | Upkeep a tick, at 10 energy | Hazard factor | Expected life, ticks | Today's life |
|---|---|---|---|---|---|
| 0 | 0.250 | 0.0190 | 1.000 | 4000 | 4000 |
| 0.25 | 0.188 | 0.0152 | 0.762 | 5246 | 4000 |
| 0.5 | 0.125 | 0.0114 | 0.525 | 7619 | 4000 |
| 0.75 | 0.062 | 0.0076 | 0.287 | 13913 | 4000 |
| 0.9 | 0.025 | 0.0053 | 0.145 | 27586 | 4000 |
| 0.99 | 0.003 | 0.0040 | 0.060 | 67227 | 4000 |

