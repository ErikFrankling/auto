# Energy-only evolution toy

A small CPU sim for building intuition, written 2026-10-31. It is not the GPU design and doesn't test open-ended evolution.

Each cell has its own 80-weight RNN. There is no score. Weights only change at birth, with Gaussian noise. Every world gets the same 300 energy per tick. Random cells rain in for the first 1,500 ticks, and after that an extinction is final.

Each tick a cell splits its energy across rest, move, eat, divide and steal. An inherited body trait sets which food it digests. The trade-off is convex, so eating both foods is worse than specialising.

## Run it

```
nix develop -c node research/toy/batch.js biomes 1 60000
nix develop -c node research/toy/batch.js sealed 11 30000 15000 0.99
nix develop -c node research/toy/analyse.js research/toy/main.jsonl
```

Arguments are world, seed, ticks, then optionally the tick and fraction of a mass kill. Worlds are `plain`, `mixed`, `biomes`, `drift`, `sealed` and `islands`.

## Results

`main.jsonl` has 8 seeds per world at 60,000 ticks. `shock.jsonl` has 8 other seeds with 99% of cells killed at tick 15,000, measured at tick 30,000.

| World | Alive | Two kinds coexist | Follows food | Energy from stealing | Empty habitats | Alive after kill | Two kinds after kill | Empty after kill |
|---|---|---|---|---|---|---|---|---|
| One food, even (`plain`) | 8/8 | n/a | 0.31 | 10% | | 8/8 | n/a | |
| Two foods, mixed everywhere (`mixed`) | 7/8 | 2/7 | 0.29 | 4% | | not run | not run | |
| Two foods, separate regions (`biomes`) | 8/8 | 8/8 | 0.43 | 12% | | 8/8 | 5/8 | |
| Two foods, drifting patches (`drift`) | 8/8 | 5/8 | 0.46 | 25% | | 6/8 | 0/6 | |
| Sealed islands (`sealed`) | 8/8 | 6/8 | 0.42 | 17% | 10/32 | 7/8 | 6/7 | 11/28 |
| Islands with rare migrants (`islands`) | 8/8 | 7/8 | 0.38 | 15% | 2/32 | 8/8 | 6/8 | 4/32 |

- **Two kinds coexist:** at least 10% of cells eat mostly A and at least 10% eat mostly B.
- **Follows food:** cosine between a cell's movement and the smell of its edible food. Fresh random brains score about 0.05.

The constants are the first ones tried, not tuned towards a result. With a few hundred cells and 8 runs, a gap of one or two runs could be chance. No real predator species formed, only a share of stealing across the population.
