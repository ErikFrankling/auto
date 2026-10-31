# Fraser Price's evolution sim

Researched 2026-10-31. Source post: https://x.com/fraserpricee/status/2116330338083348905

`post.mp4` is the video from that post (not committed), `sheet1.png` is a contact sheet of it.

## What he says it is

No code or paper is public. All of this comes from his posts.

- Each particle is an agent with its own small recurrent neural net. "1-10 million 1-10k param neural nets" at "200+ steps/s", in CUDA written for his own hardware. [Oct 10](https://x.com/fraserpricee/status/2108837968353857800)
- Each tick the net senses K background fields and allocates energy across A actions. [Jul 5](https://x.com/fraserpricee/status/2073560856726519857), [Jul 7](https://x.com/fraserpricee/status/2074454162146787791)
- Actions: sensing, signalling, moving, bonding, splitting, applying forces, harvesting, sharing, fighting. [Sep 16](https://x.com/fraserpricee/status/2100357688613908707)
- Everything costs energy, including existing. Zero energy means death. [Sep 16](https://x.com/fraserpricee/status/2100357686856438217)
- No loss or backprop. Random spawn, random mutation on reproduction. Colour is approximately the genome. [Jul 9](https://x.com/fraserpricee/status/2075243081222275080)
- Heat emission is an action with consequences: organisms kill lower-energy swarms with it, then harvest them. [Jul 9](https://x.com/fraserpricee/status/2075243083579445331)
- Training the particles with backprop made them "more boring". [Oct 28](https://x.com/fraserpricee/status/2115446610981191980)
- Same config on a larger grid (100k to 400k particles) gives more dynamics and hierarchy. [Oct 28](https://x.com/fraserpricee/status/2115462098461581699)
- An earlier version used fly connectomes as the controller. [Sep 15](https://x.com/fraserpricee/status/2099865719823339912)
- Hardware: 4 x RTX Pro 6000 Max-Q, Threadripper 9960X. [Jun 30](https://x.com/fraserpricee/status/2071957463373410806)
- Not released: "will fix/open source it in future" (Oct 10), "not for now" ([Oct 28](https://x.com/fraserpricee/status/2115504766289604761)).

The best single read is his [14-post thread of Sep 16](https://x.com/fraserpricee/status/2100357683060613259). Long runs: [8 h timelapse](https://x.com/fraserpricee/status/2115389281057276092).

## What the video shows

- Starts as uniform noise at tick ~300 with population still climbing, and clusters by colour within a few thousand ticks.
- Population then sits at about 999,9xx. That looks like a fixed 1M slot limit (my inference).
- HUD reads about 120-155 steps/s against a target that changes between clips (32 to 1024). Target-capped clips hit the cap, so roughly 150 steps/s looks like the real ceiling at 1M.
- HUD ticks reach about 177k within the 125 s video, so it is cut or sped up.
- No speech, only music.

## Open alternatives to read or run

- [Fluoddity](https://github.com/aphid91/Fluoddity): MIT, 132 stars, has a [WebGL demo](https://aphid91.github.io/Fluoddity-Core/). 80-parameter brains, particles interact only through trails. Evolution is by user selection, with no energy or death. Extends Sage Jenson's [physarum model](https://cargocollective.com/sagejenson/physarum).
- [ALIEN](https://github.com/chrxh/alien): BSD-3, 5.5k stars, active. Closest in spirit but CUDA, so it will not run on the pc's AMD GPU.
- [Flow-Lenia](https://arxiv.org/abs/2212.07906): grid-based, per-location rule parameters, no individual agents.
- [Microcosmos](https://arxiv.org/abs/2607.02954): GPU-first filament organisms in fluid, ALIFE 2026.
- [Emergent Garden](https://x.com/max_romana/status/2081811621538640061): YouTube channel Fraser recommends.

## Plan for our version

The pc has an RX 7900 XT (20 GB), so no CUDA. The homelab has no GPU listed. Use WGSL compute shaders.

Per-agent state in storage buffers: position, velocity, energy, hidden state, weights. Each step:

1. Deposit particles' energy, heat and signal onto 2D grids, then diffuse and decay them.
2. Sample K field values per particle.
3. Run the per-particle RNN with its own weights.
4. Softmax outputs into an energy budget across A actions; apply them.
5. Charge the cost of existing, kill at zero energy, free the slot.
6. Divide into a free slot, with Gaussian mutation of the child's weights.

Render with colour from a fixed random projection of the weights to RGB.

Start at 100k agents with about 256 parameters each: about 100 MB of weights (1M agents is about 1 GB, at 4 bytes per parameter). His upper range of 1M x 10k parameters is 40 GB and does not fit on 20 GB.

Start with four actions: move, harvest, divide, emit heat. Add bonding last, since it needs neighbour lookups and is the hardest to run on a GPU.

The first thing to check is whether a fixed config keeps producing novelty or settles down, since that is his main claim. He says he has not found a way to measure higher-level evolution; genome clustering over time is his suggestion.
