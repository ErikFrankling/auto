//! Gosper's Hashlife. The universe is a quadtree in which identical squares are
//! the same node, and each node remembers its future. Repetition in space and in
//! time collapses, so patterns with 10^20 cells or 10^15 generations stay cheap.
//! On a random soup it would just be a slow dense simulator.

use std::collections::HashMap;

type Id = u32;

#[derive(Clone, Copy)]
struct Node {
    nw: Id,
    ne: Id,
    sw: Id,
    se: Id,
    level: u8,
    population: u128,
    result: Id,
}

const NONE: Id = u32::MAX;

pub struct HashLife {
    nodes: Vec<Node>,
    cache: HashMap<(Id, Id, Id, Id), Id>,
    empty: Vec<Id>,
    slow_memo: HashMap<(Id, u8), Id>,
    root: Id,
    pub generation: u128,
}

impl Default for HashLife {
    fn default() -> Self {
        Self::new()
    }
}

impl HashLife {
    pub fn new() -> Self {
        let dead = Node {
            nw: 0,
            ne: 0,
            sw: 0,
            se: 0,
            level: 0,
            population: 0,
            result: NONE,
        };
        let live = Node {
            nw: 0,
            ne: 0,
            sw: 0,
            se: 0,
            level: 0,
            population: 1,
            result: NONE,
        };
        let mut h = Self {
            nodes: vec![dead, live],
            cache: HashMap::new(),
            empty: vec![0],
            slow_memo: HashMap::new(),
            root: 0,
            generation: 0,
        };
        h.root = h.empty(3);
        h
    }

    /// Build from live cell coordinates, relative to the universe's centre.
    pub fn from_cells(cells: &[(i64, i64)]) -> Self {
        let mut h = Self::new();
        let reach = cells
            .iter()
            .map(|(x, y)| x.abs().max(y.abs()))
            .max()
            .unwrap_or(1)
            + 1;
        let level = (64 - (reach as u64).leading_zeros() + 1).max(3) as u8;
        let half = 1i64 << (level - 1);
        let mut sorted: Vec<(i64, i64)> = cells.iter().map(|(x, y)| (x + half, y + half)).collect();
        sorted.sort_unstable();
        sorted.dedup();
        h.root = h.build(level, 0, 0, &sorted);
        h
    }

    fn build(&mut self, level: u8, x0: i64, y0: i64, cells: &[(i64, i64)]) -> Id {
        if cells.is_empty() {
            return self.empty(level);
        }
        if level == 0 {
            return 1;
        }
        let half = 1i64 << (level - 1);
        let (mut nw, mut ne, mut sw, mut se) = (vec![], vec![], vec![], vec![]);
        for c in cells {
            let east = c.0 >= x0 + half;
            let south = c.1 >= y0 + half;
            match (east, south) {
                (false, false) => nw.push(*c),
                (true, false) => ne.push(*c),
                (false, true) => sw.push(*c),
                (true, true) => se.push(*c),
            }
        }
        let a = self.build(level - 1, x0, y0, &nw);
        let b = self.build(level - 1, x0 + half, y0, &ne);
        let c = self.build(level - 1, x0, y0 + half, &sw);
        let d = self.build(level - 1, x0 + half, y0 + half, &se);
        self.join(a, b, c, d)
    }

    fn empty(&mut self, level: u8) -> Id {
        while self.empty.len() <= level as usize {
            let e = *self.empty.last().expect("level zero exists");
            let up = self.join(e, e, e, e);
            self.empty.push(up);
        }
        self.empty[level as usize]
    }

    fn join(&mut self, nw: Id, ne: Id, sw: Id, se: Id) -> Id {
        if let Some(id) = self.cache.get(&(nw, ne, sw, se)) {
            return *id;
        }
        let n = &self.nodes;
        let population = n[nw as usize].population
            + n[ne as usize].population
            + n[sw as usize].population
            + n[se as usize].population;
        let level = n[nw as usize].level + 1;
        let id = self.nodes.len() as Id;
        self.nodes.push(Node {
            nw,
            ne,
            sw,
            se,
            level,
            population,
            result: NONE,
        });
        self.cache.insert((nw, ne, sw, se), id);
        id
    }

    fn pad(&mut self) {
        let r = self.nodes[self.root as usize];
        let e = self.empty(r.level - 1);
        let nw = self.join(e, e, e, r.nw);
        let ne = self.join(e, e, r.ne, e);
        let sw = self.join(e, r.sw, e, e);
        let se = self.join(r.se, e, e, e);
        self.root = self.join(nw, ne, sw, se);
    }

    fn get(&self, id: Id, quad: usize) -> Id {
        let n = &self.nodes[id as usize];
        [n.nw, n.ne, n.sw, n.se][quad]
    }

    /// The centre of a level `k` node, 2^(k-2) generations later.
    fn future(&mut self, id: Id) -> Id {
        let node = self.nodes[id as usize];
        if node.result != NONE {
            return node.result;
        }
        let result = if node.population == 0 {
            self.empty(node.level - 1)
        } else if node.level == 2 {
            self.base(id)
        } else {
            let (nw, ne, sw, se) = (node.nw, node.ne, node.sw, node.se);
            let n = self.join(
                self.get(nw, 1),
                self.get(ne, 0),
                self.get(nw, 3),
                self.get(ne, 2),
            );
            let w = self.join(
                self.get(nw, 2),
                self.get(nw, 3),
                self.get(sw, 0),
                self.get(sw, 1),
            );
            let c = self.join(
                self.get(nw, 3),
                self.get(ne, 2),
                self.get(sw, 1),
                self.get(se, 0),
            );
            let e = self.join(
                self.get(ne, 2),
                self.get(ne, 3),
                self.get(se, 0),
                self.get(se, 1),
            );
            let s = self.join(
                self.get(sw, 1),
                self.get(se, 0),
                self.get(sw, 3),
                self.get(se, 2),
            );
            let r00 = self.future(nw);
            let r01 = self.future(n);
            let r02 = self.future(ne);
            let r10 = self.future(w);
            let r11 = self.future(c);
            let r12 = self.future(e);
            let r20 = self.future(sw);
            let r21 = self.future(s);
            let r22 = self.future(se);
            let a = self.join(r00, r01, r10, r11);
            let b = self.join(r01, r02, r11, r12);
            let c = self.join(r10, r11, r20, r21);
            let d = self.join(r11, r12, r21, r22);
            let ra = self.future(a);
            let rb = self.future(b);
            let rc = self.future(c);
            let rd = self.future(d);
            self.join(ra, rb, rc, rd)
        };
        self.nodes[id as usize].result = result;
        result
    }

    /// A 4x4 node's centre 2x2, one generation on.
    fn base(&mut self, id: Id) -> Id {
        let mut grid = [[0u8; 4]; 4];
        for y in 0..4 {
            for x in 0..4 {
                let q = self.get(id, (y / 2) * 2 + x / 2);
                let cell = self.get(q, (y % 2) * 2 + x % 2);
                grid[y][x] = cell as u8;
            }
        }
        let next = |x: usize, y: usize| -> Id {
            let mut n = 0;
            for dy in 0..3 {
                for dx in 0..3 {
                    if dx != 1 || dy != 1 {
                        n += grid[y + dy - 1][x + dx - 1];
                    }
                }
            }
            u32::from(n == 3 || (n == 2 && grid[y][x] == 1))
        };
        let (a, b, c, d) = (next(1, 1), next(2, 1), next(1, 2), next(2, 2));
        self.join(a, b, c, d)
    }

    /// Advance by exactly `gens` generations by combining power-of-two steps.
    pub fn advance(&mut self, gens: u128) {
        for bit in 0..128 {
            if (gens >> bit) & 1 == 1 {
                self.step_power(bit as u8);
            }
        }
    }

    /// Advance by exactly 2^k generations.
    fn step_power(&mut self, k: u8) {
        // `future` advances a node 2^(level-2) generations, so to take a smaller
        // step on a big universe we walk slow steps down to the wanted level.
        loop {
            let root = self.nodes[self.root as usize];
            let (nw, ne, sw, se) = (root.nw, root.ne, root.sw, root.se);
            let inner = self.nodes[self.get(nw, 3) as usize].population
                + self.nodes[self.get(ne, 2) as usize].population
                + self.nodes[self.get(sw, 1) as usize].population
                + self.nodes[self.get(se, 0) as usize].population;
            if root.level >= k + 2 && inner == root.population {
                break;
            }
            self.pad();
        }
        self.pad();
        self.root = self.slow(self.root, k);
        self.generation += 1u128 << k;
    }

    /// The centre of a node, exactly 2^k generations later.
    fn slow(&mut self, id: Id, k: u8) -> Id {
        let node = self.nodes[id as usize];
        if node.level == k + 2 {
            return self.future(id);
        }
        if node.population == 0 {
            return self.empty(node.level - 1);
        }
        if let Some(r) = self.slow_memo.get(&(id, k)) {
            return *r;
        }
        let (nw, ne, sw, se) = (node.nw, node.ne, node.sw, node.se);
        let n = self.join(
            self.get(nw, 1),
            self.get(ne, 0),
            self.get(nw, 3),
            self.get(ne, 2),
        );
        let w = self.join(
            self.get(nw, 2),
            self.get(nw, 3),
            self.get(sw, 0),
            self.get(sw, 1),
        );
        let c = self.join(
            self.get(nw, 3),
            self.get(ne, 2),
            self.get(sw, 1),
            self.get(se, 0),
        );
        let e = self.join(
            self.get(ne, 2),
            self.get(ne, 3),
            self.get(se, 0),
            self.get(se, 1),
        );
        let s = self.join(
            self.get(sw, 1),
            self.get(se, 0),
            self.get(sw, 3),
            self.get(se, 2),
        );
        let parts = [nw, n, ne, w, c, e, sw, s, se].map(|p| self.slow(p, k));
        let centre = |h: &mut Self, a: Id, b: Id, c: Id, d: Id| -> Id {
            let j1 = h.get(a, 3);
            let j2 = h.get(b, 2);
            let j3 = h.get(c, 1);
            let j4 = h.get(d, 0);
            h.join(j1, j2, j3, j4)
        };
        let a = centre(self, parts[0], parts[1], parts[3], parts[4]);
        let b = centre(self, parts[1], parts[2], parts[4], parts[5]);
        let c2 = centre(self, parts[3], parts[4], parts[6], parts[7]);
        let d = centre(self, parts[4], parts[5], parts[7], parts[8]);
        let r = self.join(a, b, c2, d);
        self.slow_memo.insert((id, k), r);
        r
    }

    pub fn population(&self) -> u128 {
        self.nodes[self.root as usize].population
    }

    /// log2 of the universe's side length.
    pub fn universe_level(&self) -> u8 {
        self.nodes[self.root as usize].level
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn memory_bytes(&self) -> usize {
        self.nodes.len() * (std::mem::size_of::<Node>() + 32)
    }
}

/// Bill Gosper's glider gun as live cell coordinates.
pub fn gosper_gun() -> Vec<(i64, i64)> {
    const ROWS: [&str; 9] = [
        "........................O...........",
        "......................O.O...........",
        "............OO......OO............OO",
        "...........O...O....OO............OO",
        "OO........O.....O...OO..............",
        "OO........O...O.OO....O.O...........",
        "..........O.....O.......O...........",
        "...........O...O....................",
        "............OO......................",
    ];
    parse(&ROWS)
}

/// The R-pentomino. Settles at generation 1103 with 116 cells.
pub fn r_pentomino() -> Vec<(i64, i64)> {
    parse(&[".OO", "OO.", ".O."])
}

fn parse(rows: &[&str]) -> Vec<(i64, i64)> {
    let mut cells = Vec::new();
    for (y, row) in rows.iter().enumerate() {
        for (x, c) in row.bytes().enumerate() {
            if c == b'O' {
                cells.push((x as i64, y as i64));
            }
        }
    }
    cells
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap as Map, HashSet};

    fn sparse_step(live: &HashSet<(i64, i64)>) -> HashSet<(i64, i64)> {
        let mut counts: Map<(i64, i64), u8> = Map::new();
        for (x, y) in live {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx != 0 || dy != 0 {
                        *counts.entry((x + dx, y + dy)).or_insert(0) += 1;
                    }
                }
            }
        }
        counts
            .into_iter()
            .filter(|(c, n)| *n == 3 || (*n == 2 && live.contains(c)))
            .map(|(c, _)| c)
            .collect()
    }

    #[test]
    fn r_pentomino_settles_at_116_cells() {
        let mut h = HashLife::from_cells(&r_pentomino());
        h.advance(1103);
        assert_eq!(h.population(), 116);
        h.advance(1000);
        // Six gliders fly off and nothing else changes the count.
        assert_eq!(h.population(), 116);
    }

    #[test]
    fn matches_a_plain_simulation_at_every_size() {
        let mut live: HashSet<_> = gosper_gun().into_iter().collect();
        let mut h = HashLife::from_cells(&gosper_gun());
        let mut done = 0u128;
        for jump in [1u128, 2, 3, 8, 13, 100, 256, 500] {
            h.advance(jump);
            for _ in 0..jump {
                live = sparse_step(&live);
            }
            done += jump;
            assert_eq!(h.population(), live.len() as u128, "at generation {done}");
        }
    }

    #[test]
    fn gun_reaches_huge_generations() {
        let mut h = HashLife::from_cells(&gosper_gun());
        h.advance(1 << 30);
        // One new 5-cell glider every 30 generations, on top of the gun.
        let gliders = ((1u128 << 30) / 30) * 5;
        assert!(
            h.population().abs_diff(gliders) < 100,
            "population {}",
            h.population()
        );
    }
}
