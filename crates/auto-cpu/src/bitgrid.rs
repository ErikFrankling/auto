//! Dense Life grids, one bit per cell, in 64-bit words. Bit i of a word is x offset i.

#[cfg(feature = "threads")]
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// One byte per cell, eight reads per cell.
    Bytes,
    /// 64 cells at a time with scalar u64 logic.
    Swar,
    /// 256 cells at a time.
    Avx2,
    /// 512 cells at a time, using VPTERNLOG for full adders.
    Avx512,
}

impl Kernel {
    pub fn available(self) -> bool {
        match self {
            Kernel::Bytes | Kernel::Swar => true,
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx2 => std::is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Kernel::Avx512 => std::is_x86_feature_detected!("avx512f"),
            #[cfg(not(target_arch = "x86_64"))]
            _ => false,
        }
    }
}

pub struct BitGrid {
    pub width: usize,
    pub height: usize,
    words_per_row: usize,
    cur: Vec<u64>,
    next: Vec<u64>,
}

impl BitGrid {
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width % 64 == 0 && width > 0 && height > 0);
        let words_per_row = width / 64;
        let n = words_per_row * height;
        Self {
            width,
            height,
            words_per_row,
            cur: vec![0; n],
            next: vec![0; n],
        }
    }

    pub fn random(width: usize, height: usize, seed: u64) -> Self {
        let mut g = Self::new(width, height);
        let mut s = seed | 1;
        for w in &mut g.cur {
            let mut v = 0u64;
            for b in 0..64 {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s % 100 < 35 {
                    v |= 1 << b;
                }
            }
            *w = v;
        }
        g
    }

    pub fn get(&self, x: usize, y: usize) -> bool {
        (self.cur[y * self.words_per_row + x / 64] >> (x % 64)) & 1 == 1
    }

    pub fn set(&mut self, x: usize, y: usize, v: bool) {
        let i = y * self.words_per_row + x / 64;
        if v {
            self.cur[i] |= 1 << (x % 64);
        } else {
            self.cur[i] &= !(1 << (x % 64));
        }
    }

    pub fn population(&self) -> u64 {
        self.cur.iter().map(|w| w.count_ones() as u64).sum()
    }

    /// Advance one generation. `threaded` uses all rayon threads.
    pub fn step(&mut self, kernel: Kernel, threaded: bool) {
        assert!(
            kernel.available(),
            "{kernel:?} is not available on this CPU"
        );
        let (w, h) = (self.words_per_row, self.height);
        let cur = &self.cur;
        let row = |y: usize, out: &mut [u64]| {
            let up = &cur[((y + h - 1) % h) * w..][..w];
            let mid = &cur[y * w..][..w];
            let dn = &cur[((y + 1) % h) * w..][..w];
            match kernel {
                Kernel::Bytes => bytes_row(up, mid, dn, out),
                Kernel::Swar => swar_row(up, mid, dn, out, 0),
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `available` checked the CPU feature above.
                Kernel::Avx2 => unsafe { avx2_row(up, mid, dn, out) },
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `available` checked the CPU feature above.
                Kernel::Avx512 => unsafe { avx512_row(up, mid, dn, out) },
                #[cfg(not(target_arch = "x86_64"))]
                _ => unreachable!(),
            }
        };
        #[cfg(feature = "threads")]
        if threaded {
            self.next
                .par_chunks_mut(w)
                .enumerate()
                .for_each(|(y, out)| row(y, out));
            std::mem::swap(&mut self.cur, &mut self.next);
            return;
        }
        let _ = threaded;
        for (y, out) in self.next.chunks_mut(w).enumerate() {
            row(y, out);
        }
        std::mem::swap(&mut self.cur, &mut self.next);
    }
}

#[inline(always)]
fn word(tl: u64, tc: u64, tr: u64, ml: u64, mc: u64, mr: u64, bl: u64, bc: u64, br: u64) -> u64 {
    let n0 = (tc << 1) | (tl >> 63);
    let n2 = (tc >> 1) | (tr << 63);
    let n3 = (mc << 1) | (ml >> 63);
    let n4 = (mc >> 1) | (mr << 63);
    let n5 = (bc << 1) | (bl >> 63);
    let n7 = (bc >> 1) | (br << 63);
    let x01 = n0 ^ tc;
    let s0 = x01 ^ n2;
    let c0 = (n0 & tc) | (x01 & n2);
    let x34 = n3 ^ n4;
    let s1 = x34 ^ n5;
    let c1 = (n3 & n4) | (x34 & n5);
    let s2 = bc ^ n7;
    let c2 = bc & n7;
    let x = s0 ^ s1;
    let ones = x ^ s2;
    let c3 = (s0 & s1) | (x & s2);
    let twos = c0 ^ c1 ^ c2 ^ c3;
    let many = (c0 & c1) | (c2 & c3) | ((c0 | c1) & (c2 | c3));
    !many & twos & (ones | mc)
}

fn swar_row(up: &[u64], mid: &[u64], dn: &[u64], out: &mut [u64], from: usize) {
    swar_range(up, mid, dn, out, from, mid.len());
}

fn swar_range(up: &[u64], mid: &[u64], dn: &[u64], out: &mut [u64], from: usize, to: usize) {
    let w = mid.len();
    for x in from..to {
        let l = if x == 0 { w - 1 } else { x - 1 };
        let r = if x + 1 == w { 0 } else { x + 1 };
        out[x] = word(
            up[l], up[x], up[r], mid[l], mid[x], mid[r], dn[l], dn[x], dn[r],
        );
    }
}

/// The baseline: unpack to one byte per cell, loop, pack.
fn bytes_row(up: &[u64], mid: &[u64], dn: &[u64], out: &mut [u64]) {
    let w = mid.len() * 64;
    let read = |row: &[u64], x: usize| ((row[x / 64] >> (x % 64)) & 1) as u8;
    out.fill(0);
    for x in 0..w {
        let (l, r) = ((x + w - 1) % w, (x + 1) % w);
        let n = read(up, l)
            + read(up, x)
            + read(up, r)
            + read(mid, l)
            + read(mid, r)
            + read(dn, l)
            + read(dn, x)
            + read(dn, r);
        if n == 3 || (n == 2 && read(mid, x) == 1) {
            out[x / 64] |= 1 << (x % 64);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_row(up: &[u64], mid: &[u64], dn: &[u64], out: &mut [u64]) {
    use std::arch::x86_64::*;
    let w = mid.len();
    if w < 6 {
        return swar_row(up, mid, dn, out, 0);
    }
    swar_range(up, mid, dn, out, 0, 1);
    let mut x = 1;
    // SAFETY: x-1..x+5 stays inside each slice while x + 5 <= w.
    unsafe {
        while x + 5 <= w {
            let load =
                |row: &[u64], i: usize| _mm256_loadu_si256(row.as_ptr().add(i) as *const __m256i);
            let (tl, tc, tr) = (load(up, x - 1), load(up, x), load(up, x + 1));
            let (ml, mc, mr) = (load(mid, x - 1), load(mid, x), load(mid, x + 1));
            let (bl, bc, br) = (load(dn, x - 1), load(dn, x), load(dn, x + 1));
            let west = |c: __m256i, l: __m256i| {
                _mm256_or_si256(_mm256_slli_epi64(c, 1), _mm256_srli_epi64(l, 63))
            };
            let east = |c: __m256i, r: __m256i| {
                _mm256_or_si256(_mm256_srli_epi64(c, 1), _mm256_slli_epi64(r, 63))
            };
            let n0 = west(tc, tl);
            let n2 = east(tc, tr);
            let n3 = west(mc, ml);
            let n4 = east(mc, mr);
            let n5 = west(bc, bl);
            let n7 = east(bc, br);
            let x01 = _mm256_xor_si256(n0, tc);
            let s0 = _mm256_xor_si256(x01, n2);
            let c0 = _mm256_or_si256(_mm256_and_si256(n0, tc), _mm256_and_si256(x01, n2));
            let x34 = _mm256_xor_si256(n3, n4);
            let s1 = _mm256_xor_si256(x34, n5);
            let c1 = _mm256_or_si256(_mm256_and_si256(n3, n4), _mm256_and_si256(x34, n5));
            let s2 = _mm256_xor_si256(bc, n7);
            let c2 = _mm256_and_si256(bc, n7);
            let xx = _mm256_xor_si256(s0, s1);
            let ones = _mm256_xor_si256(xx, s2);
            let c3 = _mm256_or_si256(_mm256_and_si256(s0, s1), _mm256_and_si256(xx, s2));
            let twos = _mm256_xor_si256(_mm256_xor_si256(c0, c1), _mm256_xor_si256(c2, c3));
            let many = _mm256_or_si256(
                _mm256_or_si256(_mm256_and_si256(c0, c1), _mm256_and_si256(c2, c3)),
                _mm256_and_si256(_mm256_or_si256(c0, c1), _mm256_or_si256(c2, c3)),
            );
            let res = _mm256_andnot_si256(many, _mm256_and_si256(twos, _mm256_or_si256(ones, mc)));
            _mm256_storeu_si256(out.as_mut_ptr().add(x) as *mut __m256i, res);
            x += 4;
        }
    }
    swar_row(up, mid, dn, out, x);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn avx512_row(up: &[u64], mid: &[u64], dn: &[u64], out: &mut [u64]) {
    use std::arch::x86_64::*;
    let w = mid.len();
    if w < 10 {
        return swar_row(up, mid, dn, out, 0);
    }
    swar_range(up, mid, dn, out, 0, 1);
    let mut x = 1;
    // SAFETY: x-1..x+9 stays inside each slice while x + 9 <= w.
    unsafe {
        while x + 9 <= w {
            let load = |row: &[u64], i: usize| _mm512_loadu_si512(row.as_ptr().add(i) as *const _);
            let (tl, tc, tr) = (load(up, x - 1), load(up, x), load(up, x + 1));
            let (ml, mc, mr) = (load(mid, x - 1), load(mid, x), load(mid, x + 1));
            let (bl, bc, br) = (load(dn, x - 1), load(dn, x), load(dn, x + 1));
            let west = |c: __m512i, l: __m512i| {
                _mm512_or_si512(_mm512_slli_epi64::<1>(c), _mm512_srli_epi64::<63>(l))
            };
            let east = |c: __m512i, r: __m512i| {
                _mm512_or_si512(_mm512_srli_epi64::<1>(c), _mm512_slli_epi64::<63>(r))
            };
            let n0 = west(tc, tl);
            let n2 = east(tc, tr);
            let n3 = west(mc, ml);
            let n4 = east(mc, mr);
            let n5 = west(bc, bl);
            let n7 = east(bc, br);
            // VPTERNLOG truth tables: 0x96 is three-way XOR, 0xE8 is majority.
            let s0 = _mm512_ternarylogic_epi64::<0x96>(n0, tc, n2);
            let c0 = _mm512_ternarylogic_epi64::<0xE8>(n0, tc, n2);
            let s1 = _mm512_ternarylogic_epi64::<0x96>(n3, n4, n5);
            let c1 = _mm512_ternarylogic_epi64::<0xE8>(n3, n4, n5);
            let s2 = _mm512_xor_si512(bc, n7);
            let c2 = _mm512_and_si512(bc, n7);
            let ones = _mm512_ternarylogic_epi64::<0x96>(s0, s1, s2);
            let c3 = _mm512_ternarylogic_epi64::<0xE8>(s0, s1, s2);
            let t = _mm512_ternarylogic_epi64::<0x96>(c0, c1, c2);
            let twos = _mm512_xor_si512(t, c3);
            let carry = _mm512_ternarylogic_epi64::<0xE8>(c0, c1, c2);
            let many = _mm512_or_si512(carry, _mm512_and_si512(t, c3));
            let res = _mm512_andnot_si512(many, _mm512_and_si512(twos, _mm512_or_si512(ones, mc)));
            _mm512_storeu_si512(out.as_mut_ptr().add(x) as *mut _, res);
            x += 8;
        }
    }
    swar_row(up, mid, dn, out, x);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_agrees_and_threads_do_too() {
        let reference = {
            let mut g = BitGrid::random(1024, 300, 42);
            for _ in 0..40 {
                g.step(Kernel::Bytes, false);
            }
            g.cur.clone()
        };
        for kernel in [Kernel::Swar, Kernel::Avx2, Kernel::Avx512] {
            if !kernel.available() {
                continue;
            }
            for threaded in [false, true] {
                let mut g = BitGrid::random(1024, 300, 42);
                for _ in 0..40 {
                    g.step(kernel, threaded);
                }
                assert_eq!(g.cur, reference, "{kernel:?} threaded={threaded}");
            }
        }
    }

    #[test]
    fn glider_moves_one_cell_per_four_generations() {
        let mut g = BitGrid::new(64, 64);
        for (x, y) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            g.set(x, y, true);
        }
        for _ in 0..4 {
            g.step(Kernel::Swar, false);
        }
        for (x, y) in [(1, 0), (2, 1), (0, 2), (1, 2), (2, 2)] {
            assert!(g.get(x + 1, y + 1));
        }
        assert_eq!(g.population(), 5);
    }
}
