// A tiny energy-only ecology. One RNN per cell, no fitness function.
function World(preset, seed) {
  const G = 64, N = 2400, I = 9, H = 4, O = 7, P = I * H + H * H + H * O, FCAP = 4;
  let s = seed >>> 0;
  const rnd = () => { s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-9)) * Math.cos(6.2832 * rnd());
  const C = G * G, fa = new Float32Array(C), fb = new Float32Array(C), ra = new Float32Array(C), rb = new Float32Array(C), wall = new Uint8Array(C), head = new Int32Array(C), next = new Int32Array(N), other = new Int32Array(N);
  const w = { preset, G, N, I, H, O, P, FCAP, n: 0, tick: 0, born: 0, fa, fb, ra, rb, wall, x: new Float32Array(N), y: new Float32Array(N), e: new Float32Array(N), womb: new Float32Array(N), t: new Float32Array(N), tag: new Float32Array(N), atk: new Float32Array(N), hit: new Uint8Array(N), id: new Int32Array(N), g: new Float32Array(N * P), h: new Float32Array(N * H), hist: [], sensors: new Float32Array(N * I), shares: new Float32Array(N * 5), rain: 1, extinct: -1, islandsDead: 0, killedBy: [0, 0], sunIn: 0, eaten: 0, stolen: 0, cosSum: 0, cosN: 0 };
  const SUN = 300 / C;
  for (let y = 0; y < G; y++) for (let x = 0; x < G; x++) {
    const i = y * G + x, u = x / G * 6.2832, v = y / G * 6.2832;
    if (preset === 'plain') ra[i] = SUN;
    if (preset === 'mixed') { ra[i] = SUN / 2; rb[i] = SUN / 2; }
    if (preset === 'biomes') { const q = Math.sin(u * 2 + 1) + Math.sin(v * 2 + 2 * Math.sin(u)); const m = q > 0 ? 1 : 0; ra[i] = SUN * m; rb[i] = SUN * (1 - m); }
    if (preset === 'islands' || preset === 'sealed') { const ix = x < G / 2 ? 0 : 1, iy = y < G / 2 ? 0 : 1; wall[i] = (x % (G / 2) < 2 || y % (G / 2) < 2) ? 1 : 0; const k = (ix + iy) % 2; if (!wall[i]) { if (k) ra[i] = 1; else rb[i] = 1; } }
  }
  if (preset === 'islands' || preset === 'sealed') { let tot = 0; for (let i = 0; i < C; i++) tot += ra[i] + rb[i]; const f = SUN * C / tot; for (let i = 0; i < C; i++) { ra[i] *= f; rb[i] *= f; } }
  const blobs = []; if (preset === 'drift') for (let k = 0; k < 6; k++) blobs.push({ x: rnd() * G, y: rnd() * G, a: rnd() * 6.28, f: k % 2 });
  const paintBlobs = () => { ra.fill(0); rb.fill(0); let tot = 0; const R = 7; for (const b of blobs) { b.a += (rnd() - 0.5) * 0.3; b.x = (b.x + Math.cos(b.a) * 0.6 + G) % G; b.y = (b.y + Math.sin(b.a) * 0.6 + G) % G; for (let dy = -R; dy <= R; dy++) for (let dx = -R; dx <= R; dx++) { if (dx * dx + dy * dy > R * R) continue; const i = ((b.y + dy + G) % G | 0) * G + ((b.x + dx + G) % G | 0); const r = b.f ? rb : ra; r[i] = 1; tot++; } } const f = SUN * C / Math.max(1, tot); for (let i = 0; i < C; i++) { ra[i] *= f; rb[i] *= f; } };
  if (preset === 'drift') paintBlobs();
  for (let i = 0; i < C; i++) { fa[i] = ra[i] > 0 ? FCAP : 0; fb[i] = rb[i] > 0 ? FCAP : 0; }
  w.spawn = (x, y, e, t, tag, parent) => {
    if (w.n >= N) return -1;
    const a = w.n++; w.x[a] = x; w.y[a] = y; w.e[a] = e; w.womb[a] = 0; w.atk[a] = 0; w.hit[a] = 0; w.t[a] = t; w.tag[a] = tag; w.id[a] = ++w.born;
    for (let j = 0; j < H; j++) w.h[a * H + j] = 0;
    if (parent < 0) for (let j = 0; j < P; j++) w.g[a * P + j] = gauss() * 0.7;
    else for (let j = 0; j < P; j++) w.g[a * P + j] = w.g[parent * P + j] + gauss() * 0.04;
    return a;
  };
  const free = () => { for (let k = 0; k < 40; k++) { const x = rnd() * G, y = rnd() * G; if (!wall[(y | 0) * G + (x | 0)]) return [x, y]; } return [G / 2 + 3, G / 2 + 3]; };
  const seedRain = () => { for (let k = 0; k < 8; k++) { const [x, y] = free(); w.spawn(x, y, 20, rnd(), rnd(), -1); } };
  const kill = (a) => { const i = (w.y[a] | 0) * G + (w.x[a] | 0); fa[i] = Math.min(FCAP, fa[i] + 0.5 * Math.max(0, w.e[a] + w.womb[a])); w.e[a] = -1e9; };
  const move = (a, b) => { w.x[a] = w.x[b]; w.y[a] = w.y[b]; w.e[a] = w.e[b]; w.womb[a] = w.womb[b]; w.t[a] = w.t[b]; w.tag[a] = w.tag[b]; w.atk[a] = w.atk[b]; w.hit[a] = w.hit[b]; w.id[a] = w.id[b]; for (let j = 0; j < P; j++) w.g[a * P + j] = w.g[b * P + j]; for (let j = 0; j < H; j++) w.h[a * H + j] = w.h[b * H + j]; for (let j = 0; j < I; j++) w.sensors[a * I + j] = w.sensors[b * I + j]; for (let j = 0; j < 5; j++) w.shares[a * 5 + j] = w.shares[b * 5 + j]; };
  const inp = new Float32Array(I), hn = new Float32Array(H), out = new Float32Array(O);
  w.step = () => {
    w.tick++;
    if (preset === 'drift' && w.tick % 10 === 0) paintBlobs();
    if (w.rain && w.n < 150) seedRain();
    if (w.tick >= 1500) w.rain = 0;
    for (let i = 0; i < C; i++) { if (ra[i]) fa[i] = Math.min(FCAP, fa[i] + ra[i]); if (rb[i]) fb[i] = Math.min(FCAP, fb[i] + rb[i]); }
    head.fill(-1);
    for (let a = 0; a < w.n; a++) { const i = (w.y[a] | 0) * G + (w.x[a] | 0); next[a] = head[i]; head[i] = a; }
    const n0 = w.n;
    for (let a = 0; a < n0; a++) {
      if (w.e[a] <= 0) continue;
      const xi = w.x[a] | 0, yi = w.y[a] | 0, i = yi * G + xi, t = w.t[a], ea = t * t, eb = (1 - t) * (1 - t);
      const ed = (j) => ea * fa[j] + eb * fb[j], ix = (xi + 1) % G, dx = (xi + G - 1) % G, iy = (yi + 1) % G, dy = (yi + G - 1) % G;
      let crowd = 0, nb = -1;
      for (let b = head[i]; b >= 0 && crowd < 8; b = next[b]) if (b !== a && w.e[b] > 0) { crowd++; nb = b; }
      inp[0] = Math.min(2, w.e[a] / 25); inp[1] = ed(i) / FCAP; inp[2] = (ed(yi * G + ix) - ed(yi * G + dx)) / FCAP; inp[3] = (ed(iy * G + xi) - ed(dy * G + xi)) / FCAP; inp[4] = crowd / 4; inp[5] = w.hit[a]; inp[6] = nb >= 0 ? 1 - 4 * Math.abs(w.tag[nb] - w.tag[a]) : 0; inp[7] = rnd() * 2 - 1; inp[8] = 1;
      for (let j = 0; j < I; j++) w.sensors[a * I + j] = inp[j];
      w.hit[a] = 0;
      const g = a * P, ho = a * H;
      for (let j = 0; j < H; j++) { let v = 0; for (let k = 0; k < I; k++) v += w.g[g + j * I + k] * inp[k]; for (let k = 0; k < H; k++) v += w.g[g + I * H + j * H + k] * w.h[ho + k]; hn[j] = Math.tanh(v); }
      for (let j = 0; j < H; j++) w.h[ho + j] = hn[j];
      for (let j = 0; j < O; j++) { let v = 0; for (let k = 0; k < H; k++) v += w.g[g + I * H + H * H + j * H + k] * hn[k]; out[j] = v; }
      let m = -1e9; for (let j = 0; j < 5; j++) if (out[j] > m) m = out[j];
      let z = 0; for (let j = 0; j < 5; j++) z += Math.exp(out[j] - m);
      const sh = (j) => Math.exp(out[j] - m) / z;
      const rest = sh(0), mv = sh(1), eat = sh(2), div = sh(3), atk = sh(4);
      const q = a * 5; w.shares[q] = rest; w.shares[q + 1] = mv; w.shares[q + 2] = eat; w.shares[q + 3] = div; w.shares[q + 4] = atk;
      w.atk[a] += (atk - w.atk[a]) * 0.02;
      const B = 1;
      w.e[a] -= 0.05 + B * (0.3 * mv + eat + div + atk);
      const want = eat * B * 6, ga = Math.min(fa[i], want * ea), gb = Math.min(fb[i], want * eb);
      fa[i] -= ga; fb[i] -= gb; w.e[a] += ga + gb; w.eaten += ga + gb;
      if (atk > 0.02 && nb >= 0 && w.e[nb] > 0) { const take = Math.min(w.e[nb], atk * B * 10); w.e[nb] -= take; w.e[a] += take * 0.6; w.hit[nb] = 1; w.stolen += take * 0.6; }
      const l = Math.hypot(out[5], out[6]) + 1e-6, sp = 2.2 * mv; { const gm = Math.hypot(inp[2], inp[3]); if (gm > 0.05) { w.cosSum += (out[5] * inp[2] + out[6] * inp[3]) / (l * gm); w.cosN++; } }
      let nx = w.x[a] + out[5] / l * sp, ny = w.y[a] + out[6] / l * sp; nx = Math.fround((nx + G) % G); ny = Math.fround((ny + G) % G); if (nx >= G) nx = 0; if (ny >= G) ny = 0;
      if (!wall[(ny | 0) * G + (nx | 0)]) { w.x[a] = nx; w.y[a] = ny; }
      if (w.e[a] > 60) w.e[a] = 60;
      w.womb[a] += div * B;
      if (w.womb[a] >= 14) { let bx = Math.fround((w.x[a] + (rnd() - 0.5) * 3 + G) % G), by = Math.fround((w.y[a] + (rnd() - 0.5) * 3 + G) % G); if (bx >= G) bx = 0; if (by >= G) by = 0; if (wall[(by | 0) * G + (bx | 0)]) { bx = w.x[a]; by = w.y[a]; } let tt = t + gauss() * 0.02; tt = tt < 0 ? 0 : tt > 1 ? 1 : tt; let tg = w.tag[a] + gauss() * 0.01; tg = tg < 0 ? 0 : tg > 1 ? 1 : tg; if (w.spawn(bx, by, w.womb[a], tt, tg, a) >= 0) w.womb[a] = 0; else w.womb[a] = 14; }
    }
    if (preset === 'islands' && w.n > 0 && rnd() < 0.01) { const a = (rnd() * w.n) | 0, [x, y] = free(); w.x[a] = x; w.y[a] = y; }
    for (let a = w.n - 1; a >= 0; a--) if (w.e[a] <= 0 || rnd() < 1 / 2500) { if (w.e[a] > -1e8) kill(a); w.n--; if (a !== w.n) move(a, w.n); }
    if (w.n === 0 && !w.rain && w.extinct < 0) w.extinct = w.tick;
  };
  w.shock = (frac) => { for (let a = 0; a < w.n; a++) if (rnd() < frac) w.e[a] = 0; };
  w.stats = () => {
    const n = w.n; let spec = [0, 0], gen = 0, atk = 0, e = 0, q = [0, 0, 0, 0];
    for (let a = 0; a < n; a++) { const t = w.t[a]; if (t > 0.75) spec[0]++; else if (t < 0.25) spec[1]++; else gen++; atk += w.atk[a]; e += w.e[a]; }
    if (preset === 'islands' || preset === 'sealed') for (let a = 0; a < n; a++) q[(w.x[a] < G / 2 ? 0 : 1) + (w.y[a] < G / 2 ? 0 : 2)]++;
    const sh = [0,0,0,0,0]; let onN=0,onE=0,offN=0,offE=0,pred=0,match=0,spc=0;
    for (let a = 0; a < n; a++) { for (let j=0;j<5;j++) sh[j]+=w.shares[a*5+j]; const fh=w.sensors[a*I+1], ee=w.shares[a*5+2]; if (fh>0.25){onN++;onE+=ee;} else {offN++;offE+=ee;} if (w.atk[a]>0.3) pred++; const t=w.t[a], i=(w.y[a]|0)*G+(w.x[a]|0); if (t>0.75||t<0.25){spc++; if ((t>0.75&&ra[i]>0)||(t<0.25&&rb[i]>0)) match++;} }
    const chase = w.cosN ? +(w.cosSum / w.cosN).toFixed(3) : 0; w.cosSum = 0; w.cosN = 0;
    return { chase, shares: sh.map(v=>n?+(v/n).toFixed(3):0), eatOn: onN?+(onE/onN).toFixed(3):0, eatOff: offN?+(offE/offN).toFixed(3):0, onFrac: n?+(onN/n).toFixed(2):0, pred: n?+(pred/n).toFixed(3):0, match: spc?+(match/spc).toFixed(2):0, tick: w.tick, n, A: spec[0], B: spec[1], gen, atk: n ? atk / n : 0, e: n ? e / n : 0, q, extinct: w.extinct, eaten: w.eaten, stolen: w.stolen };
  };
  return w;
}
if (typeof module !== 'undefined') module.exports = World;
