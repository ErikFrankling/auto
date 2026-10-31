const World = require('./engine.js');
const [preset, seed, T, shockAt, shockFrac] = [process.argv[2], +process.argv[3], +process.argv[4], +process.argv[5] || 0, +process.argv[6] || 0];
const w = World(preset, seed); const marks = [2000, 10000, 30000, 60000, 120000]; const pts = []; const series = [];
let minAfter = 1e9, chaseWin = 0;
for (let i = 1; i <= T; i++) {
  w.step();
  if (shockAt && i === shockAt) w.shock(shockFrac);
  if (shockAt && i > shockAt) minAfter = Math.min(minAfter, w.n);
  if (i % 500 === 0) series.push(w.n);
  if (marks.includes(i) || i === T) { const s = w.stats(); pts.push({ t: i, n: s.n, A: s.A, B: s.B, gen: s.gen, chase: s.chase, pred: s.pred, match: s.match, shares: s.shares, eatOn: s.eatOn, eatOff: s.eatOff, onFrac: s.onFrac, q: s.q, ate: Math.round(s.eaten), stole: Math.round(s.stolen) }); w.cosSum = 0; w.cosN = 0; }
  if (w.extinct >= 0) break;
}
console.log(JSON.stringify({ preset, seed, T, shockAt, shockFrac, extinct: w.extinct, born: w.born, minAfter: minAfter === 1e9 ? null : minAfter, pts, series }));
