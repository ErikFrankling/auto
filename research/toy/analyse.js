const fs = require('fs');
const f = process.argv[2]; const rows = fs.readFileSync(f, 'utf8').trim().split('\n').map(JSON.parse);
const by = {}; for (const r of rows) (by[r.preset] ||= []).push(r);
const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0, sd = (a) => { const m = mean(a); return Math.sqrt(mean(a.map(v => (v - m) ** 2))); };
for (const [p, rs] of Object.entries(by)) {
  const alive = rs.filter(r => r.extinct < 0), last = alive.map(r => r.pts.at(-1));
  const early = alive.map(r => r.pts[0]);
  const both = last.filter(s => s.A >= 0.1 * s.n && s.B >= 0.1 * s.n).length, genDom = last.filter(s => s.gen > 0.5 * s.n).length, oneSpec = last.filter(s => (s.A > 0.9 * s.n || s.B > 0.9 * s.n)).length;
  const row = { p, runs: rs.length, alive: alive.length, n: Math.round(mean(last.map(s => s.n))), both, genDom, oneSpec, chase: +mean(last.map(s => s.chase)).toFixed(3), chaseSd: +sd(last.map(s => s.chase)).toFixed(3), chase2k: +mean(early.map(s => s.chase)).toFixed(3), match: +mean(last.map(s => s.match)).toFixed(2), eatOn: +mean(last.map(s => s.eatOn)).toFixed(3), eatOff: +mean(last.map(s => s.eatOff)).toFixed(3), stole: +mean(last.map(s => s.stole / (s.ate + s.stole))).toFixed(3), shares: [0, 1, 2, 3, 4].map(j => +mean(last.map(s => s.shares[j])).toFixed(2)), born: Math.round(mean(rs.map(r => r.born))) , q: last.map(s => s.q.filter(v => v === 0).length).join('')};
  if (rs[0].shockAt) { row.minAfter = rs.map(r => r.minAfter).join(','); row.ext = rs.filter(r => r.extinct >= 0).length; row.n_ = rs.map(r => r.pts.at(-1)?.n).join(','); }
  console.log(JSON.stringify(row));
}
