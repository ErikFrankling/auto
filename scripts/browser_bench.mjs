// Runs the browser build's benchmark in a real browser and appends the results.
// The page posts its result back, so any browser works.
// BROWSER=chrome-agent opens real Chrome on the agent monitor, which is what it takes to get
// the real GPU: headless Chrome falls back to its software renderer.
// usage: BROWSER=chrome|chrome-agent|firefox node scripts/browser_bench.mjs <dist dir> <out.jsonl> id:WxH [id:WxH ...]
import { spawn, execFileSync } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile, appendFile, mkdir, writeFile, rm } from 'node:fs/promises';
import { extname, join, dirname } from 'node:path';

const [dist, out, ...jobs] = process.argv.slice(2);
const browser = process.env.BROWSER ?? 'chrome';
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
let waiting = null;
const server = createServer(async (req, res) => {
  const path = new URL(req.url, 'http://x').pathname;
  if (req.method === 'POST' && path === '/report') {
    let body = '';
    for await (const chunk of req) body += chunk;
    res.writeHead(204);
    res.end();
    waiting?.(body);
    return;
  }
  try {
    const data = await readFile(join(dist, path === '/' ? 'index.html' : path));
    res.writeHead(200, { 'content-type': types[extname(path)] ?? 'text/html' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(0, '127.0.0.1');
await new Promise((r) => server.once('listening', r));
const port = server.address().port;
const seconds = process.env.SECONDS_PER_RUN ?? '3';
const extra = (process.env.BROWSER_FLAGS ?? '').split(' ').filter(Boolean);

async function launch(url) {
  const profile = `/tmp/auto-${browser}-${process.pid}`;
  if (browser === 'firefox') {
    await rm(profile, { recursive: true, force: true });
    await mkdir(profile, { recursive: true });
    await writeFile(join(profile, 'user.js'), [
      'user_pref("dom.webgpu.enabled", true);',
      'user_pref("gfx.webgpu.ignore-blocklist", true);',
      'user_pref("gfx.webrender.all", true);',
      'user_pref("browser.shell.checkDefaultBrowser", false);',
      'user_pref("datareporting.policy.dataSubmissionEnabled", false);',
    ].join('\n'));
    return spawn('firefox', ['--headless', '--no-remote', '--profile', profile, url], { stdio: 'ignore' });
  }
  if (browser === 'chrome-agent') {
    execFileSync('hyprctl', ['-i', '0', 'dispatch', 'exec',
      `[workspace name:agent silent] env HYPRLAND_AGENT_SEAT=1 VK_DRIVER_FILES=/run/opengl-driver/share/vulkan/icd.d/radeon_icd.x86_64.json google-chrome --user-data-dir=${profile} --no-first-run --no-default-browser-check --ozone-platform=wayland --hyprland-agent-seat --class=auto-chrome-test --enable-unsafe-webgpu --enable-features=Vulkan --use-vulkan=native --use-webgpu-adapter=vulkan --ignore-gpu-blocklist '${url}'`]);
    return {
      kill() {
        const clients = JSON.parse(execFileSync('hyprctl', ['-i', '0', 'clients', '-j']).toString());
        for (const c of clients.filter((c) => c.class === 'auto-chrome-test')) {
          execFileSync('hyprctl', ['-i', '0', 'dispatch', 'closewindow', `address:${c.address}`]);
        }
      },
    };
  }
  return spawn('google-chrome', [
    '--headless=new', `--user-data-dir=${profile}`, '--no-first-run', '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist', '--disable-background-timer-throttling', ...extra, url,
  ], { stdio: 'ignore' });
}

await mkdir(dirname(out), { recursive: true });
for (const job of jobs) {
  const [id, size] = job.split(':');
  // `view:lenia` runs the live viewer instead of a benchmark and reports what it drew.
  const url = id === 'view'
    ? `http://127.0.0.1:${port}/?a=${size}&report=1`
    : `http://127.0.0.1:${port}/?bench=${encodeURIComponent(id)}&size=${size}&seconds=${seconds}&report=1`;
  const result = new Promise((r) => {
    waiting = r;
    setTimeout(() => r(null), 180000);
  });
  const child = await launch(url);
  const value = await result;
  child.kill();
  await new Promise((r) => setTimeout(r, 1000));
  if (id === 'view' && value) {
    const row = JSON.parse(value);
    await writeFile(join(dirname(out), `view-${size}.png`), Buffer.from(row.image.split(',')[1], 'base64'));
    console.log(row.stats);
    continue;
  }
  console.log(value ?? `${job}: timed out`);
  if (value && !value.includes('"error"')) await appendFile(out, value + '\n');
}
server.close();
process.exit(0);
