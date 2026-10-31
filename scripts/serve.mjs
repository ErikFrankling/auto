// Serves the browser build. usage: node scripts/serve.mjs <dist dir> <port>
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';

const [dist, port] = process.argv.slice(2);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
createServer(async (req, res) => {
  const path = new URL(req.url, 'http://x').pathname;
  try {
    const data = await readFile(join(dist, path === '/' ? 'index.html' : path));
    res.writeHead(200, { 'content-type': types[extname(path)] ?? 'text/html' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end();
  }
}).listen(Number(port), '0.0.0.0');
