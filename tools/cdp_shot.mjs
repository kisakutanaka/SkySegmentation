// ヘッドレス Chrome にページを開かせ、待ってからスクリーンショットを保存する最小の CDP クライアント。
// 使い方: node tools/cdp_shot.mjs <url> <out.png> [waitMs]
// 事前に Chrome を --headless=new --remote-debugging-port=<port> で起動しておくこと（既定 9333、CDP_PORT で変更可）。
const [, , url, outPath, waitMsArg] = process.argv;
const waitMs = Number(waitMsArg || 15000);
const port = process.env.CDP_PORT || '9333';
const tab = await (await fetch(`http://127.0.0.1:${port}/json/new?` + encodeURIComponent(url), { method: 'PUT' })).json();
const ws = new WebSocket(tab.webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const send = (method, params = {}) => new Promise(res => { pending.set(++id, res); ws.send(JSON.stringify({ id, method, params })); });
ws.onmessage = e => {
  const m = JSON.parse(e.data);
  if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  if (m.method === 'Runtime.consoleAPICalled') console.log('[console]', m.params.args.map(a => a.value).join(' '));
  if (m.method === 'Runtime.exceptionThrown') console.log('[exception]', m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
};
await new Promise(r => (ws.onopen = r));
await send('Runtime.enable');
await send('Page.enable');
await new Promise(r => setTimeout(r, 1500));
await send('Runtime.evaluate', { expression: "document.getElementById('start')?.click()" });
await new Promise(r => setTimeout(r, waitMs));
const stats = await send('Runtime.evaluate', { expression: "document.getElementById('stats')?.textContent + ' | ' + document.getElementById('status')?.textContent" });
console.log('[stats]', stats.result?.value);
const shot = await send('Page.captureScreenshot', { format: 'png' });
(await import('node:fs')).writeFileSync(outPath, Buffer.from(shot.data, 'base64'));
console.log('saved', outPath);
process.exit(0);
