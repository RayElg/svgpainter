// Paints off the main thread. In: init, switchBackend, start, credit, stop.
// Out: progress, status, ready, group, idle, done, error.
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';
import { Grammar, OrtBackend, EdgeHead, paint } from './core.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
const defaultThreads = () => (self.crossOriginIsolated ? Math.max(1, Math.min(8, (navigator.hardwareConcurrency || 4) - 1)) : 1);

let g, head, backend, encBytes = null, decBytes = null;
let credit = 0, wake = null, run = 0, abortCtrl = null, current = Promise.resolve();
const post = (m) => postMessage(m);

async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get('content-length') || 0, parts = [];
  let loaded = 0;
  const rd = r.body.getReader();
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    parts.push(value); loaded += value.length;
    onProgress(value.length, total);
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

async function gpuAdapterName() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    const info = adapter?.info || (await adapter?.requestAdapterInfo?.());
    return info?.description || info?.device || info?.vendor || '';
  } catch {
    return '';
  }
}

// Workaround for ORT WebGPU Conv2d shader bug when C_in=9 (not a multiple of 4):
// pad enc.onnx's first Concat from 9 -> 12 channels (appending `ref`) and zero-pad enc.stem.0.weight [48,9,3,3] -> [48,12,3,3].
// No-op if enc.onnx was already exported with 12 input channels.
function patchEncConv9To12(buf) {
  const cat = (...arrs) => { const n = arrs.reduce((s, a) => s + a.length, 0), o = new Uint8Array(n); let k = 0; for (const a of arrs) { o.set(a, k); k += a.length; } return o; };
  const vi = (n) => { const r = []; while (n > 127) { r.push((n & 127) | 128); n >>>= 7; } r.push(n); return Uint8Array.from(r); };
  const vi64 = (n) => { let b = BigInt(n); if (b < 0n) b = (1n << 64n) + b; const r = []; while (b > 127n) { r.push(Number((b & 127n) | 128n)); b >>= 7n; } r.push(Number(b)); return Uint8Array.from(r); };
  const decStr = (b) => new TextDecoder().decode(b);
  const encField = (x) => {
    const t = vi((x.f << 3) | x.w);
    if (x.w === 0) return cat(t, vi64(x.v));
    if (x.w === 2) return cat(t, vi(x.b.length), x.b);
    return cat(t, x.b);
  };
  const readFields = (b) => {
    const out = [];
    for (let p = 0; p < b.length;) {
      let tag = 0, sh = 0;
      for (;;) { const c = b[p++]; tag |= (c & 127) << sh; if (!(c & 128)) break; sh += 7; }
      const f = tag >>> 3, w = tag & 7;
      if (w === 0) {
        let v = 0n, s = 0n;
        for (;;) { const c = BigInt(b[p++]); v |= (c & 127n) << s; if (!(c & 128n)) break; s += 7n; }
        out.push({ f, w, v });
      } else if (w === 2) {
        let len = 0, s = 0;
        for (;;) { const c = b[p++]; len |= (c & 127) << s; if (!(c & 128)) break; s += 7; }
        out.push({ f, w, b: b.subarray(p, p + len) });
        p += len;
      } else if (w === 5) { out.push({ f, w, b: b.subarray(p, p + 4) }); p += 4; }
      else if (w === 1) { out.push({ f, w, b: b.subarray(p, p + 8) }); p += 8; }
      else return buf;
    }
    return out;
  };
  const m = readFields(buf);
  const gIdx = m.findIndex((x) => x.f === 7);
  if (gIdx < 0) return buf;
  const g = readFields(m[gIdx].b);
  let patched = 0;
  for (const item of g) {
    if (item.f === 1) {
      const nf = readFields(item.b);
      const op = nf.find((x) => x.f === 4);
      const outs = nf.filter((x) => x.f === 2).map((x) => decStr(x.b));
      const ins = nf.filter((x) => x.f === 1);
      if (op && decStr(op.b) === 'Concat' && outs[0] === '/Concat_output_0' && ins.length === 3) {
        const lastIn = nf.map((x) => x.f).lastIndexOf(1);
        nf.splice(lastIn + 1, 0, { f: 1, w: 2, b: new TextEncoder().encode('ref') });
        item.b = cat(...nf.map(encField));
        patched++;
      }
    } else if (item.f === 5) {
      const tf = readFields(item.b);
      const name = tf.find((x) => x.f === 8);
      if (name && decStr(name.b) === 'enc.stem.0.weight') {
        const dims = tf.filter((x) => x.f === 1);
        if (dims.length === 4 && dims[1].v === 9n) {
          dims[1].v = 12n;
          const raw = tf.find((x) => x.f === 9);
          const oldF = new Float32Array(Uint8Array.from(raw.b).buffer);
          const newF = new Float32Array(48 * 12 * 3 * 3);
          for (let o = 0; o < 48; o++) newF.set(oldF.subarray(o * 9 * 9, (o + 1) * 9 * 9), o * 12 * 9);
          raw.b = new Uint8Array(newF.buffer);
          item.b = cat(...tf.map(encField));
          patched++;
        }
      }
    }
  }
  if (patched !== 2) return buf;
  m[gIdx].b = cat(...g.map(encField));
  return cat(...m.map(encField));
}

async function createBackend(pref) {
  const order = pref === 'auto' ? ['webgpu', 'wasm'] : (pref === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm']);
  const notes = [];
  if (backend) { await backend.dispose?.(); backend = null; }
  for (const ep of order) {
    if (ep === 'webgpu' && !('gpu' in navigator)) { notes.push('WebGPU is not available in this browser'); continue; }
    try {
      const isGpu = ep === 'webgpu';
      // Run the heavy 512x512 ViT encoder on WebGPU and the tiny 1-token causal decoder on WASM
      // to avoid 106 synchronous GPU->CPU logits readbacks per group.
      const encSession = { executionProviders: isGpu ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' };
      const decSession = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
      post({ type: 'status', text: `initializing ${isGpu ? 'WebGPU' : 'WASM'} encoder…` });
      const enc = await ort.InferenceSession.create(encBytes, encSession);
      post({ type: 'status', text: 'initializing WASM decoder…' });
      const dec = await ort.InferenceSession.create(decBytes, decSession);
      backend = new OrtBackend(ort, g, enc, dec, false);
      const device = isGpu ? await gpuAdapterName() : (self.crossOriginIsolated ? 'SIMD' : 'single-thread');
      post({ type: 'ready', backend: ep, threads: ort.env.wasm.numThreads, device, notes });
      return;
    } catch (e) { notes.push(`${ep} failed: ${e.message || e}`); }
  }
  post({ type: 'error', message: 'The model could not be loaded. ' + notes.join(' · ') });
}

async function init(pref, threads) {
  try {
    ort.env.wasm.numThreads = self.crossOriginIsolated && threads ? threads : defaultThreads();
    const base = new URL('../model/', import.meta.url);
    g = new Grammar(await (await fetch(new URL('meta.json', base))).json());
    head = new EdgeHead(g, await (await fetch(new URL('head.bin', base))).arrayBuffer());
    let loaded = 0, total = 0;
    const seen = new Set();
    const prog = (key) => (n, t) => {
      if (!seen.has(key)) { seen.add(key); total += t; }
      loaded += n;
      post({ type: 'progress', loaded, total });
    };
    [encBytes, decBytes] = await Promise.all([
      fetchBytes(new URL('enc.onnx', base), prog('enc')),
      fetchBytes(new URL('dec.onnx', base), prog('dec')),
    ]);
    encBytes = patchEncConv9To12(encBytes);
    await createBackend(pref);
  } catch (e) {
    post({ type: 'error', message: String(e.message || e) });
  }
}

async function switchBackend(pref) {
  run = -1;
  abortCtrl?.abort();
  wake?.(); wake = null;
  await current;
  if (!encBytes || !decBytes) return;
  try {
    await createBackend(pref);
  } catch (e) {
    post({ type: 'error', message: String(e.message || e) });
  }
}

async function start(m) {
  const my = m.id;
  run = my;
  abortCtrl?.abort();
  wake?.(); wake = null;
  await current;
  if (my !== run) return;
  const ctrl = new AbortController();
  abortCtrl = ctrl;
  credit = m.credit || 32;
  current = (async () => {
    let sent = 1, made = 0, activeMs = 0;
    let groupStart = performance.now();
    try {
      for await (const step of paint(g, backend, m.ref, m.plan, { head: m.head ? head : null, guard: m.guard, signal: ctrl.signal })) {
        if (my !== run || ctrl.signal.aborted) return;
        const groupMs = Math.max(1, performance.now() - groupStart);
        const nEl = step.elements.length;
        made += nEl;
        activeMs += groupMs;
        const avgRate = made / (activeMs / 1000);
        const instRate = nEl > 0 ? nEl / (groupMs / 1000) : avgRate;
        const msPerShape = nEl > 0 ? groupMs / nEl : 0;
        post({
          type: 'group', run: my, z: step.z, tx: step.tx, ty: step.ty,
          elements: step.elements.map((e) => ({ C: Array.from(e.C), tile: e.tile })),
          newTiles: step.tiles.slice(sent), psnr: -10 * Math.log10(Math.max(step.mse, 1e-10)),
          rate: avgRate, instRate, groupMs, msPerShape,
        });
        sent = step.tiles.length;
        credit -= Math.max(1, nEl);
        if (credit <= 0 && my === run && !ctrl.signal.aborted) {
          post({ type: 'idle', run: my });
          while (credit <= 0 && my === run && !ctrl.signal.aborted) await new Promise((r) => { wake = r; });
        }
        if (my !== run || ctrl.signal.aborted) return;
        groupStart = performance.now();
      }
      if (my === run && !ctrl.signal.aborted) post({ type: 'done', run: my });
    } catch (e) {
      if (my === run && !ctrl.signal.aborted) post({ type: 'error', message: String(e.message || e) });
    }
  })();
}

onmessage = (e) => {
  const m = e.data;
  if (m.type === 'init') init(m.backend || 'auto', +m.threads || 0);
  else if (m.type === 'switchBackend') switchBackend(m.backend || 'auto');
  else if (m.type === 'start') start(m);
  else if (m.type === 'credit') { credit += m.n; wake?.(); wake = null; }
  else if (m.type === 'stop') { run = -1; abortCtrl?.abort(); wake?.(); wake = null; }
};
