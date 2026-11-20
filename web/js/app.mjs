import { Grammar, T_RECT, T_ELL, T_TRI, T_BLOB, toSvg } from './core.mjs';

const $ = (id) => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';
const S = 512;
const LEVELS = [[1, 128], [2, 64], [4, 24], [8, 8], [16, 8], [32, 8]];
const AHEAD = 48; // shapes computed ahead of playback

let g = null, ready = false, runId = 0, ref = null, refName = '';
let els = [], tiles = [[0, 0, S, S]], groupOf = [], groups = [];
let planTotal = 0;
let shown = 0, granted = 0, playing = false, done = false;
let lastT = 0, acc = 0, rate = 0, lastTile = -1, curG = null;
let viewMode = 'side';
let splitPct = 50;

const worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
worker.onmessage = (e) => onWorker(e.data);
worker.onerror = (e) => fail(`Worker failed: ${e.message || 'unknown error'}`);
const params = new URLSearchParams(location.search);
const initialBackend = params.get('backend') || 'auto';
worker.postMessage({ type: 'init', backend: initialBackend, threads: params.get('threads') });
fetch('model/meta.json').then((r) => r.json()).then((m) => { g = new Grammar(m); });

const fmt = (x) => {
  const s = x.toFixed(2).replace(/0+$/, '').replace(/\.$/, '');
  return s === '-0' || s === '' ? '0' : s;
};

function createShapeNode(C) {
  const t = C[0] | 0;
  const rgb = `rgb(${Math.round(C[g.CR] * 255)},${Math.round(C[g.CR + 1] * 255)},${Math.round(C[g.CR + 2] * 255)})`;
  const a = fmt(C[g.CA]);
  if (t === T_RECT || t === T_ELL) {
    const [cx, cy, hw, hh, ang] = [C[1], C[2], C[3], C[4], C[5]];
    const el = document.createElementNS(SVG_NS, t === T_RECT ? 'rect' : 'ellipse');
    if (t === T_RECT) {
      el.setAttribute('x', fmt(cx - hw));
      el.setAttribute('y', fmt(cy - hh));
      el.setAttribute('width', fmt(2 * hw));
      el.setAttribute('height', fmt(2 * hh));
    } else {
      el.setAttribute('cx', fmt(cx));
      el.setAttribute('cy', fmt(cy));
      el.setAttribute('rx', fmt(hw));
      el.setAttribute('ry', fmt(hh));
    }
    if (ang) el.setAttribute('transform', `rotate(${fmt(ang * 180 / Math.PI)} ${fmt(cx)} ${fmt(cy)})`);
    el.setAttribute('fill', rgb);
    el.setAttribute('fill-opacity', a);
    return el;
  }
  if (t === T_TRI) {
    const el = document.createElementNS(SVG_NS, 'polygon');
    el.setAttribute('points', `${fmt(C[1])},${fmt(C[2])} ${fmt(C[3])},${fmt(C[4])} ${fmt(C[5])},${fmt(C[6])}`);
    el.setAttribute('fill', rgb);
    el.setAttribute('fill-opacity', a);
    return el;
  }
  const el = document.createElementNS(SVG_NS, 'path');
  if (t === T_BLOB) {
    const P = [[C[1], C[2]], [C[3], C[4]], [C[5], C[6]], [C[7], C[8]]];
    const mid = P.map((p, i) => [(p[0] + P[(i + 1) % 4][0]) / 2, (p[1] + P[(i + 1) % 4][1]) / 2]);
    let d = `M${fmt(mid[3][0])} ${fmt(mid[3][1])}`;
    for (let i = 0; i < 4; i++) d += `Q${fmt(P[i][0])} ${fmt(P[i][1])} ${fmt(mid[i][0])} ${fmt(mid[i][1])}`;
    el.setAttribute('d', d + 'Z');
    el.setAttribute('fill', rgb);
    el.setAttribute('fill-opacity', a);
    el.setAttribute('fill-rule', 'evenodd');
    return el;
  }
  el.setAttribute('d', `M${fmt(C[1])} ${fmt(C[2])}Q${fmt(C[3])} ${fmt(C[4])} ${fmt(C[5])} ${fmt(C[6])}`);
  el.setAttribute('fill', 'none');
  el.setAttribute('stroke', rgb);
  el.setAttribute('stroke-opacity', a);
  el.setAttribute('stroke-width', fmt(C[7]));
  el.setAttribute('stroke-linecap', 'round');
  return el;
}

function fail(text) {
  const m = $('msg');
  m.textContent = text;
  m.className = 'msg err';
  const b = $('backend');
  b.textContent = 'Model error';
  b.className = 'badge err';
}

function note(text) {
  const m = $('msg');
  m.textContent = text;
  m.className = 'msg';
}

function updateScrubberBars() {
  const denom = Math.max(planTotal, els.length, 1);
  $('scrubBuf').style.width = Math.min(100, (100 * els.length) / denom) + '%';
  $('scrubPlayed').style.width = Math.min(100, (100 * shown) / Math.max(els.length, 1)) * (els.length / denom) + '%';
}

function onWorker(m) {
  if (m.type === 'progress') {
    const pct = m.total ? (100 * m.loaded) / m.total : 50;
    $('prog').firstElementChild.style.width = pct + '%';
    $('backend').textContent = `Runtime: loading ${(m.loaded / 1e6).toFixed(0)} MB…`;
  } else if (m.type === 'status') {
    $('backend').textContent = `Runtime: ${m.text}`;
    $('runtimeSub').textContent = m.text;
  } else if (m.type === 'ready') {
    ready = true;
    $('prog').classList.add('done');
    $('engine').disabled = false;
    $('engine').value = m.backend === 'webgpu' ? 'webgpu' : 'wasm';
    const b = $('backend');
    if (m.backend === 'webgpu') {
      b.textContent = m.device ? `Runtime: WebGPU + WASM (${m.device})` : 'Runtime: WebGPU + WASM';
    } else {
      b.textContent = `Runtime: WASM · ${m.threads} thread${m.threads > 1 ? 's' : ''}`;
    }
    b.className = 'badge ok';
    b.title = m.notes.length ? m.notes.join(' · ') : `Execution provider: ${m.backend}`;
    $('runtimeSub').textContent = m.notes.length ? m.notes[0] : 'ONNX Runtime Web';
    if (m.notes.length) note(m.notes.join(' · '));
    if (ref) begin();
  } else if (m.type === 'group' && m.run === runId) {
    tiles.push(...m.newTiles);
    const startIdx = els.length;
    groups.push({ z: m.z, tx: m.tx, ty: m.ty, psnr: m.psnr, end: startIdx + m.elements.length });
    for (const e of m.elements) addElement(e, groups.length - 1);
    rate = m.rate;
    $('made').textContent = els.length.toLocaleString();
    $('rate').textContent = `${m.instRate.toFixed(1)} / s`;
    $('rateSub').textContent = `avg ${m.rate.toFixed(1)}/s · ${Math.round(m.msPerShape)} ms/shape`;
    $('runtimeSub').textContent = `decoding · ${Math.round(m.groupMs)} ms / group`;
    $('pos').max = els.length;
    $('save').disabled = !shown;
    updateScrubberBars();
    drawChart();
  } else if (m.type === 'idle' && m.run === runId) {
    $('runtimeSub').textContent = 'buffered ahead · waiting for playback';
  } else if (m.type === 'done' && m.run === runId) {
    done = true;
    planTotal = els.length;
    updateScrubberBars();
    $('runtimeSub').textContent = `finished · avg ${rate.toFixed(1)} shapes/s`;
    note(`Done: painted ${els.length.toLocaleString()} shapes.`);
  } else if (m.type === 'error') {
    fail(m.message);
  }
}

function addElement(e, gi) {
  const C = Float64Array.from(e.C);
  if (e.tile !== lastTile) {
    if (e.tile && !$('t' + e.tile)) {
      const [x, y, w, h] = tiles[e.tile];
      const cp = document.createElementNS(SVG_NS, 'clipPath');
      cp.id = 't' + e.tile;
      const r = document.createElementNS(SVG_NS, 'rect');
      r.setAttribute('x', String(x));
      r.setAttribute('y', String(y));
      r.setAttribute('width', String(w));
      r.setAttribute('height', String(h));
      cp.appendChild(r);
      $('defs').appendChild(cp);
    }
    curG = document.createElementNS(SVG_NS, 'g');
    if (e.tile) curG.setAttribute('clip-path', `url(#t${e.tile})`);
    $('els').appendChild(curG);
    lastTile = e.tile;
  }
  const node = createShapeNode(C);
  node.style.display = 'none';
  curG.appendChild(node);
  const counts = els.length ? els[els.length - 1].counts.slice() : [0, 0, 0, 0, 0];
  const typeId = C[0] | 0;
  if (typeId >= 0 && typeId < 5) counts[typeId] += 1;
  els.push({ C, tile: e.tile, node, counts });
  groupOf.push(gi);
}

function updateLoupe(t, gr) {
  const [x, y, w, h] = t;
  const ctx = $('zoomRef').getContext('2d');
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, 160, 160);
  if (ref) {
    ctx.imageSmoothingEnabled = w >= 160;
    ctx.drawImage($('ref'), x, y, w, h, 0, 0, 160, 160);
  }
  $('zoomSvg').setAttribute('viewBox', `${x} ${y} ${w} ${h}`);
  if (!gr || gr.z === 1) {
    $('loupeLabel').textContent = 'Full canvas (1×)';
  } else {
    $('loupeLabel').textContent = `${gr.z}× · tile ${gr.ty * gr.z + gr.tx + 1}/${gr.z * gr.z}`;
  }
}

function drawChart() {
  const c = $('chart');
  const ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  if (!groups.length) {
    ctx.fillStyle = '#8a8a82';
    ctx.font = '11px Menlo, Consolas, monospace';
    ctx.fillText('Waiting for first group…', 10, H / 2 + 4);
    return;
  }

  let lo = Infinity, hi = -Infinity;
  for (const gr of groups) {
    if (gr.psnr < lo) lo = gr.psnr;
    if (gr.psnr > hi) hi = gr.psnr;
  }
  lo = Math.floor(lo - 1);
  hi = Math.ceil(hi + 1);
  if (hi - lo < 4) hi = lo + 4;
  $('psnrPeak').textContent = `peak ${Math.max(...groups.map((x) => x.psnr)).toFixed(2)} dB`;

  const padL = 34, padR = 10, padT = 10, padB = 14;
  const cw = W - padL - padR, ch = H - padT - padB;
  const maxX = Math.max(els.length, 1);
  const xOf = (n) => padL + (n / maxX) * cw;
  const yOf = (v) => padT + (1 - (v - lo) / (hi - lo)) * ch;

  // Horizontal grid lines
  ctx.strokeStyle = '#e3e3dc';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#7c7c74';
  ctx.font = '10px Menlo, Consolas, monospace';
  for (const val of [lo, Math.round((lo + hi) / 2), hi]) {
    const yy = Math.round(yOf(val)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(W - padR, yy);
    ctx.stroke();
    ctx.fillText(String(val), 4, yy + 3);
  }

  // Zoom level transition lines
  let prevZ = 0;
  for (const gr of groups) {
    if (gr.z !== prevZ) {
      if (prevZ !== 0) {
        const xx = Math.round(xOf(gr.end - 1)) + 0.5;
        ctx.strokeStyle = '#d4d4cc';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(xx, padT);
        ctx.lineTo(xx, H - padB);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#6e6e66';
        ctx.fillText(`${gr.z}×`, xx + 3, padT + 9);
      }
      prevZ = gr.z;
    }
  }

  // Soft area fill under curve
  ctx.beginPath();
  ctx.moveTo(xOf(groups[0].end), H - padB);
  for (const gr of groups) ctx.lineTo(xOf(gr.end), yOf(gr.psnr));
  ctx.lineTo(xOf(groups[groups.length - 1].end), H - padB);
  ctx.closePath();
  ctx.fillStyle = 'rgba(43, 108, 176, 0.12)';
  ctx.fill();

  // Curve stroke
  ctx.beginPath();
  groups.forEach((gr, i) => {
    const xx = xOf(gr.end), yy = yOf(gr.psnr);
    if (i === 0) ctx.moveTo(xx, yy);
    else ctx.lineTo(xx, yy);
  });
  ctx.strokeStyle = '#2b6cb0';
  ctx.lineWidth = 1.75;
  ctx.stroke();

  // Current playback marker
  if (shown > 0) {
    const curGr = groups[groupOf[shown - 1]];
    const cx = xOf(shown), cy = yOf(curGr.psnr);
    ctx.strokeStyle = 'rgba(43, 108, 176, 0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, padT);
    ctx.lineTo(cx, H - padB);
    ctx.stroke();

    ctx.fillStyle = '#2b6cb0';
    ctx.beginPath();
    ctx.arc(cx, cy, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.2;
    ctx.stroke();
  }
}

function show(n, animate = false) {
  n = Math.max(0, Math.min(els.length, Math.round(n)));
  const prev = shown;
  const doPop = animate && $('animStrokes').checked && n > prev && n - prev <= 8;
  for (let k = Math.min(prev, n); k < Math.max(prev, n); k++) {
    const node = els[k].node;
    if (k < n) {
      if (doPop) node.classList.add('pop');
      else node.classList.remove('pop');
      node.style.display = '';
    } else {
      node.classList.remove('pop');
      node.style.display = 'none';
    }
  }
  shown = n;
  $('pos').value = n;
  $('count').textContent = n.toLocaleString();
  $('save').disabled = !n;
  $('stepBack').disabled = !n;
  $('stepFwd').disabled = n >= els.length;
  updateScrubberBars();

  const gr = n ? groups[groupOf[n - 1]] : null;
  $('psnr').textContent = gr ? `${gr.psnr.toFixed(2)} dB` : '–';
  const t = tiles[n ? els[n - 1].tile : 0];
  const whole = !gr || gr.z === 1;
  $('view').textContent = whole ? '1× full' : `${gr.z}× (${gr.ty * gr.z + gr.tx + 1}/${gr.z * gr.z})`;

  const counts = n ? els[n - 1].counts : [0, 0, 0, 0, 0];
  for (let i = 0; i < 5; i++) $('mix' + i).textContent = counts[i];

  const showBox = !whole && $('showTile').checked;
  for (const [id, tagId] of [['tileA', 'tileTagA'], ['tileB', 'tileTagB']]) {
    const s = $(id).style;
    s.display = showBox ? 'block' : 'none';
    s.left = t[0] / 5.12 + '%';
    s.top = t[1] / 5.12 + '%';
    s.width = t[2] / 5.12 + '%';
    s.height = t[3] / 5.12 + '%';
    $(tagId).textContent = gr ? `${gr.z}×` : '';
  }

  updateLoupe(t, gr);
  drawChart();

  if (!done && granted - n < AHEAD / 2) {
    worker.postMessage({ type: 'credit', n: AHEAD });
    granted += AHEAD;
  }
}

function setPlaying(p) {
  playing = p && ready && !!ref;
  $('play').textContent = playing ? 'Pause' : (shown && shown >= els.length && done ? 'Replay' : 'Play');
  if (playing) {
    if (done && shown >= els.length) show(0);
    lastT = performance.now();
    acc = 0;
    requestAnimationFrame(tick);
  }
}

function tick(now) {
  if (!playing) return;
  const sp = $('speed').value;
  const r = sp === 'live' ? Math.max(rate, 2) * (els.length - shown > 16 ? 1.5 : 1.0) : +sp;
  acc += ((now - lastT) / 1000) * r;
  lastT = now;
  const step = Math.floor(acc);
  if (step && shown < els.length) {
    acc -= step;
    show(shown + step, true);
  }
  if (shown >= els.length) acc = Math.min(acc, 1);
  if (done && shown >= els.length) {
    setPlaying(false);
    return;
  }
  requestAnimationFrame(tick);
}

function plan() {
  const deepest = +$('depth').value;
  return LEVELS.filter(([z]) => z <= deepest);
}

function reset() {
  runId++;
  worker.postMessage({ type: 'stop' });
  $('els').replaceChildren();
  $('defs').replaceChildren();
  els = [];
  groupOf = [];
  groups = [];
  tiles = [[0, 0, S, S]];
  lastTile = -1;
  curG = null;
  shown = 0;
  granted = 0;
  done = false;
  rate = 0;
  $('pos').max = 0;
  $('pos').value = 0;
  $('count').textContent = '0';
  $('made').textContent = '0';
  $('psnr').textContent = '–';
  $('rate').textContent = '–';
  $('rateSub').textContent = '';
  $('view').textContent = '–';
  $('psnrPeak').textContent = 'PSNR curve';
  for (let i = 0; i < 5; i++) $('mix' + i).textContent = '0';
  $('stepBack').disabled = true;
  $('stepFwd').disabled = true;
  updateScrubberBars();
  updateLoupe([0, 0, S, S], null);
  drawChart();
  note('');
}

function begin() {
  reset();
  const p = plan();
  planTotal = p.reduce((s, [z, per]) => s + z * z * per, 0);
  $('svgInfo').textContent = `up to ${planTotal.toLocaleString()} shapes`;
  $('runtimeSub').textContent = 'encoding first group…';
  granted = AHEAD;
  worker.postMessage({
    type: 'start',
    id: runId,
    ref: ref.slice(),
    plan: p,
    head: $('sharpen').checked,
    guard: true,
    credit: AHEAD,
  });
  $('play').disabled = false;
  $('restart').disabled = false;
  setPlaying(true);
}

function useImage(img, name) {
  const c = $('ref');
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, S, S);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const s = Math.min(w, h);
  ctx.drawImage(img, (w - s) / 2, (h - s) / 2, s, s, 0, 0, S, S);

  // Sync swipe reference canvas
  const sctx = $('swipeRef').getContext('2d');
  sctx.drawImage(c, 0, 0);

  const d = ctx.getImageData(0, 0, S, S).data;
  const P = S * S;
  ref = new Float32Array(3 * P);
  for (let i = 0; i < P; i++) {
    for (let k = 0; k < 3; k++) ref[k * P + i] = d[4 * i + k] / 255;
  }
  refName = name;
  $('dropHint').style.display = 'none';
  if (ready) begin();
  else note('Model is still loading; painting will start once ready.');
}

function loadUrl(url, name) {
  const img = new Image();
  img.onload = () => useImage(img, name);
  img.onerror = () => fail(`Could not load "${name}".`);
  img.src = url;
}

function loadFile(f) {
  if (!f || !f.type.startsWith('image/')) return;
  document.querySelectorAll('.thumb').forEach((b) => b.classList.remove('on'));
  const url = URL.createObjectURL(f);
  loadUrl(url, f.name.replace(/\.[^.]+$/, ''));
}

// View mode switching & swipe divider
function setViewMode(mode) {
  viewMode = mode;
  const stage = $('stage');
  stage.classList.toggle('mode-solo', mode === 'solo');
  stage.classList.toggle('mode-swipe', mode === 'swipe');
  document.querySelectorAll('.seg button').forEach((btn) => {
    const on = btn.dataset.mode === mode;
    btn.classList.toggle('active', on);
    btn.setAttribute('aria-selected', String(on));
  });
  applySplit(splitPct);
}

function applySplit(pct) {
  splitPct = Math.max(0, Math.min(100, pct));
  $('swipeDivider').style.left = splitPct + '%';
  $('swipeRef').style.clipPath = `inset(0 ${100 - splitPct}% 0 0)`;
}

let draggingSplit = false;
const updateSplitFromEvent = (e) => {
  const rect = $('svgPane').getBoundingClientRect();
  const clientX = e.touches ? e.touches[0].clientX : e.clientX;
  applySplit(((clientX - rect.left) / rect.width) * 100);
};
$('swipeDivider').addEventListener('pointerdown', (e) => {
  draggingSplit = true;
  $('swipeDivider').setPointerCapture(e.pointerId);
  updateSplitFromEvent(e);
});
$('swipeDivider').addEventListener('pointermove', (e) => {
  if (draggingSplit) updateSplitFromEvent(e);
});
$('swipeDivider').addEventListener('pointerup', () => { draggingSplit = false; });
$('swipeDivider').addEventListener('pointercancel', () => { draggingSplit = false; });
applySplit(50);

document.querySelectorAll('.seg button').forEach((btn) => {
  btn.addEventListener('click', () => setViewMode(btn.dataset.mode));
});

// Load sample gallery safely using DOM APIs
fetch('samples/samples.json')
  .then((r) => (r.ok ? r.json() : []))
  .then((list) => {
    let firstBtn = null;
    for (const s of list) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'thumb';
      b.title = s.credit ? `${s.title} (${s.credit})` : s.title;
      const im = document.createElement('img');
      im.src = `samples/${s.file}`;
      im.alt = s.title;
      b.appendChild(im);
      b.onclick = () => {
        document.querySelectorAll('.thumb').forEach((x) => x.classList.toggle('on', x === b));
        loadUrl(`samples/${s.file}`, s.file.replace(/\.[^.]+$/, ''));
      };
      $('pick').appendChild(b);
      if (!firstBtn) firstBtn = b;
    }
    // Auto-select the macaw or first sample so the page comes alive right away
    const macaw = [...$('pick').querySelectorAll('button.thumb')].find((b) => b.title.toLowerCase().includes('macaw'));
    (macaw || firstBtn)?.click();
  })
  .catch(() => {});

$('file').onchange = (e) => loadFile(e.target.files[0]);

for (const paneId of ['refPane', 'svgPane']) {
  const pane = $(paneId);
  pane.ondragover = (e) => { e.preventDefault(); pane.classList.add('dragging'); };
  pane.ondragleave = () => pane.classList.remove('dragging');
  pane.ondrop = (e) => {
    e.preventDefault();
    pane.classList.remove('dragging');
    loadFile(e.dataTransfer.files[0]);
  };
}
document.addEventListener('paste', (e) => {
  const f = [...(e.clipboardData?.files || [])][0];
  if (f) loadFile(f);
});

$('play').onclick = () => setPlaying(!playing);
$('stepBack').onclick = () => { setPlaying(false); show(shown - 1); };
$('stepFwd').onclick = () => { setPlaying(false); show(shown + 1, true); };
$('restart').onclick = () => { if (ref && ready) begin(); };
$('engine').onchange = (e) => {
  const next = e.target.value;
  ready = false;
  $('engine').disabled = true;
  $('play').disabled = true;
  $('restart').disabled = true;
  $('backend').textContent = `Runtime: switching to ${next === 'webgpu' ? 'WebGPU' : 'WASM'}…`;
  $('backend').className = 'badge';
  $('runtimeSub').textContent = 'initializing session…';
  reset();
  worker.postMessage({ type: 'switchBackend', backend: next });
};
$('depth').onchange = () => { if (ref && ready) begin(); };
$('sharpen').onchange = () => { if (ref && ready) begin(); };
$('pos').oninput = (e) => { setPlaying(false); show(+e.target.value); };
$('showTile').onchange = () => show(shown);

$('save').onclick = () => {
  const svg = toSvg(g, els.slice(0, shown), tiles);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }));
  a.download = `${refName || 'painting'}-${shown}.svg`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};

document.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT'].includes(e.target.tagName) && e.target.type !== 'range') return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!playing); }
  if (e.code === 'ArrowRight') { setPlaying(false); show(shown + 1, true); }
  if (e.code === 'ArrowLeft') { setPlaying(false); show(shown - 1); }
});
