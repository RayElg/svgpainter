// The painter in JavaScript: grammar, greedy decoding over the ONNX step graph, rasteriser,
// magnified tile views and the edge head. Images are Float32Array (3, S, S) in [0, 1].

export const K_START = 0, K_POS = 1, K_SIZE = 2, K_ANG = 3, K_COL = 4, K_ALPHA = 5, K_NONE = 6;
export const T_RECT = 0, T_ELL = 1, T_TRI = 2, T_CURVE = 3, T_BLOB = 4;
export const TYPE_NAMES = ['rect', 'ellipse', 'tri', 'curve', 'blob'];

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

export class Grammar {
  constructor(meta) {
    Object.assign(this, meta);
    this.COL_KIND = [];
    for (let t = 0; t < this.N_TYPES; t++) {
      const row = new Array(this.W).fill(K_NONE);
      for (let j = 0; j < this.TOK_LEN[t]; j++) row[this.TOK_COL[t][j]] = this.TOK_KIND[t][j];
      this.COL_KIND.push(row);
    }
  }

  deq(kind, q) {
    switch (kind) {
      case K_POS: return this.POS_MIN + this.POS_STEP * q;
      case K_SIZE: { const u = q / (this.N_SIZE - 1); return this.SIZE_MIN + this.SIZE_SPAN * u * u; }
      case K_ANG: return q * (Math.PI / this.N_ANG);
      case K_COL: return q / (this.N_COL - 1);
      case K_ALPHA: return (q + 1) / this.N_ALPHA;
      case K_START: return q;
      default: return 0;
    }
  }

  q2c(Q) {
    const t = clamp(Q[0], 0, this.N_TYPES - 1), C = new Float64Array(this.W), ck = this.COL_KIND[t];
    for (let c = 0; c < this.W; c++) C[c] = this.deq(ck[c], Q[c]);
    C[0] = Q[0];
    return C;
  }

  anchor(C, j) {
    const t = C[0] | 0, x1 = C[1], y1 = C[2], x2 = C[3], y2 = C[4], x3 = C[5], y3 = C[6], x4 = C[7], y4 = C[8];
    if (j < 2) return [0, 0, 0, 0];
    if (j === 2) return [x1, 0, 1, 0];
    if (t <= T_ELL || j === 3) return [x1, y1, 1, 1];
    if (j === 4) return [x2, 0, 1, 0];
    if (j === 5) return [x2, y2, 1, 1];
    if (j === 6) return [x3, 0, 1, 0];
    if (t === T_TRI) return [(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, 1, 1];
    if (t === T_CURVE) return [0.25 * x1 + 0.5 * x2 + 0.25 * x3, 0.25 * y1 + 0.5 * y2 + 0.25 * y3, 1, 1];
    if (j === 7) return [x3, y3, 1, 1];
    if (j === 8) return [x4, 0, 1, 0];
    return [(x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4, 1, 1];
  }

  clampShape(C) {
    const ck = this.COL_KIND[C[0] | 0], hi = this.POS_MIN + this.POS_STEP * (this.N_POS - 1);
    for (let c = 1; c < this.W; c++) {
      if (ck[c] === K_POS) C[c] = clamp(C[c], this.POS_MIN, hi);
      else if (ck[c] === K_SIZE) C[c] = clamp(C[c], this.SIZE_MIN, this.SIZE_MIN + this.SIZE_SPAN);
    }
    return C;
  }

  viewToImage(C, s, ox, oy) {
    const out = Float64Array.from(C), ck = this.COL_KIND[C[0] | 0];
    for (let c = 1; c < this.W; c++) {
      if (ck[c] === K_POS) out[c] = C[c] * s + (c % 2 === 1 ? ox : oy);
      else if (ck[c] === K_SIZE) out[c] = C[c] * s;
    }
    return out;
  }
}

export class Stepper {
  constructor(g, K) {
    this.g = g; this.K = K;
    this.Q = Array.from({ length: K }, () => new Int32Array(g.W));
    this.n = 0; this.j = 0; this.t = 0; this.done = false;
  }
  nextSlot() {
    const g = this.g;
    if (this.done) return g.SLOT_PAD;
    if (this.j === 0) return g.SLOT_START;
    return 3 + this.t * (g.MAXTOK - 1) + (this.j - 1);
  }
  nextKidx() { return this.done ? this.K : this.n; }
  nextAnchor() {
    if (this.done) return [0, 0, 0, 0];
    const C = this.g.q2c(this.Q[Math.min(this.n, this.K - 1)]);
    return this.g.anchor(C, Math.min(this.j, this.g.MAXTOK - 1));
  }
  pick(logits) {
    const g = this.g;
    if (this.done) return g.PAD;
    let best = -1, bv = -Infinity;
    const scan = (lo, hi) => { for (let i = lo; i < hi; i++) if (logits[i] > bv) { bv = logits[i]; best = i; } };
    if (this.j === 0) {
      scan(g.END, g.END + 1);
      if (this.n < this.K) scan(g.TYPE0, g.TYPE0 + g.N_TYPES);
    } else {
      const kind = g.TOK_KIND[this.t][this.j];
      scan(g.KIND_BASE[kind], g.KIND_BASE[kind] + g.KIND_N[kind]);
    }
    return best;
  }
  push(tok) {
    const g = this.g;
    if (this.done) return;
    const nn = Math.min(this.n, this.K - 1);
    if (this.j === 0) {
      if (tok === g.END) { this.done = true; return; }
      this.t = tok - g.TYPE0;
      this.Q[nn][0] = this.t;
    } else {
      const jj = Math.min(this.j, g.MAXTOK - 1), kind = g.TOK_KIND[this.t][jj];
      this.Q[nn][g.TOK_COL[this.t][jj]] = tok - g.KIND_BASE[kind];
    }
    this.j += 1;
    if (this.j >= g.TOK_LEN[this.t]) { this.n += 1; this.j = 0; }
  }
}

// clip = [x0, y0, x1, y1)
export function drawElement(g, img, S, C, clip) {
  const a = C[g.CA], r = C[g.CR], gg = C[g.CR + 1], b = C[g.CR + 2], P = S * S;
  forCoverage(g, S, C, clip, (i, cov) => {
    const w = cov * a;
    img[i] += w * (r - img[i]); img[i + P] += w * (gg - img[i + P]); img[i + 2 * P] += w * (b - img[i + 2 * P]);
  });
}

// change in summed squared error if C were drawn (img unchanged)
export function errorDelta(g, ref, img, S, C, clip) {
  const a = C[g.CA], col = [C[g.CR], C[g.CR + 1], C[g.CR + 2]], P = S * S;
  let d = 0;
  forCoverage(g, S, C, clip, (i, cov) => {
    const w = cov * a;
    for (let c = 0; c < 3; c++) {
      const k = i + c * P, v = img[k], e0 = ref[k] - v, e1 = ref[k] - (v + w * (col[c] - v));
      d += e1 * e1 - e0 * e0;
    }
  });
  return d;
}

export function forCoverage(g, S, C, clip, fn) {
  const t = C[0] | 0;
  const cx0 = clip ? clip[0] : 0, cy0 = clip ? clip[1] : 0, cx1 = clip ? clip[2] : S, cy1 = clip ? clip[3] : S;
  const put = (i, cov) => { if (cov > 0) fn(i, cov); };
  const box = (xa, xb, ya, yb, fn) => {
    const i0 = Math.max(cx0, Math.floor(xa)), i1 = Math.min(cx1, Math.ceil(xb));
    const j0 = Math.max(cy0, Math.floor(ya)), j1 = Math.min(cy1, Math.ceil(yb));
    for (let y = j0; y < j1; y++) for (let x = i0; x < i1; x++) put(y * S + x, fn(x + 0.5, y + 0.5));
  };
  if (t === T_RECT || t === T_ELL) {
    const cx = C[1], cy = C[2], hw = C[3], hh = C[4], c = Math.cos(C[5]), s = Math.sin(C[5]);
    const ac = Math.abs(c), as = Math.abs(s);
    if (t === T_RECT) {
      const ex = ac * (hw + 0.5) + as * (hh + 0.5) + 1, ey = as * (hw + 0.5) + ac * (hh + 0.5) + 1;
      box(cx - ex, cx + ex, cy - ey, cy + ey, (X, Y) => {
        const dx = X - cx, dy = Y - cy, lx = dx * c + dy * s, ly = -dx * s + dy * c;
        const ox = clamp(Math.min(hw, lx + 0.5) - Math.max(-hw, lx - 0.5), 0, 1);
        const oy = clamp(Math.min(hh, ly + 0.5) - Math.max(-hh, ly - 0.5), 0, 1);
        return ox * oy;
      });
    } else {
      const rx = hw, ry = hh, ex = Math.hypot(rx * c, ry * s) + 3, ey = Math.hypot(rx * s, ry * c) + 3;
      const thin = Math.min(2 * rx, 1) * Math.min(2 * ry, 1), mr = Math.min(rx, ry);
      box(cx - ex, cx + ex, cy - ey, cy + ey, (X, Y) => {
        const dx = X - cx, dy = Y - cy, lx = dx * c + dy * s, ly = -dx * s + dy * c;
        const ax = lx / rx, ay = ly / ry, k1 = Math.sqrt(ax * ax + ay * ay);
        const k2 = Math.sqrt((ax / rx) ** 2 + (ay / ry) ** 2);
        const d = k2 > 1e-8 ? (k1 - 1) * k1 / Math.max(k2, 1e-8) : -mr;
        return clamp(0.5 - d, 0, 1) * thin;
      });
    }
    return;
  }
  if (t === T_TRI) {
    const x = [C[1], C[3], C[5]], y = [C[2], C[4], C[6]];
    const area2 = (x[1] - x[0]) * (y[2] - y[0]) - (y[1] - y[0]) * (x[2] - x[0]);
    const sgn = Math.sign(area2);
    if (sgn === 0) return;
    const E = [];
    let lmax = 0;
    for (let e = 0; e < 3; e++) {
      const ax = x[e], ay = y[e], ex = x[(e + 1) % 3] - ax, ey = y[(e + 1) % 3] - ay;
      const ln = Math.max(Math.hypot(ex, ey), 1e-6);
      lmax = Math.max(lmax, ln);
      E.push([ax, ay, ex, ey, ln]);
    }
    const hgt = Math.min(Math.abs(area2) / lmax, 1);
    // per row, the pixels inside every edge form one interval
    for (let yy = cy0; yy < cy1; yy++) {
      const Y = yy + 0.5;
      let lo = -Infinity, hi = Infinity;
      for (const [ax, ay, ex, ey, ln] of E) {
        const k = -sgn * ey / ln, m = sgn * (ex * (Y - ay) + ey * ax) / ln;
        if (k > 0) lo = Math.max(lo, (-0.5 - m) / k);
        else if (k < 0) hi = Math.min(hi, (-0.5 - m) / k);
        else if (m <= -0.5) { lo = Infinity; break; }
      }
      const i0 = Math.max(cx0, Math.floor(lo - 1)), i1 = Math.min(cx1, Math.ceil(hi + 1));
      for (let xx = i0; xx < i1; xx++) {
        const X = xx + 0.5;
        let dmin = Infinity;
        for (const [ax, ay, ex, ey, ln] of E) dmin = Math.min(dmin, sgn * (ex * (Y - ay) - ey * (X - ax)) / ln);
        put(yy * S + xx, clamp(0.5 + dmin, 0, 1) * hgt);
      }
    }
    return;
  }
  const V = [];
  let w = 0;
  if (t === T_CURVE) {
    const [x0, y0, x1, y1, x2, y2] = [C[1], C[2], C[3], C[4], C[5], C[6]];
    w = C[7];
    V.push([x0, y0]);
    for (let i = 1; i <= g.CURVE_SEGS; i++) {
      const u = i / g.CURVE_SEGS, p = (1 - u) ** 2, q = 2 * u * (1 - u), rr = u * u;
      V.push([p * x0 + q * x1 + rr * x2, p * y0 + q * y1 + rr * y2]);
    }
  } else {
    const Pt = [[C[1], C[2]], [C[3], C[4]], [C[5], C[6]], [C[7], C[8]]];
    for (let i = 0; i < 4; i++) {
      const pv = Pt[(i + 3) % 4], P0 = Pt[i], nx = Pt[(i + 1) % 4];
      const A = [(pv[0] + P0[0]) / 2, (pv[1] + P0[1]) / 2], B = [(P0[0] + nx[0]) / 2, (P0[1] + nx[1]) / 2];
      for (let k = 0; k < g.BLOB_SEGS; k++) {
        const u = k / g.BLOB_SEGS, p = (1 - u) ** 2, q = 2 * u * (1 - u), rr = u * u;
        V.push([p * A[0] + q * P0[0] + rr * B[0], p * A[1] + q * P0[1] + rr * B[1]]);
      }
    }
  }
  const closed = t === T_BLOB, nE = closed ? V.length : V.length - 1;
  let xa = Infinity, xb = -Infinity, ya = Infinity, yb = -Infinity;
  for (const [vx, vy] of V) { xa = Math.min(xa, vx); xb = Math.max(xb, vx); ya = Math.min(ya, vy); yb = Math.max(yb, vy); }
  const m = closed ? 1 : w / 2 + 1;
  const segs = [];
  for (let i = 0; i < nE; i++) {
    const [ax, ay] = V[i], [bx, by] = V[(i + 1) % V.length];
    const ex = bx - ax, ey = by - ay;
    segs.push([ax, ay, bx, by, ex, ey, Math.max(ex * ex + ey * ey, 1e-8)]);
  }
  let fac;
  if (closed) {
    let area2 = 0;
    for (let i = 0; i < V.length; i++) { const p = V[i], q = V[(i + 1) % V.length]; area2 += p[0] * q[1] - q[0] * p[1]; }
    const Pt = [C[1], C[3], C[5], C[7]], Qt = [C[2], C[4], C[6], C[8]];
    const diam = Math.max(Math.hypot(Math.max(...Pt) - Math.min(...Pt), Math.max(...Qt) - Math.min(...Qt)), 1e-6);
    fac = Math.min((2 / Math.PI) * Math.abs(area2) / diam, 1);
  } else fac = Math.min(w, 1);
  box(xa - m, xb + m, ya - m, yb + m, (X, Y) => {
    let d2 = Infinity, inside = false;
    for (const [ax, ay, bx, by, ex, ey, l2] of segs) {
      const wx = X - ax, wy = Y - ay, h = clamp((wx * ex + wy * ey) / l2, 0, 1);
      const rx = wx - h * ex, ry = wy - h * ey, dd = rx * rx + ry * ry;
      if (dd < d2) d2 = dd;
      if (closed && ((ay > Y) !== (by > Y)) && (((wy * ex - wx * ey) > 0) === (by > ay))) inside = !inside;
    }
    if (closed) {
      const d = Math.sqrt(Math.max(d2, 1e-12));
      return clamp(0.5 + (inside ? d : -d), 0, 1) * fac;
    }
    return clamp(0.5 - (Math.sqrt(d2) - w / 2), 0, 1) * fac;
  });
}

export function mse(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s / a.length;
}
export const psnr = (m) => -10 * Math.log10(Math.max(m, 1e-10));

// bilinear upsample of a tile, as F.interpolate(align_corners=False)
export function tileView(src, S, z, tx, ty, out) {
  const t = S / z, x0 = tx * t, y0 = ty * t, P = S * S;
  out = out || new Float32Array(3 * P);
  const idx = new Int32Array(2 * S), lam = new Float64Array(S);
  for (let d = 0; d < S; d++) {
    const s = Math.max(0, (d + 0.5) / z - 0.5), i0 = Math.floor(s);
    idx[2 * d] = i0; idx[2 * d + 1] = i0 + (i0 < t - 1 ? 1 : 0); lam[d] = s - i0;
  }
  for (let c = 0; c < 3; c++) {
    const base = c * P;
    for (let y = 0; y < S; y++) {
      const ra = base + (y0 + idx[2 * y]) * S + x0, rb = base + (y0 + idx[2 * y + 1]) * S + x0, ly = lam[y];
      for (let x = 0; x < S; x++) {
        const i0 = idx[2 * x], i1 = idx[2 * x + 1], lx = lam[x];
        out[base + y * S + x] = (1 - ly) * ((1 - lx) * src[ra + i0] + lx * src[ra + i1]) + ly * ((1 - lx) * src[rb + i0] + lx * src[rb + i1]);
      }
    }
  }
  return out;
}

function pool6(ref, canvas, S, r, signed) {
  const k = S / r, P = S * S, out = new Float32Array(6 * r * r), inv = 1 / (k * k);
  for (let c = 0; c < 6; c++) {
    const src = c < 3 ? ref : canvas, base = (c % 3) * P;
    for (let j = 0; j < r; j++) for (let i = 0; i < r; i++) {
      let s = 0;
      for (let y = j * k; y < (j + 1) * k; y++) for (let x = i * k; x < (i + 1) * k; x++) s += src[base + y * S + x];
      const v = s * inv;
      out[(c * r + j) * r + i] = signed ? v * 2 - 1 : v;
    }
  }
  return out;
}

function sampleMap(map, ch, r, S, x, y, out, o) {
  const ix = clamp(x / S * r - 0.5, 0, r - 1), iy = clamp(y / S * r - 0.5, 0, r - 1);
  const x0 = Math.floor(ix), y0 = Math.floor(iy), x1 = Math.min(x0 + 1, r - 1), y1 = Math.min(y0 + 1, r - 1);
  const fx = ix - x0, fy = iy - y0, rr = r * r;
  for (let c = 0; c < ch; c++) {
    const b = c * rr;
    out[o + c] = (1 - fy) * ((1 - fx) * map[b + y0 * r + x0] + fx * map[b + y0 * r + x1]) + fy * ((1 - fx) * map[b + y1 * r + x0] + fx * map[b + y1 * r + x1]);
  }
}

// pixel pyramid (18) and glimpse windows (972) around the anchor
export class AnchorSampler {
  constructor(g, ref, canvas) {
    const S = g.S;
    this.g = g;
    this.pyr = g.pix_res.map((r) => [r, r === S ? null : pool6(ref, canvas, S, r, false)]);
    this.pyr = this.pyr.map(([r, m]) => [r, m || interleave(ref, canvas, S, false)]);
    this.gl = g.glimpse.src.map(([r, step]) => [r, step, r === S ? interleave(ref, canvas, S, true) : pool6(ref, canvas, S, r, true)]);
  }
  pix(anc) {
    const out = new Float32Array(18), has = anc[2] * anc[3];
    if (!has) return out;
    this.pyr.forEach(([r, m], l) => sampleMap(m, 6, r, this.g.S, anc[0], anc[1], out, 6 * l));
    return out;
  }
  glimpse(anc) {
    const n = this.g.glimpse.n, N = n * n, out = new Float32Array(this.g.glimpse.dim), has = anc[2] * anc[3];
    if (!has) return out;
    const tmp = new Float32Array(6), h = (n - 1) / 2;
    this.gl.forEach(([r, step, m], si) => {
      for (let k = 0; k < N; k++) {
        const col = k % n, row = Math.floor(k / n);
        sampleMap(m, 6, r, this.g.S, anc[0] + (col - h) * step, anc[1] + (row - h) * step, tmp, 0);
        for (let c = 0; c < 6; c++) out[si * N * 6 + k * 6 + c] = tmp[c];
      }
    });
    return out;
  }
}

function interleave(ref, canvas, S, signed) {
  const P = S * S, out = new Float32Array(6 * P);
  out.set(ref, 0); out.set(canvas, 3 * P);
  if (signed) for (let i = 0; i < out.length; i++) out[i] = out[i] * 2 - 1;
  return out;
}

function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.5 * a);
  const y = 1 - t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return s * y;
}
const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));

export class EdgeHead {
  constructor(g, buf) {
    const h = g.head, sizes = [h.n_in, ...h.layers], f = new Float32Array(buf);
    this.g = g; this.h = h; this.L = [];
    let o = 0;
    for (let l = 0; l < sizes.length - 1; l++) {
      const n = sizes[l], m = sizes[l + 1];
      this.L.push({ n, m, W: f.subarray(o, o + n * m), b: f.subarray(o + n * m, o + n * m + m) });
      o += n * m + m;
    }
    if (o !== f.length) throw new Error(`head.bin: expected ${o} floats, got ${f.length}`);
  }

  forward(x) {
    this.L.forEach((l, li) => {
      const y = new Float64Array(l.m);
      for (let i = 0; i < l.m; i++) {
        let s = l.b[i];
        const row = i * l.n;
        for (let k = 0; k < l.n; k++) s += l.W[row + k] * x[k];
        y[i] = li < this.L.length - 1 ? gelu(s) : s;
      }
      x = y;
    });
    return x;
  }

  axes(C) { const c = Math.cos(C[5]), s = Math.sin(C[5]); return [[c, s], [-s, c]]; }

  boundary(C) {
    const P = this.h.P, hw = Math.max(C[3], 0.25), hh = Math.max(C[4], 0.25), [a, b] = this.axes(C);
    const pts = [], nrm = [], isEl = (C[0] | 0) === T_ELL, q = P / 4;
    for (let i = 0; i < P; i++) {
      let lu, lv, nu, nv;
      if (isEl) {
        const phi = (i + 0.5) * (2 * Math.PI / P) - Math.PI / 4, eu = Math.cos(phi), ev = Math.sin(phi);
        nu = eu / hw; nv = ev / hh;
        const nn = Math.hypot(nu, nv);
        nu /= nn; nv /= nn; lu = hw * eu; lv = hh * ev;
      } else {
        const side = Math.floor(i / q), s = ((i % q) + 0.5) / q * 2 - 1;
        const ru = [1, -s, -1, s][side], rv = [s, 1, -s, -1][side];
        nu = [1, 0, -1, 0][side]; nv = [0, 1, 0, -1][side];
        lu = hw * ru; lv = hh * rv;
      }
      pts.push([C[1] + lu * a[0] + lv * b[0], C[2] + lu * a[1] + lv * b[1]]);
      nrm.push([nu * a[0] + nv * b[0], nu * a[1] + nv * b[1]]);
    }
    return [pts, nrm];
  }

  features(ref, canvas, C, pass) {
    const g = this.g, S = g.S, P = this.h.P, R = this.h.R_OFF, nR = R.length, PR = P * nR;
    const [pts, nrm] = this.boundary(C);
    const f = new Float64Array(this.h.n_in);
    const col = [C[g.CR], C[g.CR + 1], C[g.CR + 2]], al = C[g.CA];
    const rs = new Float32Array(3), cs = new Float32Array(3);
    const sq = (v) => Math.sign(v) * Math.sqrt(Math.abs(v));
    for (let p = 0; p < P; p++) for (let k = 0; k < nR; k++) {
      const x = pts[p][0] + nrm[p][0] * R[k], y = pts[p][1] + nrm[p][1] * R[k];
      sampleImg(ref, S, x, y, rs); sampleImg(canvas, S, x, y, cs);
      const inside = x >= 0 && x <= S && y >= 0 && y <= S ? 1 : 0;
      let res = 0, after = 0;
      for (let c = 0; c < 3; c++) {
        const d = rs[c] - cs[c], e = rs[c] - (cs[c] + al * (col[c] - cs[c]));
        res += d * d; after += e * e;
      }
      const i = p * nR + k;
      f[i] = sq(res - after) * inside; f[PR + i] = sq(res) * inside; f[2 * PR + i] = inside;
    }
    f[3 * PR] = Math.log(Math.max(C[3], 0.25)) / 5; f[3 * PR + 1] = Math.log(Math.max(C[4], 0.25)) / 5;
    f[3 * PR + 2] = al; f[3 * PR + 3] = (C[0] | 0) === T_ELL ? 1 : 0;
    f[3 * PR + 4 + Math.min(pass, this.h.PASSES - 1)] = 1;
    for (let i = 0; i < f.length; i++) if (!Number.isFinite(f[i])) f[i] = 0;
    return f;
  }

  applyLocal(C0, o) {
    const g = this.g, L = this.h.CLAMP, [a, b] = this.axes(C0);
    o = Array.from(o, (v) => (Number.isFinite(v) ? clamp(v, -L, L) : 0));
    const C = Float64Array.from(C0);
    C[1] = C0[1] + o[0] * a[0] + o[1] * b[0];
    C[2] = C0[2] + o[0] * a[1] + o[1] * b[1];
    C[3] = Math.max(C0[3] + o[2], g.SIZE_MIN);
    C[4] = Math.max(C0[4] + o[3], g.SIZE_MIN);
    C[5] = C0[5] + o[4] / Math.max(Math.max(C0[3], C0[4]), 1);
    return g.clampShape(C);
  }

  // guard: keep whichever of C and each pass's result lowers the error most
  refine(ref, canvas, C, passes, guard = false) {
    if ((C[0] | 0) > T_ELL) return C;
    let best = C, bd = guard ? errorDelta(this.g, ref, canvas, this.g.S, C, null) : 0;
    for (let t = 0; t < passes; t++) {
      C = this.applyLocal(C, this.forward(this.features(ref, canvas, C, t)));
      if (guard) { const d = errorDelta(this.g, ref, canvas, this.g.S, C, null); if (d < bd) { bd = d; best = C; } }
    }
    return guard ? best : C;
  }
}

function sampleImg(img, S, x, y, out) {
  const ix = clamp(x - 0.5, 0, S - 1), iy = clamp(y - 0.5, 0, S - 1);
  const x0 = Math.floor(ix), y0 = Math.floor(iy), x1 = Math.min(x0 + 1, S - 1), y1 = Math.min(y0 + 1, S - 1);
  const fx = ix - x0, fy = iy - y0, P = S * S;
  for (let c = 0; c < 3; c++) {
    const b = c * P;
    out[c] = (1 - fy) * ((1 - fx) * img[b + y0 * S + x0] + fx * img[b + y0 * S + x1]) + fy * ((1 - fx) * img[b + y1 * S + x0] + fx * img[b + y1 * S + x1]);
  }
}

// gpu: optional WebGPU execution provider; avoid persistent gpu-buffer KV-cache aliasing bugs
export class OrtBackend {
  constructor(ort, g, enc, dec, gpu = false) {
    Object.assign(this, { ort, g, enc, dec, gpu });
    const Ld = g.dec_layers, h = g.heads, dh = g.d / g.heads;
    this.cacheDims = [Ld, h, g.Lmax, dh];
    this.zeros = new Float32Array(Ld * h * g.Lmax * dh);
    this.cross = null;
    this.ck = null;
    this.cv = null;
  }
  static async create(ort, g, encSrc, decSrc, opts = {}) {
    const gpu = !!opts.gpu, base = opts.session || {};
    const encOpts = { ...(opts.encSession || base), ...(gpu ? { preferredOutputLocation: { cross: 'gpu-buffer' } } : {}) };
    const decOpts = { ...(opts.decSession || base), ...(gpu ? { preferredOutputLocation: { ck_out: 'gpu-buffer', cv_out: 'gpu-buffer' } } : {}) };
    const enc = await ort.InferenceSession.create(encSrc, encOpts);
    const dec = await ort.InferenceSession.create(decSrc, decOpts);
    return new OrtBackend(ort, g, enc, dec, gpu);
  }
  _drop(t) {
    if (t && t.location === 'gpu-buffer') t.dispose?.();
  }
  async dispose() {
    this._drop(this.cross); this._drop(this.ck); this._drop(this.cv);
    this.cross = this.ck = this.cv = null;
    await this.enc?.release?.();
    await this.dec?.release?.();
  }
  async encode(ref, canvas) {
    const S = this.g.S, T = this.ort.Tensor;
    this._drop(this.cross); this._drop(this.ck); this._drop(this.cv);
    this.cross = this.ck = this.cv = null;
    const out = await this.enc.run({ ref: new T('float32', ref, [1, 3, S, S]), canvas: new T('float32', canvas, [1, 3, S, S]) });
    this.cross = out.cross;
    this.ck = new T('float32', new Float32Array(this.zeros.length), this.cacheDims);
    this.cv = new T('float32', new Float32Array(this.zeros.length), this.cacheDims);
  }
  async step(ids, anc, pixv, glv) {
    const T = this.ort.Tensor;
    const out = await this.dec.run({
      ids: new T('int32', Int32Array.from(ids), [5]), anc: new T('float32', Float32Array.from(anc), [1, 4]),
      pixv: new T('float32', pixv, [1, 18]), glv: new T('float32', glv, [1, glv.length]), ck: this.ck, cv: this.cv, cross: this.cross,
    });
    this._drop(this.ck); this._drop(this.cv);
    this.ck = out.ck_out; this.cv = out.cv_out;
    return out.logits.data;
  }
}

export async function decodeGroup(g, backend, rv, cv, signal) {
  if (signal?.aborted) return null;
  await backend.encode(rv, cv);
  if (signal?.aborted) return null;
  const smp = new AnchorSampler(g, rv, cv), st = new Stepper(g, g.K);
  let tok = g.BOS, slot = g.SLOT_BOS;
  for (let i = 0; i < g.Lmax; i++) {
    if (signal?.aborted) return null;
    const nxt = st.nextSlot(), anc = st.nextAnchor();
    const ids = [tok, slot, nxt, Math.min(st.nextKidx(), g.K), i];
    const logits = await backend.step(ids, anc, smp.pix(anc), smp.glimpse(anc));
    if (signal?.aborted) return null;
    const t = st.pick(logits);
    st.push(t);
    tok = t; slot = nxt;
    if (st.done) break;
  }
  return { Q: st.Q, n: st.n };
}

export function parsePlan(text) {
  return text.split(',').map((p) => p.split(':').map(Number));
}

// plan [[z, per tile], ...]. Yields after every group: { elements: [{ C, tile }], z, tx, ty, tiles, canvas, mse }
export async function* paint(g, backend, ref, plan, opts = {}) {
  const S = g.S, K = g.K, head = opts.head || null, passes = opts.passes ?? g.head.PASSES, guard = !!opts.guard;
  const canvas = opts.canvas || new Float32Array(3 * S * S).fill(1);
  const tiles = [[0, 0, S, S]];
  for (const [z, per] of plan) {
    const t = S / z;
    for (let ty = 0; ty < z; ty++) for (let tx = 0; tx < z; tx++) {
      let tid = 0;
      if (z > 1) { tiles.push([tx * t, ty * t, t, t]); tid = tiles.length - 1; }
      for (let r = 0; r < Math.floor(per / K); r++) {
        if (opts.signal?.aborted) return;
        const rv = z === 1 ? ref : tileView(ref, S, z, tx, ty);
        const cv = z === 1 ? canvas : tileView(canvas, S, z, tx, ty);
        const decoded = await decodeGroup(g, backend, rv, cv, opts.signal);
        if (!decoded || opts.signal?.aborted) return;
        const { Q, n } = decoded;
        const elements = [];
        for (let k = 0; k < n; k++) {
          if (opts.signal?.aborted) return;
          let C = g.q2c(Q[k]);
          if (head) C = head.refine(rv, cv, C, passes, guard);
          drawElement(g, cv, S, C, null);
          if (z > 1) {
            const Ci = g.viewToImage(C, 1 / z, tx * t, ty * t);
            drawElement(g, canvas, S, Ci, [tx * t, ty * t, (tx + 1) * t, (ty + 1) * t]);
            elements.push({ C: Ci, tile: tid });
          } else elements.push({ C, tile: 0 });
        }
        if (opts.signal?.aborted) return;
        yield { elements, z, tx, ty, tiles, canvas, mse: opts.measure === false ? null : mse(ref, canvas) };
      }
    }
  }
}

const fmt = (x) => { const s = x.toFixed(2).replace(/0+$/, '').replace(/\.$/, ''); return s === '-0' || s === '' ? '0' : s; };

export function shapeToSvg(g, C) {
  const t = C[0] | 0, rgb = `rgb(${Math.round(C[g.CR] * 255)},${Math.round(C[g.CR + 1] * 255)},${Math.round(C[g.CR + 2] * 255)})`;
  const a = fmt(C[g.CA]);
  if (t === T_RECT || t === T_ELL) {
    const [cx, cy, hw, hh, ang] = [C[1], C[2], C[3], C[4], C[5]];
    const tr = ang ? ` transform="rotate(${fmt(ang * 180 / Math.PI)} ${fmt(cx)} ${fmt(cy)})"` : '';
    return t === T_RECT
      ? `<rect x="${fmt(cx - hw)}" y="${fmt(cy - hh)}" width="${fmt(2 * hw)}" height="${fmt(2 * hh)}"${tr} fill="${rgb}" fill-opacity="${a}"/>`
      : `<ellipse cx="${fmt(cx)}" cy="${fmt(cy)}" rx="${fmt(hw)}" ry="${fmt(hh)}"${tr} fill="${rgb}" fill-opacity="${a}"/>`;
  }
  if (t === T_TRI) return `<polygon points="${fmt(C[1])},${fmt(C[2])} ${fmt(C[3])},${fmt(C[4])} ${fmt(C[5])},${fmt(C[6])}" fill="${rgb}" fill-opacity="${a}"/>`;
  if (t === T_BLOB) {
    const P = [[C[1], C[2]], [C[3], C[4]], [C[5], C[6]], [C[7], C[8]]];
    const mid = P.map((p, i) => [(p[0] + P[(i + 1) % 4][0]) / 2, (p[1] + P[(i + 1) % 4][1]) / 2]);
    let d = `M${fmt(mid[3][0])} ${fmt(mid[3][1])}`;
    for (let i = 0; i < 4; i++) d += `Q${fmt(P[i][0])} ${fmt(P[i][1])} ${fmt(mid[i][0])} ${fmt(mid[i][1])}`;
    return `<path d="${d}Z" fill="${rgb}" fill-opacity="${a}" fill-rule="evenodd"/>`;
  }
  return `<path d="M${fmt(C[1])} ${fmt(C[2])}Q${fmt(C[3])} ${fmt(C[4])} ${fmt(C[5])} ${fmt(C[6])}" fill="none" stroke="${rgb}" stroke-opacity="${a}" stroke-width="${fmt(C[7])}" stroke-linecap="round"/>`;
}

export function toSvg(g, elements, tiles) {
  const S = g.S, used = new Set();
  let body = '', cur = 0;
  for (const { C, tile } of elements) {
    if (tile !== cur) { body += (cur ? '</g>\n' : '') + (tile ? `<g clip-path="url(#t${tile})">\n` : ''); cur = tile; }
    if (tile) used.add(tile);
    body += shapeToSvg(g, C) + '\n';
  }
  if (cur) body += '</g>\n';
  const defs = used.size ? '<defs>' + [...used].sort((a, b) => a - b).map((i) => {
    const [x, y, w, h] = tiles[i];
    return `<clipPath id="t${i}"><rect x="${x}" y="${y}" width="${w}" height="${h}"/></clipPath>`;
  }).join('') + '</defs>\n' : '';
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${S} ${S}" width="${S}" height="${S}">\n${defs}<rect width="${S}" height="${S}" fill="#fff"/>\n${body}</svg>\n`;
}
