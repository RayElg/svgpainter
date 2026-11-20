// The JavaScript painter against PyTorch traces from export_onnx.py: npm test
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as ort from 'onnxruntime-node';
import { Grammar, OrtBackend, EdgeHead, decodeGroup, paint, parsePlan, drawElement, mse, psnr } from '../js/core.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(here, '..', 'model');
const g = new Grammar(JSON.parse(fs.readFileSync(path.join(md, 'meta.json'))));
const headBuf = fs.readFileSync(path.join(md, 'head.bin'));
const head = new EdgeHead(g, headBuf.buffer.slice(headBuf.byteOffset, headBuf.byteOffset + headBuf.byteLength));
const backend = await OrtBackend.create(ort, g, path.join(md, 'enc.onnx'), path.join(md, 'dec.onnx'));
const tests = JSON.parse(fs.readFileSync(path.join(here, 'expect.json')));
const S = g.S, P = S * S;
let bad = 0;
const check = (ok, msg) => { console.log((ok ? 'ok   ' : 'FAIL ') + msg); if (!ok) bad++; };

for (const tc of tests) {
  const u8 = fs.readFileSync(path.join(here, `ref_${tc.idx}.bin`));
  const ref = new Float32Array(3 * P);
  for (let i = 0; i < ref.length; i++) ref[i] = u8[i] / 255;

  const blank = new Float32Array(3 * P).fill(1);
  const t0 = Date.now();
  const { Q, n } = await decodeGroup(g, backend, ref, blank);
  const ms = Date.now() - t0;
  const same = n === tc.first_n && Q.slice(0, n).every((row, k) => row.every((v, c) => v === tc.first_Q[k][c]));
  check(same, `image ${tc.idx}: first group, ${n} elements, tokens equal torch (${ms} ms)`);

  for (const mode of ['none', 'edge3']) {
    const ex = tc[mode];
    const canvas = new Float32Array(3 * P).fill(1);
    ex.C.forEach((C, i) => {
      if (!ex.keep[i]) return;
      const t = ex.tile[i], clip = t ? [ex.tiles[t][0], ex.tiles[t][1], ex.tiles[t][0] + ex.tiles[t][2], ex.tiles[t][1] + ex.tiles[t][3]] : null;
      drawElement(g, canvas, S, Float64Array.from(C), clip);
    });
    const pr = psnr(mse(ref, canvas)), pt = ex.psnr[ex.psnr.length - 1];
    check(Math.abs(pr - pt) < 0.05, `image ${tc.idx} ${mode}: torch elements, JS raster ${pr.toFixed(3)} dB vs torch ${pt} dB`);

    const curve = [psnr(mse(ref, blank))];
    const t1 = Date.now();
    let els = 0, firstDiff = -1, i0 = 0;
    for await (const step of paint(g, backend, ref, parsePlan(tc.plan), { head: mode === 'edge3' ? head : null, passes: 3 })) {
      curve.push(psnr(step.mse));
      for (const e of step.elements) {
        while (i0 < ex.keep.length && !ex.keep[i0]) i0++;
        const tC = ex.C[i0++];
        if (firstDiff < 0 && tC && e.C.some((v, c) => Math.abs(v - tC[c]) > 0.02)) firstDiff = els;
        els++;
      }
    }
    const dt = (Date.now() - t1) / 1000, last = curve[curve.length - 1];
    console.log(`      JS  ${curve.map((v) => v.toFixed(2)).join(' ')}\n      torch ${ex.psnr.map((v) => v.toFixed(2)).join(' ')}`);
    check(Math.abs(last - pt) < 0.1, `image ${tc.idx} ${mode}: loop ${els} elements in ${dt.toFixed(1)} s, final ${last.toFixed(3)} vs ${pt} dB` +
      (firstDiff < 0 ? ', every element matches' : `, first element that differs: #${firstDiff}`));
  }
}
console.log(bad ? `${bad} check(s) failed` : 'all checks passed');
process.exit(bad ? 1 : 0);
