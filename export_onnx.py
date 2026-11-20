"""Export the painter for the browser (web/).

    enc.onnx   ref, canvas (1,3,512,512) -> keys then values of every decoder layer's cross-attention
    dec.onnx   one decoding step with fixed-size caches: ids int32 (5,) = token, slot, next slot,
               element index, position; anc (1,4); pixv (1,18); glv (1,972); ck, cv; cross
               -> logits (1,VOCAB), ck_out, cv_out
    head.bin   the edge offset head, float32, weight then bias per layer
    meta.json  grammar and model constants
    test/      references and PyTorch traces for web/test/run_node.mjs
"""
import argparse
import json
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from svg_painter import grammar as G
from svg_painter import model as M
from svg_painter import offset as O
from svg_painter.data import TeacherData
from svg_painter.metrics import psnr
from svg_painter.model import load_model
from svg_painter.offset import load_heads
from svg_painter.paint import paint, parse_plan


def attend(q, k, v, bias=None):
    s = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    return torch.softmax(s if bias is None else s + bias, -1) @ v


class EncExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.enc, self.blocks = model.enc, model.dec.blocks
        # Pad the first Conv2d from 9 -> 12 input channels (multiple of 4) to avoid an ORT WebGPU Conv2d shader bug.
        c0 = self.enc.stem[0]
        if c0.in_channels == 9:
            c12 = nn.Conv2d(12, c0.out_channels, c0.kernel_size, c0.stride, c0.padding, bias=c0.bias is not None)
            with torch.no_grad():
                c12.weight.zero_()
                c12.weight[:, :9].copy_(c0.weight)
                if c0.bias is not None:
                    c12.bias.copy_(c0.bias)
            self.stem0 = c12
            self.stem_rest = self.enc.stem[1:]
        else:
            self.stem0 = None

    def forward(self, ref, canvas):
        e = self.enc
        if self.stem0 is not None:
            x = torch.cat([ref * 2 - 1, canvas * 2 - 1, (ref - canvas) * 2, ref], 1)
            x = self.stem_rest(self.stem0(x))
        else:
            x = torch.cat([ref * 2 - 1, canvas * 2 - 1, (ref - canvas) * 2], 1)
            x = e.stem(x)
        x = x.flatten(2).transpose(1, 2) + e.pos + e.pos_proj(e.pos_feat)
        for b in e.blocks:
            y = b.n1(x)
            B, L, D = y.shape
            q, k, v = b.attn.split(y)
            x = x + b.attn.o(attend(q, k, v).transpose(1, 2).reshape(B, L, D))
            x = x + b.mlp(b.n2(x))
        kv = [b.cross.memory(e.norm(x)) for b in self.blocks]
        return torch.cat([k for k, _ in kv] + [v for _, v in kv], 0)


class DecExport(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.dec = model.dec
        self.Lmax = G.max_seq_len(model.c.k_shapes) - 1
        self.register_buffer("ar", torch.arange(self.Lmax).float(), persistent=False)

    def forward(self, ids, anc, pixv, glv, ck, cv, cross):
        d = self.dec
        D, h, Ld = d.c.d, d.c.heads, len(d.blocks)
        dh = D // h
        ids = ids.long()
        pos = ids[4:5]
        x = d.embed(ids[0:1], ids[1:2], ids[2:3], ids[3:4], pos)[:, None]
        has = anc[:, 2:3] * anc[:, 3:4]
        x = x + d.pix_in(torch.cat([M.col_fourier(pixv).flatten(-2) * has, has], -1))[:, None]
        x = x + d.gl_b(F.gelu(d.gl_a(glv)))[:, None]
        d2 = M.anchor_dist2(anc, d.g)
        sel = (self.ar == pos.float()).view(1, 1, self.Lmax, 1)
        bias = ((self.ar > pos.float()).float() * -1e9).view(1, 1, 1, self.Lmax)
        ko, vo = [], []
        for li, b in enumerate(d.blocks):
            q, k, v = b.attn.split(b.n1(x))
            kl, vl = torch.where(sel, k, ck[li:li + 1]), torch.where(sel, v, cv[li:li + 1])
            x = x + b.attn.o(attend(q, kl, vl, bias).transpose(1, 2).reshape(1, 1, D))
            qc = b.cross.q(b.n2(x)).reshape(1, 1, h, dh).transpose(1, 2)
            cb = -0.5 * d2[:, None, None, :] * (b.cross.inv_sigma ** 2)[None, :, None, None]
            y = attend(qc, cross[li:li + 1], cross[Ld + li:Ld + li + 1], cb)
            x = x + b.cross.o(y.transpose(1, 2).reshape(1, 1, D))
            x = x + b.mlp(b.n3(x))
            ko.append(kl)
            vo.append(vl)
        return d.logits(x[:, 0]), torch.cat(ko, 0), torch.cat(vo, 0)


def meta(model):
    c = model.c
    return dict(
        S=G.S, K=c.k_shapes, W=G.W, CR=G.CR, CA=G.CA, MAXTOK=G.MAXTOK, VOCAB=G.VOCAB,
        Lmax=G.max_seq_len(c.k_shapes) - 1, d=c.d, heads=c.heads, dec_layers=c.dec_layers,
        M=(c.img // c.patch) ** 2, pix_res=list(M.PIX_RES),
        POS_MIN=G.POS_MIN, POS_STEP=G.POS_STEP, N_POS=G.N_POS, N_SIZE=G.N_SIZE, SIZE_MIN=G.SIZE_MIN,
        SIZE_SPAN=G.SIZE_SPAN, N_ANG=G.N_ANG, N_COL=G.N_COL, N_ALPHA=G.N_ALPHA,
        PAD=G.PAD, BOS=G.BOS, END=G.END, TYPE0=G.TYPE0, N_TYPES=G.N_TYPES,
        KIND_BASE=G.KIND_BASE, KIND_N=G.KIND_N, TOK_LEN=G.TOK_LEN.tolist(), TOK_KIND=G.TOK_KIND.tolist(),
        TOK_COL=G.TOK_COL.tolist(), SLOT_PAD=G.SLOT_PAD, SLOT_BOS=G.SLOT_BOS, SLOT_START=G.SLOT_START,
        CURVE_SEGS=12, BLOB_SEGS=8,
        glimpse=dict(n=M.GLIMPSE_N, src=[list(g) for g in M.GLIMPSE], dim=M.GLIMPSE_DIM),
        head=dict(P=O.P, R_OFF=list(O.R_OFF), CLAMP=O.CLAMP, PASSES=O.PASSES, n_in=O.N_FEAT + O.PASSES, layers=[512, 256, O.N_OUT]),
    )


def onnx_group(enc, dec, mt, ref, canvas):
    """The browser's decoding loop on the exported graphs, for the parity check."""
    K, Lmax, Ld, h = mt["K"], mt["Lmax"], mt["dec_layers"], mt["heads"]
    cross = enc.run(None, {"ref": ref.numpy(), "canvas": canvas.numpy()})[0]
    pyr, gsrc = M.pixel_pyramid(ref, canvas), M.glimpse_sources(ref, canvas)
    st = G.Stepper(1, K, "cpu")
    ck = np.zeros((Ld, h, Lmax, mt["d"] // h), np.float32)
    cv = np.zeros_like(ck)
    tok, slot = G.BOS, G.SLOT_BOS
    for i in range(Lmax):
        nxt, anc = int(st.next_slot()), st.next_anchor()
        ids = np.array([tok, slot, nxt, min(int(st.next_kidx()), K), i], np.int32)
        feeds = {"ids": ids, "anc": anc.numpy(), "pixv": M.sample_pyramid(pyr, anc[None])[0][0].numpy(),
                 "glv": M.sample_glimpse(gsrc, anc[None])[0].numpy(), "ck": ck, "cv": cv, "cross": cross}
        logits, ck, cv = dec.run(None, feeds)
        new = torch.from_numpy(logits).masked_fill(~st.mask(), float("-inf")).argmax(-1)
        st.push(new)
        tok, slot = int(new), nxt
        if bool(st.done.all()):
            break
    return st.Q[0], int(st.n[0])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--heads", required=True)
    p.add_argument("--out", default="web")
    p.add_argument("--eval_pack", required=True)
    p.add_argument("--test_idx", default="3,25,52")
    p.add_argument("--test_plan", default="1:32,2:8")
    a = p.parse_args()
    model = load_model(a.ckpt, "cpu")[0].eval().float()
    assert model.c.glimpse
    heads = load_heads(a.heads, "cpu")
    md, td = os.path.join(a.out, "model"), os.path.join(a.out, "test")
    os.makedirs(md, exist_ok=True)
    os.makedirs(td, exist_ok=True)
    mt = meta(model)
    S, Lmax, Ld, h = mt["S"], mt["Lmax"], mt["dec_layers"], mt["heads"]
    enc, dec = EncExport(model).eval(), DecExport(model).eval()
    ref, canvas = torch.rand(1, 3, S, S), torch.ones(1, 3, S, S)
    with torch.no_grad():
        cross = enc(ref, canvas)
    cache = torch.zeros(Ld, h, Lmax, mt["d"] // h)
    step_in = (torch.tensor([G.BOS, G.SLOT_BOS, G.SLOT_START, 0, 0], dtype=torch.int32), torch.zeros(1, 4), torch.zeros(1, 18),
               torch.zeros(1, M.GLIMPSE_DIM), cache, cache, cross)
    kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
    torch.onnx.export(enc, (ref, canvas), os.path.join(md, "enc.onnx"), input_names=["ref", "canvas"], output_names=["cross"], **kw)
    torch.onnx.export(dec, step_in, os.path.join(md, "dec.onnx"), input_names=["ids", "anc", "pixv", "glv", "ck", "cv", "cross"],
                      output_names=["logits", "ck_out", "cv_out"], **kw)
    with open(os.path.join(md, "head.bin"), "wb") as f:
        for i in (0, 2, 4):
            f.write(heads.edge[i].weight.detach().numpy().astype("<f4").tobytes())
            f.write(heads.edge[i].bias.detach().numpy().astype("<f4").tobytes())
    json.dump(mt, open(os.path.join(md, "meta.json"), "w"))

    import onnxruntime as ort
    es = ort.InferenceSession(os.path.join(md, "enc.onnx"), providers=["CPUExecutionProvider"])
    ds = ort.InferenceSession(os.path.join(md, "dec.onnx"), providers=["CPUExecutionProvider"])
    data = TeacherData(a.eval_pack, "cpu")
    plan, tests = parse_plan(a.test_plan), []
    for i in [int(v) for v in a.test_idx.split(",")]:
        u8 = (data.ref(torch.tensor([i]))[0] * 255).round().clamp(0, 255).byte()
        u8.numpy().tofile(os.path.join(td, f"ref_{i}.bin"))
        r = u8.float()[None] / 255
        blank = torch.ones_like(r)
        with torch.no_grad():
            Qt, nt = model.dec.sample(model.enc(r, blank), 1, greedy_first=True)
        Qo, no = onnx_group(es, ds, mt, r, blank)
        same = int(nt[0, 0]) == no and bool((Qt[0, 0][:no] == Qo[:no]).all())
        print(f"image {i}: first group, onnx == torch: {same} ({no} elements)", flush=True)
        exp = {"idx": i, "first_Q": Qt[0, 0].tolist(), "first_n": int(nt[0, 0]), "plan": a.test_plan}
        for mode, hd in (("none", None), ("edge3", heads)):
            out = paint(model, r, plan, hd, "edge", guard=False)
            exp[mode] = {"C": [[round(v, 4) for v in row] for row in out["C"][0].tolist()], "keep": out["keep"][0].tolist(),
                         "tile": out["tile"].tolist(), "tiles": [list(t) for t in out["tiles"]],
                         "psnr": [round(v, 3) for v in psnr(out["mse"][0]).tolist()]}
            print(f"  {mode}: psnr after each group", exp[mode]["psnr"], flush=True)
        tests.append(exp)
    json.dump(tests, open(os.path.join(td, "expect.json"), "w"))
    for f in ("enc.onnx", "dec.onnx", "head.bin"):
        print(f, round(os.path.getsize(os.path.join(md, f)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
