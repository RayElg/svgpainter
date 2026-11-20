"""Held-out evaluation: PSNR after each level of the plan, split into flat graphics and photos;
SSIM and LPIPS of the result; a grid image and a few SVGs."""
import argparse
import io
import itertools
import json
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from svg_painter import grammar as G
from svg_painter.data import TeacherData
from svg_painter.metrics import psnr, ssim
from svg_painter.model import load_model
from svg_painter.offset import load_heads
from svg_painter.paint import paint, parse_plan, to_svg


def to_img(x):
    return (x.clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--heads")
    p.add_argument("--head", default="grad", choices=["grad", "edge"])
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--guard", type=int, default=1)
    p.add_argument("--pack", required=True)
    p.add_argument("--plan", default="1:128,2:64,4:8")
    p.add_argument("--at", default="", help="more element counts to report, such as 16,32,64")
    p.add_argument("--n", type=int, default=96, help="images, spread over the pack (first half flat graphics)")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--out", required=True)
    p.add_argument("--show", type=int, default=8, help="images in the grid and as SVG")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, dev)[0].eval()
    heads = load_heads(a.heads, dev) if a.heads else None
    try:
        from svg_painter.metrics import Lpips
        lp = Lpips(dev)
    except ImportError:
        lp = None
    plan, K = parse_plan(a.plan), model.c.k_shapes
    cum = list(itertools.accumulate(min(K, per - j * K) for z, per in plan for _ in range(z * z) for j in range(math.ceil(per / K))))
    ends = list(itertools.accumulate(z * z * per for z, per in plan))
    want = ends + [int(v) for v in a.at.split(",") if v]
    marks = {m: cum.index(m) + 1 for m in sorted(set(want)) if m in cum}
    d = TeacherData(a.pack, dev)
    idx = torch.linspace(0, d.N - 1, a.n).long()
    show = set(torch.linspace(0, a.n - 1, min(a.show, a.n)).long().tolist())
    mses, ss, lps, rows, types = [], [], [], [], torch.zeros(5)
    for i in range(0, a.n, a.bs):
        ref = d.ref(idx[i:i + a.bs])
        out = paint(model, ref, plan, heads, a.head, a.passes, bool(a.guard))
        mses.append(out["mse"][:, list(marks.values())])
        types += torch.bincount(out["C"][..., 0][out["keep"]].long().cpu(), minlength=5)[:5].float()
        r, c = [F.interpolate(x, size=(256, 256), mode="area") for x in (ref, out["canvas"])]
        ss.append(ssim(r, c))
        if lp is not None:
            lps.append(lp(r, c))
        for b in range(ref.shape[0]):
            if i + b in show:
                rows.append(np.concatenate([to_img(ref[b]), to_img(out["canvas"][b])], 1))
                svg = to_svg(out, b)
                open(os.path.join(a.out, f"{i + b}.svg"), "w").write(svg)
                try:
                    import resvg_py
                    png = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg)))).convert("RGB")
                    diff = np.abs(np.asarray(png, np.float32) - to_img(out["canvas"][b]).astype(np.float32)).mean()
                    print(f"image {i + b}: resvg render of the SVG vs internal canvas, mean abs diff {diff:.3f} / 255")
                except ImportError:
                    pass
    e, h = torch.cat(mses), a.n // 2
    res = {"plan": a.plan, "head": f"{a.head}{a.passes}" if heads else "none", "guard": bool(a.guard), "images": a.n,
           "psnr [flat, photo]": {m: [round(psnr(e[:h, j]).mean().item(), 2), round(psnr(e[h:, j]).mean().item(), 2)]
                                  for j, m in enumerate(marks)},
           "ssim": round(torch.cat(ss).mean().item(), 4),
           "types": {t: round(v, 3) for t, v in zip(G.TYPE_NAMES, (types / types.sum()).tolist())}}
    if lps:
        res["lpips"] = round(torch.cat(lps).mean().item(), 4)
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(a.out, "grid.jpg"), quality=90)


if __name__ == "__main__":
    main()
