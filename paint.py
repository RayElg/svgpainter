"""Paint images as SVG.

python paint.py photo.jpg logo.png --ckpt model/painter.pt --heads model/heads.pt --out out/
"""
import argparse
import os
import numpy as np
import torch
from PIL import Image
from svg_painter.metrics import psnr
from svg_painter.model import load_model
from svg_painter.offset import load_heads
from svg_painter.paint import paint, parse_plan, plan_size, to_svg


def load_image(path, S=512):
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((l, t, l + s, t + s)).resize((S, S), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float() / 255


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--heads")
    p.add_argument("--head", default="grad", choices=["grad", "edge"])
    p.add_argument("--guard", type=int, default=1)
    p.add_argument("--plan", default="1:128,2:64,4:8", help="levels as zoom:elements per tile")
    p.add_argument("--out", default="out")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(a.ckpt, dev)[0].eval()
    heads = load_heads(a.heads, dev) if a.heads else None
    plan = parse_plan(a.plan)
    for path in a.images:
        ref = load_image(path)[None].to(dev)
        out = paint(model, ref, plan, heads, a.head, guard=bool(a.guard))
        name = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0])
        open(name + ".svg", "w").write(to_svg(out))
        Image.fromarray((out["canvas"][0].clamp(0, 1).permute(1, 2, 0) * 255).round().byte().cpu().numpy()).save(name + ".png")
        print(f"{name}.svg  {int(out['keep'].sum())} of {plan_size(plan)} elements  {psnr(out['mse'][0, -1]).item():.2f} dB")


if __name__ == "__main__":
    main()
