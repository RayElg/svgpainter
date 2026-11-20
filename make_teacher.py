"""Run the search teacher over images: one .npz of shapes (Q) and gains per image.

python make_teacher.py data/img/twemoji data/teacher/twemoji --shapes 192 --first 100 --count 960
"""
import argparse
import os
import time
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from svg_painter import grammar as G
from svg_painter.teacher import Fitter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("out")
    p.add_argument("--shapes", type=int, required=True)
    p.add_argument("--first", type=int, default=0, help="index into the sorted file list")
    p.add_argument("--count", type=int)
    p.add_argument("--batch", type=int, default=96)
    p.add_argument("--types", default=",".join(G.TYPE_NAMES), help="primitives the search may propose")
    p.add_argument("--m0", type=int, default=384, help="random proposals per element")
    p.add_argument("--top", type=int, default=16)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--grad", type=int, default=12)
    p.add_argument("--gk", type=int, default=4)
    p.add_argument("--grad_hi", type=int, default=8, help="gradient steps at 256 px on the best two")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(os.listdir(a.src))[a.first:]
    files = files[:a.count] if a.count else files
    files = [f for f in files if not os.path.exists(os.path.join(a.out, f + ".npz"))]
    allowed = [G.TYPE_NAMES.index(t) for t in a.types.split(",")]
    from svg_painter.synth import TYPE_P
    type_p = [TYPE_P[t] if t in allowed else 0.0 for t in range(G.N_TYPES)]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load = lambda f: np.asarray(Image.open(os.path.join(a.src, f)).convert("RGB"))
    pool, t0 = ThreadPoolExecutor(8), time.time()
    for i in range(0, len(files), a.batch):
        fs = files[i:i + a.batch]
        ref = torch.from_numpy(np.stack(list(pool.map(load, fs)))).to(dev).permute(0, 3, 1, 2).float() / 255
        fit = Fitter(ref, m0=a.m0, top=a.top, rounds=a.rounds, grad=a.grad, gk=a.gk, grad_hi=a.grad_hi, type_p=type_p)
        Q, gain = fit.fit(a.shapes)
        Q, gain = Q.cpu().numpy().astype(np.int16), gain.cpu().numpy().astype(np.float32)
        for j, f in enumerate(fs):
            tmp = os.path.join(a.out, f + ".tmp.npz")
            np.savez(tmp, Q=Q[j], gain=gain[j])
            os.replace(tmp, os.path.join(a.out, f + ".npz"))
        done = i + len(fs)
        dt = time.time() - t0
        print(f"{done}/{len(files)}  psnr {(-10 * torch.log10(fit.mse())).mean().item():.2f}  "
              f"{done * a.shapes / dt:.0f} shapes/s  eta {(len(files) - done) * dt / done / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
