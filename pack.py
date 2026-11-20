"""Pack teacher trajectories with their references for training. Each trajectory is cut where the
teacher stopped improving. An optional third field keeps the first n trajectories of a set.

python pack.py data/pack data/img/twemoji:data/teacher/twemoji data/img/coco_train:data/teacher/coco_train:500
"""
import json
import os
import sys
import numpy as np
from multiprocessing import Pool
from PIL import Image
from svg_painter import grammar as G


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    items = []
    for spec in sys.argv[2:]:
        img_dir, t_dir, *n = spec.split(":")
        if n and int(n[0]) == 0:
            continue
        fs = sorted(f for f in os.listdir(t_dir) if f.endswith(".npz") and ".tmp" not in f)
        items += [(os.path.join(img_dir, f[:-4]), os.path.join(t_dir, f)) for f in fs[:int(n[0]) if n else None]]
    N = len(items)
    trajs = []
    for _, t in items:
        with np.load(t) as z:
            trajs.append((z["Q"], z["gain"]))
    T = max(len(q) for q, _ in trajs)
    traj, gain, n = np.zeros((N, T, G.W), np.int16), np.zeros((N, T), np.float32), np.zeros(N, np.int32)
    for i, (q, g) in enumerate(trajs):
        bad = np.nonzero(g <= 0)[0]
        k = int(bad[0]) if len(bad) else len(g)
        traj[i, :k], gain[i, :k], n[i] = q[:k], g[:k], k
    with open(os.path.join(out, "refs.u8"), "wb") as f, Pool(16) as p:
        for im in p.imap(load, [a for a, _ in items], chunksize=16):
            f.write(np.ascontiguousarray(im).tobytes())
    np.save(os.path.join(out, "traj.npy"), traj)
    np.save(os.path.join(out, "n.npy"), n)
    np.save(os.path.join(out, "gain.npy"), gain)
    json.dump({"n": N, "T": T, "files": [a for a, _ in items]}, open(os.path.join(out, "meta.json"), "w"))
    print(f"packed {N} trajectories, T={T}, mean length {n.mean():.1f}")


if __name__ == "__main__":
    main()
