"""Training states: (reference, canvas after t shapes, the next K shapes as tokens)."""
import json
import os
import numpy as np
import torch
from . import grammar as G
from .raster import render
from .synth import random_scene


def make_batch(Q, n, J, K, img, ref=None, end_ok=True, gen=None, early=0.0):
    """J states from each sequence Q (B,T,W), n (B,). ref None: the reference is the render of
    all n shapes (synthetic scenes). early: share of states drawn from the first 2K shapes.
    -> ref (B*J,3,img,img), canvas, seq dict of (B*J,1,L) plus raw targets Q, n."""
    B, T, _ = Q.shape
    dev = Q.device
    hi = (n + 1) if end_ok else (n - K + 1).clamp(min=1)
    t = (torch.rand((B, J), generator=gen, device=dev) * hi[:, None]).long()
    if early > 0:
        e = torch.rand((B, J), generator=gen, device=dev) < early
        te = torch.randint(0, 2 * K + 1, (B, J), generator=gen, device=dev)
        te = torch.where(torch.rand((B, J), generator=gen, device=dev) < 0.4, torch.zeros_like(te), te)
        t = torch.where(e, torch.minimum(te, hi[:, None] - 1), t)
    cv = render(G.q2c(Q), n, img, snapshots=t if ref is not None else torch.cat([t, n[:, None]], 1))
    if ref is None:
        ref, cv = cv[:, J], cv[:, :J]
    idx = (t[..., None] + torch.arange(K, device=dev)).clamp(max=T - 1)
    Qt = torch.gather(Q[:, None].expand(B, J, T, G.W), 2, idx[..., None].expand(B, J, K, G.W)).reshape(B * J, K, G.W)
    nt = (n[:, None] - t).clamp(0, K).reshape(-1)
    seq = {k: v[:, None] for k, v in G.q2tokens(Qt, nt).items()}
    seq["Q"], seq["n"] = Qt, nt
    ref = ref[:, None].expand(B, J, 3, img, img).reshape(B * J, 3, img, img)
    return ref, cv.reshape(B * J, 3, img, img), seq


def synth_batch(B, J, K, img, device, n_max=64, gen=None, early=0.0):
    Q, n = random_scene(B, n_max, gen, device)
    return make_batch(Q, n, J, K, img, None, True, gen, early)


class TeacherData:
    """A pack from pack.py: refs.u8 (N,512,512,3), traj.npy (N,T,W), n.npy."""

    def __init__(self, pack_dir, device, img=512):
        self.N = json.load(open(os.path.join(pack_dir, "meta.json")))["n"]
        self.refs = np.memmap(os.path.join(pack_dir, "refs.u8"), dtype=np.uint8, mode="r", shape=(self.N, 512, 512, 3))
        self.gpu_refs = None
        if torch.device(device).type == "cuda" and self.N * 786432 < 9e9:
            self.gpu_refs = torch.empty(self.N, 512, 512, 3, dtype=torch.uint8, device=device)
            for i in range(0, self.N, 256):
                self.gpu_refs[i:i + 256] = torch.from_numpy(np.ascontiguousarray(self.refs[i:i + 256])).to(device)
        self.traj = torch.from_numpy(np.load(os.path.join(pack_dir, "traj.npy")).astype(np.int64))
        self.n = torch.from_numpy(np.load(os.path.join(pack_dir, "n.npy")).astype(np.int64))
        self.dev, self.img = device, img

    def ref(self, idx):
        if self.gpu_refs is not None:
            x = self.gpu_refs[idx.to(self.dev)]
        else:
            x = torch.from_numpy(np.stack([self.refs[i] for i in idx.tolist()])).to(self.dev)
        x = x.permute(0, 3, 1, 2).float() / 255
        if self.img != 512:
            x = torch.nn.functional.interpolate(x, size=(self.img, self.img), mode="area")
        return x

    def spread(self, n):
        """n references evenly spread over the pack (packs list flat graphics first, then photos)."""
        return self.ref(torch.linspace(0, self.N - 1, n).long())

    def sample(self, B, gen=None, flip=True):
        idx = torch.randint(0, self.N, (B,), generator=gen)
        ref, Q, n = self.ref(idx), self.traj[idx].to(self.dev), self.n[idx].to(self.dev)
        if flip:
            f = torch.rand(B, device=self.dev) < 0.5
            ref = torch.where(f[:, None, None, None], ref.flip(-1), ref)
            Q = torch.where(f[:, None, None], G.hflip(Q), Q)
        return ref, Q, n
