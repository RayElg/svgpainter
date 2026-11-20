"""Sub-token refinement of rectangles and ellipses. The painter places an element on its token
grid; a small head then moves it by a pixel or two, in the element's own frame:
du, dv (centre), dhw, dhh (half-sizes), dth (rotation as arc length at the far end), in px.

    edge   ref and canvas sampled across the outline: 32 points, 8 offsets along the normal
    grad   the same plus the gradient of the error with respect to the five numbers
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import grammar as G
from .raster import composite, coverage
from .synth import clamp_shapes

P = 32
R_OFF = (-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0)
N_OUT = 5
CLAMP = 6.0
N_FEAT = 3 * P * len(R_OFF) + 4
PASSES = 3
HEADS = ("edge", "grad")


def _axes(C):
    th = C[..., 5]
    return torch.stack([th.cos(), th.sin()], -1), torch.stack([-th.sin(), th.cos()], -1)


def boundary(C):
    """P outline points and outward normals, (M,P,2) each."""
    dev = C.device
    hw, hh = C[:, 3].clamp(min=0.25)[:, None], C[:, 4].clamp(min=0.25)[:, None]
    a, b = _axes(C)
    phi = (torch.arange(P, device=dev) + 0.5) * (2 * math.pi / P) - math.pi / 4
    eu, ev = phi.cos()[None], phi.sin()[None]
    en = torch.stack([eu / hw, ev / hh], -1)
    en = en / en.norm(dim=-1, keepdim=True)
    el = torch.stack([hw * eu, hh * ev], -1)
    q = P // 4
    s = (torch.arange(q, device=dev) + 0.5) / q * 2 - 1
    one, z = torch.ones(q, device=dev), torch.zeros(q, device=dev)
    rl = torch.stack([hw * torch.cat([one, -s, -one, s])[None], hh * torch.cat([s, one, -s, -one])[None]], -1)
    rn = torch.stack([torch.cat([one, z, -one, z]), torch.cat([z, one, z, -one])], -1)[None].expand(C.shape[0], P, 2)
    is_el = (C[:, 0].long() == G.T_ELL)[:, None, None]
    loc, nl = torch.where(is_el, el, rl), torch.where(is_el, en, rn)
    pts = C[:, None, 1:3] + loc[..., :1] * a[:, None] + loc[..., 1:] * b[:, None]
    return pts, nl[..., :1] * a[:, None] + nl[..., 1:] * b[:, None]


def edge_features(ref, canvas, C):
    """(M,3,S,S), (M,3,S,S), (M,W) -> (M,N_FEAT)."""
    S = ref.shape[-1]
    pts, nrm = boundary(C)
    xy = pts[:, :, None] + nrm[:, :, None] * torch.tensor(R_OFF, device=C.device)[None, None, :, None]
    grid = xy / S * 2 - 1
    rs = F.grid_sample(ref, grid, mode="bilinear", padding_mode="border", align_corners=False)
    cs = F.grid_sample(canvas, grid, mode="bilinear", padding_mode="border", align_corners=False)
    inside = ((xy >= 0) & (xy <= S)).all(-1).float()
    fill, al = C[:, G.CR:G.CR + 3, None, None], C[:, G.CA, None, None, None]
    res = ((rs - cs) ** 2).sum(1)
    gain = res - ((rs - (cs + al * (fill - cs))) ** 2).sum(1)
    sq = lambda v: v.sign() * v.abs().sqrt()
    extra = torch.stack([C[:, 3].clamp(min=0.25).log() / 5, C[:, 4].clamp(min=0.25).log() / 5, C[:, G.CA],
                         (C[:, 0].long() == G.T_ELL).float()], 1)
    f = torch.cat([(sq(gain) * inside).flatten(1), (sq(res) * inside).flatten(1), inside.flatten(1), extra], 1)
    return torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def grad_features(ref, canvas, C):
    """Gradient of the squared error w.r.t. the five numbers, per px of outline, signed sqrt."""
    S = ref.shape[-1]
    with torch.enable_grad():
        X = C.detach().clone().requires_grad_(True)
        w = coverage(X, S)[:, None] * X[:, G.CA, None, None, None]
        dc = X[:, G.CR:G.CR + 3, None, None].detach() - canvas
        g, = torch.autograd.grad((w * dc * (w * dc - 2 * (ref - canvas))).sum(), X)
    a, b = _axes(C)
    arm = torch.maximum(C[:, 3], C[:, 4]).clamp(min=1.0)
    gl = torch.stack([(g[:, 1:3] * a).sum(-1), (g[:, 1:3] * b).sum(-1), g[:, 3], g[:, 4], g[:, 5] / arm], 1)
    gl = gl / (4 * (C[:, 3] + C[:, 4]).clamp(min=1.0) * C[:, G.CA].clamp(min=0.1))[:, None]
    gl = torch.nan_to_num(gl, nan=0.0, posinf=0.0, neginf=0.0)
    return (gl.sign() * gl.abs().sqrt()).clamp(-10, 10)


def to_local(C0, C1):
    """The move from C0 to C1 as (M,5)."""
    a, b = _axes(C0)
    d = C1[:, 1:3] - C0[:, 1:3]
    dth = torch.remainder(C1[:, 5] - C0[:, 5] + math.pi / 2, math.pi) - math.pi / 2
    out = torch.stack([(d * a).sum(-1), (d * b).sum(-1), C1[:, 3] - C0[:, 3], C1[:, 4] - C0[:, 4],
                       dth * torch.maximum(C0[:, 3], C0[:, 4])], 1)
    return out.clamp(-CLAMP, CLAMP)


def apply_local(C0, o):
    """Inverse of to_local. Other primitives come back unchanged."""
    a, b = _axes(C0)
    o = torch.nan_to_num(o, nan=0.0, posinf=0.0, neginf=0.0).clamp(-CLAMP, CLAMP).to(C0.dtype)
    C1 = C0.clone()
    C1[:, 1:3] = C0[:, 1:3] + o[:, :1] * a + o[:, 1:2] * b
    C1[:, 3] = (C0[:, 3] + o[:, 2]).clamp(min=G.SIZE_MIN)
    C1[:, 4] = (C0[:, 4] + o[:, 3]).clamp(min=G.SIZE_MIN)
    C1[:, 5] = C0[:, 5] + o[:, 4] / torch.maximum(C0[:, 3], C0[:, 4]).clamp(min=1.0)
    return torch.where((C0[:, 0].long() <= G.T_ELL)[:, None], clamp_shapes(C1), C0)


class OffsetHeads(nn.Module):
    def __init__(self):
        super().__init__()
        mk = lambda n: nn.Sequential(nn.Linear(n, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, N_OUT))
        self.edge, self.grad = mk(N_FEAT + PASSES), mk(N_FEAT + N_OUT + PASSES)
        for h in (self.edge, self.grad):
            nn.init.zeros_(h[-1].weight)
            nn.init.zeros_(h[-1].bias)

    def predict(self, which, ref, canvas, C, t):
        one = F.one_hot(torch.full((C.shape[0],), min(t, PASSES - 1), device=C.device), PASSES).float()
        f = edge_features(ref, canvas, C)
        if which == "edge":
            return self.edge(torch.cat([f, one], 1))
        return self.grad(torch.cat([f, grad_features(ref, canvas, C), one], 1))

    @torch.no_grad()
    def refine(self, which, ref, canvas, C, passes=PASSES, guard=False):
        """guard: keep whichever of C and each pass's result lowers the error most."""
        cands = [C]
        for t in range(passes):
            C = apply_local(C, self.predict(which, ref, canvas, C, t).float())
            cands.append(C)
        if not guard:
            return C
        err = torch.stack([((ref - composite(canvas, c)) ** 2).sum((1, 2, 3)) for c in cands], 1)
        return torch.stack(cands, 1)[torch.arange(C.shape[0], device=C.device), err.argmin(1)]


def load_heads(path, device):
    heads = OffsetHeads().to(device).eval()
    heads.load_state_dict(torch.load(path, map_location=device)["heads"])
    return heads
