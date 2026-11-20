"""Random shapes, and synthetic scenes with exact ground truth."""
import math
import torch
from . import grammar as G
from .raster import coverage

TYPE_P = (0.25, 0.25, 0.2, 0.12, 0.18)


def _rand(shape, gen, dev):
    return torch.rand(shape, generator=gen, device=dev)


def _randn(shape, gen, dev):
    return torch.randn(shape, generator=gen, device=dev)


def random_shapes(M, gen=None, types=None, centers=None, size=None, smin=2.0, smax=300.0, device="cpu", type_p=None):
    """(M,W) continuous shapes. size (M,) sets the half-extent."""
    dev = device
    if types is None:
        t = torch.multinomial(torch.tensor(type_p or TYPE_P, device=dev), M, replacement=True, generator=gen)
    else:
        t = torch.tensor(types, device=dev)[torch.randint(0, len(types), (M,), generator=gen, device=dev)]
    c = centers if centers is not None else _rand((M, 2), gen, dev) * G.S
    s = size if size is not None else torch.exp(math.log(smin) + _rand((M,), gen, dev) * math.log(smax / smin))
    out = torch.zeros(M, G.W, device=dev)
    out[:, 0] = t.float()
    asp = torch.exp(_rand((M,), gen, dev) * math.log(0.15))
    ang = _rand((M,), gen, dev) * math.pi
    ang = torch.where(_rand((M,), gen, dev) < 0.35, (ang > math.pi / 2).float() * (math.pi / 2), ang)
    re = torch.stack([c[:, 0], c[:, 1], s, s * asp, ang], 1)
    th = _rand((M, 3), gen, dev) * (2 * math.pi)
    rad = (0.4 + 0.9 * _rand((M, 3), gen, dev)) * s[:, None]
    tri = torch.stack([c[:, 0, None] + rad * torch.cos(th), c[:, 1, None] + rad * torch.sin(th)], 2).reshape(M, 6)
    d = _randn((M, 2), gen, dev)
    d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6) * s[:, None]
    p1 = c + _randn((M, 2), gen, dev) * (0.5 * s[:, None])
    w = torch.exp(math.log(1.5) + _rand((M,), gen, dev) * math.log(48 / 1.5))
    cur = torch.cat([c + d, p1, c - d, torch.minimum(w, 1.5 + s)[:, None]], 1)
    th4 = _rand((M, 1), gen, dev) * (2 * math.pi) + torch.arange(4, device=dev) * (math.pi / 2) \
        + (_rand((M, 4), gen, dev) - 0.5) * 1.2
    rad4 = (0.5 + 1.0 * _rand((M, 4), gen, dev)) * s[:, None]
    blob = torch.stack([c[:, 0, None] + rad4 * torch.cos(th4), c[:, 1, None] + rad4 * torch.sin(th4)], 2).reshape(M, 8)
    out[:, 1:9] = torch.where((t == G.T_BLOB)[:, None], blob, out[:, 1:9])
    out[:, 1:6] = torch.where((t <= G.T_ELL)[:, None], re, out[:, 1:6])
    out[:, 1:7] = torch.where((t == G.T_TRI)[:, None], tri, out[:, 1:7])
    out[:, 1:8] = torch.where((t == G.T_CURVE)[:, None], cur, out[:, 1:8])
    out[:, G.CR:G.CR + 3] = _rand((M, 3), gen, dev)
    a = (torch.randint(0, G.N_ALPHA, (M,), generator=gen, device=dev) + 1).float() / G.N_ALPHA
    out[:, G.CA] = torch.where(_rand((M,), gen, dev) < 0.8, torch.ones_like(a), a)
    return clamp_shapes(out)


def clamp_shapes(C):
    C = C.clone()
    kinds = G.COL_KIND.to(C.device)[C[..., 0].long()]
    C = torch.where(kinds == G.K_POS, C.clamp(G.POS_MIN, G.POS_MIN + G.POS_STEP * (G.N_POS - 1)), C)
    return torch.where(kinds == G.K_SIZE, C.clamp(G.SIZE_MIN, G.SIZE_MIN + G.SIZE_SPAN), C)


def area(C):
    """Approximate area in px^2, (...,W) -> (...)."""
    t = C[..., 0].long()
    re = 4 * C[..., 3] * C[..., 4]
    el = math.pi * C[..., 3] * C[..., 4]
    tri = 0.5 * ((C[..., 3] - C[..., 1]) * (C[..., 6] - C[..., 2]) - (C[..., 4] - C[..., 2]) * (C[..., 5] - C[..., 1])).abs()
    cur = (((C[..., 5] - C[..., 1]) ** 2 + (C[..., 6] - C[..., 2]) ** 2).sqrt() + C[..., 7]) * C[..., 7]
    x, y = C[..., 1:9:2], C[..., 2:9:2]
    blob = (5 / 12) * (x * y.roll(-1, -1) - x.roll(-1, -1) * y).sum(-1).abs()
    return torch.where(t == G.T_RECT, re, torch.where(t == G.T_ELL, el, torch.where(t == G.T_TRI, tri,
                       torch.where(t == G.T_CURVE, cur, blob))))


@torch.no_grad()
def random_scene(B, N, gen=None, device="cpu", vis_res=128, min_vis=0.35):
    """A background, then shapes painted large to small; shapes that end up mostly hidden are
    dropped. -> Q (B,N,W), n (B,)."""
    dev = device
    lo = torch.exp(math.log(5) + _rand((B, 1), gen, dev) * math.log(10))
    hi = lo * torch.exp(math.log(3) + _rand((B, 1), gen, dev) * math.log(3))
    size = torch.exp(torch.log(lo) + _rand((B, N), gen, dev) * torch.log(hi / lo)).clamp(max=280)
    C = random_shapes(B * N, gen, size=size.reshape(-1), device=dev).reshape(B, N, G.W)
    pal = _rand((B, 6, 3), gen, dev)
    npal = torch.randint(2, 7, (B, 1), generator=gen, device=dev)
    pick = (torch.rand((B, N), generator=gen, device=dev) * npal).long()
    col = (torch.gather(pal, 1, pick[..., None].expand(B, N, 3)) + 0.06 * _randn((B, N, 3), gen, dev)).clamp(0, 1)
    C[..., G.CR:G.CR + 3] = torch.where(_rand((B, 1, 1), gen, dev) < 0.3, C[..., G.CR:G.CR + 3], col)
    bg = torch.zeros(B, G.W, device=dev)
    bg[:, 1:5] = G.S / 2
    bg[:, G.CR:G.CR + 3] = _rand((B, 3), gen, dev)
    bg[:, G.CA] = 1
    C[:, 0] = bg
    a = area(C)
    a[:, 0] = float("inf")
    C = torch.gather(C, 1, a.argsort(1, descending=True)[..., None].expand(B, N, G.W))
    count = torch.randint(3, N + 1, (B,), generator=gen, device=dev)
    Q = G.c2q(C)
    cov = coverage(G.q2c(Q).reshape(B * N, G.W), vis_res).reshape(B, N, -1) > 0.5
    cov = cov & (torch.arange(N, device=dev)[None, :, None] < count[:, None, None])
    top = (cov * torch.arange(1, N + 1, device=dev)[None, :, None]).amax(1)
    vis = torch.zeros(B, N + 1, device=dev).scatter_add_(1, top, torch.ones_like(top, dtype=torch.float))[:, 1:]
    keep = (vis >= min_vis * cov.sum(-1).float()) & (vis >= 4)
    keep[:, 0] = True
    order = ((~keep).long() * N + torch.arange(N, device=dev)[None]).argsort(1)
    return torch.gather(Q, 1, order[..., None].expand(B, N, G.W)), keep.sum(1)
