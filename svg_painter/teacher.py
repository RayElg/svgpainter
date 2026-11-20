"""Search teacher: greedy fitting by random proposals, hill climbing and gradient steps through
the rasteriser. Fill colour and opacity are solved in closed form, so the search is over geometry."""
import math
import torch
import torch.nn.functional as F
from . import grammar as G
from .raster import coverage, composite
from .synth import random_shapes, clamp_shapes, area

ALPHAS = torch.tensor([(i + 1) / G.N_ALPHA for i in range(G.N_ALPHA)])


@torch.no_grad()
def score(ref, canvas, C, chunk=4096):
    """Best colour and alpha per candidate. ref, canvas (B,3,h,h), C (B,M,W)
    -> change in MSE (B,M), negative is better; C with colour and alpha set."""
    B, M, _ = C.shape
    h = ref.shape[-1]
    resid = (ref - canvas).reshape(B, 3, -1)
    cv = canvas.reshape(B, 3, -1)
    X1 = torch.cat([resid, (resid * cv).sum(1, keepdim=True)], 1).transpose(1, 2)
    X2 = torch.cat([torch.ones_like(cv[:, :1]), cv, (cv * cv).sum(1, keepdim=True)], 1).transpose(1, 2)
    S1, S2 = [], []
    per = max(1, chunk * 4096 // (h * h * B))
    for i in range(0, M, per):
        c = C[:, i:i + per]
        cov = coverage(c.reshape(-1, G.W), h).reshape(B, c.shape[1], -1)
        S1.append(cov @ X1)
        S2.append((cov * cov) @ X2)
    S1, S2 = torch.cat(S1, 1), torch.cat(S2, 1)
    A, Bc = S1[..., None, :3], S1[..., None, 3]
    D, E, Fq = S2[..., None, 0:1].clamp(min=1e-6), S2[..., None, 1:4], S2[..., None, 4]
    al = ALPHAS.to(C.device)[None, None, :, None]
    col = (((A / al + E) / D).clamp(0, 1) * 255).round() / 255
    delta = (-2 * al * col * A + al * al * (col * col * D - 2 * col * E)).sum(-1) + 2 * al[..., 0] * Bc + al[..., 0] ** 2 * Fq
    delta = delta / (3 * h * h)
    best = (delta * (1 - 0.05 * (1 - al[..., 0]))).argmin(-1)
    out = C.clone()
    out[..., G.CR:G.CR + 3] = torch.gather(col, 2, best[..., None, None].expand(B, M, 1, 3))[:, :, 0]
    out[..., G.CA] = ALPHAS.to(C.device)[best]
    return torch.gather(delta, 2, best[..., None])[..., 0], out


def mutate(C, sp, ss, sa, gen=None):
    kinds = G.COL_KIND.to(C.device)[C[..., 0].long()]
    n = torch.randn(C.shape, generator=gen, device=C.device)
    on = torch.rand(C.shape, generator=gen, device=C.device) < 0.6
    out = torch.where((kinds == G.K_POS) & on, C + sp * n, C)
    out = torch.where((kinds == G.K_SIZE) & on, C * torch.exp(ss * n), out)
    out = torch.where((kinds == G.K_ANG) & on, C + sa * n, out)
    return clamp_shapes(out)


def _rates(C, h, lr_pos, lr_size, lr_ang):
    kinds = G.COL_KIND.to(C.device)[C[..., 0].long()]
    lr = torch.where(kinds == G.K_POS, torch.full_like(C, lr_pos * G.S / h), torch.zeros_like(C))
    lr = torch.where(kinds == G.K_SIZE, lr_size * C, lr)
    return torch.where(kinds == G.K_ANG, torch.full_like(C, lr_ang), lr)


def _loss(X, h, resid, cv):
    B, M, _ = X.shape
    w = coverage(X.reshape(-1, G.W), h).reshape(B, M, 1, h, h) * X[..., G.CA, None, None, None]
    dc = X[..., G.CR:G.CR + 3, None, None].detach() - cv
    return (w * dc * (w * dc - 2 * resid)).sum((2, 3, 4))


def refine_grad(ref, canvas, C, steps=12, lr_pos=0.35, lr_size=0.04, lr_ang=0.03):
    """Adam on geometry; colour and alpha re-solved every few steps. Returns the best iterate."""
    h = ref.shape[-1]
    lr = _rates(C, h, lr_pos, lr_size, lr_ang)
    resid, cv = (ref - canvas)[:, None], canvas[:, None]
    m, v = torch.zeros_like(C), torch.zeros_like(C)
    best_d, C = score(ref, canvas, C)
    best, X = C.clone(), C.clone()
    for i in range(steps):
        with torch.enable_grad():
            X = X.detach().requires_grad_(True)
            g, = torch.autograd.grad(_loss(X, h, resid, cv).sum(), X)
        m, v = 0.8 * m + 0.2 * g, 0.9 * v + 0.1 * g * g
        X = clamp_shapes((X - lr * (m / (1 - 0.8 ** (i + 1))) / ((v / (1 - 0.9 ** (i + 1))).sqrt() + 1e-12)).detach())
        X[..., 0], X[..., G.CR:] = C[..., 0], C[..., G.CR:]
        if i % 3 == 2 or i == steps - 1:
            d, X = score(ref, canvas, X)
            better = d < best_d
            best = torch.where(better[..., None], X, best)
            best_d = torch.where(better, d, best_d)
    return best_d, best


def refine_geom(ref, canvas, C, steps=10, lr_pos=0.25, lr_size=0.03, lr_ang=0.02):
    """Adam on geometry with colour and alpha fixed. Never worse than the input."""
    h = ref.shape[-1]
    lr = _rates(C, h, lr_pos, lr_size, lr_ang)
    resid, cv = (ref - canvas)[:, None], canvas[:, None]
    m, v = torch.zeros_like(C), torch.zeros_like(C)
    X, best, best_d = C.clone(), C.clone(), None
    for i in range(steps + 1):
        with torch.enable_grad():
            X = X.detach().requires_grad_(True)
            per = _loss(X, h, resid, cv) / (3 * h * h)
            g, = torch.autograd.grad(per.sum(), X)
        per = per.detach()
        if best_d is None:
            best_d = per.clone()
        else:
            better = per < best_d
            best = torch.where(better[..., None], X.detach(), best)
            best_d = torch.where(better, per, best_d)
        if i == steps:
            break
        g = g * (3 * h * h)
        m, v = 0.8 * m + 0.2 * g, 0.9 * v + 0.1 * g * g
        X = clamp_shapes((X - lr * (m / (1 - 0.8 ** (i + 1))) / ((v / (1 - 0.9 ** (i + 1))).sqrt() + 1e-12)).detach())
        X[..., 0], X[..., G.CR:] = C[..., 0], C[..., G.CR:]
    return best_d, best


class Fitter:
    """Greedy fitter for a batch of references (B,3,512,512).
    m0 proposals per element, the top best are mutated for some rounds; grad steps on the gk best
    at the low and middle resolution; grad_hi steps on the best two at the highest."""

    def __init__(self, ref, gen=None, m0=512, top=24, kids=8, rounds=5, res=(64, 128, 256), grad=0, gk=8, grad_hi=0, type_p=None):
        self.gen, self.dev, self.B = gen, ref.device, ref.shape[0]
        self.m0, self.top, self.kids, self.rounds = m0, top, kids, rounds
        self.grad, self.gk, self.grad_hi, self.type_p = grad, gk, grad_hi, type_p
        self.res = tuple(res)
        self.ref = {r: F.interpolate(ref, size=(r, r), mode="area") if ref.shape[-1] != r else ref for r in self.res}
        self.canvas = {r: torch.ones_like(self.ref[r]) for r in self.res}
        self.logsize = torch.full((self.B,), math.log(120.0), device=self.dev)
        self.steps = 0

    def mse(self):
        r = self.res[-1]
        return ((self.ref[r] - self.canvas[r]) ** 2).mean((1, 2, 3))

    def _rand(self, *shape):
        return torch.rand(shape, generator=self.gen, device=self.dev)

    def _proposals(self, extra=None):
        B, M, lo = self.B, self.m0, self.res[0]
        err = ((self.ref[lo] - self.canvas[lo]) ** 2).sum(1).reshape(B, -1) + 1e-8
        idx = torch.multinomial(err, M, replacement=True, generator=self.gen)
        cell = G.S / lo
        cx = (idx % lo + self._rand(*idx.shape)) * cell
        cy = (idx // lo + self._rand(*idx.shape)) * cell
        uni = self._rand(B, M) < 0.2
        cx = torch.where(uni, self._rand(B, M) * G.S, cx)
        cy = torch.where(uni, self._rand(B, M) * G.S, cy)
        broad = torch.exp(math.log(2.0) + self._rand(B, M) * math.log(150))
        near = torch.exp(self.logsize[:, None] + 0.7 * torch.randn((B, M), generator=self.gen, device=self.dev))
        size = torch.where(self._rand(B, M) < 0.5, broad, near).clamp(1.5, 300)
        C = random_shapes(B * M, self.gen, centers=torch.stack([cx, cy], -1).reshape(-1, 2),
                          size=size.reshape(-1), device=self.dev, type_p=self.type_p).reshape(B, M, G.W)
        return C if extra is None else torch.cat([extra, C], 1)

    @torch.no_grad()
    def step(self):
        """Fit and draw one shape per image -> Q (B,W), gain (B,) = drop in MSE."""
        lo, mid, hi = self.res
        B = self.B
        extra = None
        if self.steps == 0:
            extra = torch.zeros(B, 1, G.W, device=self.dev)
            extra[..., 1:5] = G.S / 2
        C = self._proposals(extra)
        d, C = score(self.ref[lo], self.canvas[lo], C)
        k = min(self.top, C.shape[1])
        C = torch.gather(C, 1, d.topk(k, largest=False).indices[..., None].expand(B, k, G.W))
        for r in range(self.rounds):
            f = 0.55 ** r
            res = lo if r < 2 else mid
            pool = torch.cat([C, mutate(C.repeat_interleave(self.kids, 1), 20 * f, 0.35 * f, 0.4 * f, self.gen)], 1)
            d, pool = score(self.ref[res], self.canvas[res], pool)
            k = max(8, self.top // (1 + r))
            C = torch.gather(pool, 1, d.topk(k, largest=False).indices[..., None].expand(B, k, G.W))
        if self.grad:
            C = C[:, :self.gk]
            _, C = refine_grad(self.ref[lo], self.canvas[lo], C, self.grad)
            if mid != lo:
                _, C = refine_grad(self.ref[mid], self.canvas[mid], C, self.grad)
        if self.grad_hi:
            d, C = score(self.ref[mid], self.canvas[mid], C)
            idx = d.topk(min(2, C.shape[1]), largest=False).indices
            C2 = torch.gather(C, 1, idx[..., None].expand(B, idx.shape[1], G.W))
            _, C2 = refine_grad(self.ref[hi], self.canvas[hi], C2, self.grad_hi, lr_pos=0.25)
            C = torch.cat([C2, C], 1)
        C = G.q2c(G.c2q(C[:, :8]))
        d, C = score(self.ref[hi], self.canvas[hi], C)
        best = d.argmin(1)
        ar = torch.arange(B, device=self.dev)
        Q = G.c2q(C[ar, best])
        C = G.q2c(Q)
        for r in self.res:
            self.canvas[r] = composite(self.canvas[r], C)
        self.logsize = 0.8 * self.logsize + 0.2 * (0.5 * torch.log(area(C).clamp(min=4.0)) - math.log(2.0))
        self.steps += 1
        return Q, -d[ar, best]

    def fit(self, n):
        Qs, gains = zip(*(self.step() for _ in range(n)))
        return torch.stack(Qs, 1), torch.stack(gains, 1)
