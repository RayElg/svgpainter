"""Parallel painting episodes for RL and head training. An episode paints the whole image, then,
with zoom, each 2x2 tile through a 2x magnified view; finished ones restart on a new reference."""
import torch
import torch.nn.functional as F
from . import grammar as G
from .paint import view_to_image
from .raster import render
from .synth import random_scene


class Episodes:
    def __init__(self, E, data, img, dev, max_shapes, k, zoom=False, zoom_per=64, synth_frac=0.25):
        self.E, self.data, self.img, self.dev = E, data, img, dev
        self.max_shapes, self.k, self.zoom, self.zoom_per, self.synth_frac = max_shapes, k, zoom, zoom_per, synth_frac
        self.stage = torch.zeros(E, dtype=torch.long, device=dev)
        self.ref = torch.zeros(E, 3, img, img, device=dev)
        self.canvas = torch.ones(E, 3, img, img, device=dev)
        self.done_shapes = torch.zeros(E, dtype=torch.long, device=dev)
        self.budget = torch.zeros(E, dtype=torch.long, device=dev)
        self.reset(torch.ones(E, dtype=torch.bool, device=dev), stagger=True)

    def _tile_budget(self, m):
        return torch.randint(2, self.zoom_per // self.k + 1, (m,), device=self.dev) * self.k

    @torch.no_grad()
    def reset(self, mask, stagger=False):
        """stagger: start part-way through teacher trajectories (and half of them in a tile), so
        the first batches cover every stage of an episode."""
        idx = mask.nonzero(as_tuple=True)[0]
        m = idx.numel()
        if not m:
            return
        ref, Qt, nt = self.data.sample(m)
        synth = torch.rand(m, device=self.dev) < self.synth_frac
        if synth.any():
            Q, n = random_scene(m, 48, device=self.dev)
            ref = torch.where(synth[:, None, None, None], render(G.q2c(Q), n, self.img), ref)
        self.ref[idx] = ref
        self.canvas[idx] = 1.0
        self.done_shapes[idx] = 0
        self.stage[idx] = 0
        self.budget[idx] = torch.randint(3, max(4, self.max_shapes // self.k + 1), (m,), device=self.dev) * self.k
        if stagger:
            t = (torch.rand(m, device=self.dev) * (nt - self.k).clamp(min=1)).long() // self.k * self.k
            t = torch.where(synth, torch.zeros_like(t), t)
            if self.zoom:
                tile = (torch.rand(m, device=self.dev) < 0.5) & ~synth
                t = torch.where(tile, nt, t)
                self.stage[idx] = torch.where(tile, torch.randint(1, 5, (m,), device=self.dev), 0)
            self.canvas[idx] = render(G.q2c(Qt), t, self.img)
            self.done_shapes[idx] = torch.where(self.stage[idx] > 0, torch.zeros_like(t), t)
            self.budget[idx] = torch.where(self.stage[idx] > 0, self._tile_budget(m), torch.maximum(self.budget[idx], t + 2 * self.k))

    def _origin(self):
        t = self.img // 2
        s = (self.stage - 1).clamp(min=0)
        on = (self.stage > 0).float()
        return (s % 2) * t * on, (s // 2) * t * on, 1.0 - 0.5 * on

    @torch.no_grad()
    def views(self):
        """What the painter looks at: the whole image or a tile magnified 2x."""
        if not self.zoom or not bool((self.stage > 0).any()):
            return self.ref, self.canvas
        t = self.img // 2
        rv, cv = self.ref.clone(), self.canvas.clone()
        for st in range(1, 5):
            i = (self.stage == st).nonzero(as_tuple=True)[0]
            if i.numel():
                x0, y0 = ((st - 1) % 2) * t, ((st - 1) // 2) * t
                up = lambda x: F.interpolate(x[i][:, :, y0:y0 + t, x0:x0 + t], size=(self.img, self.img), mode="bilinear", align_corners=False)
                rv[i], cv[i] = up(self.ref), up(self.canvas)
        return rv, cv

    @torch.no_grad()
    def commit(self, C, n):
        """Draw the chosen groups C (E,K,W), in view coordinates, and move on."""
        if self.zoom:
            ox, oy, sc = self._origin()
            C = view_to_image(C, sc[:, None, None], ox[:, None, None], oy[:, None, None])
        new = render(C, n, self.img, canvas=self.canvas)
        if self.zoom and bool((self.stage > 0).any()):
            t = self.img // 2
            yy, xx = torch.meshgrid(torch.arange(self.img, device=self.dev), torch.arange(self.img, device=self.dev), indexing="ij")
            ins = ((xx[None] >= ox[:, None, None]) & (xx[None] < ox[:, None, None] + t) &
                   (yy[None] >= oy[:, None, None]) & (yy[None] < oy[:, None, None] + t)) | (self.stage == 0)[:, None, None]
            new = torch.where(ins[:, None], new, self.canvas)
        self.canvas = new
        self.done_shapes += n
        fin = (self.done_shapes >= self.budget) | (n == 0)
        if self.zoom:
            nxt = fin & (self.stage < 4)
            self.stage = torch.where(nxt, self.stage + 1, self.stage)
            self.done_shapes = torch.where(nxt, torch.zeros_like(self.done_shapes), self.done_shapes)
            self.budget = torch.where(nxt, self._tile_budget(self.E), self.budget)
            fin = fin & ~nxt
        self.reset(fin)
