"""Encoder-decoder: (reference, canvas) -> grammar tokens for the next group of elements.

The encoder fuses the two pixel-aligned images channel-wise before patchifying. The decoder is a
small causal transformer that cross-attends to the image tokens with a Gaussian window around
the point its current element is anchored at, and also sees the pixels around that point.
"""
import math
from dataclasses import dataclass, asdict, fields
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import grammar as G


@dataclass
class Config:
    img: int = 512
    patch: int = 16
    d: int = 384
    heads: int = 6
    enc_layers: int = 8
    dec_layers: int = 6
    k_shapes: int = 8
    glimpse: int = 1


N_OCT = 8
PIX_RES = (128, 32, 8)
GLIMPSE_N = 9
GLIMPSE = ((256, 2.0), (128, 8.0))
GLIMPSE_DIM = 6 * GLIMPSE_N * GLIMPSE_N * len(GLIMPSE)


def pos_fourier(v):
    u = (v - G.POS_MIN) / (G.POS_STEP * G.N_POS)
    ph = 2 * math.pi * u[..., None] * (0.5 * 2.0 ** torch.arange(N_OCT, device=v.device, dtype=v.dtype))
    return torch.cat([ph.sin(), ph.cos()], -1)


def col_fourier(v):
    u = v * ((G.N_COL - 1) / G.N_COL)
    ph = 2 * math.pi * u[..., None] * (0.5 * 2.0 ** torch.arange(N_OCT, device=v.device, dtype=v.dtype))
    return torch.cat([ph.sin(), ph.cos()], -1)


def patch_centers(g, device):
    c = (torch.arange(g, device=device, dtype=torch.float32) + 0.5) * (G.S / g)
    return c[None].expand(g, g).reshape(-1), c[:, None].expand(g, g).reshape(-1)


def pixel_pyramid(ref, canvas):
    x = torch.cat([ref, canvas], 1).float()
    return [F.adaptive_avg_pool2d(x, r) if x.shape[-1] != r else x for r in PIX_RES]


def sample_pyramid(pyr, anc):
    """anc (B,L,4) -> ref and canvas colour at each anchor, (B,L,18), and has (B,L,1)."""
    grid = (anc[..., :2] / G.S * 2 - 1)[:, :, None]
    out = [F.grid_sample(p, grid.to(p.dtype), mode="bilinear", padding_mode="border", align_corners=False)[..., 0] for p in pyr]
    has = (anc[..., 2] * anc[..., 3])[..., None]
    return torch.cat(out, 1).transpose(1, 2) * has, has


def glimpse_sources(ref, canvas):
    x = torch.cat([ref, canvas], 1).float() * 2 - 1
    return [F.adaptive_avg_pool2d(x, r) if x.shape[-1] != r else x for r, _ in GLIMPSE]


def sample_glimpse(srcs, anc):
    """9x9 windows of ref and canvas around each anchor at 2 px and 8 px spacing, (B,L,GLIMPSE_DIM)."""
    B, L, _ = anc.shape
    o = torch.arange(GLIMPSE_N, device=anc.device, dtype=torch.float32) - (GLIMPSE_N - 1) / 2
    off = torch.stack([o[None].expand(GLIMPSE_N, GLIMPSE_N), o[:, None].expand(GLIMPSE_N, GLIMPSE_N)], -1).reshape(1, 1, -1, 2)
    out = []
    for src, (_, step) in zip(srcs, GLIMPSE):
        grid = ((anc[:, :, None, :2].float() + off * step) / G.S * 2 - 1).to(src.dtype)
        v = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=False)
        out.append(v.permute(0, 2, 3, 1).reshape(B, L, -1))
    return torch.cat(out, -1) * (anc[..., 2] * anc[..., 3])[..., None]


def anchor_dist2(anc, g):
    """Squared px distance from each patch centre to the anchor, on the axes it defines."""
    px, py = patch_centers(g, anc.device)
    return (px - anc[..., 0:1]) ** 2 * anc[..., 2:3] + (py - anc[..., 1:2]) ** 2 * anc[..., 3:4]


def fourier_table():
    """(VOCAB, 80): fixed features of each numeric token's value, one block per field kind."""
    kinds = (G.K_POS, G.K_SIZE, G.K_ANG, G.K_COL, G.K_ALPHA)
    F_ = 2 * N_OCT
    t = torch.zeros(G.VOCAB, len(kinds) * F_)
    for j, k in enumerate(kinds):
        base, n = G.KIND_BASE[k], G.KIND_N[k]
        u = torch.arange(n).float() / n
        fr = torch.arange(1, N_OCT + 1).float() if k == G.K_ANG else 0.5 * 2.0 ** torch.arange(N_OCT).float()
        ph = 2 * math.pi * u[:, None] * fr[None]
        t[base:base + n, j * F_:(j + 1) * F_] = torch.cat([ph.sin(), ph.cos()], 1)
    return t


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.a = nn.Linear(d, 4 * d)
        self.b = nn.Linear(4 * d, d)

    def forward(self, x):
        return self.b(F.gelu(self.a(x)))


class SelfAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def split(self, x):
        B, L, D = x.shape
        return self.qkv(x).reshape(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)

    def forward(self, x, causal=False):
        B, L, D = x.shape
        q, k, v = self.split(x)
        return self.o(F.scaled_dot_product_attention(q, k, v, is_causal=causal).transpose(1, 2).reshape(B, L, D))


class CrossAttn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        sig = [12.0 * 2 ** i for i in range(heads - heads // 3)]
        self.inv_sigma = nn.Parameter(torch.tensor([1 / v for v in sig] + [0.0] * (heads // 3)))
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def memory(self, mem):
        B, M, D = mem.shape
        return self.kv(mem).reshape(B, M, 2, self.h, D // self.h).permute(2, 0, 3, 1, 4)

    def forward(self, x, kv, d2):
        B, L, D = x.shape
        q = self.q(x).reshape(B, L, self.h, D // self.h).transpose(1, 2)
        bias = (-0.5 * d2[:, None] * (self.inv_sigma ** 2)[None, :, None, None]).to(q.dtype)
        y = F.scaled_dot_product_attention(q, kv[0], kv[1], attn_mask=bias)
        return self.o(y.transpose(1, 2).reshape(B, L, D))


class EncBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.mlp = SelfAttn(d, heads), MLP(d)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class DecBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.cross, self.mlp = SelfAttn(d, heads), CrossAttn(d, heads), MLP(d)


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d
        assert c.patch == 16
        chans = [9, d // 8, d // 4, d // 2, d]
        layers = []
        for i in range(4):
            layers += [nn.Conv2d(chans[i], chans[i + 1], 3, 2, 1), nn.GELU()]
        self.stem = nn.Sequential(*layers, nn.Conv2d(d, d, 1))
        g = c.img // c.patch
        self.pos = nn.Parameter(torch.randn(1, g * g, d) * 0.02)
        px, py = patch_centers(g, "cpu")
        self.register_buffer("pos_feat", torch.cat([pos_fourier(px), pos_fourier(py)], -1)[None], persistent=False)
        self.pos_proj = nn.Linear(4 * N_OCT, d, bias=False)
        self.blocks = nn.ModuleList(EncBlock(d, c.heads) for _ in range(c.enc_layers))
        self.norm = nn.LayerNorm(d)
        self.glimpse = bool(c.glimpse)

    def forward(self, ref, canvas):
        """-> (image tokens (B,M,d), pixel pyramid, glimpse sources)"""
        x = torch.cat([ref * 2 - 1, canvas * 2 - 1, (ref - canvas) * 2], 1)
        x = self.stem(x).flatten(2).transpose(1, 2) + self.pos + self.pos_proj(self.pos_feat.to(x.dtype))
        for b in self.blocks:
            x = b(x)
        return self.norm(x), pixel_pyramid(ref, canvas), glimpse_sources(ref, canvas) if self.glimpse else None


class Decoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d
        self.c = c
        self.L = G.max_seq_len(c.k_shapes)
        self.g = c.img // c.patch
        self.tok = nn.Embedding(G.VOCAB, d)
        self.slot = nn.Embedding(G.N_SLOTS, d)
        self.nxt = nn.Embedding(G.N_SLOTS, d)
        self.kidx = nn.Embedding(c.k_shapes + 1, d)
        self.pos = nn.Embedding(self.L, d)
        self.blocks = nn.ModuleList(DecBlock(d, c.heads) for _ in range(c.dec_layers))
        self.pix_in = nn.Linear(6 * len(PIX_RES) * 2 * N_OCT + 1, d, bias=False)
        if c.glimpse:
            self.gl_a = nn.Linear(GLIMPSE_DIM, d)
            self.gl_b = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, G.VOCAB)
        ft = fourier_table()
        self.register_buffer("ftab", ft, persistent=False)
        self.f_in = nn.Linear(ft.shape[1], d, bias=False)
        self.f_out = nn.Linear(d, ft.shape[1], bias=False)
        self.register_buffer("slot_kind", G.SLOT_KIND.clone(), persistent=False)
        self.register_buffer("kind_mask", G.KIND_MASK.clone(), persistent=False)

    def embed(self, tok, slot, nxt, kidx, pos):
        x = self.tok(tok) + self.slot(slot) + self.nxt(nxt) + self.kidx(kidx) + self.pos(pos)
        return x + self.f_in(self.ftab[tok].to(x.dtype))

    def look(self, x, mem, anc):
        """Add what the decoder sees around the anchors anc (B,L,4) to x."""
        _, pyr, gsrc = mem
        v, has = sample_pyramid(pyr, anc)
        x = x + self.pix_in(torch.cat([col_fourier(v).flatten(-2) * has, has], -1).to(x.dtype)).reshape(x.shape)
        if self.c.glimpse:
            x = x + self.gl_b(F.gelu(self.gl_a(sample_glimpse(gsrc, anc).to(x.dtype)))).reshape(x.shape)
        return x

    def logits(self, x):
        x = self.norm(x)
        return self.head(x) + self.f_out(x) @ self.ftab.to(x.dtype).T

    def forward(self, mem, seq):
        """Teacher-forced: seq tensors (B,N,L), N sequences per image -> masked logits (B,N,L-1,V)."""
        tok, slot, kidx = seq["tok"], seq["slot"], seq["kidx"]
        B, N, L = tok.shape
        Li = L - 1
        x = self.embed(tok[..., :-1], slot[..., :-1], slot[..., 1:], kidx[..., 1:].clamp(max=self.c.k_shapes),
                       torch.arange(Li, device=tok.device))
        anc = seq["anc"][..., 1:, :].reshape(B, N * Li, 4)
        x = self.look(x, mem, anc).reshape(B * N, Li, -1)
        d2 = anchor_dist2(anc, self.g)
        for b in self.blocks:
            x = x + b.attn(b.n1(x), causal=True)
            x = x + b.cross(b.n2(x).reshape(B, N * Li, -1), b.cross.memory(mem[0]), d2).reshape(B * N, Li, -1)
            x = x + b.mlp(b.n3(x))
        logits = self.logits(x).reshape(B, N, Li, -1).float()
        return logits.masked_fill(~self.kind_mask[self.slot_kind[slot[..., 1:]]], float("-inf"))

    @torch.no_grad()
    def sample(self, mem, n_samples=1, k_shapes=None, temperature=1.0, greedy_first=False):
        """N groups per image with fixed-size KV caches -> Q (B,N,K,W), n (B,N).
        greedy_first: sample 0 of each image is the argmax decode."""
        c = self.c
        K = k_shapes or c.k_shapes
        B, M, D = mem[0].shape
        N, dev = n_samples, mem[0].device
        h, dh = c.heads, D // c.heads
        st = G.Stepper(B * N, K, dev)
        Lmax = G.max_seq_len(K) - 1
        cross = [b.cross.memory(mem[0]) for b in self.blocks]
        ck = [torch.zeros(B * N, h, Lmax, dh, device=dev, dtype=mem[0].dtype) for _ in self.blocks]
        cv = [torch.zeros_like(t) for t in ck]
        tok = torch.full((B * N,), G.BOS, device=dev)
        slot = torch.full((B * N,), G.SLOT_BOS, device=dev)
        greedy = torch.zeros(B, N, dtype=torch.bool, device=dev)
        greedy[:, 0] = greedy_first
        greedy = greedy.reshape(-1)
        for i in range(Lmax):
            nxt = st.next_slot()
            x = self.embed(tok, slot, nxt, st.next_kidx().clamp(max=c.k_shapes), torch.full_like(tok, i))[:, None]
            anc = st.next_anchor().reshape(B, N, 4)
            x = self.look(x, mem, anc)
            d2 = anchor_dist2(anc, self.g)
            for li, b in enumerate(self.blocks):
                q, k, v = b.attn.split(b.n1(x))
                ck[li][:, :, i], cv[li][:, :, i] = k[:, :, 0], v[:, :, 0]
                y = F.scaled_dot_product_attention(q, ck[li][:, :, :i + 1], cv[li][:, :, :i + 1])
                x = x + b.attn.o(y.transpose(1, 2).reshape(B * N, 1, D))
                x = x + b.cross(b.n2(x).reshape(B, N, D), cross[li], d2).reshape(B * N, 1, D)
                x = x + b.mlp(b.n3(x))
            logits = self.logits(x[:, 0]).float().masked_fill(~st.mask(), float("-inf"))
            new = torch.multinomial(F.softmax(logits / max(temperature, 1e-4), -1), 1)[:, 0]
            new = torch.where(greedy, logits.argmax(-1), new)
            new = torch.where(st.done, torch.full_like(new, G.PAD), new)
            st.push(new)
            tok, slot = new, nxt
            if bool(st.done.all()):
                break
        return st.Q.reshape(B, N, K, G.W), st.n.reshape(B, N)


class Model(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.enc = Encoder(c)
        self.dec = Decoder(c)
        self.aux = nn.Linear(c.d, 2)
        self.apply(self._init)
        if c.glimpse:
            nn.init.zeros_(self.dec.gl_b.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ref, canvas, seq):
        return self.dec(self.enc(ref, canvas), seq)


def load_model(path, device, glimpse=None):
    """glimpse=1 on a checkpoint without glimpse windows adds them as a no-op."""
    ck = torch.load(path, map_location=device)
    names = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in ck["cfg"].items() if k in names})
    if glimpse is not None:
        cfg.glimpse = glimpse
    model = Model(cfg).to(device)
    res = model.load_state_dict(ck["model"], strict=False)
    assert not res.unexpected_keys and all(k.startswith("dec.gl_") for k in res.missing_keys), res
    return model, ck


def save_model(model, path, step):
    torch.save({"model": model.state_dict(), "cfg": asdict(model.c), "step": step}, path)


_SMOOTH = None


def _smooth_matrix():
    """Ordinal fields put a little target mass on neighbouring bins."""
    m = torch.eye(G.VOCAB)
    for kind, sigma, circ in ((G.K_POS, 0.7, False), (G.K_SIZE, 0.7, False), (G.K_ANG, 0.7, True), (G.K_COL, 1.0, False)):
        base, n = G.KIND_BASE[kind], G.KIND_N[kind]
        i = torch.arange(n)
        d = (i[:, None] - i[None]).abs().float()
        if circ:
            d = torch.minimum(d, n - d)
        w = torch.exp(-0.5 * (d / sigma) ** 2) * (d <= 3)
        m[base:base + n, base:base + n] = w / w.sum(1, keepdim=True)
    return m


def token_nll(logits, seq, smooth=True):
    """Per-token NLL (B,N,L-1) and validity mask."""
    global _SMOOTH
    tgt = seq["tok"][..., 1:]
    valid = tgt != G.PAD
    logp = F.log_softmax(logits, -1)
    if smooth:
        if _SMOOTH is None or _SMOOTH.device != logits.device:
            _SMOOTH = _smooth_matrix().to(logits.device)
        q = _SMOOTH[tgt]
        nll = -(q * logp.masked_fill(q == 0, 0)).sum(-1)
    else:
        nll = -torch.gather(logp, -1, tgt[..., None])[..., 0]
    return nll.masked_fill(~valid, 0), valid


def aux_loss(model, mem, Qt, nt):
    """Every patch predicts whether the next element, and the next K, cover it."""
    from .raster import coverage
    B, K, _ = Qt.shape
    g = int(round(mem[0].shape[1] ** 0.5))
    with torch.no_grad():
        cov = coverage(G.q2c(Qt).reshape(B * K, G.W), g).reshape(B, K, g * g)
        cov = cov * (torch.arange(K, device=Qt.device)[None] < nt[:, None])[..., None]
        tgt = torch.stack([cov[:, 0], cov.amax(1)], -1).clamp(0, 1)
    return F.binary_cross_entropy_with_logits(model.aux(mem[0]).float(), tgt)
