"""Token grammar. A shape is a row of W numbers: [type, g0..g7, r, g, b, a].

    rect     cx cy hw hh ang
    ellipse  cx cy rx ry ang
    tri      x1 y1 x2 y2 x3 y3
    curve    x0 y0 x1 y1 x2 y2 width      quadratic, round caps
    blob     x1 y1 x2 y2 x3 y3 x4 y4      closed quadratic B-spline

C is continuous (px on a 512 canvas, colours in [0, 1]), Q the same layout in integer bins.
"""
import math
import torch

S = 512
T_RECT, T_ELL, T_TRI, T_CURVE, T_BLOB = range(5)
N_TYPES = 5
TYPE_NAMES = ["rect", "ellipse", "tri", "curve", "blob"]
NG = 8
W = 1 + NG + 4
CR, CA = 1 + NG, 4 + NG

K_START, K_POS, K_SIZE, K_ANG, K_COL, K_ALPHA, K_NONE = range(7)

POS_MIN, POS_STEP, N_POS = -64.0, 2.0, 320
N_SIZE, SIZE_MIN, SIZE_SPAN = 256, 0.5, 420.0
N_ANG, N_COL, N_ALPHA = 128, 256, 8

PAD, BOS, END = 0, 1, 2
TYPE0 = 3
POS0 = TYPE0 + N_TYPES
SIZE0 = POS0 + N_POS
ANG0 = SIZE0 + N_SIZE
COL0 = ANG0 + N_ANG
ALPHA0 = COL0 + N_COL
VOCAB = ALPHA0 + N_ALPHA

KIND_BASE = [TYPE0, POS0, SIZE0, ANG0, COL0, ALPHA0, 0]
KIND_N = [N_TYPES, N_POS, N_SIZE, N_ANG, N_COL, N_ALPHA, 0]
GEOM_KINDS = {
    T_RECT: [K_POS, K_POS, K_SIZE, K_SIZE, K_ANG],
    T_ELL: [K_POS, K_POS, K_SIZE, K_SIZE, K_ANG],
    T_TRI: [K_POS] * 6,
    T_CURVE: [K_POS] * 6 + [K_SIZE],
    T_BLOB: [K_POS] * 8,
}
MAXTOK = 13

_len, _kind, _col = [], [], []
for _t in range(N_TYPES):
    _g = GEOM_KINDS[_t]
    _k = [K_START] + _g + [K_COL, K_COL, K_COL, K_ALPHA]
    _c = [0] + list(range(1, 1 + len(_g))) + [CR, CR + 1, CR + 2, CA]
    _len.append(len(_k))
    _kind.append(_k + [K_NONE] * (MAXTOK - len(_k)))
    _col.append(_c + [0] * (MAXTOK - len(_k)))
TOK_LEN = torch.tensor(_len)
TOK_KIND = torch.tensor(_kind)
TOK_COL = torch.tensor(_col)
COL_KIND = torch.full((N_TYPES, W), K_NONE)
for _t in range(N_TYPES):
    for _j in range(_len[_t]):
        COL_KIND[_t, _col[_t][_j]] = _kind[_t][_j]

SLOT_PAD, SLOT_BOS, SLOT_START = 0, 1, 2
N_SLOTS = 3 + N_TYPES * (MAXTOK - 1)


def slot_id(t, j):
    return 3 + t * (MAXTOK - 1) + (j - 1)


SLOT_KIND = torch.full((N_SLOTS,), K_NONE)
SLOT_KIND[SLOT_START] = K_START
for _t in range(N_TYPES):
    for _j in range(1, _len[_t]):
        SLOT_KIND[slot_id(_t, _j)] = _kind[_t][_j]

KIND_MASK = torch.zeros(7, VOCAB, dtype=torch.bool)
KIND_MASK[K_START, TYPE0:TYPE0 + N_TYPES] = True
KIND_MASK[K_START, END] = True
for _k in (K_POS, K_SIZE, K_ANG, K_COL, K_ALPHA):
    KIND_MASK[_k, KIND_BASE[_k]:KIND_BASE[_k] + KIND_N[_k]] = True
KIND_MASK[K_NONE, PAD] = True


def max_seq_len(k):
    return 2 + MAXTOK * k


def _deq(kind, q):
    q = q.float()
    if kind == K_POS:
        return POS_MIN + POS_STEP * q
    if kind == K_SIZE:
        return SIZE_MIN + SIZE_SPAN * (q / (N_SIZE - 1)) ** 2
    if kind == K_ANG:
        return q * (math.pi / N_ANG)
    if kind == K_COL:
        return q / (N_COL - 1)
    return (q + 1) / N_ALPHA


def _quant(kind, v):
    if kind == K_POS:
        return ((v - POS_MIN) / POS_STEP).round().clamp(0, N_POS - 1)
    if kind == K_SIZE:
        return (((v - SIZE_MIN).clamp(min=0) / SIZE_SPAN).sqrt() * (N_SIZE - 1)).round().clamp(0, N_SIZE - 1)
    if kind == K_ANG:
        return torch.remainder((v * (N_ANG / math.pi)).round(), N_ANG)
    if kind == K_COL:
        return (v * (N_COL - 1)).round().clamp(0, N_COL - 1)
    return (v * N_ALPHA - 1).round().clamp(0, N_ALPHA - 1)


def _kinds(X):
    return COL_KIND.to(X.device)[X[..., 0].long().clamp(0, N_TYPES - 1)]


def q2c(Q):
    Qf = Q.float()
    out = torch.zeros_like(Qf)
    out[..., 0] = Qf[..., 0]
    kinds = _kinds(Q)
    for k in (K_POS, K_SIZE, K_ANG, K_COL, K_ALPHA):
        out = torch.where(kinds == k, _deq(k, Qf), out)
    return out


def _c2q(C):
    out = torch.zeros_like(C)
    out[..., 0] = C[..., 0]
    kinds = _kinds(C)
    for k in (K_POS, K_SIZE, K_ANG, K_COL, K_ALPHA):
        out = torch.where(kinds == k, _quant(k, C), out)
    return out.long()


def c2q(C):
    """Quantise in canonical form (twice: rounding can create ties)."""
    return _c2q(canonicalize(q2c(_c2q(canonicalize(C)))))


def canonicalize(C):
    """One representative per shape: hw >= hh, triangles and blobs clockwise from the top-most
    point, curves left to right."""
    C = C.clone()
    t = C[..., 0].long()
    re = (t == T_RECT) | (t == T_ELL)
    swap = re & (C[..., 4] > C[..., 3])
    a, b = C[..., 3].clone(), C[..., 4].clone()
    C[..., 3] = torch.where(swap, b, a)
    C[..., 4] = torch.where(swap, a, b)
    ang = torch.where(swap, C[..., 5] + math.pi / 2, C[..., 5])
    C[..., 5] = torch.where(re, torch.remainder(ang, math.pi), C[..., 5])
    tri = t == T_TRI
    if tri.any():
        P = C[..., 1:7].reshape(*C.shape[:-1], 3, 2)
        first = (P[..., 1] * 4096 + P[..., 0]).argmin(-1)
        idx = (first[..., None] + torch.arange(3, device=C.device)) % 3
        P = torch.gather(P, -2, idx[..., None].expand(*idx.shape, 2))
        cross = (P[..., 1, 0] - P[..., 0, 0]) * (P[..., 2, 1] - P[..., 0, 1]) - \
                (P[..., 1, 1] - P[..., 0, 1]) * (P[..., 2, 0] - P[..., 0, 0])
        P2 = torch.stack([P[..., 0, :], P[..., 2, :], P[..., 1, :]], -2)
        P = torch.where((cross < 0)[..., None, None], P2, P)
        C[..., 1:7] = torch.where(tri[..., None], P.reshape(*C.shape[:-1], 6), C[..., 1:7])
    bl = t == T_BLOB
    if bl.any():
        P = C[..., 1:9].reshape(*C.shape[:-1], 4, 2)
        d = P - P.mean(-2, keepdim=True)
        o1 = torch.argsort((d ** 2).sum(-1), dim=-1, stable=True)
        o2 = torch.argsort(torch.gather(torch.atan2(d[..., 1], d[..., 0]), -1, o1), dim=-1, stable=True)
        order = torch.gather(o1, -1, o2)
        P = torch.gather(P, -2, order[..., None].expand(*order.shape, 2))
        first = (P[..., 1] * 4096 + P[..., 0]).argmin(-1)
        idx = (first[..., None] + torch.arange(4, device=C.device)) % 4
        P = torch.gather(P, -2, idx[..., None].expand(*idx.shape, 2))
        C[..., 1:9] = torch.where(bl[..., None], P.reshape(*C.shape[:-1], 8), C[..., 1:9])
    cv = t == T_CURVE
    swap = cv & ((C[..., 5] * 4096 + C[..., 6]) < (C[..., 1] * 4096 + C[..., 2]))
    p0, p2 = C[..., 1:3].clone(), C[..., 5:7].clone()
    C[..., 1:3] = torch.where(swap[..., None], p2, p0)
    C[..., 5:7] = torch.where(swap[..., None], p0, p2)
    return C


def hflip(Q):
    C = q2c(Q)
    t = C[..., 0].long()
    kinds = _kinds(Q)
    xcol = torch.zeros(W, dtype=torch.bool, device=Q.device)
    xcol[[1, 3, 5, 7]] = True
    isx = (kinds == K_POS) & xcol
    isx[..., 3] = isx[..., 3] & (t >= T_TRI)
    isx[..., 5] = isx[..., 5] & (t >= T_TRI)
    C = torch.where(isx, S - C, C)
    C = torch.where(kinds == K_ANG, math.pi - C, C)
    return c2q(C)


def shape_anchors(C):
    """Where the decoder looks when predicting each token of a shape: (..., MAXTOK, 4) =
    (x, y, has_x, has_y), built only from fields already emitted."""
    t = C[..., 0].long()
    z = torch.zeros_like(C[..., 0])
    o = torch.ones_like(z)
    none = (z, z, z, z)
    x1, y1, x2, y2, x3, y3, x4, y4 = [C[..., i] for i in range(1, 9)]
    re = [none, none, (x1, z, o, z)] + [(x1, y1, o, o)] * 10
    walk = [none, none, (x1, z, o, z), (x1, y1, o, o), (x2, z, o, z), (x2, y2, o, o), (x3, z, o, z)]
    g3 = ((x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, o, o)
    mid = (0.25 * x1 + 0.5 * x2 + 0.25 * x3, 0.25 * y1 + 0.5 * y2 + 0.25 * y3, o, o)
    g4 = ((x1 + x2 + x3 + x4) / 4, (y1 + y2 + y3 + y4) / 4, o, o)
    rows = [re, walk + [g3] * 6, walk + [mid] * 6, walk + [(x3, y3, o, o), (x4, z, o, z)] + [g4] * 4]
    re, tri, cur, blob = [torch.stack([torch.stack(r, -1) for r in rs], -2) for rs in rows]
    tt = t[..., None, None]
    return torch.where(tt <= T_ELL, re, torch.where(tt == T_TRI, tri, torch.where(tt == T_CURVE, cur, blob)))


def q2tokens(Q, n, k_max=None):
    """Q (B,K,W), n (B,) -> dict of (B,L): tok, slot, kidx, anc (B,L,4). BOS, shapes, END, PAD."""
    B, K, _ = Q.shape
    dev = Q.device
    L = max_seq_len(k_max or K)
    t = Q[..., 0].clamp(0, N_TYPES - 1)
    valid = torch.arange(K, device=dev)[None] < n[:, None]
    kind = TOK_KIND.to(dev)[t]
    tok = torch.gather(Q, 2, TOK_COL.to(dev)[t]) + torch.tensor(KIND_BASE, device=dev)[kind]
    j = torch.arange(MAXTOK, device=dev)
    slot = 3 + t[..., None] * (MAXTOK - 1) + (j - 1)
    slot = torch.where(j == 0, torch.full_like(slot, SLOT_START), slot)
    ln = TOK_LEN.to(dev)[t] * valid
    pos = (1 + torch.cumsum(ln, 1) - ln)[..., None] + j
    pos = torch.where(j < ln[..., None], pos, torch.full_like(pos, L)).reshape(B, -1)
    out_tok = torch.full((B, L + 1), PAD, dtype=torch.long, device=dev)
    out_slot = torch.full((B, L + 1), SLOT_PAD, dtype=torch.long, device=dev)
    out_k = torch.full((B, L + 1), K, dtype=torch.long, device=dev)
    out_anc = torch.zeros(B, L + 1, 4, device=dev)
    out_anc.scatter_(1, pos[..., None].expand(B, K * MAXTOK, 4), shape_anchors(q2c(Q)).reshape(B, -1, 4))
    out_tok.scatter_(1, pos, tok.reshape(B, -1))
    out_slot.scatter_(1, pos, slot.reshape(B, -1))
    out_k.scatter_(1, pos, torch.arange(K, device=dev)[None, :, None].expand(B, K, MAXTOK).reshape(B, -1))
    out_tok, out_slot, out_k, out_anc = [x[:, :L].clone() for x in (out_tok, out_slot, out_k, out_anc)]
    out_tok[:, 0], out_slot[:, 0], out_anc[:, 0] = BOS, SLOT_BOS, 0
    end = (1 + ln.sum(1))[:, None]
    out_tok.scatter_(1, end, END)
    out_slot.scatter_(1, end, SLOT_START)
    out_k.scatter_(1, end, n[:, None])
    out_anc.scatter_(1, end[..., None].expand(B, 1, 4), 0.0)
    return {"tok": out_tok, "slot": out_slot, "kidx": out_k, "anc": out_anc}


class Stepper:
    """Grammar state machine for n_seq sequences being sampled."""

    def __init__(self, n_seq, k_max, device):
        self.k_max = k_max
        z = lambda: torch.zeros(n_seq, dtype=torch.long, device=device)
        self.Q = torch.zeros(n_seq, k_max, W, dtype=torch.long, device=device)
        self.n, self.j, self.t = z(), z(), z()
        self.done = torch.zeros(n_seq, dtype=torch.bool, device=device)
        self._len, self._kind, self._col = TOK_LEN.to(device), TOK_KIND.to(device), TOK_COL.to(device)
        self._base = torch.tensor(KIND_BASE, device=device)
        self._mask = KIND_MASK.to(device)
        self._ar = torch.arange(n_seq, device=device)

    def next_slot(self):
        s = 3 + self.t * (MAXTOK - 1) + (self.j - 1)
        s = torch.where(self.j == 0, torch.full_like(s, SLOT_START), s)
        return torch.where(self.done, torch.full_like(s, SLOT_PAD), s)

    def next_kidx(self):
        return torch.where(self.done, torch.full_like(self.n, self.k_max), self.n)

    def next_anchor(self):
        C = q2c(self.Q[self._ar, self.n.clamp(max=self.k_max - 1)])
        return shape_anchors(C)[self._ar, self.j.clamp(max=MAXTOK - 1)] * (~self.done)[:, None]

    def mask(self):
        kind = torch.where(self.j == 0, torch.full_like(self.j, K_START), self._kind[self.t, self.j])
        kind = torch.where(self.done, torch.full_like(kind, K_NONE), kind)
        m = self._mask[kind].clone()
        m[(self.n >= self.k_max) & (self.j == 0) & ~self.done, TYPE0:TYPE0 + N_TYPES] = False
        return m

    def push(self, tok):
        live = ~self.done
        at_start = (self.j == 0) & live
        ended = at_start & (tok == END)
        new = at_start & ~ended
        self.done = self.done | ended
        self.t = torch.where(new, tok - TYPE0, self.t)
        jj = self.j.clamp(max=MAXTOK - 1)
        col = self._col[self.t, jj]
        val = tok - self._base[self._kind[self.t, jj]]
        nn = self.n.clamp(max=self.k_max - 1)
        self.Q[self._ar, nn, col] = torch.where(live & ~at_start, val, self.Q[self._ar, nn, col])
        self.Q[self._ar, nn, 0] = torch.where(new, tok - TYPE0, self.Q[self._ar, nn, 0])
        self.j = torch.where(live & ~ended, self.j + 1, self.j)
        fin = live & (self.j >= self._len[self.t]) & ~ended
        self.n = self.n + fin.long()
        self.j = torch.where(fin, torch.zeros_like(self.j), self.j)


def _f(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s and s != "-0" else "0"


def shape_to_svg(c):
    c = [float(v) for v in c]
    t = int(c[0])
    fill = "rgb(%d,%d,%d)" % tuple(int(round(v * 255)) for v in c[CR:CR + 3])
    a = _f(c[CA])
    if t in (T_RECT, T_ELL):
        cx, cy, hw, hh, ang = c[1:6]
        tr = f' transform="rotate({_f(math.degrees(ang))} {_f(cx)} {_f(cy)})"' if ang else ""
        if t == T_RECT:
            return (f'<rect x="{_f(cx - hw)}" y="{_f(cy - hh)}" width="{_f(2 * hw)}" height="{_f(2 * hh)}"'
                    f'{tr} fill="{fill}" fill-opacity="{a}"/>')
        return f'<ellipse cx="{_f(cx)}" cy="{_f(cy)}" rx="{_f(hw)}" ry="{_f(hh)}"{tr} fill="{fill}" fill-opacity="{a}"/>'
    if t == T_TRI:
        p = [_f(v) for v in c[1:7]]
        return f'<polygon points="{p[0]},{p[1]} {p[2]},{p[3]} {p[4]},{p[5]}" fill="{fill}" fill-opacity="{a}"/>'
    if t == T_BLOB:
        P = [(c[1 + 2 * i], c[2 + 2 * i]) for i in range(4)]
        mid = [((P[i][0] + P[(i + 1) % 4][0]) / 2, (P[i][1] + P[(i + 1) % 4][1]) / 2) for i in range(4)]
        d = f"M{_f(mid[3][0])} {_f(mid[3][1])}" + "".join(
            f"Q{_f(P[i][0])} {_f(P[i][1])} {_f(mid[i][0])} {_f(mid[i][1])}" for i in range(4))
        return f'<path d="{d}Z" fill="{fill}" fill-opacity="{a}" fill-rule="evenodd"/>'
    p = [_f(v) for v in c[1:8]]
    return (f'<path d="M{p[0]} {p[1]}Q{p[2]} {p[3]} {p[4]} {p[5]}" fill="none" stroke="{fill}" stroke-opacity="{a}"'
            f' stroke-width="{p[6]}" stroke-linecap="round"/>')


def to_svg(C, tile=None, tiles=None, size=S):
    """C (N,W) -> SVG document. tile (N,) indexes tiles [(x, y, w, h)]; 0 means unclipped."""
    ids = [0] * C.shape[0] if tile is None else [int(v) for v in tile]
    lines, cur = [], 0
    for c, i in zip(C.tolist(), ids):
        if i != cur:
            lines += (["</g>"] if cur else []) + ([f'<g clip-path="url(#t{i})">'] if i else [])
            cur = i
        lines.append(shape_to_svg(c))
    lines += ["</g>"] if cur else []
    used = sorted(set(ids) - {0})
    defs = ""
    if used:
        defs = "<defs>" + "".join(f'<clipPath id="t{i}"><rect x="{tiles[i][0]}" y="{tiles[i][1]}" width="{tiles[i][2]}" '
                                  f'height="{tiles[i][3]}"/></clipPath>' for i in used) + "</defs>\n"
    body = "\n".join(lines)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {S} {S}" width="{size}" height="{size}">\n'
            f'{defs}<rect width="{S}" height="{S}" fill="#fff"/>\n{body}\n</svg>\n')
