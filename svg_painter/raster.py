"""Batched anti-aliased rasteriser, differentiable in geometry. Coverage comes from signed
distances in pixel units, so one code path serves every resolution."""
import math
import torch
from .grammar import S, T_RECT, T_ELL, T_TRI, T_CURVE, T_BLOB, W, CR, CA

CURVE_SEGS = 12
BLOB_SEGS = 8
_grid_cache = {}


def grid(res, device, dtype=torch.float32):
    key = (res, str(device), dtype)
    if key not in _grid_cache:
        v = (torch.arange(res, device=device, dtype=dtype) + 0.5) * (S / res)
        _grid_cache[key] = (v[None, None, :], v[None, :, None])
    return _grid_cache[key]


def _cov_rect(p, X, Y, sc):
    cx, cy, hw, hh, ang = [p[:, i, None, None] for i in range(5)]
    c, s = torch.cos(ang), torch.sin(ang)
    dx, dy = X - cx, Y - cy
    lx, ly = (dx * c + dy * s) * sc, (-dx * s + dy * c) * sc
    hw, hh = hw * sc, hh * sc
    ox = (torch.minimum(hw, lx + 0.5) - torch.maximum(-hw, lx - 0.5)).clamp(0, 1)
    oy = (torch.minimum(hh, ly + 0.5) - torch.maximum(-hh, ly - 0.5)).clamp(0, 1)
    return ox * oy


def _cov_ell(p, X, Y, sc):
    cx, cy, rx, ry, ang = [p[:, i, None, None] for i in range(5)]
    c, s = torch.cos(ang), torch.sin(ang)
    dx, dy = X - cx, Y - cy
    ax, ay = (dx * c + dy * s) / rx, (-dx * s + dy * c) / ry
    k1 = torch.sqrt(ax * ax + ay * ay)
    k2 = torch.sqrt((ax / rx) ** 2 + (ay / ry) ** 2)
    d = torch.where(k2 > 1e-8, (k1 - 1) * k1 / k2.clamp(min=1e-8), -torch.minimum(rx, ry).expand_as(k1))
    thin = (2 * rx * sc).clamp(max=1) * (2 * ry * sc).clamp(max=1)
    return (0.5 - d * sc).clamp(0, 1) * thin


def _cov_tri(p, X, Y, sc):
    x1, y1, x2, y2, x3, y3 = [p[:, i, None, None] for i in range(6)]
    area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    sgn = torch.sign(area2)
    dmin = lmax = None
    for ax, ay, bx, by in ((x1, y1, x2, y2), (x2, y2, x3, y3), (x3, y3, x1, y1)):
        ex, ey = bx - ax, by - ay
        ln = torch.sqrt(ex * ex + ey * ey).clamp(min=1e-6)
        d = sgn * (ex * (Y - ay) - ey * (X - ax)) / ln
        dmin = d if dmin is None else torch.minimum(dmin, d)
        lmax = ln if lmax is None else torch.maximum(lmax, ln)
    return (0.5 + dmin * sc).clamp(0, 1) * (area2.abs() / lmax * sc).clamp(max=1)


def _cov_curve(p, X, Y, sc):
    x0, y0, x1, y1, x2, y2, w = [p[:, i, None, None] for i in range(7)]
    d2 = None
    px, py = x0, y0
    for i in range(1, CURVE_SEGS + 1):
        t = i / CURVE_SEGS
        a, b, c = (1 - t) ** 2, 2 * t * (1 - t), t * t
        qx, qy = a * x0 + b * x1 + c * x2, a * y0 + b * y1 + c * y2
        ex, ey = qx - px, qy - py
        h = (((X - px) * ex + (Y - py) * ey) / (ex * ex + ey * ey).clamp(min=1e-8)).clamp(0, 1)
        dd = (X - px - h * ex) ** 2 + (Y - py - h * ey) ** 2
        d2 = dd if d2 is None else torch.minimum(d2, dd)
        px, py = qx, qy
    return (0.5 - (torch.sqrt(d2) - w / 2) * sc).clamp(0, 1) * (w * sc).clamp(max=1)


def blob_outline(p):
    """(M,8) control points -> (M, 4*BLOB_SEGS, 2) polyline of the closed quadratic B-spline."""
    P = p.reshape(-1, 4, 2)
    a, b = (P.roll(1, 1) + P) / 2, (P + P.roll(-1, 1)) / 2
    t = (torch.arange(BLOB_SEGS, device=p.device, dtype=p.dtype) / BLOB_SEGS)[None, None, :, None]
    V = (1 - t) ** 2 * a[:, :, None] + 2 * t * (1 - t) * P[:, :, None] + t * t * b[:, :, None]
    return V.reshape(-1, 4 * BLOB_SEGS, 2)


def _cov_blob(p, X, Y, sc):
    V = blob_outline(p[:, :8])
    Vn = V.roll(-1, 1)
    M, n = V.shape[:2]
    eg = int(max(1, min(n, (1 << 26) // max(1, M * X.shape[-1] * Y.shape[-2]))))
    Xe, Ye = X[:, None], Y[:, None]
    d2 = inside = None
    for i in range(0, n, eg):
        ax, ay = V[:, i:i + eg, 0, None, None], V[:, i:i + eg, 1, None, None]
        bx, by = Vn[:, i:i + eg, 0, None, None], Vn[:, i:i + eg, 1, None, None]
        ex, ey = bx - ax, by - ay
        wx, wy = Xe - ax, Ye - ay
        h = ((wx * ex + wy * ey) / (ex * ex + ey * ey).clamp(min=1e-8)).clamp(0, 1)
        dd = ((wx - h * ex) ** 2 + (wy - h * ey) ** 2).amin(1)
        d2 = dd if d2 is None else torch.minimum(d2, dd)
        c = ((ay > Ye) != (by > Ye)) & (((wy * ex - wx * ey) > 0) == (by > ay))
        c = (c.sum(1) & 1).bool()
        inside = c if inside is None else inside ^ c
    area2 = (V[..., 0] * Vn[..., 1] - Vn[..., 0] * V[..., 1]).sum(1)[:, None, None]
    d = torch.sqrt(d2.clamp(min=1e-12))
    P = p[:, :8].reshape(-1, 4, 2)
    diam = (P.amax(1) - P.amin(1)).norm(dim=-1).clamp(min=1e-6)[:, None, None]
    thick = (2 / math.pi) * area2.abs() / diam
    return (0.5 + torch.where(inside, d, -d) * sc).clamp(0, 1) * (thick * sc).clamp(max=1)


_COV = {T_RECT: _cov_rect, T_ELL: _cov_ell, T_TRI: _cov_tri, T_CURVE: _cov_curve, T_BLOB: _cov_blob}


def coverage(C, res):
    """(M,W) -> (M,res,res) in [0,1], alpha not applied."""
    X, Y = grid(res, C.device, C.dtype)
    sc = res / S
    t = C[:, 0].long()
    geom = C[:, 1:CR]
    if len(t) and bool((t == t[0]).all()):
        return _COV[int(t[0])](geom, X, Y, sc)
    out = torch.zeros(C.shape[0], res, res, device=C.device, dtype=C.dtype)
    for ty, fn in _COV.items():
        idx = (t == ty).nonzero(as_tuple=True)[0]
        if idx.numel():
            out[idx] = fn(geom[idx], X, Y, sc)
    return out


def composite(canvas, C, keep=None):
    """Draw one shape per canvas. canvas (M,3,H,W), C (M,W), keep (M,) bool."""
    w = coverage(C, canvas.shape[-1]) * C[:, CA, None, None]
    if keep is not None:
        w = w * keep[:, None, None].to(w.dtype)
    return canvas + w[:, None] * (C[:, CR:CR + 3, None, None] - canvas)


def render(C, n, res, canvas=None, snapshots=None, chunk=8):
    """Draw the first n[b] shapes of C (B,K,W) -> (B,3,res,res). snapshots (B,J): return the
    canvases after that many shapes instead, (B,J,3,res,res)."""
    B, K, _ = C.shape
    if canvas is None:
        canvas = torch.ones(B, 3, res, res, device=C.device, dtype=C.dtype)
    snaps, snap_at = None, {}
    if snapshots is not None:
        snaps = canvas[:, None].repeat(1, snapshots.shape[1], 1, 1, 1)
        for b, row in enumerate(snapshots.tolist()):
            for j, k in enumerate(row):
                snap_at.setdefault(k, ([], []))
                snap_at[k][0].append(b)
                snap_at[k][1].append(j)
    kmax = int(n.max()) if n.numel() else 0
    ar = torch.arange(K, device=C.device)
    for k0 in range(0, kmax, chunk):
        ch = min(chunk, kmax - k0)
        cc = C[:, k0:k0 + ch]
        w = coverage(cc.reshape(-1, W), res).reshape(B, ch, 1, res, res)
        w = w * (cc[..., CA] * (ar[k0:k0 + ch][None] < n[:, None]))[..., None, None, None]
        col = cc[..., CR:CR + 3, None, None]
        for i in range(ch):
            if k0 + i in snap_at:
                b, j = snap_at[k0 + i]
                snaps[b, j] = canvas[b]
            canvas = torch.lerp(canvas, col[:, i].expand_as(canvas), w[:, i])
    if snaps is None:
        return canvas
    for k, (b, j) in snap_at.items():
        if k >= kmax:
            snaps[b, j] = canvas[b]
    return snaps
