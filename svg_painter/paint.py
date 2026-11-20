"""The closed loop: look at (reference, canvas), emit a group of elements, draw them, repeat.

A plan paints coarse to fine: (z, per) splits the image into z x z tiles and paints `per`
elements in each, looking at the tile magnified z times. Tile elements are clipped to the tile.
"""
import torch
import torch.nn.functional as F
from . import grammar as G
from .raster import composite, render


def parse_plan(text):
    """'1:128,2:64' -> [(1, 128), (2, 64)]"""
    return [tuple(int(v) for v in part.split(":")) for part in text.split(",")]


def plan_size(plan):
    return sum(z * z * per for z, per in plan)


def mse(a, b):
    return ((a - b) ** 2).mean((1, 2, 3))


def view_to_image(C, s, ox, oy):
    """Shapes drawn in a magnified view -> image coordinates. s, ox, oy scalars or broadcastable."""
    kinds = G.COL_KIND.to(C.device)[C[..., 0].long().clamp(0, G.N_TYPES - 1)]
    col = torch.arange(G.W, device=C.device).expand_as(kinds)
    out = torch.where((kinds == G.K_POS) & (col % 2 == 1), C * s + ox, C)
    out = torch.where((kinds == G.K_POS) & (col % 2 == 0), C * s + oy, out)
    return torch.where(kinds == G.K_SIZE, C * s, out)


def tile_view(x, z, tx, ty):
    S = x.shape[-1]
    t = S // z
    return F.interpolate(x[:, :, ty * t:(ty + 1) * t, tx * t:(tx + 1) * t], size=(S, S), mode="bilinear", align_corners=False)


@torch.no_grad()
def paint(model, ref, plan, heads=None, head="grad", passes=3, guard=True, canvas=None):
    """ref (B,3,512,512) -> canvas, mse after every group (B,turns+1), and the elements in drawing
    order in image px: C (B,total,W), keep (B,total), tile (total,) indexing tiles [(x, y, w, h)]."""
    dev, S, K = ref.device, ref.shape[-1], model.c.k_shapes
    canvas = torch.ones_like(ref) if canvas is None else canvas
    hist, Cs, keeps, ids, tiles = [mse(ref, canvas)], [], [], [], [(0, 0, S, S)]
    for z, per in plan:
        t = S // z
        for ty in range(z):
            for tx in range(z):
                tid = 0
                if z > 1:
                    tiles.append((tx * t, ty * t, t, t))
                    tid = len(tiles) - 1
                left = per
                while left > 0:
                    k = min(K, left)
                    left -= k
                    rv, cv = (ref, canvas) if z == 1 else (tile_view(ref, z, tx, ty), tile_view(canvas, z, tx, ty))
                    with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        Q, n = model.dec.sample(model.enc(rv, cv), 1, k, greedy_first=True)
                    C, n = G.q2c(Q[:, 0]), n[:, 0]
                    for j in range(k):
                        if heads is not None:
                            C[:, j] = heads.refine(head, rv, cv, C[:, j], passes, guard)
                        cv = composite(cv, C[:, j], keep=j < n)
                    if z == 1:
                        canvas = cv
                    else:
                        C = view_to_image(C, 1.0 / z, tx * t, ty * t)
                        drawn = render(C, n, S, canvas=canvas)
                        canvas = canvas.clone()
                        canvas[:, :, ty * t:(ty + 1) * t, tx * t:(tx + 1) * t] = drawn[:, :, ty * t:(ty + 1) * t, tx * t:(tx + 1) * t]
                    Cs.append(C)
                    keeps.append(torch.arange(k, device=dev)[None] < n[:, None])
                    ids += [tid] * k
                    hist.append(mse(ref, canvas))
    return {"canvas": canvas, "mse": torch.stack(hist, 1), "C": torch.cat(Cs, 1), "keep": torch.cat(keeps, 1),
            "tile": torch.tensor(ids), "tiles": tiles}


def to_svg(out, b=0):
    keep = out["keep"][b].cpu()
    return G.to_svg(out["C"][b].cpu()[keep], out["tile"][keep], out["tiles"])
