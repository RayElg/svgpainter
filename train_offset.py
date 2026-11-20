"""Train the offset heads with the painter frozen. Every rect and ellipse the painter emits is
polished by gradient steps on its geometry; each head gets PASSES tries to predict what is still
missing to the polished element. Half the episodes advance with the grad head's result, half
with the polish."""
import argparse
import time
import os
import torch
import torch.nn.functional as F
from svg_painter import grammar as G
from svg_painter.data import TeacherData
from svg_painter.episodes import Episodes
from svg_painter.metrics import psnr
from svg_painter.model import load_model
from svg_painter.offset import HEADS, PASSES, OffsetHeads, apply_local, to_local
from svg_painter.paint import paint
from svg_painter.raster import composite
from svg_painter.teacher import refine_geom
from train_sl import Log, setup


@torch.no_grad()
def evaluate(model, heads, ref, n_shapes, bs=16):
    """PSNR [flat, photo] at n_shapes, whole view, with no head and each head."""
    h, out = ref.shape[0] // 2, {}
    for name, hd in (("none", None), ("edge", heads), ("grad", heads)):
        e = torch.cat([paint(model, ref[i:i + bs], [(1, n_shapes)], heads=hd, head=name, guard=False)["mse"][:, -1]
                       for i in range(0, ref.shape[0], bs)])
        out[name] = [round(psnr(e[:h]).mean().item(), 2), round(psnr(e[h:]).mean().item(), 2)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="painter checkpoint")
    p.add_argument("--pack", required=True)
    p.add_argument("--eval_pack", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--heads", help="heads to continue from")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--max_shapes", type=int, default=256)
    p.add_argument("--zoom_per", type=int, default=64)
    p.add_argument("--polish_steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=300)
    p.add_argument("--eval_n", type=int, default=32)
    p.add_argument("--eval_shapes", type=int, default=64)
    a = p.parse_args()
    dev = setup(0)
    model = load_model(a.init, dev)[0].eval()
    model.requires_grad_(False)
    K, img = model.c.k_shapes, model.c.img
    heads = OffsetHeads().to(dev)
    if a.heads:
        heads.load_state_dict(torch.load(a.heads, map_location=dev)["heads"])
    opt = torch.optim.AdamW(heads.parameters(), lr=a.lr, weight_decay=0.0)
    ev = TeacherData(a.eval_pack, dev, img).spread(a.eval_n)
    eps = Episodes(a.envs, TeacherData(a.pack, dev, img), img, dev, a.max_shapes, K, True, a.zoom_per)
    log = Log(a.out)
    hub = lambda x, y, w: (F.huber_loss(x, y, reduction="none") * w).sum()
    E, stats, t0 = a.envs, {}, time.time()
    for step in range(a.steps):
        lr = a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * (1 - step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        ref, canvas = eps.views()
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            Q, n = model.dec.sample(model.enc(ref, canvas), 1, K, greedy_first=True)
        Q, n = Q[:, 0], n[:, 0]
        C0 = G.q2c(Q)
        use_pred = torch.rand(E, device=dev) < 0.5
        cv, Cg, loss, cnt, tot = canvas, C0.clone(), 0.0, 0.0, {}
        for k in range(K):
            Ck = C0[:, k]
            with torch.no_grad():
                Cp = refine_geom(ref, cv, Ck[:, None], steps=a.polish_steps)[1][:, 0]
                w = ((k < n) & (Ck[:, 0].long() <= G.T_ELL)).float()[:, None]
                zero = torch.zeros(E, 5, device=dev)
                tot["left_0"] = tot.get("left_0", 0.0) + hub(zero, to_local(Ck, Cp), w).item()
            cur = {h: Ck for h in HEADS}
            for t in range(PASSES):
                for h in HEADS:
                    o = heads.predict(h, ref, cv, cur[h], t)
                    loss = loss + hub(o, torch.nan_to_num(to_local(cur[h], Cp)), w)
                    with torch.no_grad():
                        cur[h] = apply_local(cur[h], o.detach().float())
                        tot[f"left_{h}_{t + 1}"] = tot.get(f"left_{h}_{t + 1}", 0.0) + hub(zero, to_local(cur[h], Cp), w).item()
            cnt += w.sum().item() * 5
            with torch.no_grad():
                Cn = torch.where(use_pred[:, None] & (w > 0), cur["grad"], torch.where(w > 0, Cp, Ck))
                Cg[:, k] = Cn
                cv = composite(cv, Cn, keep=k < n)
        opt.zero_grad(set_to_none=True)
        (loss / max(cnt, 1.0)).backward()
        torch.nn.utils.clip_grad_norm_(heads.parameters(), 1.0)
        opt.step()
        eps.commit(Cg, n)
        for k, v in tot.items():
            stats[k] = stats.get(k, 0.0) + v / max(cnt, 1.0)
        if (step + 1) % a.log_every == 0:
            log({"step": step + 1, **{k: round(v / a.log_every, 4) for k, v in sorted(stats.items())}, "min": round((time.time() - t0) / 60, 2)})
            stats = {}
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            torch.save({"heads": heads.state_dict(), "step": step + 1}, os.path.join(a.out, "heads.pt"))
            log({"step": step + 1, "psnr": evaluate(model, heads.eval(), ev, a.eval_shapes)})
            heads.train()


if __name__ == "__main__":
    main()
