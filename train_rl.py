"""RL stage: REINFORCE in the closed loop. Each step samples N groups per episode state, draws
them, and credits every element with the drop in log distance it and the rest of its group bring,
against the other samples (leave-one-out). One of the groups is then committed."""
import argparse
import math
import os
import time
import torch
import torch.nn.functional as F
from svg_painter import grammar as G
from svg_painter.data import TeacherData, make_batch, synth_batch
from svg_painter.episodes import Episodes
from svg_painter.model import load_model, save_model, token_nll
from svg_painter.raster import render
from train_sl import Log, evaluate, load_eval, setup


@torch.no_grad()
def snapshots(ref, canvas, Q, n, res):
    """Canvas before and after each element of every group, at res -> ref, (E,N,K+1,3,res,res)."""
    E, N, K, _ = Q.shape
    down = lambda x: F.interpolate(x, size=(res, res), mode="area") if x.shape[-1] != res else x
    r, c = down(ref), down(canvas)
    cc = c[:, None].expand(E, N, 3, res, res).reshape(E * N, 3, res, res)
    at = torch.arange(K + 1, device=Q.device)[None].expand(E * N, K + 1)
    return r, render(G.q2c(Q.reshape(E * N, K, G.W)), n.reshape(-1), res, canvas=cc, snapshots=at).reshape(E, N, K + 1, 3, res, res)


def log_dists(r, snaps, rw, lp, amp):
    """log distance to the reference, per metric, (E,N,K+1)."""
    E, N, K1 = snaps.shape[:3]
    out = {"mse": torch.log(((snaps - r[:, None, None]) ** 2).mean((3, 4, 5)).clamp(min=1e-8))}
    if "lpips" in rw:
        d = []
        for e in range(0, E, 8):
            with torch.autocast(r.device.type, dtype=torch.bfloat16, enabled=amp):
                fr, fs = lp.feats(r[e:e + 8]), lp.feats(snaps[e:e + 8].reshape(-1, *r.shape[1:]))
            d.append(lp.dist([f.float().reshape(-1, N, K1, *f.shape[1:]) for f in fs], [f.float()[:, None, None] for f in fr]))
        out["lpips"] = torch.log(torch.cat(d).clamp(min=1e-5))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--eval_pack")
    p.add_argument("--envs", type=int, default=32)
    p.add_argument("--n", type=int, default=8, help="samples per state; sample 0 is greedy")
    p.add_argument("--max_shapes", type=int, default=256, help="longest episode on the whole image")
    p.add_argument("--zoom", type=int, default=0, help="1: then paint each 2x2 tile magnified 2x")
    p.add_argument("--zoom_per", type=int, default=64, help="most elements per tile")
    p.add_argument("--reward", default="mse", help="mse, lpips, or a mix such as mse:0.5,lpips:0.5")
    p.add_argument("--reward_res", type=int, default=256)
    p.add_argument("--sl_weight", type=float, default=0.5, help="supervised loss kept alongside")
    p.add_argument("--sl_bs", type=int, default=4)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--decay", type=int, default=1, help="cosine decay to 10%")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_n", type=int, default=32)
    p.add_argument("--eval_shapes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = setup(a.seed)
    model = load_model(a.init, dev)[0]
    K, img = model.c.k_shapes, model.c.img
    rw = {}
    for part in a.reward.split(","):
        name, _, w = part.partition(":")
        rw[name] = float(w or 1)
    rw = {k: v / sum(rw.values()) for k, v in rw.items()}
    lp = None
    if "lpips" in rw:
        from svg_painter.metrics import Lpips
        lp = Lpips(dev, a.reward_res)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0, fused=dev.type == "cuda")
    data = TeacherData(a.pack, dev, img)
    ev = load_eval(a.eval_pack, a.eval_n, a.eval_shapes, dev) if a.eval_pack else None
    eps = Episodes(a.envs, data, img, dev, a.max_shapes, K, bool(a.zoom), a.zoom_per)
    amp = dev.type == "cuda"
    log = Log(a.out)
    log({"step": 0, **evaluate(model, ev)})
    E, N = a.envs, a.n
    ar = torch.arange(E, device=dev)
    stats, t0 = {}, time.time()
    for step in range(a.steps):
        frac = step / a.steps
        lr = a.lr * min(1, (step + 1) / a.warmup) * ((0.5 * (1 + math.cos(math.pi * frac)) * 0.9 + 0.1) if a.decay else 1.0)
        for g in opt.param_groups:
            g["lr"] = lr
        ref, canvas = eps.views()
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            mem = model.enc(ref, canvas)
            with torch.no_grad():
                Q, n = model.dec.sample(mem, N, K, greedy_first=True)
        r, snaps = snapshots(ref, canvas, Q, n, a.reward_res)
        lms = log_dists(r, snaps, rw, lp, amp)
        del snaps
        Rm = {k: lms[k][..., 0] - lms[k][..., -1] for k in rw}
        R = sum(w * Rm[k] / (Rm[k].std() + 1e-6) for k, w in rw.items()) if list(rw) != ["mse"] else Rm["mse"]
        seq = G.q2tokens(Q.reshape(E * N, K, G.W), n.reshape(-1))
        seq = {k: v.reshape(E, N, *v.shape[1:]) for k, v in seq.items()}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            logits = model.dec(mem, seq)
        adv = 0
        for name, w in rw.items():
            togo = (lms[name][..., :-1] - lms[name][..., -1:])[:, 1:]
            base = (togo.sum(1, keepdim=True) - togo) / (N - 2)
            am = togo - base
            adv = adv + w * am / (am.std() + 1e-6)
        adv = adv / (adv.std() + 1e-6)
        ntok, valid = token_nll(logits[:, 1:], {k: v[:, 1:] for k, v in seq.items()}, smooth=False)
        kid, tgt = seq["kidx"][:, 1:, 1:], seq["tok"][:, 1:, 1:]
        wt = torch.gather(adv, 2, kid.clamp(max=K - 1)) * (kid < K) * (tgt != G.END)
        pg = (wt.detach() * ntok).sum() / valid.sum()
        sl =torch.zeros((), device=dev)
        if a.sl_weight > 0:
            with torch.no_grad():
                r2, Q2, n2 = data.sample(a.sl_bs)
                p1, p2 = make_batch(Q2, n2, 4, K, img, r2, end_ok=False), synth_batch(a.sl_bs, 4, K, img, dev)
                sseq = {k: torch.cat([p1[2][k], p2[2][k]]) for k in ("tok", "slot", "kidx", "anc")}
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                slog = model(torch.cat([p1[0], p2[0]]), torch.cat([p1[1], p2[1]]), sseq)
            snll, sval = token_nll(slog, sseq)
            sl = snll.sum() / sval.sum()
        opt.zero_grad(set_to_none=True)
        (pg + a.sl_weight * sl).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        u = torch.rand(E, device=dev)
        pick = torch.where(u < 0.5, R.argmax(1), torch.where(u < 0.75, torch.zeros_like(ar), torch.randint(1, N, (E,), device=dev)))
        eps.commit(G.q2c(Q[ar, pick]), n[ar, pick])
        Rmse = lms["mse"][..., 0] - lms["mse"][..., -1]
        s = {"R_greedy": Rmse[:, 0].mean().item(), "R_sample": Rmse[:, 1:].mean().item(), "n_shapes": n.float().mean().item(),
             "sl": sl.item(), "gn": gn.item()}
        for k, v in s.items():
            stats[k] = stats.get(k, 0) + v
        if (step + 1) % a.log_every == 0:
            log({"step": step + 1, **{k: round(v / a.log_every, 5) for k, v in stats.items()}, "min": round((time.time() - t0) / 60, 2)})
            stats = {}
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            log({"step": step + 1, **evaluate(model, ev)})
            save_model(model, os.path.join(a.out, "ckpt.pt"), step + 1)


if __name__ == "__main__":
    main()
