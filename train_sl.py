"""Supervised stage: synthetic scenes (exact targets) and teacher trajectories."""
import argparse
import json
import math
import os
import time
import torch
from svg_painter import grammar as G
from svg_painter.data import TeacherData, make_batch, synth_batch
from svg_painter.metrics import psnr
from svg_painter.model import Config, Model, aux_loss, load_model, save_model, token_nll
from svg_painter.paint import paint
from svg_painter.raster import render


def setup(seed):
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return dev


def load_eval(pack, n, n_shapes, dev):
    d = TeacherData(pack, dev)
    idx = torch.linspace(0, d.N - 1, n).long()
    ref = d.ref(idx)
    Q, k = d.traj[idx].to(dev), d.n[idx].to(dev).clamp(max=n_shapes)
    cv = torch.cat([render(G.q2c(Q[i:i + 8]), k[i:i + 8], 512) for i in range(0, n, 8)])
    return {"ref": ref, "teacher": psnr(((cv - ref) ** 2).mean((1, 2, 3))).mean().item(), "shapes": n_shapes}


@torch.no_grad()
def evaluate(model, ev, bs=16):
    """Held-out PSNR at ev['shapes'] elements, greedy, whole view; the teacher's for scale."""
    if ev is None:
        return {}
    model.eval()
    e = torch.cat([paint(model, ev["ref"][i:i + bs], [(1, ev["shapes"])])["mse"][:, -1] for i in range(0, len(ev["ref"]), bs)])
    model.train()
    return {"psnr": round(psnr(e).mean().item(), 3), "psnr_teacher": round(ev["teacher"], 3)}


class Log:
    def __init__(self, out):
        os.makedirs(out, exist_ok=True)
        self.f = open(os.path.join(out, "log.jsonl"), "a")

    def __call__(self, rec):
        line = json.dumps(rec)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--eval_pack")
    p.add_argument("--init", help="checkpoint to continue from")
    p.add_argument("--glimpse", type=int, help="1: add glimpse windows (to a checkpoint without them)")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--bs_synth", type=int, default=8)
    p.add_argument("--bs_teacher", type=int, default=8)
    p.add_argument("--snaps", type=int, default=4, help="states per scene")
    p.add_argument("--early", type=float, default=0.25, help="share of states from the first turns")
    p.add_argument("--aux", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2500)
    p.add_argument("--eval_n", type=int, default=32)
    p.add_argument("--eval_shapes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = setup(a.seed)
    if a.init:
        model = load_model(a.init, dev, a.glimpse)[0]
    else:
        model = Model(Config(glimpse=a.glimpse or 0)).to(dev)
    K, img = model.c.k_shapes, model.c.img
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters on {dev}", flush=True)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    rest = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=dev.type == "cuda")
    data = TeacherData(a.pack, dev, img)
    ev = load_eval(a.eval_pack, a.eval_n, a.eval_shapes, dev) if a.eval_pack else None
    amp = dev.type == "cuda"
    log = Log(a.out)
    t0 = time.time()
    for step in range(a.steps):
        lr = a.lr * min(1, (step + 1) / a.warmup) * (0.5 * (1 + math.cos(math.pi * step / a.steps)) * 0.95 + 0.05)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.no_grad():
            parts = []
            if a.bs_synth:
                parts.append(synth_batch(a.bs_synth, a.snaps, K, img, dev, early=a.early))
            if a.bs_teacher:
                r, Q, n = data.sample(a.bs_teacher)
                parts.append(make_batch(Q, n, a.snaps, K, img, r, end_ok=False, early=a.early))
            ref, cv = torch.cat([x[0] for x in parts]), torch.cat([x[1] for x in parts])
            seq = {k: torch.cat([x[2][k] for x in parts]) for k in parts[0][2]}
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            mem = model.enc(ref, cv)
            logits = model.dec(mem, seq)
        nll, valid = token_nll(logits, seq)
        loss = nll.sum() / valid.sum()
        la = aux_loss(model, mem, seq["Q"], seq["n"]) if a.aux else torch.zeros((), device=dev)
        opt.zero_grad(set_to_none=True)
        (loss + a.aux * la).backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.log_every == 0:
            ns = a.bs_synth * a.snaps
            part = lambda s: round((nll[s].sum() / valid[s].sum().clamp(min=1)).item(), 4)
            log({"step": step, "loss": round(loss.item(), 4), "synth": part(slice(0, ns)), "teacher": part(slice(ns, None)),
                 "aux": round(la.item(), 4), "lr": lr, "gn": round(gn.item(), 3), "min": round((time.time() - t0) / 60, 2)})
        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            log({"step": step + 1, **evaluate(model, ev)})
            save_model(model, os.path.join(a.out, "ckpt.pt"), step + 1)


if __name__ == "__main__":
    main()
