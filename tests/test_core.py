"""python -m tests.test_core"""
import io
import numpy as np
import torch
from svg_painter import grammar as G
from svg_painter import raster as R
from svg_painter.model import Config, Model
from svg_painter.synth import random_scene, random_shapes


def test_roundtrip():
    g = torch.Generator().manual_seed(0)
    Q = G.c2q(random_shapes(64 * 8, g).reshape(64, 8, G.W))
    assert (G.c2q(G.q2c(Q)) == Q).all()
    n = torch.randint(0, 9, (64,), generator=g)
    seq = G.q2tokens(Q, n)
    st = G.Stepper(64, 8, "cpu")
    for i in range(1, seq["tok"].shape[1]):
        live = ~st.done
        assert st.mask()[live, seq["tok"][live, i]].all()
        assert (st.next_slot()[live] == seq["slot"][live, i]).all()
        st.push(seq["tok"][:, i])
    assert st.done.all() and (st.n == n).all()
    for b in range(64):
        assert (st.Q[b, :n[b]] == Q[b, :n[b]]).all()


def test_resvg():
    import resvg_py
    from PIL import Image
    g = torch.Generator().manual_seed(1)
    for ty in range(G.N_TYPES):
        C = G.q2c(G.c2q(random_shapes(24, g, types=[ty])))
        for i in range(0, 24, 6):
            mine = R.render(C[None, i:i + 6], torch.tensor([6]), 512)[0]
            png = bytes(resvg_py.svg_to_bytes(svg_string=G.to_svg(C[i:i + 6])))
            ref = torch.from_numpy(np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).copy()).permute(2, 0, 1) / 255
            err = (mine - ref).abs().mean().item() * 255
            assert err < 1.0, (G.TYPE_NAMES[ty], err)


def test_hflip():
    Q, n = random_scene(4, 16)
    a = R.render(G.q2c(Q), n, 128).flip(-1)
    b = R.render(G.q2c(G.hflip(Q)), n, 128)
    assert (a - b).abs().mean().item() * 255 < 1.0


def test_sampler():
    """Cached sampling and the teacher-forced pass pick the same tokens."""
    torch.manual_seed(0)
    m = Model(Config(img=64, d=96, heads=6, enc_layers=1, dec_layers=2, k_shapes=4)).eval()
    torch.nn.init.normal_(m.dec.gl_b.weight, std=0.02)
    for p in m.parameters():
        p.data.mul_(4.0)
    ref, cv = torch.rand(3, 3, 64, 64), torch.rand(3, 3, 64, 64)
    with torch.no_grad():
        mem = m.enc(ref, cv)
        Q, n = m.dec.sample(mem, 2, 4, greedy_first=True)
        seq = {k: v.reshape(3, 2, *v.shape[1:]) for k, v in G.q2tokens(Q.reshape(6, 4, G.W), n.reshape(-1)).items()}
        logits = m.dec(mem, seq)
    tgt = seq["tok"][..., 1:]
    valid = (tgt != G.PAD) & ~((tgt == G.END) & (n == 4)[..., None])
    assert (logits.argmax(-1) == tgt)[:, 0][valid[:, 0]].all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
