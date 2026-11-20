"""SSIM and LPIPS. LPIPS needs the `lpips` package."""
import torch
import torch.nn.functional as F

_win = {}


def psnr(mse):
    return -10 * torch.log10(mse.clamp(min=1e-10))


def ssim(a, b, size=11, sigma=1.5):
    """Mean SSIM per image, a, b (B,3,H,W) in [0,1]."""
    key = (str(a.device), a.dtype)
    if key not in _win:
        x = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-x * x / (2 * sigma * sigma))
        g = g / g.sum()
        _win[key] = (g[:, None] * g[None])[None, None].expand(3, 1, size, size).contiguous().to(a.device, a.dtype)
    w = _win[key]
    mu_a, mu_b = F.conv2d(a, w, groups=3), F.conv2d(b, w, groups=3)
    va = F.conv2d(a * a, w, groups=3) - mu_a * mu_a
    vb = F.conv2d(b * b, w, groups=3) - mu_b * mu_b
    cab = F.conv2d(a * b, w, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cab + c2)) / ((mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2))
    return s.mean((1, 2, 3))


class Lpips:
    """LPIPS (AlexNet) split into features and distance, so one reference is encoded once and
    compared with many canvases."""

    def __init__(self, device, res=256):
        import lpips
        self.lp = lpips
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.w = [lin.model[-1].weight.clamp(min=0).sqrt() for lin in self.net.lins]
        self.res = res

    @torch.no_grad()
    def feats(self, x, chunk=512):
        if x.shape[-1] != self.res:
            x = F.interpolate(x, size=(self.res, self.res), mode="area")
        out = None
        for i in range(0, x.shape[0], chunk):
            h = self.net.net.forward(self.net.scaling_layer(x[i:i + chunk].float() * 2 - 1))
            f = [self.lp.normalize_tensor(v) * w for v, w in zip(h, self.w)]
            out = [[v] for v in f] if out is None else [o + [v] for o, v in zip(out, f)]
        return [torch.cat(o) for o in out]

    @staticmethod
    def dist(fa, fb):
        return sum(((a - b) ** 2).sum(-3).mean((-2, -1)) for a, b in zip(fa, fb))

    @torch.no_grad()
    def __call__(self, a, b):
        return self.dist(self.feats(a), self.feats(b))
