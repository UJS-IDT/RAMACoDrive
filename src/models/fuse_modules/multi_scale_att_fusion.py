import torch
import torch.nn as nn
import torch.nn.functional as F
from models.fuse_modules.self_attn import AttFusion

class MultiAttFusion(nn.Module):
    def __init__(self, input_channels=256, pool_scales=(1, 2, 4), use_proj=True):
        super().__init__()
        self.C = input_channels
        self.scales = tuple(sorted(set(pool_scales)))

        self.fusions = nn.ModuleDict()
        for s in self.scales:
            self.fusions[str(s)] = AttFusion(self.C)

        self.use_proj = use_proj
        if use_proj:
            self.post_proj = nn.ModuleDict({
                str(s): nn.Sequential(
                    nn.Conv2d(self.C, self.C, kernel_size=1, bias=False),
                    nn.BatchNorm2d(self.C),
                    nn.ReLU(inplace=True)
                ) for s in self.scales
            })

        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

        self.single = AttFusion(self.C)

        self.refine = nn.Sequential(
            nn.Conv2d(self.C, self.C, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.C),
            nn.ReLU(inplace=True)
        )

    def _pool(self, x, s):
        if s == 1:
            return x
        return F.avg_pool2d(x, kernel_size=s, stride=s, ceil_mode=False)

    def _upsample_to(self, x, size_hw):
        if x.shape[-2:] == size_hw:
            return x
        return F.interpolate(x, size=size_hw, mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _check_shapes(self, outs):
        h, w = outs[0].shape[-2:]
        for t in outs:
            assert t.shape[-2:] == (h, w)

    def forward(self, x, record_len):
        """
        x: [sum(n_i), C, H, W]
        record_len: [B]
        """
        Bflat, C, H, W = x.shape

        base = self.single(x, record_len)

        outs = []
        for s in self.scales:
            x_s = self._pool(x, s)
            f_s = self.fusions[str(s)](x_s, record_len)
            f_s = self._upsample_to(f_s, (H, W))
            if self.use_proj:
                f_s = self.post_proj[str(s)](f_s)
            outs.append(f_s)

        weights = torch.softmax(self.scale_logits, dim=0)
        fused = sum(w * o for w, o in zip(weights, outs))

        out = self.refine(fused) + base
        return out