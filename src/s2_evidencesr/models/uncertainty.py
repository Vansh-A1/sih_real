import torch
from torch import nn


class UncertaintyHead(nn.Module):
    """Predict conditional error scale in reflectance units, not pure sensor noise."""
    def __init__(self, members=1):
        super().__init__()
        self.log_scale = nn.Conv2d(32, 4, 3, padding=1)
        nn.init.zeros_(self.log_scale.weight); nn.init.constant_(self.log_scale.bias, -3.)
        self.adapters = nn.ModuleList([nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 4, 3, padding=1)) for _ in range(members - 1)])
        for head in self.adapters:
            nn.init.normal_(head[-1].weight, std=.001); nn.init.zeros_(head[-1].bias)

    def forward(self, decoded):
        scale = self.log_scale(decoded).clamp(-9, 1).exp().clamp_min(1e-5)
        corrections = [head(decoded) for head in self.adapters]
        return scale, corrections if corrections else None
