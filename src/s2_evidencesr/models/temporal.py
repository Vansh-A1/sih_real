"""Target-anchored temporal evidence with global and dense local alignment."""
import torch
from torch import nn
from torch.nn import functional as F
from ..metrics.quality import phase_translation


def warp(image, shifts, offsets=None):
    """Apply (dy,dx) translation then learned SOURCE-coordinate local offsets.

    Portable differentiable bilinear sampling, zero exterior, align_corners=False.
    Dense offsets are a local feature warp, not a renamed resize or a DCNv2 kernel.
    """
    b, _, h, w = image.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing="ij")
    gx = xx[None].expand(b, -1, -1).float() - shifts[:, 1, None, None]
    gy = yy[None].expand(b, -1, -1).float() - shifts[:, 0, None, None]
    if offsets is not None:
        gx = gx + offsets[:, 0]; gy = gy + offsets[:, 1]
    grid = torch.stack([2 * (gx + .5) / w - 1, 2 * (gy + .5) / h - 1], -1)
    return F.grid_sample(image.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)


class TemporalEvidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.offset = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 2, 3, padding=1))
        nn.init.zeros_(self.offset[-1].weight); nn.init.zeros_(self.offset[-1].bias)
        self.stability = nn.Sequential(nn.Conv2d(10, 24, 3, padding=1), nn.GELU(), nn.Conv2d(24, 1, 1))
        self.inject = nn.Conv2d(64, 64, 1, bias=False)
        self.strength = nn.Parameter(torch.tensor(.1))

    def forward(self, target_features, target, target_weights, auxiliaries, model):
        b, _, h, w = target.shape
        support = target.new_zeros(b, 1, h, w)
        if not auxiliaries:
            return target_features, support, None
        numerator = torch.zeros_like(target_features, dtype=torch.float32)
        gate_losses = []
        for aux in auxiliaries:
            x, weight = aux["image"], aux["weights"].expand_as(aux["image"])
            if x.shape != target.shape:
                raise ValueError("Auxiliary image must match target batch and native grid")
            if not torch.isfinite(weight).all() or ((weight < 0) | (weight > 1)).any():
                raise ValueError("Auxiliary quality weights must be finite in [0,1]")
            if not (weight > 0).any():
                continue
            if not torch.isfinite(torch.where(weight > 0, x, torch.zeros_like(x))).all():
                raise ValueError("Nonfinite auxiliary values need zero weight")
            safe = torch.where(weight > 0, x, model.mean).float()
            with torch.no_grad():
                shifts, confidence = phase_translation(target[:, [2, 3]], safe[:, [2, 3]],
                                                       torch.minimum(target_weights[:, [2, 3]], weight[:, [2, 3]]), self.config.max_shift)
            learning = model.normalize(safe)
            _, features = model.encode_bands(learning)
            condition = model.degradation(learning) if model.degradation is not None else None
            for i, block in enumerate(model.naf):
                features = block(features)
                if str(i) in model.films:
                    features = model.films[str(i)](features, condition)
            features = model.swin(features)
            globally_aligned = warp(features, shifts)
            offsets = self.config.max_local_offset * torch.tanh(self.offset(torch.cat([target_features.float(), globally_aligned], 1)))
            aligned = warp(features, shifts, offsets)
            image_aligned = warp(safe, shifts, offsets)
            valid_support = warp((weight > 0).float(), shifts, offsets).amin(1, keepdim=True) >= 1 - 1e-6
            quality = warp(weight, shifts, offsets).amin(1, keepdim=True) * target_weights.amin(1, keepdim=True) * valid_support
            quality = quality * (confidence > .01)[:, None, None, None]
            difference = (target - image_aligned).abs().mean(1, keepdim=True)
            prior_stability = torch.exp(-difference / self.config.stability_tau)
            # Explicit target-date safety cutoff; proxy is not a learned real-world change detector.
            compatible = difference < 4 * self.config.stability_tau
            logits = self.stability(torch.cat([target, image_aligned, quality, difference], 1))
            gate = quality * compatible * prior_stability * torch.sigmoid(logits)
            numerator = numerator + gate * aligned
            support = support + gate
            if bool((quality > 0).any()):
                proxy = prior_stability.detach()
                loss = F.binary_cross_entropy_with_logits(logits, proxy, reduction="none")
                gate_losses.append((loss * quality).sum() / quality.sum())
        if not bool((support > 0).any()):
            return target_features, support, None
        fused = numerator / support.clamp_min(1e-8)
        addition = self.inject(fused) * support.clamp(max=1)
        return target_features + self.strength * addition, support, torch.stack(gate_losses).mean() if gate_losses else None
