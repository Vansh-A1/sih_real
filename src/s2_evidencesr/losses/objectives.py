import torch
from torch.nn import functional as F


class NoValidData(ValueError):
    """An unavailable loss must not be logged as a successful zero error."""


def masked_mean(values, weights):
    weights = weights.expand_as(values).float()
    if not torch.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any():
        raise ValueError("Loss weights must be finite in [0,1]")
    if not bool((weights > 0).any()):
        raise NoValidData("No valid support for this loss or metric")
    safe = torch.where(weights > 0, values.float(), torch.zeros_like(values, dtype=torch.float32))
    if not torch.isfinite(safe).all():
        raise FloatingPointError("Nonfinite value on valid loss support")
    return (safe * weights).sum() / weights.sum()


def footprint_weights(weights, radius=1):
    # Exclude artificial exterior values from filtered reference supervision.
    bad = F.pad(1 - weights.float(), [radius] * 4, value=1)
    return 1 - F.max_pool2d(bad, 2 * radius + 1, stride=1)


def highpass(y):
    return y - F.avg_pool2d(y, 3, stride=1, padding=1)


def spectral_angle(pred, target, weights, differentiable=True):
    p, y = pred.float(), target.float()
    dot = (p * y).sum(1, keepdim=True)
    pn, yn = p.norm(dim=1, keepdim=True), y.norm(dim=1, keepdim=True)
    mask = weights.amin(1, keepdim=True) * ((pn > 1e-6) & (yn > 1e-6))
    cosine = dot / (pn * yn).clamp_min(1e-12)
    delta = 1e-6 if differentiable else 0
    return masked_mean(torch.acos(cosine.clamp(-1 + delta, 1 - delta)), mask)


def reconstruction_loss(result, target, hr_weights, observation, lr_weights, sensor, config):
    """Four V1 losses act before correction. All operations/reductions FP32."""
    hr_weights = hr_weights * result.valid.to(hr_weights.dtype)
    y = torch.where(hr_weights > 0, target.float(), torch.zeros_like(target, dtype=torch.float32))
    x = torch.where(lr_weights > 0, observation.float(), torch.zeros_like(observation, dtype=torch.float32))
    p = result.pre_correction.float()
    filtered = footprint_weights(hr_weights)
    # Cycle supervision needs both observation quality and valid reference filter support.
    cycle_weights = lr_weights * sensor.observation_weights(hr_weights)
    raw = {"radiometric": masked_mean(torch.sqrt((p - y).square() + 1e-6), hr_weights),
           "cycle": masked_mean((sensor(p) - x).abs(), cycle_weights),
           "sam": spectral_angle(p, y, hr_weights)}
    hf = masked_mean((highpass(p) - highpass(y)).abs(), filtered)
    dx = (p[..., 1:] - p[..., :-1]) - (y[..., 1:] - y[..., :-1])
    dy = (p[..., 1:, :] - p[..., :-1, :]) - (y[..., 1:, :] - y[..., :-1, :])
    hf += .5 * (masked_mean(dx.abs(), torch.minimum(hr_weights[..., 1:], hr_weights[..., :-1])) +
                masked_mean(dy.abs(), torch.minimum(hr_weights[..., 1:, :], hr_weights[..., :-1, :])))
    raw["high_frequency"] = hf
    if result.uncertainty_scale is not None:
        scale = result.uncertainty_scale.float().clamp_min(1e-5)
        raw["uncertainty"] = masked_mean((result.corrected.float() - y).abs() / scale + scale.log(), hr_weights)
    if result.temporal_gate_loss is not None:
        raw["temporal_gate"] = result.temporal_gate_loss
    weighted = {key: value * getattr(config, key) for key, value in raw.items()}
    total = sum(weighted.values())
    # Shared-backbone ensemble members each receive physical reconstruction supervision.
    if result.ensemble_predictions is not None:
        extra = torch.stack([masked_mean((member - y).abs(), hr_weights) for member in result.ensemble_predictions[1:]]).mean()
        raw["ensemble_reconstruction"] = extra
        weighted["ensemble_reconstruction"] = extra
        total = total + extra
    return total, {"raw": raw, "weighted": weighted, "valid_weight": hr_weights.sum().float()}
