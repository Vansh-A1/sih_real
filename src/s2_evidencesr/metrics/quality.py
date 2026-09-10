"""Physical metrics with explicit masks/ranges and unavailable-metric reporting."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from ..config import BANDS
from ..losses.objectives import masked_mean, spectral_angle, footprint_weights, NoValidData


def phase_translation(target, auxiliary, weights=None, max_shift=None):
    """Integer phase correlation: shift (dy,dx) to APPLY to auxiliary.

    Confidence is a peak fraction, an engineering diagnostic, not calibrated probability.
    Returns [B,2] shifts and [B] scores. No wrap is performed by the alignment warp.
    """
    if weights is None:
        weights = torch.ones_like(target)
    reference = (torch.nan_to_num(target) * weights).mean(1)
    candidate = (torch.nan_to_num(auxiliary) * weights).mean(1)
    reference = reference - reference.mean((-2, -1), keepdim=True)
    candidate = candidate - candidate.mean((-2, -1), keepdim=True)
    product = torch.fft.fft2(reference.float()) * torch.fft.fft2(candidate.float()).conj()
    cross = product / product.abs().clamp_min(1e-8)
    correlation = torch.fft.ifft2(cross).real
    b, h, w = correlation.shape
    peak, flat = correlation.flatten(1).max(1)
    yy, xx = flat // w, flat % w
    yy = torch.where(yy > h // 2, yy - h, yy)
    xx = torch.where(xx > w // 2, xx - w, xx)
    shifts = torch.stack([yy, xx], 1).float()
    confidence = peak.clamp_min(0) / correlation.abs().sum((-2, -1)).clamp_min(1e-8)
    if max_shift is not None:
        confidence = confidence * (shifts.abs().amax(1) <= max_shift)
    return shifts, confidence


def ssim(pred, target, weights, data_range=1.0, window=7):
    """Uniform-window SSIM, population covariance, full valid footprint only."""
    window = min(window, min(pred.shape[-2:]))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        raise NoValidData("Image too small for SSIM")
    radius = window // 2
    pool = lambda z: F.avg_pool2d(z, window, stride=1, padding=radius)
    mu_x, mu_y = pool(pred), pool(target)
    var_x = (pool(pred.square()) - mu_x.square()).clamp_min(0)
    var_y = (pool(target.square()) - mu_y.square()).clamp_min(0)
    cov = pool(pred * target) - mu_x * mu_y
    c1, c2 = (.01 * data_range) ** 2, (.03 * data_range) ** 2
    val = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))
    return masked_mean(val, footprint_weights(weights, radius))


def image_metrics(pred, target, weights, data_range=1., scale=None, interval=None):
    pred, target, weights = pred.float(), target.float(), weights.float()
    target = torch.where(weights > 0, target, torch.zeros_like(target))
    metrics = {"physical_data_range": data_range, "opensr_correctness": None,
               "opensr_status": "unavailable: no validated OpenSR adapter installed"}
    for b, name in enumerate(BANDS):
        w, error = weights[:, b:b + 1], (pred - target)[:, b:b + 1]
        try:
            mae = float(masked_mean(error.abs(), w))
            mse = float(masked_mean(error.square(), w))
            bias = float(masked_mean(error, w))
            metrics[name] = {"mae": mae, "rmse": math.sqrt(mse), "bias": bias,
                             "psnr": 10 * math.log10(data_range ** 2 / mse) if mse else None,
                             "exact_match": mse == 0, "valid_weight": float(w.sum())}
        except NoValidData:
            metrics[name] = {"status": "no_valid_support"}
    for name, fn in (("sam_radians", lambda: spectral_angle(pred, target, weights, False)),
                     ("ssim", lambda: ssim(pred, target, weights, data_range))):
        try:
            metrics[name] = float(fn())
        except NoValidData:
            metrics[name] = None
    denom_p, denom_y = pred[:, 3] + pred[:, 2], target[:, 3] + target[:, 2]
    ndvi_valid = weights[:, [2, 3]].amin(1) * ((denom_p.abs() > 1e-5) & (denom_y.abs() > 1e-5))
    safe_p = torch.where(denom_p.abs() > 1e-5, denom_p, torch.ones_like(denom_p))
    safe_y = torch.where(denom_y.abs() > 1e-5, denom_y, torch.ones_like(denom_y))
    try:
        metrics["ndvi_mae"] = float(masked_mean(((pred[:, 3] - pred[:, 2]) / safe_p - (target[:, 3] - target[:, 2]) / safe_y).abs(), ndvi_valid))
    except NoValidData:
        metrics["ndvi_mae"] = None
    shifts, score = phase_translation(target, pred, weights)
    metrics["displacement_hr_pixels_dy_dx"] = shifts.tolist()
    metrics["phase_peak_fraction"] = score.tolist()
    if scale is not None:
        metrics["uncertainty"] = uncertainty_metrics(pred, target, weights, scale, interval)
    return metrics


def uncertainty_metrics(pred, target, weights, scale, interval=None):
    error = (pred - target).abs()
    selected = weights > 0
    if not selected.any():
        return {"status": "no_valid_support"}
    scores, errors = scale[selected].float(), error[selected].float()
    order = scores.argsort()
    risk = errors[order].cumsum(0) / torch.arange(1, len(order) + 1, device=error.device)
    result = {"nll": float(masked_mean(error / scale.clamp_min(1e-5) + scale.clamp_min(1e-5).log(), weights)),
              "aurc": float(risk.mean()), "scale_is_calibrated": interval is not None,
              "risk_at_coverage": {str(c): float(risk[max(0, math.ceil(c * len(risk)) - 1)]) for c in (.1, .25, .5, .75, 1.)}}
    if interval is not None:
        lo, hi = interval
        result["coverage"] = float(masked_mean(((target >= lo) & (target <= hi)).float(), weights))
        result["mean_interval_width"] = float(masked_mean(hi - lo, weights))
    return result


def aggregate_scenes(rows, seed=42, bootstrap=1000):
    """Macro means and bootstrap CIs over whole scenes, not independent pixels."""
    rng = np.random.default_rng(seed)
    flattened = {}
    def flatten(obj, prefix=""):
        for key, value in obj.items():
            full = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                yield from flatten(value, full)
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value):
                yield full, float(value)
    for row in rows:
        for key, value in flatten(row["metrics"]):
            flattened.setdefault(key, {}).setdefault(row["scene"], []).append(value)
    out = {}
    for key, scenes in flattened.items():
        values = np.array([np.mean(v) for v in scenes.values()])
        boot = rng.choice(values, (bootstrap, len(values)), replace=True).mean(1)
        out[key] = {"scene_macro_mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(), "scenes": len(values)}
    return out
