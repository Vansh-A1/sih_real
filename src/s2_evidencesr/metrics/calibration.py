"""Separate-split finite-sample conformal calibration with declared units."""
from pathlib import Path
import json
import math
import tempfile
import numpy as np
import torch
from ..config import BANDS


def finite_quantile(scores, alpha):
    """Order ceil((n+1)(1-alpha)); infinity if that order exceeds n."""
    scores = np.asarray(scores, dtype="float64")
    if not 0 < alpha < 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs finite scores and 0 < alpha < 1")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    return float(np.partition(scores, rank - 1)[rank - 1]) if rank <= len(scores) else None


def intervals(prediction, scale, calibration):
    if scale is None:
        raise ValueError("This model has no uncertainty scale")
    values = calibration["quantiles"]
    q = prediction.new_tensor([float("inf") if v is None else v for v in values]).view(1, 4, 1, 1)
    return prediction - q * scale, prediction + q * scale


@torch.no_grad()
def calibrate(model, dataset, output, alpha=.1, unit="scene_max", product="tiled", device="cpu"):
    from ..data.manifest import read_cube, collate_samples
    from ..inference.tiled import infer_tiled, product_signature
    from ..training.engine import forward_batch
    if dataset.split != "calibration":
        raise ValueError("Conformal calibration requires a separate calibration split")
    if model.uncertainty is None:
        raise ValueError("Calibration requires a V3/V4 uncertainty head")
    if unit not in ("scene_max", "pixel") or product not in ("tiled", "whole_chip"):
        raise ValueError("Calibration unit is scene_max or pixel; product is tiled or whole_chip")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    model = model.to(device).eval()
    grouped = [{}, {}, {}, {}]
    for i, desc in enumerate(dataset.samples):
        sample = dataset[i]
        if product == "tiled":
            import rasterio
            with tempfile.TemporaryDirectory(prefix="s2sr-calibration-") as temp:
                result = infer_tiled(model, desc["lr"], dataset.root, Path(temp) / "sr.tif", device=device,
                                     auxiliaries=desc.get("auxiliaries", []) if model.temporal is not None else [],
                                     metadata=desc.get("sensor_metadata"), availability=desc.get("metadata_available"),
                                     cutoff=str(dataset.cutoff) if dataset.cutoff else None)
                with rasterio.open(result["outputs"]["sr"]) as src:
                    pred = src.read()
                with rasterio.open(result["outputs"]["scale"]) as src:
                    scale = src.read()
        else:
            batch = collate_samples([sample], device)
            result = forward_batch(model, batch)
            pred = result.corrected[0].cpu().numpy(); scale = result.uncertainty_scale[0].cpu().numpy()
        target, valid = sample["y"].numpy(), sample["valid"].numpy()
        for b in range(4):
            good = (valid[b] == 1) & np.isfinite(pred[b]) & np.isfinite(target[b]) & np.isfinite(scale[b])
            score = np.abs(pred[b][good] - target[b][good]) / np.maximum(scale[b][good], 1e-5)
            if not len(score):
                continue
            grouped[b].setdefault(sample["scene"], []).append(float(score.max()) if unit == "scene_max" else score)
    quantiles, counts = [], []
    for groups in grouped:
        scores = [max(v) for v in groups.values()] if unit == "scene_max" else np.concatenate([a for v in groups.values() for a in v]) if groups else []
        quantiles.append(finite_quantile(scores, alpha)); counts.append(len(scores))
    report = {"alpha": alpha, "unit": unit, "grouping": "scene", "bands": list(BANDS), "quantiles": quantiles, "score_counts": counts,
              "infinite_quantiles_encoded_as_null": True, "product": product,
              "product_signature": product_signature(model, model.config.tiling if product == "tiled" else None),
              "coverage_target": "marginal per-band across scenes, simultaneous within each scene" if unit == "scene_max" else "marginal per-band pixel; spatial exchangeability not established",
              "assumptions": "Exchangeable calibration/deployment units. Spatial dependence and geographic shift can invalidate coverage.",
              "synthetic": dataset.doc.get("synthetic", False)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
