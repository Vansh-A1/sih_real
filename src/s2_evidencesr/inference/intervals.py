from pathlib import Path
import numpy as np
from .tiled import product_signature


def export_intervals(model, inference_report, calibration):
    import rasterio
    if calibration["product_signature"] != product_signature(model, model.config.tiling):
        raise ValueError("Calibration does not match this correction/tiling product")
    if "scale" not in inference_report["outputs"]:
        raise ValueError("Model did not produce an uncertainty scale")
    if any(v is None for v in calibration["quantiles"]):
        raise ValueError("Calibration has insufficient independent units for finite intervals; gather more calibration scenes or change alpha")
    path = Path(inference_report["outputs"]["sr"])
    lower, upper = path.with_name(path.stem + "_lower.tif"), path.with_name(path.stem + "_upper.tif")
    if lower.exists() or upper.exists():
        raise FileExistsError("Interval output already exists")
    q = np.asarray(calibration["quantiles"], dtype="float32")[:, None, None]
    with rasterio.open(path) as src, rasterio.open(inference_report["outputs"]["scale"]) as scale, rasterio.open(lower, "w", **src.profile) as lo, rasterio.open(upper, "w", **src.profile) as hi:
        for _, window in src.block_windows(1):
            prediction, width = src.read(window=window), scale.read(window=window) * q
            lo.write(prediction - width, window=window); hi.write(prediction + width, window=window)
        for dst in (lo, hi):
            dst.descriptions = src.descriptions
            dst.update_tags(units="reflectance", calibration_unit=calibration["unit"], alpha=calibration["alpha"], product_signature=calibration["product_signature"])
    return {"lower": str(lower), "upper": str(upper)}
