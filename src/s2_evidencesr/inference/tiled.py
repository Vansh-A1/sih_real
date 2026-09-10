"""Windowed reads and disk-backed overlap accumulation; bounded working RAM."""
from dataclasses import asdict
from pathlib import Path
import json
import tempfile
import numpy as np
import torch
from ..config import BANDS
from ..data.manifest import read_cube


def starts(length, core, overlap):
    stride = core - overlap
    return range(0, length, stride)


def blend_weights(h, w):
    # Strictly positive endpoints prevent holes at exterior boundaries.
    yy = np.sin(np.pi * (np.arange(h) + .5) / h) ** 2
    xx = np.sin(np.pi * (np.arange(w) + .5) / w) ** 2
    return np.maximum(yy[:, None] * xx[None], 1e-3).astype("float32")


def model_fingerprint(model):
    import hashlib
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def product_signature(model, tiling=None):
    import hashlib
    payload = {"correction": asdict(model.config.correction), "sensor": asdict(model.config.sensor),
               "tiling": asdict(tiling) if tiling is not None else None, "mode": "analytical", "scale_blend": "linear_weighted_scale", "model_fingerprint": model_fingerprint(model)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@torch.no_grad()
def infer_tiled(model, descriptor, root, output, tiling=None, device="cpu", auxiliaries=None, metadata=None, availability=None,
                cutoff=None):
    """Export physical GeoTIFF, native 10 m cycle diagnostic, available scales.

    This routine evaluates analytical output only. Memory is O(tile²) plus bounded
    output stripe; disk accumulator scales with scene size. Separate per-band nodata.
    """
    import rasterio
    from affine import Affine
    from rasterio.windows import Window
    from ..data.manifest import parse_date
    cfg = tiling or model.config.tiling
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    h, w = descriptor["shape"]
    transform = Affine(*descriptor["transform"])
    out_transform = transform @ Affine.scale(.25, .25)
    if not descriptor.get("crs"):
        raise ValueError("Output requires an explicit CRS")
    auxiliary_descriptors = []
    for aux in auxiliaries or []:
        if aux["shape"] != descriptor["shape"] or aux["crs"] != descriptor["crs"] or aux["transform"] != descriptor["transform"]:
            raise ValueError("Auxiliary inputs must be registered to the target grid")
        if parse_date(aux["acquired"]) <= parse_date(cutoff or descriptor["acquired"]):
            auxiliary_descriptors.append(aux)
    if auxiliary_descriptors and model.temporal is None:
        raise ValueError("Temporal inputs require a temporal model")
    paths = {"sr": str(output), "cycle": str(output.with_name(output.stem + "_cycle_10m.tif"))}
    if model.uncertainty is not None:
        paths["scale"] = str(output.with_name(output.stem + "_scale.tif"))
    if model.uncertainty is not None and model.config.architecture.ensemble_members > 1:
        paths["ensemble_disagreement"] = str(output.with_name(output.stem + "_ensemble_disagreement.tif"))
    if model.temporal is not None:
        paths["temporal_support"] = str(output.with_name(output.stem + "_temporal_support.tif"))
    for p in paths.values():
        if Path(p).exists():
            raise FileExistsError(p)
    profile = dict(driver="GTiff", height=h * 4, width=w * 4, count=4, dtype="float32", crs=descriptor["crs"],
                   transform=out_transform, nodata=float("nan"), tiled=True, blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
    # Memmaps do not allocate a full scene-sized RAM tensor.
    with tempfile.TemporaryDirectory(prefix="s2sr-mosaic-", dir=output.parent) as scratch:
        shape = (4, h * 4, w * 4)
        accum = np.memmap(Path(scratch) / "sum.dat", mode="w+", dtype="float32", shape=shape)
        weight_sum = np.memmap(Path(scratch) / "weight.dat", mode="w+", dtype="float32", shape=shape)
        scale_sum = np.memmap(Path(scratch) / "scale.dat", mode="w+", dtype="float32", shape=shape) if "scale" in paths else None
        support_sum = np.memmap(Path(scratch) / "support.dat", mode="w+", dtype="float32", shape=shape) if "temporal_support" in paths else None
        member_sum = np.memmap(Path(scratch) / "members.dat", mode="w+", dtype="float32", shape=(model.config.architecture.ensemble_members, *shape)) if "ensemble_disagreement" in paths else None
        tile_count, tile_cycle = 0, []
        for y in starts(h, cfg.core, cfg.overlap):
            for x in starts(w, cfg.core, cfg.overlap):
                ch, cw = min(cfg.core, h - y), min(cfg.core, w - x)
                x0, y0 = max(0, x - cfg.halo), max(0, y - cfg.halo)
                x1, y1 = min(w, x + cw + cfg.halo), min(h, y + ch + cfg.halo)
                window = (x0, y0, x1 - x0, y1 - y0)
                values, mask = read_cube(descriptor, root, window)
                auxiliary = []
                for desc in auxiliary_descriptors:
                    ax, aw = read_cube(desc, root, window)
                    auxiliary.append({"image": torch.from_numpy(ax)[None].to(device), "weights": torch.from_numpy(aw)[None].to(device)})
                tx, tw = torch.from_numpy(values)[None].to(device), torch.from_numpy(mask)[None].to(device)
                md = torch.tensor(metadata, device=device, dtype=torch.float32)[None] if metadata is not None else None
                av = torch.tensor(availability, device=device, dtype=torch.float32)[None] if availability is not None else None
                result = model(tx, tw, md, av, auxiliary, mode="analytical")
                cy, cx = 4 * (y - y0), 4 * (x - x0)
                crop = (..., slice(cy, cy + ch * 4), slice(cx, cx + cw * 4))
                values = result.corrected[0][crop].cpu().numpy()
                valid = result.valid[0][crop].cpu().numpy()
                weight = blend_weights(ch * 4, cw * 4)[None] * valid
                target = (slice(None), slice(y * 4, (y + ch) * 4), slice(x * 4, (x + cw) * 4))
                accum[target] += np.where(valid, values, 0) * weight
                weight_sum[target] += weight
                if scale_sum is not None:
                    scale_sum[target] += result.uncertainty_scale[0][crop].cpu().numpy() * weight
                if support_sum is not None:
                    up_support = torch.nn.functional.interpolate(result.temporal_support.float(), scale_factor=4, mode="nearest")
                    support_sum[target] += up_support[0][crop].cpu().numpy() * weight
                if member_sum is not None:
                    for member_id, member in enumerate(result.ensemble_predictions):
                        member_sum[(member_id, *target)] += member[0][crop].cpu().numpy() * weight
                tile_count += 1
                tile_cycle.append(result.correction_log[-1])
        writers = {"sr": rasterio.open(output, "w", **profile)}
        for key in ("scale", "temporal_support", "ensemble_disagreement"):
            if key in paths:
                writers[key] = rasterio.open(paths[key], "w", **profile)
        try:
            for yy in range(0, h * 4, 256):
                for xx in range(0, w * 4, 256):
                    hh, ww = min(256, h * 4 - yy), min(256, w * 4 - xx)
                    region = (slice(None), slice(yy, yy + hh), slice(xx, xx + ww))
                    weight = np.asarray(weight_sum[region])
                    for key, source in (("sr", accum), ("scale", scale_sum), ("temporal_support", support_sum)):
                        if source is not None:
                            values = np.divide(source[region], weight, out=np.full(weight.shape, np.nan, dtype="float32"), where=weight > 0)
                            writers[key].write(values, window=Window(xx, yy, ww, hh))
                    if member_sum is not None:
                        members = np.divide(member_sum[(slice(None), *region)], weight[None], out=np.full((model.config.architecture.ensemble_members, *weight.shape), np.nan, dtype="float32"), where=weight[None] > 0)
                        writers["ensemble_disagreement"].write(members.std(axis=0).astype("float32"), window=Window(xx, yy, ww, hh))
            for key, writer in writers.items():
                writer.descriptions = BANDS
                writer.update_tags(units="reflectance" if key != "temporal_support" else "weighted auxiliary support", product=key,
                                   model_version=model.config.version, estimated_not_observed="true", normalization=model.config.normalization.provenance,
                                   uncertainty_status="uncalibrated" if key == "scale" else "not_applicable",
                                   product_signature=product_signature(model, cfg))
        finally:
            for writer in writers.values():
                writer.close()
        del accum, weight_sum, scale_sum, support_sum, member_sum
    # Re-evaluate the delivered mosaic through D in windows with sufficient PSF halo.
    cycle_profile = dict(profile, height=h, width=w, transform=transform)
    total_error, total_weight = 0., 0.
    halo = math_ceil_div(model.sensor.radius + 4, 4)
    with rasterio.open(output) as sr, rasterio.open(paths["cycle"], "w", **cycle_profile) as cycle:
        for y in range(0, h, cfg.core):
            for x in range(0, w, cfg.core):
                ch, cw = min(cfg.core, h - y), min(cfg.core, w - x)
                x0, y0 = max(0, x - halo), max(0, y - halo)
                x1, y1 = min(w, x + cw + halo), min(h, y + ch + halo)
                hr = sr.read(window=Window(x0 * 4, y0 * 4, (x1 - x0) * 4, (y1 - y0) * 4))
                lr, weights = read_cube(descriptor, root, (x0, y0, x1 - x0, y1 - y0))
                yh = torch.from_numpy(np.nan_to_num(hr))[None].to(device)
                good = torch.from_numpy(np.isfinite(hr).astype("float32"))[None].to(device)
                mask = torch.from_numpy(weights)[None].to(device) * model.sensor.observation_weights(good)
                error = model.sensor(yh) - torch.from_numpy(np.nan_to_num(lr))[None].to(device)
                cut = (..., slice(y - y0, y - y0 + ch), slice(x - x0, x - x0 + cw))
                error, mask = error[cut], mask[cut]
                total_error += float((error.abs() * mask).sum()); total_weight += float(mask.sum())
                data = torch.where(mask > 0, error, float("nan"))[0].cpu().numpy()
                cycle.write(data, window=Window(x, y, cw, ch))
        cycle.descriptions = BANDS
        cycle.update_tags(units="reflectance", grid_resolution="native_10m", diagnostic="D(final_mosaic)-X")
    report = {"outputs": paths, "tiles": tile_count, "final_mosaic_cycle_mae": total_error / total_weight if total_weight else None,
              "cycle_valid_weight": total_weight, "tile_correction_diagnostics": tile_cycle, "tiling": asdict(cfg),
              "product_signature": product_signature(model, cfg), "footprint_preserved": True,
              "note": "Finite-halo tiles need not equal full-scene inference because SCA and degradation conditioning pool globally."}
    if cfg.cog:
        from rasterio.shutil import copy
        with rasterio.Env() as env:
            if "COG" not in env.drivers():
                raise RuntimeError("GDAL COG driver unavailable; GeoTIFF outputs were retained")
        cog = output.with_name(output.stem + "_cog.tif")
        if cog.exists():
            raise FileExistsError(cog)
        copy(output, cog, driver="COG", compress="DEFLATE")
        report["outputs"]["cog"] = str(cog)
    output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def math_ceil_div(a, b):
    return (a + b - 1) // b
