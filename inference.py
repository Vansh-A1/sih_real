"""Edit the settings below, then run: python inference.py

Run this file inside your S2-EvidenceSR-4X checkout on the computer with last.pt.
Use the Python environment used for training. No command-line arguments or
training are involved. The saved configuration, normalization and EMA weights
are restored from the checkpoint; HR is only a visual reference.

Default decoding expects PREPARED floating-point WorldStrat proxy images in the
same units as training. For original satellite DN files, set the band selections,
scales and offsets to the values used during preparation. Do not normalize an
image by its own maximum before inference. This script does not recreate an
unknown WorldStrat reference-harmonization procedure.
"""

from pathlib import Path


# ======================== EDIT THESE PATHS ========================
# Relative paths are resolved against the folder containing this script.
CHECKPOINT_PATH = "artifacts/worldstrat_v1_cuda_20260911/last.pt"
LR_IMAGE_PATH = "PUT_YOUR_LR_IMAGE_PATH_HERE.tif"
HR_IMAGE_PATH = "PUT_YOUR_HR_IMAGE_PATH_HERE.tif"
OUTPUT_DIRECTORY = "artifacts/inference_results"

# ======================== IMAGE SETTINGS =========================
# Band numbers are ONE-BASED. The model needs four MEASURED bands ordered
# blue, green, red, NIR (B2, B3, B4, B8). Prepared B/G/R/NIR files use (1,2,3,4).
# If your four-band file is stored R/G/B/NIR, use (3,2,1,4).
LR_BAND_INDICES = (1, 2, 3, 4)

# Select HR red, green and blue in that order. Prepared B/G/R/NIR uses (3,2,1).
# For an ordinary RGB image, or a file stored R/G/B/NIR, use (1,2,3).
HR_RGB_BAND_INDICES = (3, 2, 1)

# decoded = stored_value * SCALE + OFFSET. A scalar or one value per selected
# band is accepted. Defaults: already prepared floating-point proxy values.
# Example ONLY for DN decoded that way during training: LR_SCALE = 0.0001.
# HR scaling must match the prepared reference; /255 is only for 8-bit RGB.
LR_SCALE = 1.0
LR_OFFSET = 0.0
HR_SCALE = 1.0
HR_OFFSET = 0.0
LR_NODATA = None  # Optional extra stored nodata value; raster masks are read too.
HR_NODATA = None
LR_NPY_LAYOUT = "CHW"  # .npy only: "CHW" or "HWC"; no layout guessing.
HR_NPY_LAYOUT = "CHW"

# Optional .npy reliability mask, in [0,1], shape HxW or 4xHxW in MODEL band
# order. Use the same cloud/quality preparation as training when applicable.
LR_WEIGHTS_PATH = None

# ======================== RUN / DISPLAY ==========================
DEVICE = "auto"  # "auto", "cpu", "cuda", or "cuda:0"
USE_EMA = True  # Matches this repository's validation checkpoint loader.
TILE_SIZE = 128  # LR pixels per core. Small images use one forward pass.
TILE_HALO = 16
TILE_OVERLAP = 16
SHOW_PLOTS = True  # False on a server; both PNGs are saved either way.
DISPLAY_PERCENTILES = (2, 98)  # One LR-derived RGB stretch shared by all panels.
DISPLAY_GAMMA = 1.0  # Display only; numeric SR output is never stretched.
DISPLAY_MAX_SIDE = 1600  # Bound preview resolution; saved SR retains all pixels.

# Optional sensor metadata: copy the values AND availability flags from the
# training manifest for this LR sample. None represents missing metadata.
SENSOR_METADATA = None
METADATA_AVAILABILITY = None


import json
import sys
import warnings
from datetime import datetime

import numpy as np
import torch

# Prefer the source next to this script, including changes used on the training
# computer, rather than an older installation elsewhere in the environment.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from s2_evidencesr.config import BANDS
from s2_evidencesr.inference.tiled import blend_weights, starts
from s2_evidencesr.training.state import model_from_checkpoint


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_image(path, bands, scale, offset, layout="CHW", nodata=None):
    """Read selected channels and explicit decoding; retain nodata and geography."""
    import rasterio
    from rasterio.errors import NotGeoreferencedWarning

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    if not bands or any(not isinstance(i, int) or i < 1 for i in bands) or len(set(bands)) != len(bands):
        raise ValueError("Band indices must be distinct positive one-based integers.")
    geo = None
    if path.suffix.lower() == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if raw.ndim != 3 or layout not in ("CHW", "HWC"):
            raise ValueError(f"{path}: expected a 3D array with CHW or HWC layout.")
        raw = np.moveaxis(raw, -1, 0) if layout == "HWC" else raw
        if max(bands) > raw.shape[0]:
            raise ValueError(f"{path}: requested band {max(bands)}, but only {raw.shape[0]} exist.")
        raw = np.asarray(raw[np.array(bands) - 1], dtype=np.float32)
        valid = np.isfinite(raw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with rasterio.open(path) as source:
                if max(bands) > source.count:
                    raise ValueError(f"{path}: requested band {max(bands)}, but only {source.count} exist. LR needs measured NIR too.")
                raw = source.read(list(bands)).astype(np.float32)
                valid = (source.read_masks(list(bands)) > 0) & np.isfinite(raw)
                if source.crs is not None:
                    geo = {"crs": source.crs, "transform": source.transform,
                           "height": source.height, "width": source.width}
    if min(raw.shape[-2:]) < 1:
        raise ValueError(f"{path}: image is empty.")
    if nodata is not None:
        valid &= raw != nodata
    factors = []
    for name, value in (("scale", scale), ("offset", offset)):
        factor = np.asarray(value, dtype=np.float32)
        if factor.ndim == 0:
            factor = np.repeat(factor, len(bands))
        if factor.shape != (len(bands),) or not np.isfinite(factor).all():
            raise ValueError(f"{name} must be finite: scalar or {len(bands)} values in selected-band order.")
        if name == "scale" and (factor <= 0).any():
            raise ValueError("Decoding scales must be positive.")
        factors.append(factor[:, None, None])
    decoded = raw * factors[0] + factors[1]
    if not np.isfinite(decoded[valid]).all():
        raise ValueError(f"{path}: decoding overflowed; check scale and offset.")
    if not valid.any():
        raise ValueError(f"{path}: no valid pixels.")
    return np.where(valid, decoded, np.nan).astype(np.float32), valid.astype(np.float32), geo


def check_footprints(lr_geo, hr_geo):
    """Permit different pixel sizes, but require matching footprints if known."""
    if lr_geo is None or hr_geo is None:
        print("No complete georeferencing for this pair; the images must be crops of the same area.")
        return
    def corners(geo):
        w, h = geo["width"], geo["height"]
        return np.array([geo["transform"] @ p for p in ((0, 0), (w, 0), (w, h), (0, h))])
    a = lr_geo["transform"]
    tolerance = max(np.hypot(a.a, a.d), np.hypot(a.b, a.e)) * 1e-3
    if lr_geo["crs"] != hr_geo["crs"] or not np.allclose(corners(lr_geo), corners(hr_geo), rtol=0, atol=tolerance):
        raise ValueError("LR and HR footprints differ. Use crops registered to the same area/CRS before comparison.")


@torch.no_grad()
def predict_sr(model, lr, weights, device, tile_size=128, halo=16, overlap=16,
               metadata=None, availability=None):
    """Analytical SR from LR alone; overlap blending bounds each GPU forward."""
    if lr.ndim != 3 or lr.shape[0] != 4 or weights.shape != lr.shape:
        raise ValueError("LR and weights must have matching [4,H,W] shapes.")
    if tile_size < 1 or halo < 0 or not 0 <= overlap < tile_size:
        raise ValueError("Need TILE_SIZE > 0, TILE_HALO >= 0, and 0 <= TILE_OVERLAP < TILE_SIZE.")
    if not np.isfinite(weights).all() or ((weights < 0) | (weights > 1)).any() or not (weights > 0).any():
        raise ValueError("LR weights need valid support and finite values in [0,1].")
    if not np.isfinite(lr[weights > 0]).all():
        raise ValueError("Nonfinite LR values must have zero reliability weight.")
    if (metadata is None) != (availability is None):
        raise ValueError("Set both SENSOR_METADATA and METADATA_AVAILABILITY, or leave both None.")
    md = av = None
    if metadata is not None:
        md = torch.as_tensor(metadata, dtype=torch.float32, device=device)
        av = torch.as_tensor(availability, dtype=torch.float32, device=device)
        expected = (model.config.architecture.metadata_dim,)
        if md.shape != expected or av.shape != expected or not torch.isfinite(md).all() or not torch.isfinite(av).all() or ((av < 0) | (av > 1)).any():
            raise ValueError(f"Metadata and availability must have shape {expected}; flags must be in [0,1].")
        md, av = md[None], av[None]
    model = model.to(device).eval()
    h, w = lr.shape[-2:]
    accum = np.zeros((4, h * 4, w * 4), dtype=np.float32)
    denominator = np.zeros_like(accum)
    # CPU input/output arrays remain image-sized. Use the package's disk-backed
    # infer_tiled entry point for very large full-scene products.
    ys = [0] if h <= tile_size else list(starts(h, tile_size, overlap))
    xs = [0] if w <= tile_size else list(starts(w, tile_size, overlap))
    count = 0
    for y in ys:
        for x in xs:
            ch, cw = min(tile_size, h - y), min(tile_size, w - x)
            x0, y0 = max(0, x - halo), max(0, y - halo)
            x1, y1 = min(w, x + cw + halo), min(h, y + ch + halo)
            image = torch.from_numpy(np.ascontiguousarray(lr[:, y0:y1, x0:x1]))[None].to(device)
            mask = torch.from_numpy(np.ascontiguousarray(weights[:, y0:y1, x0:x1]))[None].to(device)
            result = model(image, mask, md, av, mode="analytical")
            cy, cx = 4 * (y - y0), 4 * (x - x0)
            crop = (slice(None), slice(cy, cy + ch * 4), slice(cx, cx + cw * 4))
            values = result.corrected[0].cpu().numpy()[crop]
            valid = result.valid[0].cpu().numpy()[crop].astype(bool)
            if not np.isfinite(values[valid]).all():
                raise FloatingPointError("The checkpoint produced nonfinite SR at valid pixels.")
            blend = blend_weights(ch * 4, cw * 4)[None] * valid
            region = (slice(None), slice(y * 4, (y + ch) * 4), slice(x * 4, (x + cw) * 4))
            accum[region] += np.where(valid, values, 0) * blend
            denominator[region] += blend
            count += 1
            print(f"Inference tile {count}/{len(xs) * len(ys)}", flush=True)
            del image, mask, result
    return np.divide(accum, denominator, out=np.full_like(accum, np.nan), where=denominator > 0)


def save_numeric_sr(sr, directory, geo):
    """Save all four bands without display scaling or clipping."""
    import rasterio
    from affine import Affine

    directory = Path(directory)
    np.save(directory / "sr.npy", sr, allow_pickle=False)
    if geo is not None:
        with rasterio.open(directory / "sr.tif", "w", driver="GTiff", count=4,
                           width=sr.shape[2], height=sr.shape[1], dtype="float32",
                           crs=geo["crs"], transform=geo["transform"] @ Affine.scale(0.25, 0.25),
                           nodata=float("nan"), compress="deflate", BIGTIFF="IF_SAFER") as dst:
            dst.write(sr)
            dst.descriptions = BANDS
            dst.update_tags(units="experimental_worldstrat_training_proxy_units",
                            physical_reflectance_accuracy="not_independently_calibrated",
                            product="analytical_corrected_sr", estimated_not_observed="true")


def display_limits(lr_rgb, percentiles=(2, 98)):
    """Fit once to LR so matching values have matching colors in every panel."""
    if len(percentiles) != 2 or not 0 <= percentiles[0] < percentiles[1] <= 100:
        raise ValueError("DISPLAY_PERCENTILES must satisfy 0 <= low < high <= 100.")
    low, high = [], []
    for band in lr_rgb:
        finite = band[np.isfinite(band)]
        if not finite.size:
            raise ValueError("An LR RGB band has no valid pixels to display.")
        a, b = np.percentile(finite, percentiles)
        low.append(a)
        high.append(max(b, a + 1e-6))
    return np.array(low)[:, None, None], np.array(high)[:, None, None]


def rgb_preview(rgb, limits, gamma=1.0, max_side=1600):
    if not np.isfinite(gamma) or gamma <= 0 or max_side < 1:
        raise ValueError("Display gamma and maximum side length must be positive.")
    stride = max(1, int(np.ceil(max(rgb.shape[-2:]) / max_side)))
    rgb = rgb[:, ::stride, ::stride]
    alpha = np.isfinite(rgb).all(0).astype(np.float32)
    low, high = limits
    colors = np.nan_to_num(np.clip((rgb - low) / (high - low), 0, 1)) ** (1 / gamma)
    return np.concatenate((np.moveaxis(colors, 0, -1), alpha[..., None]), axis=-1)


def save_and_display(lr, hr_rgb, sr, directory, show=True):
    import matplotlib
    import matplotlib.pyplot as plt

    limits = display_limits(lr[[2, 1, 0]], DISPLAY_PERCENTILES)
    preview = lambda rgb: rgb_preview(rgb, limits, DISPLAY_GAMMA, DISPLAY_MAX_SIDE)
    interactive = str(matplotlib.get_backend()).lower() not in ("agg", "pdf", "ps", "svg", "pgf", "template")
    if show and not interactive:
        print("No interactive plot backend is active. Open the saved sr.png and comparison.png instead.")
    # First display: SR alone. Closing its window opens the comparison below.
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(preview(sr[[2, 1, 0]]), interpolation="nearest")
    ax.set_title("S2-EvidenceSR-4X — SR output (RGB)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(directory) / "sr.png", dpi=160)
    if show and interactive:
        print("Showing SR first. Close this figure to open LR | HR | SR.")
        plt.show(block=True)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    images = (lr[[2, 1, 0]], hr_rgb, sr[[2, 1, 0]])
    names = ("LR input", "HR reference", "SR output (4×)")
    for ax, rgb, name in zip(axes, images, names):
        ax.imshow(preview(rgb), interpolation="nearest")
        ax.set_title(f"{name}\n{rgb.shape[2]} × {rgb.shape[1]} pixels")
        ax.axis("off")
    fig.suptitle("LR | HR | SR — shared LR-derived RGB display stretch\nExperimental WorldStrat proxy; visual comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(Path(directory) / "comparison.png", dpi=160)
    if show and interactive:
        plt.show(block=True)
    plt.close(fig)


def main():
    checkpoint_path, lr_path, hr_path = map(resolve_path, (CHECKPOINT_PATH, LR_IMAGE_PATH, HR_IMAGE_PATH))
    for label, path in (("CHECKPOINT_PATH", checkpoint_path), ("LR_IMAGE_PATH", lr_path), ("HR_IMAGE_PATH", hr_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Edit {label} at the top of inference.py. File not found: {path}")
    if len(LR_BAND_INDICES) != 4 or len(HR_RGB_BAND_INDICES) != 3:
        raise ValueError("Select exactly four LR bands (B/G/R/NIR) and three HR display bands (R/G/B).")
    if lr_path.suffix.lower() not in (".tif", ".tiff", ".jp2", ".npy"):
        raise ValueError("LR must be a multispectral TIFF/JP2 or NumPy array with measured NIR; an RGB/RGBA photograph is insufficient.")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment. Set DEVICE = 'cpu'.")
    print(f"Loading checkpoint on {device}: {checkpoint_path}")
    try:
        model, checkpoint = model_from_checkpoint(checkpoint_path, device=device, ema=USE_EMA)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Could not restore this checkpoint. Use the same S2-EvidenceSR source/configuration as training and a complete last.pt containing config, model and ema.") from exc
    step = checkpoint.get("step", "unknown")
    torch.set_num_threads(model.config.train.threads)
    print(f"Loaded step {step}; {'EMA' if USE_EMA else 'model'} weights; saved normalization restored.")
    del checkpoint  # Release CPU optimizer/RNG payload; no optimizer is constructed.

    lr, weights, lr_geo = load_image(lr_path, LR_BAND_INDICES, LR_SCALE, LR_OFFSET, LR_NPY_LAYOUT, LR_NODATA)
    if LR_WEIGHTS_PATH is not None:
        extra = np.load(resolve_path(LR_WEIGHTS_PATH), allow_pickle=False)
        if extra.shape not in (lr.shape, lr.shape[-2:], (1, *lr.shape[-2:])) or not np.isfinite(extra).all() or ((extra < 0) | (extra > 1)).any():
            raise ValueError("LR_WEIGHTS_PATH must contain matching HxW, 1xHxW or 4xHxW reliability values in [0,1].")
        weights *= extra
        lr = np.where(weights > 0, lr, np.nan).astype(np.float32)
    hr_rgb, _, hr_geo = load_image(hr_path, HR_RGB_BAND_INDICES, HR_SCALE, HR_OFFSET, HR_NPY_LAYOUT, HR_NODATA)
    check_footprints(lr_geo, hr_geo)
    print(f"LR shape: {lr.shape}; HR RGB shape: {hr_rgb.shape}")
    print(f"Decoded LR range: {np.nanmin(lr):.6g} to {np.nanmax(lr):.6g}. Check this matches training preprocessing.")
    # HR is intentionally absent from predict_sr's inputs.
    sr = predict_sr(model, lr, weights, device, TILE_SIZE, TILE_HALO, TILE_OVERLAP,
                    SENSOR_METADATA, METADATA_AVAILABILITY)
    output = resolve_path(OUTPUT_DIRECTORY) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    save_numeric_sr(sr, output, lr_geo)
    report = {"checkpoint": str(checkpoint_path), "step": step, "ema": USE_EMA,
              "lr": str(lr_path), "hr_display_only": str(hr_path), "sr_shape": list(sr.shape),
              "lr_bands": LR_BAND_INDICES, "hr_rgb_bands": HR_RGB_BAND_INDICES,
              "lr_scale": LR_SCALE, "lr_offset": LR_OFFSET, "hr_scale": HR_SCALE, "hr_offset": HR_OFFSET,
              "lr_weights": str(resolve_path(LR_WEIGHTS_PATH)) if LR_WEIGHTS_PATH else None,
              "metadata": SENSOR_METADATA, "metadata_availability": METADATA_AVAILABILITY,
              "mode": "analytical", "tile_core": TILE_SIZE, "tile_halo": TILE_HALO, "tile_overlap": TILE_OVERLAP,
              "display_percentiles": DISPLAY_PERCENTILES, "display_gamma": DISPLAY_GAMMA,
              "reference_used_for_inference": False, "physical_accuracy_calibrated": False,
              "note": "Experimental WorldStrat harmonized-reference proxy. Finite-halo tiled outputs can differ from whole-image inference."}
    (output / "inference.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SR shape: {sr.shape}. Saving outputs to: {output}")
    save_and_display(lr, hr_rgb, sr, output, show=SHOW_PLOTS)
    print(f"Done. Open {output / 'sr.png'} and {output / 'comparison.png'}")


if __name__ == "__main__":
    main()
