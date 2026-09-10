"""Explicit four-band data contract; no guessed decoding or silent resampling."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import BANDS


def parse_date(value):
    if not value:
        raise ValueError("Acquisition date is required (ISO 8601)")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def resolve(root, path):
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def read_cube(desc, root, window=None):
    """Return float32 reflectance and [4,H,W] reliability; window uses LR pixels."""
    if tuple(desc.get("bands", [])) != BANDS:
        raise ValueError("Band order must be explicitly B2,B3,B4,B8; automatic reordering is disabled")
    units = desc.get("units")
    if units not in ("reflectance", "dn"):
        raise ValueError("Declare units as reflectance or dn and provide decoding metadata")
    parse_date(desc.get("acquired"))
    path = resolve(root, desc["path"])
    source_scales, source_offsets = None, None
    if path.suffix == ".npy":
        raw = np.load(path, mmap_mode="r", allow_pickle=False)
        if window is not None:
            x, y, w, h = map(int, window)
            raw = raw[:, y:y + h, x:x + w]
        raw = np.array(raw, copy=True)
        weights = np.isfinite(raw).astype("float32")
    else:
        import rasterio
        from rasterio.windows import Window
        with rasterio.open(path) as src:
            if src.count != 4:
                raise ValueError(f"{path}: requires all four bands including measured NIR")
            if tuple(src.descriptions) != BANDS:
                raise ValueError(f"{path}: raster band descriptions must be B2/B3/B4/B8")
            if src.crs is None or str(src.crs) != desc.get("crs"):
                raise ValueError(f"{path}: CRS is absent or disagrees with manifest")
            if not np.allclose(tuple(src.transform)[:6], desc.get("transform", []), rtol=0, atol=1e-8):
                raise ValueError(f"{path}: transform disagrees with manifest")
            if desc.get("shape") != [src.height, src.width]:
                raise ValueError(f"{path}: dimensions disagree with manifest")
            win = Window(*window) if window is not None else None
            raw = src.read(window=win)
            weights = (src.read_masks(window=win) > 0).astype("float32")
            source_scales, source_offsets = list(src.scales), list(src.offsets)
    if raw.ndim != 3 or raw.shape[0] != 4:
        raise ValueError(f"{path}: expected [4,H,W], found {raw.shape}")
    if window is None and desc.get("shape") != list(raw.shape[-2:]):
        raise ValueError(f"{path}: manifest shape is wrong")
    weights *= np.isfinite(raw)
    if "nodata" in desc and desc["nodata"] is not None:
        weights *= raw != desc["nodata"]
    values = raw.astype("float32")
    if units == "reflectance":
        if desc.get("scale") is not None or desc.get("offset") is not None:
            raise ValueError("Already decoded reflectance cannot also declare scale/offset")
    else:
        scale, offset = desc.get("scale"), desc.get("offset")
        if desc.get("decode_from_raster", False):
            scale, offset = source_scales, source_offsets
            if scale == [1.0] * 4 and offset == [0.0] * 4:
                raise ValueError("Raster has default scale/offset; supply verified product decoding explicitly")
        if scale is None or offset is None or len(scale) != 4 or len(offset) != 4:
            raise ValueError("DN input needs four verified scales and offsets")
        if not np.isfinite(scale).all() or not np.isfinite(offset).all() or np.any(np.array(scale) <= 0):
            raise ValueError("Decoding scales must be positive finite values and offsets finite")
        values = values * np.asarray(scale, dtype="float32")[:, None, None] + np.asarray(offset, dtype="float32")[:, None, None]
    for key in ("quality", "cloud", "shadow", "saturation", "reference_confidence"):
        if key not in desc:
            continue
        mask_path = resolve(root, desc[key])
        if mask_path.suffix == ".npy":
            mask = np.load(mask_path, mmap_mode="r", allow_pickle=False)
            if window is not None:
                x, y, w, h = map(int, window)
                mask = mask[..., y:y + h, x:x + w]
            mask = np.asarray(mask)
        else:
            import rasterio
            from rasterio.windows import Window
            with rasterio.open(mask_path) as src:
                if str(src.crs) != desc.get("crs") or not np.allclose(tuple(src.transform)[:6], desc["transform"], atol=1e-8, rtol=0):
                    raise ValueError(f"{key} mask is on a different grid")
                mask = src.read(window=Window(*window) if window else None)
        if mask.ndim == 2:
            mask = mask[None]
        if mask.shape[-2:] != values.shape[-2:] or mask.shape[0] not in (1, 4):
            raise ValueError(f"{key} mask dimensions are incompatible")
        if not np.isfinite(mask).all():
            raise ValueError(f"{key} mask contains nonfinite values")
        if key == "quality":
            if not np.isin(mask, [0, 1, 2]).all():
                raise ValueError("Quality states must be 0 invalid, 1 uncertain, 2 valid")
            weights *= np.choose(mask.astype(int), [0., .25, 1.])
        elif key == "reference_confidence":
            if ((mask < 0) | (mask > 1)).any():
                raise ValueError("Reference confidence must be in [0,1]")
            weights *= mask
        else:
            # Dilate cloud/shadow boundaries by the specified number of pixels.
            bad = torch.from_numpy(np.array(mask > 0, dtype="float32"))[None]
            radius = int(desc.get("mask_dilation", 1)) if key in ("cloud", "shadow") else 0
            if radius < 0:
                raise ValueError("mask_dilation cannot be negative")
            if radius:
                bad = torch.nn.functional.max_pool2d(bad, 2 * radius + 1, stride=1, padding=radius)
            weights *= 1 - bad[0].numpy()
    values = np.where(weights > 0, values, np.nan).astype("float32")
    return values, weights.astype("float32")


def load_manifest(path):
    path = Path(path).resolve()
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != 1 or not isinstance(doc.get("samples"), list):
        raise ValueError("Manifest requires schema_version=1 and samples list")
    return doc, path.parent


def validate_manifest(path, read_arrays=True):
    doc, root = load_manifest(path)
    samples = doc["samples"]
    ids, groups, scenes, footprints = set(), {}, {}, []
    for sample in samples:
        sid = sample["id"]
        if sid in ids:
            raise ValueError(f"Duplicate sample id {sid}")
        ids.add(sid)
        split = sample["split"]
        if split not in ("train", "validation", "calibration", "test"):
            raise ValueError("Unknown data role; keep train/validation/calibration/test separate")
        for mapping, key in ((groups, "group"), (scenes, "scene")):
            value = sample[key]
            if value in mapping and mapping[value] != split:
                raise ValueError(f"Split leakage: {key} {value} occurs in different roles")
            mapping[value] = split
        lr, hr = sample["lr"], sample.get("hr")
        for desc in (lr, hr) if hr else (lr,):
            if tuple(desc.get("bands", [])) != BANDS or desc.get("units") not in ("reflectance", "dn"):
                raise ValueError(f"{sid}: declare B2/B3/B4/B8 and explicit physical units")
            parse_date(desc.get("acquired"))
            if not desc.get("crs") or len(desc.get("transform", [])) != 6:
                raise ValueError(f"{sid}: explicit CRS and six affine coefficients required")
            if len(desc.get("shape", [])) != 2 or min(desc["shape"]) < 1:
                raise ValueError("Invalid raster dimensions")
            if read_arrays:
                read_cube(desc, root)
        from affine import Affine
        if hr:
            if hr["shape"] != [4 * n for n in lr["shape"]] or hr["crs"] != lr["crs"]:
                raise ValueError(f"{sid}: LR/HR dimensions or CRS disagree")
            expected = Affine(*lr["transform"]) @ Affine.scale(.25, .25)
            if not np.allclose(tuple(expected)[:6], hr["transform"], atol=1e-8, rtol=0):
                raise ValueError(f"{sid}: LR/HR footprints or pixel phases disagree")
        footprint = sample.get("footprint")
        if footprint is not None:
            if len(footprint) != 4 or footprint[0] >= footprint[2] or footprint[1] >= footprint[3]:
                raise ValueError("footprint must be [xmin,ymin,xmax,ymax]")
            for other, other_crs, role in footprints:
                if role != split and other_crs == lr["crs"] and max(other[0], footprint[0]) < min(other[2], footprint[2]) and max(other[1], footprint[1]) < min(other[3], footprint[3]):
                    raise ValueError("Geographic footprint overlap across splits")
            footprints.append((footprint, lr["crs"], split))
        for aux in sample.get("auxiliaries", []):
            if aux["shape"] != lr["shape"] or aux["crs"] != lr["crs"] or aux["transform"] != lr["transform"]:
                raise ValueError("Auxiliary dates must already share the target grid; no silent reprojection")
            if read_arrays:
                read_cube(aux, root)
    return {"samples": len(samples), "roles": {role: sum(s["split"] == role for s in samples) for role in ("train", "validation", "calibration", "test")},
            "synthetic": doc.get("synthetic", False), "geographic_footprints_checked": len(footprints)}


def fingerprint(path):
    doc, root = load_manifest(path)
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode())
    paths = set()
    for s in doc["samples"]:
        for d in [s["lr"], *([s["hr"]] if s.get("hr") else []), *s.get("auxiliaries", [])]:
            for k in ("path", "quality", "cloud", "shadow", "saturation", "reference_confidence"):
                if k in d:
                    paths.add(resolve(root, d[k]))
    for p in sorted(paths):
        digest.update(str(p).encode())
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class PairedDataset(Dataset):
    """Paired chips; variable-sized samples require batch_size=1 or equal-size chips."""
    def __init__(self, path, split, cutoff=None):
        self.doc, self.root = load_manifest(path)
        self.samples = [s for s in self.doc["samples"] if s["split"] == split]
        self.split = split
        self.cutoff = parse_date(cutoff) if cutoff else None
        if not self.samples:
            raise ValueError(f"Manifest has no {split} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        s = self.samples[index]
        x, w = read_cube(s["lr"], self.root)
        if not s.get("hr"):
            raise ValueError(f"{s['id']} has no four-band HR target")
        y, valid = read_cube(s["hr"], self.root)
        auxiliary = []
        cutoff = self.cutoff or parse_date(s["lr"]["acquired"])
        for desc in s.get("auxiliaries", []):
            if parse_date(desc["acquired"]) > cutoff:
                continue
            ax, aw = read_cube(desc, self.root)
            auxiliary.append({"image": torch.from_numpy(ax), "weights": torch.from_numpy(aw), "acquired": desc["acquired"]})
        return {"id": s["id"], "scene": s["scene"], "group": s["group"], "x": torch.from_numpy(x), "weights": torch.from_numpy(w),
                "y": torch.from_numpy(y), "valid": torch.from_numpy(valid), "auxiliaries": auxiliary,
                "metadata": s.get("sensor_metadata"), "metadata_available": s.get("metadata_available")}


def collate_samples(samples, device="cpu"):
    keys = ("x", "y", "weights", "valid")
    try:
        batch = {k: torch.stack([s[k] for s in samples]).to(device) for k in keys}
    except RuntimeError as exc:
        raise ValueError("Batch chips must have equal spatial dimensions; use batch_size=1") from exc
    batch["ids"] = [s["id"] for s in samples]
    # Pad date count with explicitly invalid acquisitions, retaining a ragged input contract.
    count = max(len(s["auxiliaries"]) for s in samples)
    batch["auxiliaries"] = []
    for date in range(count):
        ims, masks = [], []
        for s in samples:
            a = s["auxiliaries"][date] if date < len(s["auxiliaries"]) else None
            ims.append(a["image"] if a else torch.zeros_like(s["x"]))
            masks.append(a["weights"] if a else torch.zeros_like(s["weights"]))
        batch["auxiliaries"].append({"image": torch.stack(ims).to(device), "weights": torch.stack(masks).to(device)})
    batch["metadata"], batch["availability"] = None, None
    if any(s["metadata"] is not None for s in samples):
        template = next(s["metadata"] for s in samples if s["metadata"] is not None)
        n = len(template)
        if any(s["metadata"] is not None and s["metadata_available"] is None for s in samples):
            raise ValueError("Sensor metadata must carry availability indicators")
        batch["metadata"] = torch.tensor([s["metadata"] or [0.] * n for s in samples], device=device, dtype=torch.float32)
        batch["availability"] = torch.tensor([s["metadata_available"] or [0.] * n for s in samples], device=device, dtype=torch.float32)
    return batch
