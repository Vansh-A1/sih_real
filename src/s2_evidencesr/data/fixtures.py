"""Procedural engineering fixtures. These are never real-reference evaluations."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import torch
from ..config import BANDS, Sensor
from ..physics import SensorOperator


def phantom(kind, size, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype("float32") / max(size - 1, 1)
    if kind == "constant":
        base = np.ones_like(xx) * .35
    elif kind == "gradient":
        base = .15 + .25 * xx + .15 * yy
    elif kind == "impulse":
        base = np.zeros_like(xx) + .15; base[size // 2, size // 2] = 1
    elif kind == "slanted_edge":
        base = .2 + .4 * (xx + .7 * yy > .8)
    else:
        base = .25 + .12 * np.sin((5 + seed % 3) * np.pi * xx) * np.cos(4 * np.pi * yy)
        base += .15 * ((xx > .25) & (xx < .55) & (yy > .2) & (yy < .8))
    gains = np.array([.7, .85, 1., 1.25]) + rng.uniform(-.06, .06, 4)
    out = np.stack([base * gains[b] + .015 * b for b in range(4)]).astype("float32")
    out[3] += .1 * (xx > .65)  # Distinct NIR-only boundary.
    return torch.from_numpy(out)


def write_raster(path, array, transform, crs="EPSG:32643", nodata=float("nan")):
    import rasterio
    from affine import Affine
    with rasterio.open(path, "w", driver="GTiff", count=4, dtype="float32", width=array.shape[-1], height=array.shape[-2],
                       transform=Affine(*transform), crs=crs, nodata=nodata, compress="deflate") as dst:
        dst.write(array.astype("float32"))
        dst.descriptions = BANDS
        dst.update_tags(units="reflectance", synthetic="true")


def generate_fixtures(directory, count=40, lr_size=16, seed=42, sensor_config=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        raise FileExistsError("Fixture manifest already exists; choose a new directory")
    if count < 8 or lr_size < 4:
        raise ValueError("Use at least eight scenes and four LR pixels per axis")
    sensor_config = sensor_config or Sensor()
    operator = SensorOperator(sensor_config)
    doc = {"schema_version": 1, "synthetic": True, "sensor": asdict(sensor_config), "seed": seed, "samples": []}
    kinds = ["constant", "gradient", "impulse", "slanted_edge", "repeated"]
    for i in range(count):
        split = "train" if i < count - 6 else ("validation" if i < count - 4 else "calibration" if i < count - 2 else "test")
        hr = phantom(kinds[i % len(kinds)], 4 * lr_size, seed + i)
        with torch.no_grad():
            lr = operator(hr[None])[0]
        qlr = np.full((4, lr_size, lr_size), 2, dtype="uint8")
        qhr = np.full(hr.shape, 2, dtype="uint8")
        if i % 7 == 0:
            qlr[:, :2, :2] = 0
            qhr[:, :8, :8] = 0
            lr[:, :2, :2] = float("nan")
        np.save(directory / f"lr_{i}.npy", lr.numpy())
        np.save(directory / f"hr_{i}.npy", hr.numpy())
        np.save(directory / f"qlr_{i}.npy", qlr)
        np.save(directory / f"qhr_{i}.npy", qhr)
        origin_x, origin_y = 500000 + i * 10000, 3000000
        def descriptor(name, shape, spacing, acquired):
            return {"path": name, "bands": list(BANDS), "units": "reflectance", "acquired": acquired,
                    "crs": "EPSG:32643", "transform": [spacing, 0, origin_x, 0, -spacing, origin_y], "shape": list(shape)}
        ld = descriptor(f"lr_{i}.npy", lr.shape[-2:], 10, "2026-01-10")
        hd = descriptor(f"hr_{i}.npy", hr.shape[-2:], 2.5, "2026-01-10")
        ld["quality"], hd["quality"] = f"qlr_{i}.npy", f"qhr_{i}.npy"
        from ..models.temporal import warp
        known_shift = torch.tensor([[1.25, -.5]])
        aux = warp(torch.nan_to_num(lr)[None], known_shift)[0]
        aq = (warp(torch.from_numpy((qlr > 0).astype("float32"))[None], known_shift)[0] >= 1 - 1e-6).numpy().astype("uint8") * 2
        np.save(directory / f"qaux_{i}.npy", aq)
        if i % 2:
            aux[:, lr_size // 3:lr_size // 2, lr_size // 3:lr_size // 2] += .4
        np.save(directory / f"aux_{i}.npy", aux.numpy())
        ad = descriptor(f"aux_{i}.npy", aux.shape[-2:], 10, "2026-01-05")
        ad["quality"] = f"qaux_{i}.npy"
        ad["known_synthetic_shift_dy_dx"] = [1.25, -.5]
        doc["samples"].append({"id": f"phantom_{i}", "scene": f"scene_{i}", "group": f"synthetic_region_{i}", "split": split,
                               "kind": kinds[i % 5], "lr": ld, "hr": hd, "auxiliaries": [ad],
                               "footprint": [origin_x, origin_y - 10 * lr_size, origin_x + 10 * lr_size, origin_y],
                               "harmonization": "exact synthetic band correspondence", "registration": "known synthetic grid"})
    first = doc["samples"][0]
    write_raster(directory / "example_lr.tif", np.load(directory / first["lr"]["path"]), first["lr"]["transform"])
    (directory / "manifest.json").write_text(json.dumps(doc, indent=2))
    return directory / "manifest.json"
