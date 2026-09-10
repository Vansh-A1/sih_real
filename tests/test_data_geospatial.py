import json
import numpy as np
import pytest
import rasterio
from affine import Affine
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.data.fixtures import generate_fixtures
from s2_evidencesr.data.manifest import validate_manifest, read_cube, PairedDataset
from s2_evidencesr.data.normalization import fit_normalization
from s2_evidencesr.inference import infer_tiled


def test_fixture_split_decode_and_geospatial(tmp_path):
    manifest = generate_fixtures(tmp_path / "fixtures", count=8, lr_size=8)
    assert validate_manifest(manifest)["samples"] == 8
    doc = json.loads(manifest.read_text())
    train = PairedDataset(manifest, "train")
    assert fit_normalization(train).provenance.startswith("training_")
    with pytest.raises(ValueError): fit_normalization(PairedDataset(manifest, "test"))
    desc = doc["samples"][0]["lr"]
    # Arbitrarily rotated/sheared affine axes must both scale by 1/4.
    desc["transform"] = [10, 2, 500000, 1, -10, 3000000]
    cfg = Config(); cfg.tiling.core = 6; cfg.tiling.halo = 4; cfg.tiling.overlap = 2
    cfg.correction.mode = "landweber"
    model = EvidenceSR(cfg).eval()
    out = tmp_path / "sr.tif"
    report = infer_tiled(model, desc, manifest.parent, out)
    with rasterio.open(out) as ds:
        assert (ds.height, ds.width, ds.count) == (32, 32, 4)
        assert ds.crs.to_epsg() == 32643
        assert ds.transform == Affine(*desc["transform"]) @ Affine.scale(.25, .25)
        assert ds.transform @ (32, 32) == Affine(*desc["transform"]) @ (8, 8)
        assert np.isnan(ds.read()[:, :8, :8]).all()
        assert np.isfinite(ds.read()[:, 16:, 16:]).all()
    with rasterio.open(report["outputs"]["cycle"]) as ds:
        assert ds.shape == (8, 8)
        assert ds.transform == Affine(*desc["transform"])
    assert report["final_mosaic_cycle_mae"] is not None
    with pytest.raises(FileExistsError): infer_tiled(model, desc, manifest.parent, out)


def test_scale_offset_and_split_leakage(tmp_path):
    manifest = generate_fixtures(tmp_path, 8, 8)
    doc = json.loads(manifest.read_text())
    desc = doc["samples"][1]["lr"]
    original, _ = read_cube(desc, tmp_path)
    desc = dict(desc, units="dn", scale=[.01] * 4, offset=[-.2] * 4)
    decoded, _ = read_cube(desc, tmp_path)
    assert np.allclose(decoded, original * .01 - .2)
    desc["units"] = "reflectance"
    with pytest.raises(ValueError): read_cube(desc, tmp_path)
    doc["samples"][-1]["group"] = doc["samples"][0]["group"]
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="leakage"): validate_manifest(manifest, read_arrays=False)
