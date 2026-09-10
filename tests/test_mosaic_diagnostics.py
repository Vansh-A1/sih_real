import json
import numpy as np
import rasterio
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.data.fixtures import generate_fixtures
from s2_evidencesr.inference import infer_tiled


def test_all_ensemble_raster_blocks_written(tmp_path):
    manifest = generate_fixtures(tmp_path / "data", 8, 68)
    doc = json.loads(manifest.read_text())
    cfg = Config(); cfg.version = "v3"
    cfg.architecture.uncertainty = True; cfg.architecture.ensemble_members = 2
    cfg.tiling.core = 36; cfg.tiling.halo = 4; cfg.tiling.overlap = 4
    model = EvidenceSR(cfg).eval()
    report = infer_tiled(model, doc["samples"][1]["lr"], manifest.parent, tmp_path / "sr.tif")
    with rasterio.open(report["outputs"]["ensemble_disagreement"]) as src:
        blocks = list(src.block_windows(1))
        assert len(blocks) == 4
        for _, win in blocks:
            values = src.read(window=win)
            assert np.isfinite(values).all()
            assert np.any(values > 0)
    with rasterio.open(report["outputs"]["scale"]) as src:
        assert np.isfinite(src.read()).all() and np.all(src.read() > 0)
