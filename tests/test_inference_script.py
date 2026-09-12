"""Inference-only checks; no optimizer steps or training checkpoints are created."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import torch
from affine import Affine
from s2_evidencesr import Config, EvidenceSR


spec = importlib.util.spec_from_file_location("inference_script", Path(__file__).resolve().parents[1] / "inference.py")
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def test_explicit_band_order_scaling_and_nodata(tmp_path):
    image = np.zeros((5, 7, 4), dtype=np.uint16)
    image[:] = [100, 200, 300, 400]  # Stored R/G/B/NIR.
    image[0, 0, 2] = 9999
    np.save(tmp_path / "lr.npy", image)
    decoded, weights, geo = inference.load_image(
        tmp_path / "lr.npy", (3, 2, 1, 4), 0.001, -0.05, "HWC", 9999)
    assert geo is None
    np.testing.assert_allclose(decoded[:, 1, 1], [.25, .15, .05, .35], atol=1e-7)
    assert np.isnan(decoded[0, 0, 0]) and weights[0, 0, 0] == 0
    with pytest.raises(ValueError, match="only 4"):
        inference.load_image(tmp_path / "lr.npy", (1, 2, 3, 5), 1, 0, "HWC")
    with pytest.raises(ValueError, match="distinct"):
        inference.load_image(tmp_path / "lr.npy", (1, 2, 3, 3), 1, 0, "HWC")


def test_geospatial_export_keeps_footprint_and_numeric_values(tmp_path):
    transform = Affine(10, 2, 500000, 1, -10, 3000000)
    data = np.full((4, 6, 8), .2, dtype=np.float32)
    data[:, 0, 0] = -9999
    source = tmp_path / "lr.tif"
    with rasterio.open(source, "w", driver="GTiff", width=8, height=6, count=4,
                       dtype="float32", crs="EPSG:32643", transform=transform, nodata=-9999) as dst:
        dst.write(data)
    lr, weights, geo = inference.load_image(source, (1, 2, 3, 4), 1, 0)
    assert (weights[:, 0, 0] == 0).all()
    sr = np.repeat(np.repeat(lr, 4, 1), 4, 2)
    sr[:, 8, 8] = 1.8  # Must not clip numeric values for display.
    inference.save_numeric_sr(sr, tmp_path, geo)
    with rasterio.open(tmp_path / "sr.tif") as dst:
        assert dst.crs.to_epsg() == 32643
        assert dst.transform == transform @ Affine.scale(.25, .25)
        assert dst.transform @ (dst.width, dst.height) == transform @ (8, 6)
        assert dst.descriptions == ("B2", "B3", "B4", "B8")
        np.testing.assert_equal(dst.read(), sr)
        hr_geo = {"crs": dst.crs, "transform": dst.transform, "height": dst.height, "width": dst.width}
    inference.check_footprints(geo, hr_geo)
    with pytest.raises(ValueError, match="footprints differ"):
        inference.check_footprints(geo, dict(hr_geo, transform=hr_geo["transform"] @ Affine.translation(1, 0)))


def test_tiled_edges_overlap_and_invalid_pixels_are_preserved():
    class KnownOutput(torch.nn.Module):
        def forward(self, x, weights, metadata, availability, mode):
            assert mode == "analytical" and not torch.is_grad_enabled()
            return SimpleNamespace(
                corrected=torch.nn.functional.interpolate(torch.nan_to_num(x), scale_factor=4, mode="nearest"),
                valid=torch.nn.functional.interpolate(weights, scale_factor=4, mode="nearest") > 0)
    lr = np.arange(4 * 9 * 13, dtype=np.float32).reshape(4, 9, 13) / 1000
    lr[1, 4, 6] = np.nan
    weights = np.isfinite(lr).astype(np.float32)
    sr = inference.predict_sr(KnownOutput(), lr, weights, "cpu", tile_size=6, halo=2, overlap=2)
    expected = np.repeat(np.repeat(lr, 4, 1), 4, 2)
    np.testing.assert_allclose(sr, expected, atol=1e-7, equal_nan=True)
    with pytest.raises(ValueError, match="valid support"):
        inference.predict_sr(KnownOutput(), lr, np.zeros_like(weights), "cpu")


def test_display_uses_shared_rgb_limits():
    rgb = np.stack([np.linspace(0, 1, 20).reshape(4, 5)] * 3)
    limits = inference.display_limits(rgb, (0, 100))
    preview = inference.rgb_preview(rgb, limits)
    darker = inference.rgb_preview(rgb * .5, limits)
    np.testing.assert_allclose(darker[..., :3], preview[..., :3] * .5)
    rgb[:, 0, 0] = np.nan
    assert inference.rgb_preview(rgb, limits)[0, 0, 3] == 0


def test_main_restores_ema_and_saves_both_views_without_training(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")

    config = Config()
    config.architecture.naf_blocks = 1
    config.architecture.swin_blocks = 0
    config.architecture.spectral_layers = 1
    config.correction.mode = "none"
    model = EvidenceSR(config).eval()
    state = model.state_dict()
    bad_raw_state = {key: value.clone() for key, value in state.items()}
    bad_raw_state["mean"].fill_(42)  # Inference must select EMA, not these raw weights.
    checkpoint = tmp_path / "last.pt"
    torch.save({"config": config.to_dict(), "ema": state, "model": bad_raw_state, "step": 10000}, checkpoint)
    lr = np.random.default_rng(7).uniform(.1, .5, (4, 8, 10)).astype(np.float32)
    np.save(tmp_path / "lr.npy", lr)
    hr = np.repeat(np.repeat(lr, 4, 1), 4, 2)
    np.save(tmp_path / "hr.npy", hr)
    for key, value in {"CHECKPOINT_PATH": str(checkpoint), "LR_IMAGE_PATH": str(tmp_path / "lr.npy"),
                       "HR_IMAGE_PATH": str(tmp_path / "hr.npy"), "OUTPUT_DIRECTORY": str(tmp_path / "out"),
                       "DEVICE": "cpu", "SHOW_PLOTS": False}.items():
        monkeypatch.setattr(inference, key, value)
    def no_optimizer(*args, **kwargs):
        raise AssertionError("Inference must never construct an optimizer")
    monkeypatch.setattr(torch.optim.AdamW, "__init__", no_optimizer)
    inference.main()
    output = next((tmp_path / "out").iterdir())
    sr = np.load(output / "sr.npy")
    with torch.no_grad():
        expected = model(torch.from_numpy(lr)[None], mode="analytical").corrected[0].numpy()
    np.testing.assert_allclose(sr, expected, atol=1e-6)
    for name in ("sr.png", "comparison.png"):
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = json.loads((output / "inference.json").read_text())
    assert report["step"] == 10000 and report["ema"] is True
    assert report["reference_used_for_inference"] is False
    assert not (output / "sr.tif").exists()  # No invented CRS for a NumPy input.


def test_display_sequence_sr_then_comparison(tmp_path, monkeypatch):
    import matplotlib
    import matplotlib.pyplot as plt
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "TkAgg")
    shown = []
    monkeypatch.setattr(plt, "show", lambda **kwargs: shown.append(len(plt.gcf().axes)))
    lr = np.linspace(.1, .6, 4 * 5 * 7, dtype=np.float32).reshape(4, 5, 7)
    sr = np.repeat(np.repeat(lr, 4, 1), 4, 2)
    inference.save_and_display(lr, sr[[2, 1, 0]], sr, tmp_path, show=True)
    assert shown == [1, 3]
