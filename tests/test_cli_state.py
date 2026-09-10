from pathlib import Path
import json
import subprocess
import sys
import pytest
import torch
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.config import load_config
from s2_evidencesr.training.state import rng_state, restore_rng, StatefulSampler, save_checkpoint, load_checkpoint


def test_configs_and_cli_import_without_training():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.json"):
        load_config(path)
    result = subprocess.run([sys.executable, "-m", "s2_evidencesr.cli.main", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "train" in result.stdout and "calibrate" in result.stdout


def test_sampler_rng_and_safe_checkpoint_roundtrip(tmp_path):
    sampler = StatefulSampler(7, 42)
    sampler.take(5)
    state = sampler.state_dict()
    expected = sampler.take(10)
    restored = StatefulSampler(7, 0); restored.load_state_dict(state)
    assert restored.take(10) == expected
    before = rng_state(); noise = torch.randn(9); restore_rng(before)
    assert torch.equal(torch.randn(9), noise)
    model = EvidenceSR(Config())
    p = tmp_path / "untrained_test_state.pt"
    save_checkpoint(p, {"model": model.state_dict(), "rng": before, "sampler": state})
    loaded = load_checkpoint(p)
    assert torch.equal(loaded["model"]["mean"], model.mean)


def test_unknown_inputs_fail_clearly():
    model = EvidenceSR(Config())
    x = torch.rand(1, 4, 8, 8)
    with pytest.raises(ValueError, match="weights"):
        model(x, torch.full_like(x, 2))
    x[..., 0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        model(x, torch.ones_like(x))


def test_calibration_is_bound_to_weights_and_product():
    from s2_evidencesr.inference.tiled import product_signature
    model = EvidenceSR(Config())
    first = product_signature(model)
    with torch.no_grad(): model.decoder.heads[0].bias.add_(.01)
    assert product_signature(model) != first
    assert product_signature(model) != product_signature(model, model.config.tiling)
