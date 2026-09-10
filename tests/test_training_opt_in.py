"""These tests deliberately do not run unless --run-training is explicitly supplied."""
import json
import numpy as np
import pytest
import torch
from s2_evidencesr import Config
from s2_evidencesr.data.fixtures import generate_fixtures
from s2_evidencesr.training.engine import train
from s2_evidencesr.training.state import load_checkpoint, model_from_checkpoint

pytestmark = pytest.mark.training


def small_config(steps):
    config = Config()
    config.train.max_steps = steps
    config.train.accumulation = 1
    config.train.warmup_steps = 0
    config.train.validate_every = 1000
    config.train.panel_every = 1000
    config.train.checkpoint_every = 2
    config.train.threads = 2
    return config


def test_training_and_exact_cpu_resume(tmp_path):
    manifest = generate_fixtures(tmp_path / "fixtures", 8, 8)
    train(small_config(4), manifest, tmp_path / "full")
    train(small_config(4), manifest, tmp_path / "resumed", stop_after=2)
    train(small_config(4), manifest, tmp_path / "resumed", resume=tmp_path / "resumed/last.pt")
    a, b = load_checkpoint(tmp_path / "full/last.pt"), load_checkpoint(tmp_path / "resumed/last.pt")
    assert a["step"] == b["step"] == 4
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), key
    for key in a["ema"]:
        assert torch.equal(a["ema"][key], b["ema"][key]), key


def test_controlled_tiny_overfit_measured_reduction(tmp_path):
    manifest = generate_fixtures(tmp_path / "fixtures", 40, 8)
    cfg = small_config(200)
    cfg.train.lr = .001
    cfg.train.time_budget_seconds = 1200
    train(cfg, manifest, tmp_path / "overfit")
    rows = [json.loads(s) for s in (tmp_path / "overfit/train.jsonl").read_text().splitlines()]
    valid = [r for r in rows if "raw" in r]
    assert len(valid) >= 20, "Insufficient completed steps for an overfit comparison"
    first = np.mean([r["raw"]["radiometric"] for r in valid[:10]])
    last = np.mean([r["raw"]["radiometric"] for r in valid[-10:]])
    assert last < .9 * first, f"Expected measured reduction: first={first}, last={last}"
