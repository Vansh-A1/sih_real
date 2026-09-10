import json
import pytest
import torch
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.config import Sensor, Correction
from s2_evidencesr.physics import SensorOperator, SensorCorrector
from s2_evidencesr.data.fixtures import generate_fixtures
from s2_evidencesr.data.manifest import PairedDataset, collate_samples


def test_fractional_pixel_center_and_asymmetric_kernel_adjoint():
    # A linear ramp sampled at 1.5 HR-center offsets matches native pixel centers.
    op = SensorOperator(Sensor(kernel_size=1, phase=[1.5, 1.5]))
    xx = torch.arange(16).float().view(1, 1, 1, 16).expand(1, 4, 12, 16)
    assert torch.equal(op(xx), torch.tensor([1.5, 5.5, 9.5, 13.5]).view(1, 1, 1, 4).expand(1, 4, 3, 4))
    kernel = torch.rand(4, 5, 5)
    operator = SensorOperator(Sensor(kernel_size=5, kernels=kernel.tolist(), phase=[1.25, 2.75])).double()
    y, x = torch.randn(1, 4, 20, 28, dtype=torch.float64), torch.randn(1, 4, 5, 7, dtype=torch.float64)
    assert torch.allclose((operator(y) * x).sum(), (y * operator.adjoint(x)).sum(), atol=1e-10)


def test_future_dates_excluded_and_ragged_batch(tmp_path):
    manifest = generate_fixtures(tmp_path, 8, 8)
    doc = json.loads(manifest.read_text())
    future = dict(doc["samples"][0]["auxiliaries"][0]); future["acquired"] = "2026-02-01"
    doc["samples"][0]["auxiliaries"].append(future)
    doc["samples"][1]["auxiliaries"] = []
    manifest.write_text(json.dumps(doc))
    dataset = PairedDataset(manifest, "train")
    assert len(dataset[0]["auxiliaries"]) == 1
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["auxiliaries"][0]["weights"][1].eq(0).all()
    assert len(PairedDataset(manifest, "train", cutoff="2025-12-01")[0]["auxiliaries"]) == 0


def test_flow_training_objectives_are_finite_without_optimizer():
    cfg = Config(); cfg.version = "v4"; cfg.architecture.flow = True
    model = EvidenceSR(cfg)
    y = torch.rand(1, 4, 32, 32)
    x = model.sensor(y).detach()
    batch = {"x": x, "y": y, "valid": torch.ones_like(y), "weights": torch.ones_like(x)}
    with torch.no_grad(): result = model(x)
    loss, _ = model.flow.stage_loss("autoencoder", result, batch, model.sensor, model.corrector)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, model.flow.autoencoder.decoder[-1].weight)[0]).all()
    with pytest.raises(RuntimeError): model.flow.stage_loss("flow", result, batch, model.sensor, model.corrector)
    # Test the flow loss wiring using explicit test-only readiness, without claiming training.
    model.flow.autoencoder_ready.fill_(True)
    loss, _ = model.flow.stage_loss("flow", result, batch, model.sensor, model.corrector)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, model.flow.velocity.output.weight)[0]).all()


def test_fourier_blend_real_and_composition_logged():
    op = SensorOperator(Sensor())
    x = torch.rand(1, 4, 8, 9)
    baseline, _ = op.baseline(x)
    y = torch.randn_like(baseline)
    correct = SensorCorrector(Correction(mode="fourier_landweber", output_range=[-.2, 1.2]))
    result, log = correct(y, x, baseline, op, torch.ones_like(x))
    assert result.dtype == torch.float32 and result.isfinite().all()
    assert {s["component"] for s in log} == {"before", "fourier", "landweber_1", "range_safeguard", "delivered"}
    assert result.min() >= -.2 and result.max() <= 1.2


def test_cutoff_keeps_exact_acquisition_time_and_timezone():
    from s2_evidencesr.data.manifest import parse_date
    assert parse_date("2026-01-10T15:00:00+05:30") == parse_date("2026-01-10T09:30:00Z")
    assert parse_date("2026-01-10T10:00:00Z") > parse_date("2026-01-10T09:30:00Z")
