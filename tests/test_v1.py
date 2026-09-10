from dataclasses import replace
import numpy as np
import pytest
import torch
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.config import Sensor, Correction, config_from_dict
from s2_evidencesr.physics import SensorOperator, SensorCorrector
from s2_evidencesr.models.blocks import SpectralAttention, SwinBlock
from s2_evidencesr.losses import reconstruction_loss, NoValidData


@pytest.mark.parametrize("shape", [(1, 1), (5, 7), (9, 11), (128, 128)])
def test_shapes_baseline_and_units(shape):
    cfg = Config(); cfg.correction.mode = "none"
    model = EvidenceSR(cfg).eval()
    x = torch.rand(1, 4, *shape)
    with torch.no_grad():
        result = model(x)
    assert result.corrected.shape == (1, 4, 4 * shape[0], 4 * shape[1])
    assert torch.equal(result.deterministic, result.baseline)
    assert torch.equal(result.deterministic_residual, torch.zeros_like(result.deterministic_residual))
    assert torch.equal(model.residual_to_physical(torch.zeros_like(x)), torch.zeros_like(x))
    assert result.uncertainty_scale is None and result.temporal_support is None and result.stochastic_residual is None
    assert model.decoder.up1[0].out_channels == 192
    assert model.decoder.up2[0].out_channels == 128


def test_attention_axis_is_only_bands():
    block = SpectralAttention().eval()
    x = torch.randn(2, 4, 16, 3, 5)
    out, weights = block(x, True)
    assert weights.shape == (2 * 3 * 5, 4, 4, 4)
    perturbed = x.clone(); perturbed[..., 0, 0] += torch.randn_like(perturbed[..., 0, 0])
    changed, _ = block(perturbed, True)  # Match the attention kernel for exact locality comparison.
    assert torch.equal(out[..., 1:, 1:], changed[..., 1:, 1:])
    assert block.attn.head_dim == 4


@pytest.mark.parametrize("phase", [[0, 0], [1, 2], [3, 3], [1.5, 1.5], [3.75, .25]])
def test_adjoint_and_autograd_transpose(phase):
    op = SensorOperator(Sensor(phase=phase)).double()
    y = torch.randn(2, 4, 36, 44, dtype=torch.float64, requires_grad=True)
    x = torch.randn(2, 4, 9, 11, dtype=torch.float64)
    a = (op(y) * x).sum()
    b = (y * op.adjoint(x)).sum()
    assert torch.allclose(a, b, atol=1e-10, rtol=1e-10)
    assert torch.allclose(torch.autograd.grad(a, y)[0], op.adjoint(x), atol=1e-12, rtol=1e-12)


def test_impulse_phase_and_coverage():
    op = SensorOperator(Sensor(kernel_size=1, phase=[2, 1]))
    y = torch.zeros(1, 4, 12, 16); y[..., 6, 9] = 1
    assert op(y)[..., 1, 2].eq(1).all()
    base, coverage = op.baseline(torch.ones(1, 4, 3, 4))
    assert coverage.eq(0).any()  # Sparse sampler has unsupported HR pixels.
    assert base[coverage == 0].eq(0).all()


def test_normalized_adjoint_constant_not_inverse():
    op = SensorOperator(Sensor())
    x = torch.ones(1, 4, 8, 8) * .4
    base, coverage = op.baseline(x)
    assert coverage.gt(0).all()
    assert torch.allclose(base, torch.full_like(base, .4), atol=1e-6)
    random = torch.rand_like(x)
    assert not torch.allclose(op(op.baseline(random)[0]), random, atol=1e-3)
    w = torch.zeros_like(x)
    base, cov = op.baseline(torch.full_like(x, float("nan")), w)
    assert base.isfinite().all() and cov.eq(0).all()


def test_masked_landweber_descent_and_zero_residual():
    op = SensorOperator(Sensor())
    correct = SensorCorrector(Correction(mode="landweber", iterations=3))
    y = torch.rand(1, 4, 36, 44)
    x = torch.rand(1, 4, 9, 11)
    w = torch.rand_like(x); w[..., :2, :3] = 0
    base, _ = op.baseline(x, w)
    out, logs = correct(y, x, base, op, w)
    errors = [l["weighted_squared_error"] for l in logs]
    assert all(a >= b for a, b in zip(errors, errors[1:]))
    fixed, _ = correct(y, op(y), base, op, w)
    assert torch.equal(fixed, y)
    assert torch.isfinite(out).all()


def test_swin_shift_and_padding_finite_backward():
    block = SwinBlock(window=8, shift=4)
    x = torch.randn(1, 64, 9, 13, requires_grad=True)
    out = block(x)
    out.square().mean().backward()
    assert out.shape == x.shape and torch.isfinite(x.grad).all()
    assert block.relative_bias.grad is not None


def test_losses_and_finite_gradients_without_optimizer():
    model = EvidenceSR(Config())
    y = torch.rand(1, 4, 32, 32)
    x = model.sensor(y).detach()
    valid = torch.ones_like(y); weights = torch.ones_like(x)
    valid[..., :4, :4] = 0; y[..., :4, :4] = float("nan")
    result = model(x, weights)
    loss, log = reconstruction_loss(result, y, valid, x, weights, model.sensor, model.config.loss)
    assert torch.isfinite(loss)
    loss.backward()  # Gradients only: no optimizer or parameter update.
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert "cycle" in log["raw"]
    with pytest.raises(NoValidData):
        reconstruction_loss(result, y, valid * 0, x, weights * 0, model.sensor, model.config.loss)


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError): config_from_dict({"unknown": 1})
    with pytest.raises(ValueError): config_from_dict({"sensor": {"phase": [4., 1]}})
    with pytest.raises(ValueError): config_from_dict({"architecture": {"mode": "generative"}})
