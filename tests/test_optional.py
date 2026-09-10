import numpy as np
import pytest
import torch
from s2_evidencesr import Config, EvidenceSR
from s2_evidencesr.metrics.quality import phase_translation
from s2_evidencesr.models.temporal import warp
from s2_evidencesr.metrics.calibration import finite_quantile, intervals


def test_known_translation_and_local_warp():
    target = torch.rand(1, 4, 32, 32)
    auxiliary = torch.roll(target, (2, -3), (-2, -1))
    shift, score = phase_translation(target, auxiliary)
    assert shift.tolist() == [[-2., 3.]]
    aligned = warp(auxiliary, shift)
    assert torch.allclose(aligned[..., 3:-3, 4:-4], target[..., 3:-3, 4:-4], atol=1e-6)
    offsets = torch.zeros(1, 2, 32, 32, requires_grad=True)
    warp(auxiliary, shift, offsets).square().mean().backward()
    assert torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0


def test_temporal_empty_invalid_and_changed_regions():
    cfg = Config(); cfg.version = "v2"; cfg.architecture.temporal = True
    m = EvidenceSR(cfg).eval()
    x = torch.rand(1, 4, 24, 24)
    with torch.no_grad():
        reference = m(x)
        invalid = m(x, auxiliaries=[{"image": x, "weights": torch.zeros_like(x)}])
    assert torch.equal(reference.corrected, invalid.corrected)
    assert invalid.temporal_support.eq(0).all()
    cfg1 = Config(); single = EvidenceSR(cfg1).eval()
    state = {k: v for k, v in m.state_dict().items() if k in single.state_dict()}
    single.load_state_dict(state)
    with torch.no_grad(): assert torch.equal(reference.corrected, single(x).corrected)
    changed = x.clone(); changed[..., 8:16, 8:16] += 1.
    with torch.no_grad():
        r = m(x, auxiliaries=[{"image": changed, "weights": torch.ones_like(x)}])
    assert r.temporal_support[..., 9:15, 9:15].eq(0).all()
    assert torch.isfinite(r.corrected).all()


def test_uncertainty_is_distinct_and_quantile_rule():
    cfg = Config(); cfg.version = "v3"; cfg.architecture.uncertainty = True; cfg.architecture.ensemble_members = 3
    m = EvidenceSR(cfg).eval()
    with torch.no_grad(): r = m(torch.rand(1, 4, 8, 8))
    assert r.uncertainty_scale.min() > 0
    assert r.ensemble_predictions.shape == (3, 1, 4, 32, 32)
    assert r.ensemble_disagreement.shape == r.corrected.shape
    assert finite_quantile(np.arange(1, 10), .1) == 9
    assert finite_quantile([1, 2], .1) is None
    lo, hi = intervals(r.corrected, r.uncertainty_scale, {"quantiles": [2., 2., 2., 2.]})
    assert torch.allclose(hi - lo, 4 * r.uncertainty_scale)


def test_flow_disabled_zero_gate_and_seeded_sampling():
    default = EvidenceSR(Config())
    assert default.flow is None
    cfg = Config(); cfg.version = "v4"; cfg.architecture.flow = True
    m = EvidenceSR(cfg).eval()
    x = torch.rand(1, 4, 8, 8)
    with torch.no_grad():
        a = m(x, mode="analytical")
        b = m(x, mode="generative", gate_override=0)
        assert torch.equal(a.corrected, b.corrected)
        condition = m.flow.condition(a.features, a.deterministic, torch.ones_like(x), None)
        first = m.flow.sample(condition, seed=5, steps=2)
        second = m.flow.sample(condition, seed=5, steps=2)
        third = m.flow.sample(condition, seed=5, steps=1)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)
    with pytest.raises(RuntimeError, match="training"):
        m(x, mode="generative")
