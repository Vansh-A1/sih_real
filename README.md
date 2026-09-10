# S2-EvidenceSR-4X

PyTorch research code for estimating Sentinel-2 **B2, B3, B4, B8** reflectance on a 2.5 m grid from 10 m observations. Input `[B,4,H,W]` produces `[B,4,4H,4W]`, preserving the geographic footprint.

**Implementation status:** V1–V4 code is present. Non-training engineering checks have been run. **No training was started, no trained checkpoints were produced, and no real-data accuracy was measured**, following the user's latest instruction. Optional uncertainty and flow outputs do not establish scientific validity.

## Install

The existing PyTorch installation was preserved. This workspace has a `.venv` that uses its compatible system packages, with Rasterio installed locally.

```bash
cd /data/projectwork/SIH
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/s2sr env
```

On a clean machine, install the appropriate official PyTorch build for your device before installing this package. Do not replace a working CUDA build merely to match the example machine. Exact inspected versions are in `docs/environment.json`; the CPU benchmark records its environment independently.

## Checks that do not train

```bash
.venv/bin/python -m pytest -q
.venv/bin/s2sr check-config --config configs/v1.json
.venv/bin/s2sr benchmark --config configs/v1.json --device cpu --size 128 --repeats 5
```

The default test suite skips every test marked `training`. Backward-gradient tests calculate derivatives but never call an optimizer or change weights. Fixtures used by tests live in temporary directories.

## Model use

```python
import torch
from s2_evidencesr import EvidenceSR, load_config

model = EvidenceSR(load_config("configs/v1.json")).eval()
x = torch.rand(1, 4, 128, 128)  # Synthetic reflectance, not real satellite data.
with torch.no_grad():
    result = model(x)
print(result.corrected.shape)  # [1,4,512,512]
```

This demonstrates the interface of an **untrained** network. For actual inference use a trained checkpoint, verified normalization statistics, and an effective sensor model appropriate to the data.

## Versions

| Configuration | Implemented components | Default output |
|---|---|---|
| `configs/v1.json` | Band stems, spectral attention, six NAF blocks, four Swin blocks, degradation FiLM, residual decoder, sensor correction | Single-date analytical |
| `configs/v2.json` | V1 plus global translation, learned dense local warp, quality/stability gating, residual feature fusion | Temporal analytical; exact no-auxiliary fallback |
| `configs/v3.json` | V2 plus predicted error scale and three shared-backbone decoder predictions | Analytical point estimate; separate scale/disagreement |
| `configs/v4.json` | V3 plus residual autoencoder, conditional velocity network, Euler/Heun sampler and two-stage gate | Analytical; stochastic sampling disabled |

The default V1 contains **654,868 trainable parameters**. The original 10–16M figure was provisional; the specified widths and explicit expansion factors yield a smaller model. Nothing was enlarged merely to meet that estimate.

V1 decomposition: `Y_det = normalized_adjoint(X_phys) + std * R_norm`, followed by configurable sensor correction. A residual never receives an additive band mean.

## Data contract

Use `examples/manifest.template.json` as a schema example, replacing all illustrative paths and metadata. It is not a real dataset. Each paired training sample needs all four HR bands, verified physical units/decoding, registered grids, acquisition dates, scene/group identifiers, and masks. Raster bands must be labeled B2/B3/B4/B8 in that order.

```bash
.venv/bin/s2sr validate-data --manifest /absolute/path/to/your/manifest.json
```

Training, validation, calibration, and test roles remain distinct. Adjacent crops must share a scene/geographic group so leakage is detected. The loader does not silently resample, infer missing NIR, or guess the DN scale. It accepts preassembled four-band `.npy` chips and GeoTIFF/other Rasterio-readable four-band rasters. Raw SAFE/XML ingestion and cross-sensor harmonization are upstream responsibilities, not implemented automatic capabilities.

See `docs/data_contract.md` for mask states, metadata slots, and decoding rules.

## Explicit training commands — not run during implementation

The following workflow is supplied for a later, intentional run. It creates synthetic data and starts training only when you invoke the `train` command.

```bash
.venv/bin/s2sr fixtures --config configs/fixture_demo.json --output artifacts/demo/data --count 40 --size 16
.venv/bin/s2sr validate-data --manifest artifacts/demo/data/manifest.json
.venv/bin/s2sr train --config configs/fixture_demo.json --manifest artifacts/demo/data/manifest.json --run artifacts/demo/run
.venv/bin/s2sr evaluate --checkpoint artifacts/demo/run/last.pt --manifest artifacts/demo/data/manifest.json --split test --output artifacts/demo/test_metrics.json
.venv/bin/s2sr infer --checkpoint artifacts/demo/run/last.pt --manifest artifacts/demo/data/manifest.json --sample phantom_38 --output artifacts/demo/sr.tif
```

An equivalent script requires an explicit switch: `bash scripts/demo.sh --allow-training`. The fixture configuration limits elapsed training time to 1,200 seconds. Neither this script nor its training command was executed here.

The production V1 configuration defaults to CPU to avoid unexpectedly taking over a shared GPU. Set `train.device` to `cuda`, choose `bf16` or `fp16` deliberately, and benchmark the actual workload before a long run. Multi-day training is not started by setup, imports, tests, benchmarks, or configuration validation.

Resume with the **saved resolved configuration** and unchanged manifest:

```bash
.venv/bin/s2sr train --config artifacts/demo/run/config.json --manifest artifacts/demo/data/manifest.json --run artifacts/demo/run --resume artifacts/demo/run/last.pt
```

For real data, set `S2_MANIFEST` to an actual validated manifest and `S2_RUN` to a new output directory, then intentionally invoke `bash scripts/train_real_data.sh`. Sensor assumptions must be reviewed first. No real manifest or real data was available in this workspace.

## Optional stages

Start V2/V3 from the earlier analytical checkpoint with `--initialize`; loaded/new parameter names are recorded. This differs from exact resume.

V4 uses three separate stages, supplied as `configs/v4_autoencoder.json`, `configs/v4_flow.json`, and `configs/v4_gate.json`. Initialize the first from the analytical checkpoint and each subsequent stage from the preceding stage's checkpoint. The analytical backbone is frozen for these stages. Flow training requires an autoencoder whose validation residual reconstruction beats a zero-residual baseline. A trained gate is required for default stochastic inference. These are engineering admission conditions, not real-reference validation.

Generative inference is an explicit Python API: call a loaded V4 model with `mode="generative", seed=42, sampling_steps=8`. `mode="change_detection"` and `mode="disaster"` always use the analytical output. The tiled CLI deliberately exports the analytical product; independent tilewise stochastic sampling is not exposed as a scientifically equivalent mosaic.

## Calibration and inference

```bash
.venv/bin/s2sr calibrate --checkpoint /path/to/v3/last.pt --manifest /path/to/manifest.json --output /path/to/calibration.json --product tiled --unit scene_max --alpha 0.1
.venv/bin/s2sr infer --checkpoint /path/to/v3/last.pt --manifest /path/to/manifest.json --sample actual_sample_id --output /path/to/sr.tif --calibration /path/to/calibration.json
```

Calibration is bound to the exact model weights, correction and tiling configuration. `scene_max` uses each scene's maximum normalized error, separately per band; its finite-sample order can be infinite when too few calibration scenes exist. `pixel` calibration is also available with explicit spatial-dependence limitations. Whole-chip evaluation requires a separate `--product whole_chip` calibration. Coverage is not automatically preserved under geographic distribution shift.

Tiled outputs include a four-band GeoTIFF, native-grid 10 m cycle residual, available error-scale, ensemble-disagreement and temporal-support files, and provenance. `--cog` additionally creates a COG when the GDAL driver is available. Overlapping tile results are blended on disk and the completed mosaic is checked through the sensor operator again.

## Experiments and reports

`s2sr ablations --config configs/v1.json --output artifacts/ablation_plan` writes configurations without training. `--execute --manifest ...` explicitly starts their training runs. Real-data OpenSR correctness and downstream tasks are **unavailable**, not approximated with misleading proxy labels.

Read:

- `docs/architecture.md`: tensor/unit interfaces and module map.
- `docs/design_decisions.md`: mathematical qualifications and implementation choices.
- `docs/data_contract.md`: inputs, masks and manifests.
- `docs/training.md`: stages, losses, checkpoints and reproducibility.
- `docs/model_card.md`: intended use and limitations.
- `docs/verification.md`: checks actually run and checks deliberately not run.

The opt-in training tests are available via `.venv/bin/python -m pytest --run-training -q`; they have **not** been exercised in this implementation session.
