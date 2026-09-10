# Verification report

## Execution boundary

The user's latest instruction was to create code without starting training. **No optimizer steps, training smoke runs, overfit experiments, real-data runs, or training demonstration scripts were executed.** Two training tests are explicitly skipped by the default test command. No trained checkpoints are delivered.

## Implemented

- Installable package and `s2sr` command.
- Strict V1–V4 JSON configurations, named tensor outputs, fixed normalization and mask handling.
- V1 band-conditioned stems, band-axis spectral attention, real NAF/Swin operations, optional degradation FiLM and correctly sized PixelShuffle decoder.
- Finite band-specific sensor kernels, fractional sampling, exact discrete adjoint, coverage-aware baseline and safeguarded approximate correction.
- Paired array/raster data loading, metadata validation, split/group/footprint checks and labeled procedural fixtures.
- Physical losses, radiometric/spectral/spatial metrics, native-grid cycle diagnostics and explicit unavailable OpenSR status.
- Training/checkpoint/resume source, logging, EMA, diagnostics, and opt-in training tests.
- V2 target-preserving temporal alignment, local warping, quality/stability gates and no-support fallback.
- V3 Laplace scale, additional decoder predictions, finite-sample calibration and raster interval export.
- V4 residual autoencoder, conditional flow objective, Euler/Heun sampling, candidate gate, and staged readiness checks.
- Windowed tiled GeoTIFF/COG path, disk-backed blending, separate optional diagnostics and final-mosaic consistency measurement.

## Checks actually run

| Command/check | Result |
|---|---|
| `.venv/bin/python -m pip install --no-deps --no-build-isolation -e .` | Local editable package installed |
| `.venv/bin/python -m pytest -q --junitxml=docs/test_results.xml` | **32 passed, 2 skipped**, 2.35 seconds |
| `.venv/bin/python -m compileall -q src tests` | All source/test files compiled |
| `s2sr check-config --config configs/v1.json` | Passed; all supplied configs also validated by tests |
| `s2sr fixtures ... --count 8 --size 8` | Passed in a temporary directory |
| `s2sr validate-data --manifest .../manifest.json` | Passed on those temporary fixtures |
| `s2sr ablations --config configs/v1.json --output ...` | Configuration generation passed; no `--execute` used |
| `s2sr env --output docs/environment.json` | Exact inspected dependency/hardware report saved |
| `s2sr benchmark --config configs/v1.json --device cpu --size 128 --repeats 5 --output docs/benchmark_cpu.json` | Forward-only benchmark completed |

Exact temporary CLI arguments and return codes are preserved in `cli_verification.json`. Their temporary outputs were removed. Test details are recorded in `test_results.xml`.

### What the tests establish

Shape checks include standard 128×128 input and small/odd sizes. Band attention uses four tokens and four heads. Locality is tested using the same numerical attention kernel. Residual initialization matches the baseline before correction, and zero residual does not add a band mean.

Float64 adjoint tests cover integer/fractional phases, asymmetric kernels, boundary taps and comparison to automatic differentiation, at approximately 1e-10 inner-product tolerance. A linear ramp verifies phase 1.5 corresponds to native pixel centers. Sparse-kernel unsupported regions remain explicit. Constant-field baseline behavior is checked without imposing an invalid inverse-closure requirement.

Weighted Landweber tests verify non-increasing fidelity, including zero residual. Fourier, range-safeguard and delivered errors are separately logged. Backward tests compute finite derivatives through Swin, model, sensor and residual objectives, **without optimizers**.

Data tests cover scale/offset decoding, prevention of double scaling, training-only normalization, masks, group leakage, future timestamps with time zones, and ragged temporal batches. Raster tests check dimensions, rotated affine footprint, CRS, nodata, edge coverage, native-grid cycle output and writing every ensemble diagnostic block.

Optional-branch checks cover known shifts, differentiable local warps, missing/corrupted auxiliary inputs, explicit synthetic changes, separate scale/disagreement, finite-sample quantile arithmetic, seeded sampling, zero-gate equivalence and rejection of untrained generative use. They test plumbing and mathematics, not learned scene understanding.

## Measured size and resources

| Configuration | Parameters |
|---|---:|
| V1 | 654,868 |
| V2 | 698,648 |
| V3 | 710,212 |
| V4 | 956,096 |

Per-module counts are in `parameter_counts.json`. The guide's provisional 10–16M size was not enforced.

The final V1 CPU benchmark used FP32, a physical batch of one, 128×128 LR input, four CPU threads, two warmup forwards and five measured forwards. Median latency was **0.276745 seconds**. Process peak RSS was **828,220 KiB** (about 809 MiB); this includes Python/PyTorch/runtime memory and is not isolated activation memory. Configured gradient accumulation is four, but no accumulation or training steps were run in the benchmark. Training latency is unavailable.

CUDA was detected on a 16 GB NVIDIA RTX 2000 Ada, but another process was using approximately 99% GPU utilization. CUDA training/AMP/performance tests were not run. The existing PyTorch 2.10.0/CUDA 12.8 installation and other processes were preserved. Rasterio was installed in the project-local environment.

## Status distinctions

| Status | Evidence |
|---|---|
| Implemented | V1–V4 source, data/training/inference paths, configurations, tests and documentation |
| Exercised by engineering tests | 32 non-training checks passed |
| Trained on synthetic fixtures | **No — user requested no training** |
| Trained on real data | **No** |
| Evaluated against independent real references | **No** |
| Training/resume equivalence tested | Source/test supplied; **not executed**, test skipped |
| Tiny overfit/loss reduction measured | **Not executed**, test skipped |
| GPU AMP/throughput verified | **Not run**; CPU checks completed |
| Real data available | **No** relevant dataset found in this workspace |
| OpenSR/downstream evaluation | Explicitly unavailable; no substitute correctness claims |

Full CLI fixture → training → checkpoint → inference demonstration is authored in `scripts/demo.sh`, but not run. Its invocation requires `--allow-training`. Trained seam quality, tile-size sensitivity, calibration coverage and one-step versus multi-step generative accuracy remain data- and training-dependent work.

## Next intentional training action

No training action is currently authorized. When the user chooses to start it, first supply verified data and sensor assumptions. The exact V1 command pattern is:

```bash
.venv/bin/s2sr train --config configs/v1.json --manifest "$S2_MANIFEST" --run "$S2_RUN"
```

`S2_MANIFEST` must be an existing validated real-data manifest and `S2_RUN` a new output directory. The helper `scripts/train_real_data.sh` requires those variables and validates the manifest first. There is no fabricated real-data path or implied real-world result.
