# Training implementation and reproducibility

No training was run in this implementation session. This document describes the supplied code, not measured convergence.

## Analytical objective

The V1 objective is `1.0 L_rad + 0.5 L_cycle + 0.1 L_SAM + 0.1 L_HF`, all evaluated on the pre-correction estimate in FP32 physical units. Raw and weighted terms are logged individually. Charbonnier epsilon is 1e-3 reflectance. SAM excludes near-zero spectral vectors and clamps the cosine away from ±1 for stable gradients. High-frequency loss combines a fixed 3×3 high-pass filter with horizontal/vertical first differences.

V2 optionally adds proxy gate supervision; V3 adds corrected-output Laplace NLL and reconstruction of additional decoder predictions. No GAN, generic RGB feature loss, or adaptive reweighting is enabled.

## State and updates

The loop uses AdamW, warmup/cosine scheduling, gradient accumulation, clipping after unscaling, optional CUDA FP16/BF16 or CPU BF16, and EMA. Scheduler and EMA advance only on successful optimizer updates. Nonfinite gradients and all-invalid data are logged and skipped. Excessive invalid attempts terminate with an actionable error.

Checkpoints contain model and EMA weights, optimizer, scheduler, AMP scaler, counters, epoch, sampling permutation/cursor/generator, Python/NumPy/Torch/CUDA RNG states, resolved config, normalization, sensor assumptions, dataset content hashes, source hash and environment details. Files are written via an atomic replacement of the run's `last.pt`. Existing nonempty run directories require explicit resume. Source/data files and unrelated checkpoints are not overwritten.

A keyboard interruption does not serialize a potentially half-updated optimizer transaction. Resume uses the last fully written checkpoint; work since that checkpoint may be lost. This is preferable to pretending an interrupted optimizer operation is resumable exactly.

`--resume` requires the same resolved configuration and dataset content fingerprint. `--initialize` loads compatible weights for a new stage and reports every new/loaded key. It preserves the earlier normalization so feature units do not change silently. Any change to the sensor model requires a separate, explicitly configured experiment.

## Reproducibility limits

The reference loader runs synchronously without worker prefetch. Its saved cursor makes CPU resume equivalence testable at optimizer-step boundaries. Exact continuation with shuffled prefetched multiworker loaders is not implemented and is not claimed. GPU kernels, backend versions, thread counts and hardware can change floating-point results; random seeds alone do not establish bitwise equivalence across environments. Some CUDA backward operations, including sampling, can be nondeterministic.

The CPU resume-equivalence and 34-training-chip overfit tests are provided in `tests/test_training_opt_in.py`. Both are skipped by default and were not run at the user's request. Their assertions remain meaningful and unweakened; failures on a future authorized run must be investigated.

## Residual stages

1. Train the analytical system and inspect real/fixture reconstruction diagnostics.
2. Train the residual autoencoder on `HP(Y_reference - Y_deterministic)` with reference-confidence masks. Its validation reconstruction must improve over predicting zero residual before `autoencoder_ready` is set.
3. Train the conditional latent velocity field on straight Gaussian/data interpolants. The analytical model and autoencoder are frozen. Fully valid reference chips are required because the encoder contains global pooling.
4. Train the two-stage gate using a soft HR-derived teacher comparing local pre-correction L1 errors for deterministic and ungated candidate estimates. HR is unavailable to the gate at inference.

Completion flags indicate successful engineering stage execution, not scientific validation. Flow quality, sample-count sensitivity, one-step versus multi-step results and real-reference hallucination remain experiments.

## Checkpoint selection

The implementation saves periodic scene-level metric reports and the latest resume checkpoint. It does **not** automatically crown a PSNR-only winner. Scientific promotion requires a separately reviewed multi-objective decision using radiometry, SAM/NDVI, displacement, cycle consistency, independently measured added-detail correctness, uncertainty and downstream utility. OpenSR and downstream adapters are explicitly unavailable in this initial repository.
