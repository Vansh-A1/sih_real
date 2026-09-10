# Model card — untrained research implementation

**Model:** S2-EvidenceSR-4X v0.1.0. **Weights:** no trained weights delivered. **Target:** B2/B3/B4/B8 reflectance, 10 m → 2.5 m sampling grid.

## Intended research use

Controlled experiments on multispectral image reconstruction, spectral fidelity, repeated observations, approximate measurement consistency, predictive error scale and optional stochastic residuals. Inputs must have verified units, georeferencing and quality masks.

## Evidence available

Engineering tests cover interfaces, exact discrete adjoints, sampling phase, correction descent, missing-data behavior, forward/backward finiteness, raster geometry, temporal plumbing and calibration arithmetic. A CPU forward-only benchmark is available. These do not establish image quality, convergence, calibrated deployment uncertainty or superiority over baselines.

No training on synthetic or real data was performed. No independent real HR comparisons, OpenSR correctness metrics or downstream application evaluations were run. The user explicitly requested code without training.

## Limitations

- A finer grid does not establish true 2.5 m resolving power.
- The deterministic network also learns priors and may invent incorrect fine structures.
- Sensor consistency cannot identify details in the operator's nullspace.
- Default PSFs/phases are synthetic engineering assumptions requiring calibration.
- Temporal differences can reflect real change; current gates need training and independent evaluation.
- NIR fidelity cannot be inferred from attractive RGB rendering.
- Error scale, ensemble variance and temporal support are distinct diagnostics, not interchangeable confidence probabilities.
- Shared-backbone ensembles can be jointly wrong.
- Calibration assumptions may fail under spatial dependence or geographic shift.
- Residual flow and its gates are experimental; one-step sampling is not guaranteed adequate.
- Raw SAFE parsing, automatic harmonization, a validated OpenSR adapter and downstream task models are outside the implemented initial scope.

## Deployment restrictions represented in code

Analytical output is the default. Change-detection/disaster modes do not run stochastic generation. Untrained flow/gate use fails clearly. No sample is selected for visual sharpness. Exports preserve the geographic footprint and label the 10 m cycle diagnostic accurately.

Before any substantive application, train on valid data, benchmark the actual hardware, evaluate geographically independent references and intended downstream tasks, and document the resulting limitations. Do not use this untrained implementation as evidence of individual buildings, damage, crop rows or other fine-scale events.
