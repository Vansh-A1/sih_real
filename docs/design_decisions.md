# Design decisions and mathematical qualifications

## Source priority

The user implementation brief defines the work. The later instruction, **“don't start the training just made the code files,”** overrides its earlier permission to run smoke training. No optimizer steps, training demos, or real-data runs were executed.

The two PDFs are references, not independent execution instructions:

- `/home/projectwork/Downloads/S2_EvidenceSR_4X_Complete_Model_Guide.pdf`
- `/home/projectwork/Downloads/S2_EvidenceSR_4X_Professor_Architecture_Explanation.pdf`

The second PDF is also present under the workspace's `output/pdf/` directory. Existing PDF outputs and their supporting files were preserved.

## Sensor model

`D = S H` is implemented using a finite nonnegative band-specific kernel, grouped zero-padded cross-correlation, and stride-four point selection with two explicit subpixel offsets. The kernel is normalized per band. Defaults are a synthetic 9×9 Gaussian approximation with sigmas 1.3/1.35/1.4/1.5 **HR pixels** and phase `(1.5,1.5)`, aligning native pixel centers with the centers of the 4×4 HR blocks. These are not validated Sentinel PSFs or acquisition geometry.

The transpose inserts observations into precisely the selected lattice and uses `conv_transpose2d` with matching padding. Therefore `<DY,X> = <Y,DᵀX>` under the implemented discrete boundary convention, including asymmetric kernels. Integer and fractional sampling phases are supported. Fractional sampling is a bilinear weighted sum of at most four lattice selections; its exact transpose scatters the same coefficients, including zero-extended boundary taps. Temporal feature warping is a separate operator.

For reliability `W`, the baseline is `Dᵀ(WX) / Dᵀ(W1)`. Coverage is returned. Unsupported positions receive the declared neutral physical value; epsilon is not treated as evidence. Constant observations yield constant baseline values on supported pixels. **This baseline is not an inverse**, and arbitrary `D(U(X)) = X` is not a required test.

Zero extension can depress a constant field's forward response at scene edges. That is a documented boundary assumption. It is not hidden by renormalizing the forward operator independently of its transpose.

## Approximate correction

Four modes are available: none, smooth Fourier anchor, Landweber, and their combination. The Fourier mask is a real radial logistic function of `fftfreq` coordinates. Cutoffs are in cycles per HR pixel (default .125); transition width .025. This preserves conjugate symmetry. FFT boundaries are circular, unlike the finite sensor operator; the complete result is measured again.

The weighted fidelity objective is `0.5 * sum W(DY-X)^2`. Its gradient is `Dᵀ W(DY-X)`, and a stable fixed-step bound uses `||sqrt(W)D||²`. Because weights lie in [0,1], the unweighted norm is an upper bound in principle. Deterministic power iteration estimates that norm; backtracking verifies actual weighted descent before accepting a step. Strict decrease is not required at zero residual or zero gradient. A rejected step leaves the estimate unchanged and is logged.

The Fourier blend, optional range safeguards and subsequent tile blending have no inherited descent guarantee. Their errors are logged separately and the final mosaic is independently checked. A single Landweber step is not an exact projection. Low cycle error is agreement with an approximate operator, not proof of true fine detail.

## Architecture details resolved

- The requested widths and block counts yield 654,868 V1 parameters with explicit expansion factor 2. The guide's provisional 10–16M target was not enforced.
- Minimal band heads are the default. Rich 48-channel heads are configurable.
- The residual's unit conversion multiplies by the band standard deviation only.
- Eight metadata slots and eight availability indicators are explicit; absent values are zeroed with availability zero. No measured MTF values are fabricated.
- Portable local alignment uses a learned dense sampling grid. It is not claimed to reproduce the full modulated deformable-convolution operator.
- Temporal stability starts from a reflectance-difference proxy and learns a refinement. Its real-world change-recognition ability remains untrained and unvalidated.
- Global pooling in NAF SCA and degradation conditioning means a finite halo cannot generally reproduce whole-scene inference exactly.
- The analytical CLI keeps stochastic tile generation out of the default mosaic. Explicit seeded generative inference is exposed through the model API.

## Uncertainty and evidence

The deterministic decoder also encodes a learned prior. Attention weights, support, cycle residuals, learned scales and ensemble spread have different meanings and remain separate.

Calibration is tied to exact model weights and the delivered correction/tiling pipeline. The finite-sample quantile is order `ceil((n+1)(1-alpha))`; if the order exceeds the number of samples the mathematical interval is infinite. Such quantiles are represented as JSON null with an explicit flag, and finite raster interval export fails with an actionable explanation.

Scene-maximum calibration targets simultaneous pixels within a scene, marginally across exchangeable scenes **per band**, not simultaneous coverage across all four bands. Pixel calibration does not establish spatial exchangeability. Geographic shift can invalidate either protocol. The calibration set is separate from training, validation and final testing.

## Implemented scope versus upstream requirements

This code consumes already assembled four-band rasters or array chips. Raw SAFE parsing, automatic cross-sensor harmonization, learned subpixel sensor calibration, a validated OpenSR adapter, and downstream segmentation/classification models are not implemented. Their absence is explicit; no switch falsely claims they exist.

Default augmentation is disabled, avoiding untracked changes to PSF orientation or sampling phase. Geographic split checks detect reused scene/group identifiers and intersecting supplied footprints in the same CRS. Cross-CRS footprint transformation is not silently inferred; use common geographic grouping or normalize footprints upstream.

## Official API references consulted

- PyTorch 2.10 AMP accumulation/unscaling: https://docs.pytorch.org/docs/2.10/notes/amp_examples.html
- PyTorch 2.10 scaled dot-product attention: https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html
- Rasterio windowed raster I/O: https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html
