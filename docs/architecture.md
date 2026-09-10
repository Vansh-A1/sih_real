# Architecture and interface map

All public model inputs use B2/B3/B4/B8 order. Unless stated otherwise, image-valued outputs are physical reflectance, not standardized features.

| Stage | Code | Shape |
|---|---|---|
| Fixed normalization | `models/network.py: normalize` | B,4,H,W |
| Shared conditioned stem | `models/blocks.py: BandStem` | B,4,16,H,W |
| Two spectral attention layers | `SpectralAttention` | B,4,16,H,W; 4 heads of dimension 4 |
| Flatten | `EvidenceSR.encode_bands` | B,64,H,W |
| Six NAF blocks | `NAFBlock` | B,64,H,W |
| Four spatial Swin blocks | `SwinBlock` | B,64,H,W; windows 8, shifts 0/4 |
| Degradation condition | `DegradationEncoder` | B,128; learned 64 + metadata 64 |
| First subpixel stage | `ResidualDecoder.up1` | 64 → 192 → PixelShuffle2 → 48 at 2H,2W |
| Second subpixel stage | `ResidualDecoder.up2` | 48 → 128 → PixelShuffle2 → 32 at 4H,4W |
| Four signed residual heads | `ResidualDecoder.heads` | B,4,4H,4W |
| Baseline and correction | `physics/` | B,4,4H,4W |

Default heads use one 3×3 convolution per band on 32 shared HR features. Rich heads concatenate 16 upsampled features from the corresponding band, giving 48 input channels. Residual heads are initialized exactly to zero, making the initial *pre-correction* estimate equal to the baseline. Correction can change that initial estimate.

NAF blocks use channel LayerNorm, expansion factor 2, depthwise 3×3 filtering, SimpleGate, globally pooled simplified channel attention, projection, and a second gated feed-forward branch. Their two residual scales start at zero. No BatchNorm or dropout is used.

Swin attention includes learned relative position bias, shifted-window region masks, and exclusion of artificial padded keys. Padded query rows remain numerically finite but are discarded. Small images disable shifts; non-divisible sizes are padded internally and cropped back. No padding becomes a physical observation.

## Structured model result

`EvidenceSR.forward` returns `SRResult`, not an unlabeled tuple.

| Field | Units/grid | Meaning |
|---|---|---|
| `baseline` | reflectance, HR | Mask-aware normalized-adjoint estimate |
| `deterministic_residual` | reflectance, HR | `std * R_norm`, no mean addition |
| `deterministic` | reflectance, HR | Baseline plus deterministic residual |
| `pre_correction` | reflectance, HR | Deterministic plus optional gated stochastic residual |
| `corrected` | reflectance, HR | Delivered point estimate |
| `pre_cycle`, `post_cycle` | reflectance, LR | Signed `D(Y)-X`, evaluated on observation support |
| `coverage` | operator coverage, HR | `DᵀW1`, not confidence |
| `valid` | boolean, HR | Target-footprint validity intersected with positive coverage |
| `correction_log` | weighted squared reflectance error | Each correction stage's measured fidelity |
| `features`, `decoder_features` | internal feature units | LR 64 and HR 32 channels |
| `temporal_support` | weighted support, LR | Auxiliary weights summed; zero is meaningful when enabled but unsupported |
| `uncertainty_scale` | reflectance, HR | Conditional Laplace error scale, initially uncalibrated |
| `ensemble_disagreement` | reflectance, HR | Population standard deviation of corrected decoder predictions |
| `stochastic_residual`, `prior_gate` | reflectance / dimensionless, HR | Separate generated contribution and gate |

Disabled optional fields are `None`, not fabricated zero confidence maps. Cycle arrays at masked observation pixels are not valid error measurements; use the original observation weights.

## Temporal injection

All dates share band stems, spectral attention, NAF and Swin weights. FFT phase correlation estimates a global translation; a learned two-channel dense offset field provides residual local alignment with bilinear sampling. This portable dense warp has explicitly different semantics from DCNv2. Quality is warped conservatively, genuine synthetic changes can be suppressed by the target-date reflectance compatibility rule, and a learned gate refines a stated stability proxy.

Fusion uses a weighted feature average. A bias-free 1×1 projection, support factor and learnable residual strength inject it into the intact target feature path. Missing or wholly invalid auxiliaries return that target feature tensor unchanged. Neither correlation scores nor learned gate values are calibrated reliability.

## Uncertainty and stochastic extension

The V3 point estimate remains the primary analytical decoder. Additional decoder adapters are independently initialized but share the feature backbone; all receive reconstruction supervision. Disagreement is distinct from the Laplace scale. Tiled disagreement is computed after separately blending each corrected member, rather than averaging local variances.

V4 learns a physical high-pass residual autoencoder with two stride-2 reductions and two PixelShuffle expansions. The latent has eight channels on the LR grid. The velocity network conditions on 64 deterministic features, four downsampled deterministic bands, four quality maps and one temporal-support map. Training uses linear Gaussian-to-data interpolants and conditional flow matching. Euler and Heun samplers are implemented; one Euler step is an experimental option, not a quality guarantee.

The preliminary gate uses inference-available condition features. Refinement additionally sees the candidate and its cycle residual. The HR teacher appears only in gate training. Sensor correction follows the gated addition. Global pooling makes strictly local latent validity assumptions unsafe; flow training therefore requires fully reliable reference chips.
