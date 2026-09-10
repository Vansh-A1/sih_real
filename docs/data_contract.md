# Data, masks and metadata

## Manifest

The top-level JSON contains `schema_version: 1`, `synthetic: true|false`, and `samples`. Synthetic fixture manifests additionally save the exact sensor configuration. Paths are absolute or relative to the manifest directory.

Each sample has `id`, `scene`, `group`, `split`, an `lr` descriptor, and an `hr` descriptor for supervised use. Roles are `train`, `validation`, `calibration` and `test`. Multiple chips of one scene must retain the same scene/group identifiers. Optional `footprint` is `[xmin,ymin,xmax,ymax]` in the descriptor CRS. Bounds are checked for intersection across splits when the CRS matches; scene and group checks apply regardless.

Each descriptor requires:

- `path`: a four-band `.npy` array or Rasterio-readable raster.
- `bands`: exactly `["B2","B3","B4","B8"]`.
- `shape`: `[height,width]`.
- `units`: `reflectance` or `dn`.
- `crs`: e.g. a verified EPSG CRS string.
- `transform`: six affine coefficients `[a,b,c,d,e,f]` with pixel-corner convention.
- `acquired`: ISO acquisition date/time.

LR and HR share a footprint. HR dimensions are four times LR dimensions. Its affine is `T_lr @ Affine.scale(.25,.25)`, scaling **both** pixel-axis vectors, including rotation/shear. Origin is unchanged; pixel centers follow their respective grids.

DN descriptors explicitly provide four `scale` and `offset` values, where `reflectance = DN * scale + offset`. Alternatively `decode_from_raster: true` uses non-default Rasterio metadata; identity/default metadata is rejected as ambiguous. Already decoded reflectance rejects simultaneous scale/offset fields to prevent double scaling. The template's values are illustrative and must be replaced from the actual product metadata. The code does not parse Sentinel SAFE XML automatically.

## Quality and reference support

Optional masks are `.npy` arrays or geospatial rasters with the same grid:

- `quality`: 0 invalid, 1 uncertain, 2 valid, mapped to weights 0/.25/1.
- `cloud`, `shadow`, `saturation`: positive means contaminated; these locations receive zero weight.
- `reference_confidence`: finite weights in [0,1] for harmonization/registration reliability.
- `mask_dilation`: cloud/shadow boundary expansion radius (one pixel by default).

Masks can be `[H,W]`, `[1,H,W]` or `[4,H,W]`. GeoTIFF masks must agree with CRS and transform. Nodata flags and nonfinite array values invalidate observations. The physical representation preserves valid unclipped reflectance; invalid values remain NaN in decoded data and are replaced by a fixed band-mean neutral value only for model operations. Observation weights remain attached.

HR reconstruction losses use reference confidence. High-pass losses erode confidence over their filter footprint; cycle supervision reduces it across every nonzero sensor-kernel tap. All-invalid loss support raises `NoValidData`, which training logs as a skipped sample group, not a zero-error success.

## Normalization

Only fully valid LR training observations enter a deterministic bounded reservoir. Fixed .1%/99.9% clipping percentiles and the mean/std of clipped samples are fitted once. Values, seed and valid counts are saved in configuration and checkpoint buffers. Validation, calibration and test pixels are never used for fitting. The example statistics shipped in configs are labeled `unfitted_example_only` and are not scientific product statistics.

## Metadata vector

The default eight slots are a caller-declared numeric convention: band MTF descriptors in slots 0–3, platform code in slot 4, sun zenith in slot 5, view zenith in slot 6, and aerosol/quality summary in slot 7. Normalize continuous slots using fixed training conventions and record them with the dataset. A matching availability vector is required. Unknown values have availability zero; they do not become invented physical measurements. Wavelength identity is represented by the band embedding, not fabricated acquisition-specific metadata.

## Temporal inputs

`auxiliaries` is a list of descriptors on the target native grid. Acquisition timestamps later than the target timestamp or an explicit deployment cutoff are excluded (timezone-aware comparison; date-only values mean midnight UTC) before loading into a batch. Ragged date counts are padded with zero-weight observations. There is no automatic reprojection or spectral harmonization.

Array fixtures include band-distinct fields, gradients, impulses, edges, repeated structures, nodata, known fractional temporal translations and artificial changed regions. They are only engineering inputs. The generated `example_lr.tif` is synthetic, not a satellite scene.

## Real-data preparation remaining

Assemble B2/B3/B4/B8 rasters, decode scale/offset from their product metadata, create reference confidence from registration and harmonization evidence, assign geographic holdouts, and document an effective sensor kernel/phase. Do not label RGB-only references as four-band supervision. No real datasets were found or downloaded during implementation.
