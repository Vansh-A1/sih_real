# RezX model connection

The HTTP adapter loads the checkpoint supplied in [Vansh-A1/sih_real](https://github.com/Vansh-A1/sih_real/blob/5663e7e14479d6627b8d5e14d7e507302c73f5be/last.pt) and calls the existing `inference.predict_sr` function. Model architecture, weights, inference code, and training code are unchanged. No training is started.

## Start

From the repository root, using the existing compatible Python environment:

```bash
.venv/bin/python -m pip install -r web_api/requirements.txt
.venv/bin/python -m web_api.fetch_checkpoint
cd frontend
npm install
npm run dev:full
```

The frontend is at http://localhost:5173. The API listens on http://127.0.0.1:8000. The launcher reuses an already-running compatible API, or starts it before Vite. Ctrl+C stops the processes the launcher started.

For a fresh Python environment, install the original project following its root README before installing the API dependencies. Preserve the appropriate PyTorch build for the machine.

The supplied checkpoint restores V1 EMA weights at step 10,000, along with its fitted normalization. The pinned checkpoint SHA-256 is:

```text
df4eb605c0b795df9f339faa5c93af9c01329984e7c0d14b6392a4254da2bfa0
```

The root `.gitignore` excludes weights. `fetch_checkpoint.py` makes a clean checkout reproducible, verifies the exact pinned download, and refuses to overwrite a different existing checkpoint.

## Separate processes / production build

```bash
# Repository root
.venv/bin/python -m uvicorn web_api.app:app --host 127.0.0.1 --port 8000

# Another terminal
cd frontend
npm run dev
```

Vite proxies `/api` to the backend. Set `REZX_API_URL` for a different backend address when using Vite. No cross-origin browser calls are needed.

For a built site, run `npm run build` in `frontend`, then start the API. The API serves both `frontend/dist` and `/api` at port 8000. Keep one API worker; the model queue and temporary records are in-process. This is a local application, not an authenticated public multi-user service.

Configuration:

| Variable | Default | Purpose |
| --- | --- | --- |
| `REZX_CHECKPOINT` | Repository root `last.pt` | Existing compatible checkpoint |
| `REZX_DEVICE` | `cpu` | Explicitly select `cuda` or `cuda:0` if desired |
| `REZX_PYTHON` | `.venv/bin/python` | Python executable used by the combined launcher |
| `REZX_API_PORT` | `8000` | Combined launcher API port |
| `REZX_FRONTEND_PORT` | `5173` | Combined launcher frontend port |

## Input contract

Upload a numeric TIFF/GeoTIFF or `.npy` array containing **measured B2, B3, B4, B8** bands. A PNG/JPEG or RGB image lacks the required NIR band and is rejected; an RGBA alpha band cannot substitute for NIR.

The Input settings panel explicitly defines:

- One-based band numbers in B2/B3/B4/B8 order. Default: `1,2,3,4`.
- NumPy layout: `CHW` (default) or `HWC`.
- Decoding: `stored × scale + offset`. Default: scale 1, offset 0 for prepared floating-point training-proxy data. Scalar or four comma-separated values are accepted.

Integer data requires explicit decoding; the API does not guess DN scaling, normalization, NIR, or reference harmonization. Use the same preparation as training. Upload limits are 20 MB, 1024 pixels per spatial dimension, and at most 16 source bands. Source masks and invalid pixels are preserved. There is no HR reference requirement.

“Try a four-band sample” creates a labeled synthetic engineering fixture and passes it through the **real model**. It is not real satellite data or an accuracy benchmark.

## Results and lifecycle

The UI polls actual job state and completed model tiles. Cancel requests stop between forward passes; an in-flight tile may finish first. A single inference executor serializes model calls and removes temporary progress/cancellation hooks afterward.

Download returns a ZIP with `sr.png`, unmodified numeric `sr.npy`, `inference.json`, and `sr.tif` when the input has georeferencing. GeoTIFF outputs retain the CRS and footprint while changing pixel spacing by 4×. PNGs are RGB display views with one original-derived 2–98 percentile stretch shared by both comparison sides. Numeric files are not display-stretched or clipped.

Temporary uploads/results expire after one hour of inactivity and are removed on clean shutdown. The frontend releases images it replaces; active jobs keep their inputs until safe to remove. Records are bounded to eight images. API URLs use random identifiers but do not replace authentication; do not expose the API publicly without deployment-level access controls.

## API and checks

Interactive endpoint documentation: http://127.0.0.1:8000/docs

- `GET /api/health`
- `POST /api/images` (multipart image plus input settings)
- `POST /api/images/sample`
- `GET /api/images/{id}/preview`
- `DELETE /api/images/{id}`
- `POST /api/jobs` with `{"image_id":"..."}`
- `GET /api/jobs/{id}` / `DELETE /api/jobs/{id}`
- `GET /api/jobs/{id}/preview` / `GET /api/jobs/{id}/download`

```bash
.venv/bin/python -m pip install -r web_api/requirements-dev.txt
.venv/bin/python -m pytest web_api/tests/test_api.py -q
cd frontend
npm run build
```

The API tests load the real pinned weights, compare returned arrays with direct inference, verify PNG color mapping, geospatial footprint/nodata, explicit decoding, invalid input cleanup, queue cancellation/retry, and missing-checkpoint responses. These are integration checks, not scientific accuracy validation.
