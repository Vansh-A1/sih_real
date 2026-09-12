# RezX web experience

React + TypeScript + Three.js frontend connected to the repository's trained S2-EvidenceSR-4X model through the local FastAPI adapter.

## Run both services

```bash
cd /data/projectwork/SIH
.venv/bin/python -m pip install -r web_api/requirements.txt
.venv/bin/python -m web_api.fetch_checkpoint
cd frontend
npm install
npm run dev:full
```

Open http://localhost:5173. The launcher starts the API on port 8000, waits for the checkpoint to load, and starts Vite. It can reuse an already-running compatible API.

For separate terminals, start the API from the repository root with `.venv/bin/python -m uvicorn web_api.app:app --host 127.0.0.1 --port 8000`, then use `npm run dev` here. `npm run build` creates `dist/`; the API serves that built site at port 8000 when started after the build. Set `REZX_API_URL` to change Vite's backend proxy target.

Full setup, endpoint documentation, input contract, and configuration: [backend README](../web_api/README.md).

## Image workflow

- Low-poly Earth and two interactive orbiting satellites lead into a cloud workspace.
- Upload four-band TIFF/GeoTIFF or NumPy data containing measured blue, green, red, and near-infrared bands.
- Input settings explicitly control band numbers, NumPy layout, scale, and offset. Defaults match prepared floating-point B/G/R/NIR inputs. Integer DN inputs require the correct decoding values from training.
- The interface displays real server job progress and supports cancellation between model tiles.
- Compare the original and model-generated RGB views with the keyboard/touch/pointer slider.
- Download a ZIP containing the PNG display, four-band NumPy data, inference report, and GeoTIFF when georeferencing is present.
- The built-in four-band sample is a labeled synthetic engineering fixture processed by the real model.

PNG/JPEG/RGB inputs cannot run this model because they lack measured NIR. Inputs are limited to 20 MB and 1024 × 1024 pixels. The model code and weights are unchanged; no training is run.

## Implementation

`src/api.ts` owns uploads, backend health, bounded requests, model jobs, cancellation, and image release. Vite proxies `/api` to the local backend. The former browser-resizing demo has been removed.

Fonts and scene visuals are local. The UI retains responsive layouts, reduced motion, animation pause, keyboard focus management, accessible dialogs, and the WebGL fallback. Temporary backend files expire after one hour or on clean shutdown.
