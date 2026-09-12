# Connected frontend verification

Verified on 2026-09-13 using the repository's pinned `last.pt` (V1, EMA, step 10,000).

- Production TypeScript/Vite build passes.
- Real API integration tests verify model output arrays against direct inference, shared PNG color mapping, GeoTIFF CRS/footprint/nodata, explicit band order/scale/offset/NumPy layout, invalid inputs, file limits, missing checkpoints, queued cancellation/retry, and cancellation during a real model forward pass.
- Browser flow: model health connection, four-band TIFF upload, missing-NIR rejection, actual server progress, real output, comparison keyboard input, ZIP download, synthetic fixture through the real model, mobile layout, offline state/reconnect, and input settings.
- Browser download verified: `sr.png`, `sr.npy`, `sr.tif`, `inference.json`; four-band 128 × 160 input becomes 512 × 640 output using checkpoint step 10,000.
- Browser Cancel stops its corresponding backend job.
- The combined `npm run dev:full` launcher was tested on isolated ports, including the frontend proxy and shutdown of its own backend process.
- Original Earth/satellite/atmosphere experience, responsive layout, reduced motion, focus behavior, and WebGL fallback are retained from the earlier frontend verification.

Screenshots and browser-check scripts are in the ignored `output/playwright/backend/` folder. The browser test deliberately aborts a health request to verify offline recovery; that intentional network error is distinct from application runtime errors (none recorded in the full flow).

The browser-interpolation demo has been removed. Existing model, inference, configuration, and training code are unchanged. No training was run. The checkpoint SHA-256 remains `df4eb605c0b795df9f339faa5c93af9c01329984e7c0d14b6392a4254da2bfa0` before and after testing. Synthetic fixtures test integration, not scientific accuracy on real satellite data.
