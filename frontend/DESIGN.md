# RezX frontend

A tiny sculpted planet is the centerpiece, with quiet product typography and an illustrated cloud workspace. The experience moves from orbit to atmosphere instead of navigating between unrelated pages.

- Palette: deep space #071a25, ocean #548f9c, land #a5caa1, mint #c5f2d8, cloud #f3f8f7, ink #23434b.
- Type: Manrope for the wordmark and generous headlines; DM Sans for interface text. Self-hosted font assets.
- Composition: left-aligned opening copy, large Earth on the right; centered floating upload and result surfaces in the atmosphere. Mobile stacks copy above the scene.
- Motion: continuous slow orbit, a user-triggered camera dive, layered cloud silhouettes and vector wind. Reduced-motion preference removes ambient movement and shortens transitions.
- The brief explicitly calls for glass, dark space, and restrained uppercase state labels. These treatments are used where requested, rather than adding unrelated dashboard or science-fiction decoration.
- Model boundary: the frontend calls the local inference API. The API loads the pinned checkpoint and reuses the original model and tiled inference functions without modifying them. Scientific inputs are four-band TIFF/NumPy data; a small synthetic fixture demonstrates the real model workflow.
