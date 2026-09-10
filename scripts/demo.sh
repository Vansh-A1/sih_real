#!/usr/bin/env bash
set -euo pipefail
# Explicitly invoked end-to-end demo. THIS SCRIPT STARTS TRAINING.
# It was authored but not executed during implementation at the user's request.
if [[ "${1:-}" != "--allow-training" ]]; then
  echo 'This demo trains on synthetic fixtures. Invoke with --allow-training only when you intend to start training.'
  exit 2
fi
S2SR="${S2SR:-.venv/bin/s2sr}"
"$S2SR" fixtures --config configs/fixture_demo.json --output artifacts/demo/data --count 40 --size 16
"$S2SR" validate-data --manifest artifacts/demo/data/manifest.json
"$S2SR" train --config configs/fixture_demo.json --manifest artifacts/demo/data/manifest.json --run artifacts/demo/run
"$S2SR" evaluate --checkpoint artifacts/demo/run/last.pt --manifest artifacts/demo/data/manifest.json --split test --output artifacts/demo/test_metrics.json
"$S2SR" infer --checkpoint artifacts/demo/run/last.pt --manifest artifacts/demo/data/manifest.json --sample phantom_38 --output artifacts/demo/sr.tif
