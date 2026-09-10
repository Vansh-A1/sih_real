#!/usr/bin/env bash
set -euo pipefail
# Set these to actual, validated paths; no real dataset was available during implementation.
: "${S2_MANIFEST:?Set S2_MANIFEST to your real-data manifest JSON}"
: "${S2_RUN:?Set S2_RUN to a new output run directory}"
S2SR="${S2SR:-.venv/bin/s2sr}"
"$S2SR" validate-data --manifest "$S2_MANIFEST"
"$S2SR" train --config configs/v1.json --manifest "$S2_MANIFEST" --run "$S2_RUN"
