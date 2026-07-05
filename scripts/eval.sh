#!/usr/bin/env bash
# Evaluate a trained checkpoint (in-domain / cross-manipulation / cross-dataset).
#   ./scripts/eval.sh ckpt_path=outputs/.../checkpoints/last.ckpt
#   ./scripts/eval.sh ckpt_path=... data.source=manifest data.manifest=cdf.csv
set -euo pipefail
source activate open_set_deepfake
cd "$(dirname "$0")/.."
python test.py "$@"
