#!/usr/bin/env bash
# Run inference on a single image or a folder.
#   ./scripts/infer.sh --ckpt model.ckpt --input image.png
#   ./scripts/infer.sh --ckpt model.ckpt --input folder/ --csv out.csv
set -euo pipefail
source activate open_set_deepfake
cd "$(dirname "$0")/.."
python predict.py "$@"
