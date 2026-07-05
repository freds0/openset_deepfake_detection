#!/usr/bin/env bash
# Train OSDFD (SigLIP 2). Extra args are forwarded to Hydra.
#   ./scripts/train.sh data.root=/data/ffpp_frames optimizer.lr=3e-5
set -euo pipefail
source activate open_set_deepfake
cd "$(dirname "$0")/.."
python train.py "$@"
