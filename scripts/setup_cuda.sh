#!/usr/bin/env bash
# Setup for a local NVIDIA GPU machine (Modal builds its own image; not needed there).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[cuda,logging,modal,dev]"
echo "Done. Try: finetune train --config configs/smoke_test.yaml --backend hf"
