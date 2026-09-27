#!/usr/bin/env bash
# One-time setup on Apple Silicon: virtualenv + MLX + logging extras.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[mac,logging,modal,dev]"
echo
echo "Done. Next:"
echo "  source .venv/bin/activate"
echo "  huggingface-cli login            # for gated models (Llama, Mistral, ...)"
echo "  wandb login                      # optional, if you enable W&B"
echo "  modal setup                      # optional, for cloud GPU training"
echo "  finetune train --config configs/smoke_test.yaml --backend mac"
