#!/bin/bash
# Usage: bash setup_runpod.sh baseline   OR   bash setup_runpod.sh injection
# Required env vars: WANDB_API_KEY, HF_TOKEN
set -e

RUN=${1:-baseline}
if [[ "$RUN" != "baseline" && "$RUN" != "injection" ]]; then
  echo "ERROR: argument must be 'baseline' or 'injection'"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN   # canonical var picked up by datasets library

pip install -r requirements.txt -q
pip install flash-attn --no-build-isolation --prefer-binary -q

wandb login "$WANDB_API_KEY" --relogin
huggingface-cli login --token "$HF_TOKEN"

mkdir -p logs

echo "=== Starting $RUN ==="
python -u run_experiment.py --config experiments/configs/${RUN}.yaml \
  2>&1 | tee logs/${RUN}_run.log
