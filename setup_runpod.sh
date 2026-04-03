#!/bin/bash
# RunPod setup — runs baseline then injection back-to-back.
# Usage: bash setup_runpod.sh
#
# Required environment variables (set in RunPod pod settings):
#   WANDB_API_KEY   — your W&B API key
#   HF_TOKEN        — HuggingFace token (for model download)
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Authenticating ==="
wandb login "$WANDB_API_KEY"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

echo "=== Creating log directory ==="
mkdir -p logs

echo "=== Run 1: Baseline ==="
python -u run_experiment.py --config experiments/configs/baseline.yaml \
  2>&1 | tee logs/baseline_run.log

echo "=== Run 2: Injection ==="
python -u run_experiment.py --config experiments/configs/injection.yaml \
  2>&1 | tee logs/injection_run.log

echo "=== Both runs complete ==="
