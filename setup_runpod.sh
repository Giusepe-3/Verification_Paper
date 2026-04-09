#!/bin/bash
# Usage: bash setup_runpod.sh <config_name>
# Example: bash setup_runpod.sh random_negatives
# For Llama-3 configs: export HF_TOKEN=hf_... before running
set -eo pipefail

CONFIG=${1}
if [ -z "$CONFIG" ]; then
  echo "ERROR: must supply a config name (without .yaml)"
  echo "Usage: bash setup_runpod.sh <config_name>"
  exit 1
fi

CONFIG_PATH="experiments/configs/${CONFIG}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: $CONFIG_PATH not found"
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# RunPod pods ship with a CUDA-matched PyTorch. Reinstalling torch via pip
# replaces it with a version that may not match the pod's CUDA driver → segfault.
# Skip torch and flash-attn; verifier.py falls back to sdpa automatically.
# Skip torch (pod ships CUDA-matched), flash-attn (segfaults), bitsandbytes
# (not needed for load_in_4bit: false configs; can segfault on CUDA mismatch).
grep -vE '^(torch|flash-attn|bitsandbytes)' requirements.txt | pip install -r /dev/stdin -q
pip uninstall flash-attn bitsandbytes -y 2>/dev/null || true

# HF auth — required for Llama-3 configs, harmless otherwise
if [ -n "$HF_TOKEN" ]; then
  export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
  python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

mkdir -p logs data

echo "=== Starting $CONFIG ==="
python -u run_experiment.py --config "$CONFIG_PATH" \
  2>&1 | tee "logs/${CONFIG}_run.log"
