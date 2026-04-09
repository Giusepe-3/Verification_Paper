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
# (even as a transitive dep of transformers/peft/accelerate) overwrites it with
# an incompatible build → segfault on first CUDA call.
# Fix: pin torch to the pod's pre-installed version as a constraint so pip
# cannot upgrade it, even transitively.
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "Pod torch: $TORCH_VER (pinned)"
echo "torch==$TORCH_VER" > /tmp/torch_pin.txt

grep -vE '^(torch|flash-attn|bitsandbytes)' requirements.txt \
  | pip install -r /dev/stdin -q -c /tmp/torch_pin.txt
pip uninstall flash-attn bitsandbytes -y 2>/dev/null || true

# Confirm torch version didn't change and CUDA is usable before starting experiment
python3 - <<'PYCHECK'
import torch, sys
print(f"Post-install torch: {torch.__version__}")
if not torch.cuda.is_available():
    print("ERROR: CUDA not available — check driver/torch mismatch")
    sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
PYCHECK

# HF auth — required for Llama-3 configs, harmless otherwise
if [ -n "$HF_TOKEN" ]; then
  export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
  python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

mkdir -p logs data

echo "=== Starting $CONFIG ==="
python -u run_experiment.py --config "$CONFIG_PATH" \
  2>&1 | tee "logs/${CONFIG}_run.log"
