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

# --- Phase 1: verify CUDA works BEFORE pip changes anything ---
python3 - <<'PRECHECK'
import torch, sys
print(f"[pre-install]  torch={torch.__version__}  CUDA={torch.version.cuda}")
try:
    ok = torch.cuda.is_available()
except Exception as e:
    print(f"[pre-install]  cuda.is_available() raised: {e}"); ok = False
if not ok:
    print("[pre-install]  CUDA NOT available — aborting (pod GPU issue, not code issue)")
    sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[pre-install]  CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
PRECHECK

# --- Phase 2: install deps, pinning torch to avoid any transitive upgrade ---
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VER" > /tmp/torch_pin.txt

# Also exclude nvidia-* CUDA library packages — these ship inside the pod's
# torch wheel already; installing a conflicting system-level version can break
# CUDA even without changing the torch version string.
grep -vE '^(torch|flash-attn|bitsandbytes|nvidia-)' requirements.txt \
  | pip install -r /dev/stdin -q \
      -c /tmp/torch_pin.txt \
      --constraint <(pip list --format=freeze | grep '^nvidia-')
pip uninstall flash-attn bitsandbytes -y 2>/dev/null || true

# --- Phase 3: verify CUDA still works AFTER pip install ---
python3 - <<'POSTCHECK'
import torch, sys
print(f"[post-install] torch={torch.__version__}")
try:
    ok = torch.cuda.is_available()
except Exception as e:
    print(f"[post-install] cuda.is_available() raised: {e}"); ok = False
if not ok:
    print("[post-install] CUDA BROKEN after pip install — a package installed a conflicting CUDA lib")
    sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[post-install] CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
POSTCHECK

# HF auth — required for Llama-3 configs, harmless otherwise
if [ -n "$HF_TOKEN" ]; then
  export HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
  python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"
fi

mkdir -p logs data

echo "=== Starting $CONFIG ==="
python -u run_experiment.py --config "$CONFIG_PATH" \
  2>&1 | tee "logs/${CONFIG}_run.log"
