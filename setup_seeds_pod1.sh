#!/bin/bash
# Pod 1: baseline seed43 → injection seed43
# Usage: bash setup_seeds_pod1.sh
set -eo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- CUDA check ---
python3 - <<'PRECHECK'
import torch, sys
print(f"[pre-install]  torch={torch.__version__}  CUDA={torch.version.cuda}")
try:
    ok = torch.cuda.is_available()
except Exception as e:
    print(f"[pre-install]  cuda.is_available() raised: {e}"); ok = False
if not ok:
    print("[pre-install]  CUDA NOT available — aborting"); sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[pre-install]  CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
PRECHECK

# --- Install deps ---
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VER" > /tmp/torch_pin.txt
grep -vE '^(torch|flash-attn|bitsandbytes|nvidia-)' requirements.txt \
  | pip install -r /dev/stdin -q \
      -c /tmp/torch_pin.txt \
      --constraint <(pip list --format=freeze | grep '^nvidia-')
pip uninstall flash-attn bitsandbytes -y 2>/dev/null || true

# --- Post-install CUDA check ---
python3 - <<'POSTCHECK'
import torch, sys
print(f"[post-install] torch={torch.__version__}")
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[post-install] CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
POSTCHECK

mkdir -p logs data

echo "=== Pod 1: baseline_seed43 ==="
python -u run_experiment.py --config experiments/configs/baseline_seed43.yaml \
  2>&1 | tee logs/baseline_seed43_run.log

echo "=== Pod 1: injection_seed43 ==="
python -u run_experiment.py --config experiments/configs/injection_seed43.yaml \
  2>&1 | tee logs/injection_seed43_run.log

echo "=== Pod 1 complete ==="
