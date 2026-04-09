#!/bin/bash
# Recovery pod pipeline: regenerates the collapsed baseline checkpoint, then runs recovery.
#
# Use this when models/baseline/iter_020 is not available (e.g. baseline pod was terminated).
# Total runtime: ~10h on A100 PCIe 80GB (~5h baseline + ~5h recovery).
#
# Usage:
#   bash setup_recovery_pod.sh
#
# Always verify CUDA before running this:
#   python3 -c "import torch; torch.ones(1, device='cuda'); print('CUDA OK')"

set -eo pipefail

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
    print("[pre-install]  CUDA NOT available — aborting (broken pod, get a new one)")
    sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[pre-install]  CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
PRECHECK

# --- Phase 2: install deps, pinning torch to avoid any transitive upgrade ---
TORCH_VER=$(python3 -c "import torch; print(torch.__version__)")
echo "torch==$TORCH_VER" > /tmp/torch_pin.txt

grep -vE '^(torch|flash-attn|bitsandbytes|nvidia-)' requirements.txt \
  | pip install -r /dev/stdin -q \
      -c /tmp/torch_pin.txt \
      --constraint <(pip list --format=freeze | grep '^nvidia-')
pip uninstall flash-attn bitsandbytes -y 2>/dev/null || true

# --- Phase 3: post-install CUDA check ---
python3 - <<'POSTCHECK'
import torch, sys
print(f"[post-install] torch={torch.__version__}")
try:
    ok = torch.cuda.is_available()
except Exception as e:
    print(f"[post-install] cuda.is_available() raised: {e}"); ok = False
if not ok:
    print("[post-install] CUDA BROKEN after pip install")
    sys.exit(1)
t = torch.ones(1, device="cuda", dtype=torch.bfloat16)
print(f"[post-install] CUDA bf16 OK on {torch.cuda.get_device_name(0)}")
POSTCHECK

mkdir -p logs data

# --- Step 1: run baseline to generate the collapsed iter_020 checkpoint ---
echo ""
echo "=========================================="
echo "  STEP 1/2: baseline (regenerating iter_020 checkpoint)"
echo "  ~5h — saves to models/baseline/iter_020"
echo "=========================================="
python -u run_experiment.py --config experiments/configs/baseline.yaml \
  2>&1 | tee logs/baseline_for_recovery_run.log

# Verify checkpoint exists before continuing
if [ ! -d "models/baseline/iter_020" ]; then
  echo "ERROR: models/baseline/iter_020 not found after baseline run — check logs/baseline_for_recovery_run.log"
  exit 1
fi
echo "Checkpoint verified: models/baseline/iter_020"

# --- Step 2: run recovery from the collapsed checkpoint ---
echo ""
echo "=========================================="
echo "  STEP 2/2: recovery"
echo "  ~5h — starts from collapsed iter_020, applies injection every iteration"
echo "=========================================="
python -u run_experiment.py --config experiments/configs/recovery.yaml \
  2>&1 | tee logs/recovery_run.log

echo ""
echo "Pipeline complete. CSVs:"
echo "  logs/baseline_for_recovery_run.log  (step 1 stdout)"
echo "  logs/recovery/metrics.csv           (recovery metrics)"
