#!/bin/bash
# RunPod setup and experiment runner for verification-collapse
# Usage: bash setup_runpod.sh
#
# Required environment variables (set in RunPod pod settings):
#   WANDB_API_KEY   — your W&B API key
#   HF_TOKEN        — HuggingFace token (for model download)
set -e

# Disable PyTorch inductor JIT compilation — bitsandbytes 0.43-0.44 uses
# pre-built CUDA kernels and doesn't need it; inductor on a fresh pod would
# spend 30+ min compiling before the first token is generated.
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton_cache
export BITSANDBYTES_NOWELCOME=1

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Patching transformers bitsandbytes integration ==="
# PyTorch 2.4 is missing nn.Module.set_submodule; replace with setattr approach
python3 /tmp/patch_bnb.py 2>/dev/null || python3 - << 'PYEOF'
path = '/usr/local/lib/python3.11/dist-packages/transformers/integrations/bitsandbytes.py'
content = open(path).read()
old = 'model.set_submodule(module_name, new_module)'
new = ('*_p, _c = module_name.split("."); '
       '_par = model.get_submodule(".".join(_p)) if _p else model; '
       'setattr(_par, _c, new_module)')
if old in content:
    open(path,'w').write(content.replace(old, new))
    print("Patched set_submodule")
else:
    print("set_submodule not found — no patch needed")
PYEOF

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
