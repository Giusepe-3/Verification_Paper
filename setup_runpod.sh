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
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN')"

mkdir -p logs data

# Dataset is shipped as data/math_subset.json (committed to the repo).
# If it is missing, the experiment cannot run — fail fast rather than silently.
if [ ! -f "data/math_subset.json" ]; then
  echo "ERROR: data/math_subset.json not found."
  echo "Generate it locally with:"
  echo "  python -c \""
  echo "    from src.math_loader import MathDataset"
  echo "    MathDataset(subset_size=250, split='train', seed=42,"
  echo "                dataset_name='hendrycks/competition_math',"
  echo "                level_filter=[3,4,5],"
  echo "                cache_path='data/math_subset.json')"
  echo "  \""
  echo "Then: git add data/math_subset.json && git commit -m 'Add MATH subset cache' && git push"
  exit 1
fi
echo "=== data/math_subset.json found — skipping dataset download ==="

echo "=== Starting $RUN ==="
python -u run_experiment.py --config experiments/configs/${RUN}.yaml \
  2>&1 | tee logs/${RUN}_run.log
