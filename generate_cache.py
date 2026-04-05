"""
Finds a working MATH dataset on HuggingFace Hub, downloads it,
filters to Level 3-5 problems, and writes data/math_subset.json.

Run once locally. Commit the resulting file so pods never need to download.

Usage:
    python generate_cache.py
"""
import json, os, random, sys

hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)

import requests
from huggingface_hub import list_repo_files, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, DisabledRepoError
import pandas as pd

SEED = 42
SUBSET_SIZE = 250
LEVEL_FILTER = {"Level 3", "Level 4", "Level 5"}
SPLIT = "train"
CACHE_PATH = "data/math_subset.json"

headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}

# ---------------------------------------------------------------------------
# Step 1: search HuggingFace for MATH datasets and find one that works
# ---------------------------------------------------------------------------
print("Searching HuggingFace Hub for MATH datasets...")
r = requests.get(
    "https://huggingface.co/api/datasets",
    params={"search": "MATH competition hendrycks", "limit": 40, "full": False},
    headers=headers,
    timeout=15,
)
r.raise_for_status()
found = r.json()
print(f"  Hub returned {len(found)} results")

# Prioritise likely candidates, then append whatever the search found
PRIORITY = [
    "EleutherAI/hendrycks_math",
    "TIGER-Lab/MATH-500",
    "HuggingFaceH4/MATH-500",
    "lighteval/MATH-Hard",
    "Maxwell-Jia/MATH",
    "hendrycks/competition_math",   # try anyway — may be re-enabled
    "lighteval/MATH",
]
search_ids = [d.get("id") or d.get("modelId") or d.get("name", "") for d in found]
candidates = PRIORITY + [s for s in search_ids if s not in PRIORITY]
print(f"  Trying {len(candidates)} candidate repos...")

def _try_repo(repo_id):
    """Return list of parquet filenames for SPLIT, or None if repo unusable."""
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset", token=hf_token))
    except (RepositoryNotFoundError, DisabledRepoError, Exception):
        return None
    parquet = [f for f in files if f.endswith(".parquet") and SPLIT in f]
    if not parquet:
        parquet = [f for f in files if f.endswith(".parquet")]
    return parquet if parquet else None

working_repo = None
working_files = None
for repo_id in candidates:
    if not repo_id:
        continue
    sys.stdout.write(f"  Checking {repo_id} ... ")
    sys.stdout.flush()
    result = _try_repo(repo_id)
    if result is not None:
        print(f"OK ({len(result)} parquet files)")
        working_repo = repo_id
        working_files = result
        break
    else:
        print("not available")

if working_repo is None:
    print("\nNo working MATH dataset found on HuggingFace Hub.")
    print("All search results tried:")
    for c in candidates:
        print(f"  {c}")
    sys.exit(1)

print(f"\nUsing: {working_repo}")
for f in working_files:
    print(f"  {f}")

# ---------------------------------------------------------------------------
# Step 2: download and load parquet files
# ---------------------------------------------------------------------------
dfs = []
for repo_filename in working_files:
    local_path = hf_hub_download(
        repo_id=working_repo,
        filename=repo_filename,
        repo_type="dataset",
        token=hf_token,
    )
    dfs.append(pd.read_parquet(local_path))
    print(f"  Loaded: {repo_filename}")

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Normalise column names (some repos use 'problem', others 'question')
if "problem" not in df.columns and "question" in df.columns:
    df["problem"] = df["question"]
if "type" not in df.columns and "subject" in df.columns:
    df["type"] = df["subject"]

# ---------------------------------------------------------------------------
# Step 3: filter by level
# ---------------------------------------------------------------------------
if "level" in df.columns:
    df = df[df["level"].isin(LEVEL_FILTER)]
    print(f"After Level 3-5 filter: {len(df)} rows")
else:
    print("WARNING: no 'level' column — skipping level filter (using all rows)")

if len(df) == 0:
    print(f"Level values present: {df['level'].unique() if 'level' in df.columns else 'N/A'}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 4: shuffle + subset
# ---------------------------------------------------------------------------
df = df.sample(n=min(SUBSET_SIZE, len(df)), random_state=SEED).reset_index(drop=True)
print(f"After subset: {len(df)} rows")

# ---------------------------------------------------------------------------
# Step 5: extract boxed answer + build records
# ---------------------------------------------------------------------------
def _extract_boxed(text):
    results = []
    i = 0
    while i < len(text):
        pos = text.find(r"\boxed{", i)
        if pos == -1:
            break
        start = pos + 7
        depth, j = 1, start
        while j < len(text) and depth > 0:
            if text[j] == "{": depth += 1
            elif text[j] == "}": depth -= 1
            j += 1
        if depth == 0:
            results.append(text[start:j-1])
        i = pos + 1
    return results[-1].strip() if results else None

PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then write your final answer inside \\boxed{{}}.\n\n"
    "Problem: {question}\n\nSolution:"
)

records = []
for _, row in df.iterrows():
    solution = str(row.get("solution") or "")
    answer = str(row.get("answer") or "").strip()
    if not answer:
        answer = _extract_boxed(solution) or ""
    problem = str(row.get("problem") or "")
    records.append({
        "question": problem,
        "solution": solution,
        "answer": answer,
        "level": str(row.get("level") or ""),
        "subject": str(row.get("type") or ""),
        "prompt": PROMPT_TEMPLATE.format(question=problem),
    })

os.makedirs("data", exist_ok=True)
with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"\nDone. {len(records)} problems written to {CACHE_PATH}")
print(f"Source dataset: {working_repo}")
print("\nNext steps:")
print("  git add data/math_subset.json")
print("  git commit -m 'Add MATH L3-5 subset cache (250 problems)'")
print("  git push")
