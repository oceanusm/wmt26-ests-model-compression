#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"
modelzip_source="${MODELZIP_SOURCE:-$(cd "$root_dir/../.." && pwd)}"
model_dir="${MODEL_DIR:-$root_dir/workdir/model}"
model_cache="${MODEL_CACHE:-/mnt/tg/data/projects/wmt26/model-compression/models}"

uv venv --python 3.12 "$venv_dir"
source "$venv_dir/bin/activate"
uv pip install -r "$root_dir/requirements.txt"
uv pip install --no-deps -e "$modelzip_source"

python3 "$root_dir/prepare_model.py" \
  --submission-config "$root_dir/submission.json" \
  --cache-dir "$model_cache" \
  --output "$model_dir"

python3 -m py_compile \
  "$root_dir/inference.py" \
  "$root_dir/pruned_gptoss.py" \
  "$root_dir/wmt26_json_adapter.py" \
  "$root_dir/wmt26_robust_inference.py"
