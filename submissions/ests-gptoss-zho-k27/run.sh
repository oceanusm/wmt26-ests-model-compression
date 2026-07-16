#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL_DIR="${MODEL_DIR:-$root_dir/workdir/model}"
source "$root_dir/.venv/bin/activate"

python3 "$root_dir/inference.py" --model "$MODEL_DIR" "$@"
