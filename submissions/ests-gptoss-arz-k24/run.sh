#!/usr/bin/env bash
set -euo pipefail

root_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
venv_dir="$root_dir/.venv"

if [[ ! -x "$venv_dir/bin/python3" ]]; then
  echo "Submission environment is missing; run $root_dir/setup.sh first." >&2
  exit 1
fi

export MODEL_DIR="${MODEL_DIR:-$root_dir/workdir/model}"
export PYTHONPATH="$root_dir${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export VLLM_KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-1073741824}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-12288}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
export VLLM_SWAP_SPACE="${VLLM_SWAP_SPACE:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec "$venv_dir/bin/python3" "$root_dir/inference.py" --model "$MODEL_DIR" "$@"
