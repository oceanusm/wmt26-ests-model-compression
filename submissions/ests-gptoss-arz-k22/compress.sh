#!/usr/bin/env bash
set -euo pipefail

# This script is not part of the evaluation contract; it is a documentation/reproducibility recipe for generating the submitted model artifact from a base model.

echo "baseline is the uncompressed Gemma baseline; no compression step is needed."
echo "Provide the original Gemma model at workdir/model, or set MODEL_DIR when running inference."