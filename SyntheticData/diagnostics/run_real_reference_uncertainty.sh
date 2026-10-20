#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# One 32-GiB GPU is sufficient. The encoder is frozen; this job never trains.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

PYTHON="${PYTHON:-$ROOT/SyntheticData/SmaExp/.venv/bin/python}"
OUT="${OUT:-$ROOT/SyntheticData/real_reference_uncertainty/outputs/mmd1024}"
CACHE="${CACHE:-$ROOT/SyntheticData/real_reference_uncertainty/outputs}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MMD_REFERENCE_SIZE="${MMD_REFERENCE_SIZE:-1024}"

exec "$PYTHON" "$ROOT/SyntheticData/diagnostics/run_real_reference_uncertainty.py" \
  --output-dir "$OUT" \
  --embedding-cache-dir "$CACHE" \
  --device cuda:0 \
  --batch-size "$BATCH_SIZE" \
  --max-length 256 \
  --pca-dim 16 \
  --clusters 8 \
  --mmd-reference-size "$MMD_REFERENCE_SIZE" \
  --bandwidth-pairs 20000 \
  --flag-metric-count 2
