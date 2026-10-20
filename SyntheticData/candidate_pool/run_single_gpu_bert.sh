#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/root/workspace
SCRIPT=${ROOT_DIR}/SyntheticData/candidate_pool/run_candidate_pool_experiments.py
DATA_DIR=${ROOT_DIR}/cloud/model/extra_data/candidate_pool/outputs
LOG_DIR=${ROOT_DIR}/cloud/model/extra_data/candidate_pool_logs
BERT_MODEL=/root/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594

export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/anaconda/lib:${LD_LIBRARY_PATH:-}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${ROOT_DIR}/cloud/model/extra_data/candidate_pool/cache/huggingface
export HUGGINGFACE_HUB_CACHE=${ROOT_DIR}/cloud/model/extra_data/candidate_pool/cache/huggingface/hub
export TRANSFORMERS_CACHE=${ROOT_DIR}/cloud/model/extra_data/candidate_pool/cache/huggingface/transformers
export TORCH_HOME=${ROOT_DIR}/cloud/model/extra_data/candidate_pool/cache/torch
# Do not enable expandable_segments under HAMI: it causes
# ``CUDA driver error: invalid argument`` during model.to(cuda).
unset PYTORCH_CUDA_ALLOC_CONF

mkdir -p "${LOG_DIR}"

# HAMI is more reliable when the experiment is loaded through stdin/runpy.
run_experiment() {
  python -u - "$@" <<'PY'
import runpy
import sys

script = "/root/workspace/SyntheticData/candidate_pool/run_candidate_pool_experiments.py"
runpy.run_path(script, run_name="__main__")
PY
}

run_with_retry() {
  local attempt
  for attempt in 1 2 3 4 5; do
    echo "[candidate-pool] attempt ${attempt}/5: $*" >&2
    if run_experiment "$@"; then
      return 0
    fi
    echo "[candidate-pool] failed; retrying after 5 seconds" >&2
    sleep 5
  done
  return 1
}

cd "${ROOT_DIR}"
run_with_retry train \
  --pool-sizes 160 250 500 \
  --model-seeds 42 43 44 \
  --shard-name single_gpu \
  --evaluate-legacy400 \
  --batch-size 16 \
  --data-dir "${DATA_DIR}" \
  --bert-model "${BERT_MODEL}"

run_with_retry finalize \
  --model-seeds 42 43 44 \
  --data-dir "${DATA_DIR}" \
  --bert-model "${BERT_MODEL}"

echo "[candidate-pool] training and finalize completed"
