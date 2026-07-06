#!/usr/bin/env bash
set -euo pipefail

DEBUG_GPU="${CUDA_VISIBLE_DEVICES:-}"
if [ -z "${DEBUG_GPU}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    DEBUG_GPU="$(
        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr \
        | head -n 1 \
        | cut -d, -f1 \
        | tr -d ' '
    )"
fi
DEBUG_GPU="${DEBUG_GPU:-0}"

echo "[run_qec_debug] CUDA_VISIBLE_DEVICES=${DEBUG_GPU}" >&2

EC_DEBUG=1 \
EC_DEBUG_LIMIT=30 \
PRUNE_METHOD=ec_pruner \
EC_SCORE_SOURCE=semantic \
EC_USE_SEMANTIC_CANDIDATE=1 \
EC_USE_SPATIAL_CANDIDATE=1 \
EC_SPATIAL_EXCLUDE_GLOBAL=1 \
EC_GLOBAL_CANDIDATE_RATIO=0.625 \
EC_CANDIDATE_RATIO=2.0 \
EC_BETA_A=0.5 \
EC_BETA_SEM=0.3 \
EC_BETA_SPATIAL=0.2 \
EC_PER_UNIT_TOPL=8 \
EC_SPATIAL_GRID_SIZE=4 \
EC_SPATIAL_TOPL=1 \
EC_USE_REPULSION=1 \
EC_USE_COMPLEMENT=1 \
EC_USE_PHI=1 \
EC_LAMBDA=0.1 \
EC_DELTA=0.1 \
EC_SOLVER=qanneal \
EC_QA_STEPS=50 \
EC_QA_T0=1.0 \
EC_QA_TEND=0.01 \
EC_QA_MAX_SWAP_RATIO=0.15 \
EC_QA_SEED=42 \
CUDA_VISIBLE_DEVICES="${DEBUG_GPU}" \
bash scripts/v1_5/eval/gqa.sh 64
