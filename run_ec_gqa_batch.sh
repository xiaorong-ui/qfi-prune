#!/usr/bin/env bash
set -euo pipefail

cd /home/gpuadmin/txr/CDPruner

RUN_ID=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/ec_gqa_${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo "[INFO] Log dir: ${LOG_DIR}"

run_exp () {
    local name="$1"
    local gpu="$2"
    shift 2

    local log_file="${LOG_DIR}/${name}.log"

    echo "[INFO] Starting ${name} on GPU ${gpu}"
    echo "[INFO] Log: ${log_file}"

    nohup env CUDA_VISIBLE_DEVICES="${gpu}" "$@" > "${log_file}" 2>&1 &

    local pid=$!
    echo "${name} ${pid} ${log_file}" >> "${LOG_DIR}/pids.txt"
    echo "[INFO] ${name} PID=${pid}"
}

# ============================================================
# Exp A: global ratio 0.625, candidate only
# 目的：看 global=0.625 的候选池本身是否优于当前 0.75
# ============================================================
run_exp "g0625_candidate_only" 0 \
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
    EC_USE_REPULSION=0 \
    EC_USE_COMPLEMENT=0 \
    bash scripts/v1_5/eval/gqa.sh 64


# ============================================================
# Exp B: global ratio 0.625, full EC-Pruner
# 目的：看 global=0.625 下 R/C/Phi 是否继续带来增益
# ============================================================
run_exp "g0625_full" 1 \
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
    bash scripts/v1_5/eval/gqa.sh 64


# ============================================================
# Exp C: global ratio 0.875, candidate only
# 目的：看 global token 多一点是否更稳
# ============================================================
run_exp "g0875_candidate_only" 2 \
    PRUNE_METHOD=ec_pruner \
    EC_SCORE_SOURCE=semantic \
    EC_USE_SEMANTIC_CANDIDATE=1 \
    EC_USE_SPATIAL_CANDIDATE=1 \
    EC_SPATIAL_EXCLUDE_GLOBAL=1 \
    EC_GLOBAL_CANDIDATE_RATIO=0.875 \
    EC_CANDIDATE_RATIO=2.0 \
    EC_BETA_A=0.5 \
    EC_BETA_SEM=0.3 \
    EC_BETA_SPATIAL=0.2 \
    EC_PER_UNIT_TOPL=8 \
    EC_SPATIAL_GRID_SIZE=4 \
    EC_SPATIAL_TOPL=1 \
    EC_USE_REPULSION=0 \
    EC_USE_COMPLEMENT=0 \
    bash scripts/v1_5/eval/gqa.sh 64


# ============================================================
# Exp D: global ratio 0.75, full EC-Pruner, 调整 lambda/delta
# 目的：看增强 C、减弱 R 是否提升 Rel/Open/Query
# ============================================================
run_exp "g075_full_l005_d015" 3 \
    PRUNE_METHOD=ec_pruner \
    EC_SCORE_SOURCE=semantic \
    EC_USE_SEMANTIC_CANDIDATE=1 \
    EC_USE_SPATIAL_CANDIDATE=1 \
    EC_SPATIAL_EXCLUDE_GLOBAL=1 \
    EC_GLOBAL_CANDIDATE_RATIO=0.75 \
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
    EC_LAMBDA=0.05 \
    EC_DELTA=0.15 \
    bash scripts/v1_5/eval/gqa.sh 64


echo "[INFO] All experiments launched."
echo "[INFO] PID file: ${LOG_DIR}/pids.txt"
echo "[INFO] Check logs:"
echo "tail -f ${LOG_DIR}/*.log"