#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-logs/qficr_k_sensitivity}"
OUT_ROOT="${OUT_ROOT:-outputs/qficr_k_sensitivity}"
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

run_one() {
  local exp="$1"
  local profile="$2"
  local gpu="$3"
  local token="$4"
  local log_path="${LOG_ROOT}/${exp}.log"

  mkdir -p "${OUT_ROOT}/${exp}"
  echo "[qficr-ksens] start exp=${exp} profile=${profile} gpu=${gpu} K=${token} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash run_gqa_profile.sh "${profile}" "${gpu}" "${token}" \
    2>&1 | tee "${log_path}"
  echo "[qficr-ksens] done exp=${exp}"
}

run_one qficr_k32 qficr_ksens_stable 0 32 &
pid0=$!
run_one qficr_k128 qficr_ksens_stable 1 128 &
pid1=$!
run_one only_reduction_k32 qficr_ksens_only_reduction 2 32 &
pid2=$!
run_one only_reduction_k128 qficr_ksens_only_reduction 3 128 &
pid3=$!

wait "${pid0}"
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

python scripts/collect_qficr_k_sensitivity_results.py
