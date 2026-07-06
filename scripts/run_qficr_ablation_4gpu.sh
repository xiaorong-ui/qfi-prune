#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOKEN="${TOKEN:-64}"
CHUNKS="${CHUNKS:-1}"
LOG_ROOT="${LOG_ROOT:-logs/ablation_qficr}"
OUT_ROOT="${OUT_ROOT:-outputs/ablation_qficr}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 1 2 3)
fi
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

EXPERIMENTS=(
  full
  only_reduction
  random_restoration
  fixed_ratio
  no_spatial_local_prior
  no_spatial
  no_local
  sem_only
  cls_only
  fixed_fusion
)

run_one() {
  local exp="$1"
  local gpu="$2"
  local profile="qficr_ablate_${exp}"
  local log_path="${LOG_ROOT}/${exp}.log"
  local out_dir="${OUT_ROOT}/${exp}"
  mkdir -p "${out_dir}"
  echo "[qficr-ablation] start exp=${exp} gpu=${gpu} token=${TOKEN} chunks=${CHUNKS} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash run_gqa_profile.sh "${profile}" "${gpu}" "${TOKEN}" \
    2>&1 | tee "${log_path}"
  echo "[qficr-ablation] done exp=${exp} gpu=${gpu}"
}

batch=()
gpu_pos=0
for exp in "${EXPERIMENTS[@]}"; do
  run_one "${exp}" "${GPUS[$gpu_pos]}" &
  batch+=("$!")
  gpu_pos=$(( (gpu_pos + 1) % ${#GPUS[@]} ))
  if [ "${#batch[@]}" -eq "${#GPUS[@]}" ]; then
    for pid in "${batch[@]}"; do
      wait "${pid}"
    done
    batch=()
    gpu_pos=0
  fi
done

for pid in "${batch[@]}"; do
  wait "${pid}"
done

python scripts/collect_qficr_ablation_results.py
