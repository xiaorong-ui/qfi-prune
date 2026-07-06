#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TOKEN="${TOKEN:-64}"
LOG_ROOT="${LOG_ROOT:-logs/qficr_tuning}"
OUT_ROOT="${OUT_ROOT:-outputs/qficr_tuning}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"
if [ "${#GPUS[@]}" -eq 0 ]; then
  GPUS=(0 1 2 3)
fi
mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

# User-requested first run set. Other candidates are emitted in commands_only.sh.
EXPERIMENTS=(
  baseline
  alpha_050
  alpha_075
  prior_lam_010
  prior_lam_025
  alpha050_prior010
  rho_max_down
  rho_max_up
)

cat > "${OUT_ROOT}/commands_only.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /home/gpuadmin/txr/CDPruner
mkdir -p logs/qficr_tuning outputs/qficr_tuning

bash run_gqa_profile.sh qficr_tune_alpha_125 0 64 2>&1 | tee logs/qficr_tuning/alpha_125.log
bash run_gqa_profile.sh qficr_tune_prior_lam_050 1 64 2>&1 | tee logs/qficr_tuning/prior_lam_050.log
bash run_gqa_profile.sh qficr_tune_alpha075_prior010 2 64 2>&1 | tee logs/qficr_tuning/alpha075_prior010.log
bash run_gqa_profile.sh qficr_tune_cap_down 3 64 2>&1 | tee logs/qficr_tuning/cap_down.log
bash run_gqa_profile.sh qficr_tune_cap_up 0 64 2>&1 | tee logs/qficr_tuning/cap_up.log
bash run_gqa_profile.sh qficr_tune_debug_prior_stats 1 64 2>&1 | tee logs/qficr_tuning/debug_prior_stats.log

# Candidate-pool topM is skipped by default because qfi_adaptive_recover_prior has EC_QFID_RECOVER_CAND_POOL=0.
# To test it explicitly, enable the pool and adjust EC_QFID_RECOVER_CAND_MULT around the baseline candidate multiplier:
# EC_QFID_RECOVER_CAND_POOL=1 EC_QFID_RECOVER_CAND_MULT=2.25 bash run_gqa_profile.sh qficr_tune_baseline 2 64 2>&1 | tee logs/qficr_tuning/topM_down.log
# EC_QFID_RECOVER_CAND_POOL=1 EC_QFID_RECOVER_CAND_MULT=3.75 bash run_gqa_profile.sh qficr_tune_baseline 3 64 2>&1 | tee logs/qficr_tuning/topM_up.log

python scripts/collect_qficr_tuning_results.py
EOF
chmod +x "${OUT_ROOT}/commands_only.sh"

run_one() {
  local exp="$1"
  local gpu="$2"
  local profile="qficr_tune_${exp}"
  local log_path="${LOG_ROOT}/${exp}.log"
  mkdir -p "${OUT_ROOT}/${exp}"
  echo "[qficr-tuning] start exp=${exp} gpu=${gpu} token=${TOKEN} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash run_gqa_profile.sh "${profile}" "${gpu}" "${TOKEN}" \
    2>&1 | tee "${log_path}"
  echo "[qficr-tuning] done exp=${exp} gpu=${gpu}"
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

python scripts/collect_qficr_tuning_results.py
