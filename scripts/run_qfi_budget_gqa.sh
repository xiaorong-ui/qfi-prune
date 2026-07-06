#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [ "${MODE}" != "dry" ] && [ "${MODE}" != "full" ]; then
    echo "Usage: bash scripts/run_qfi_budget_gqa.sh dry|full" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PROFILES_VALUE="${PROFILES:-qfid_density_depol qfid_density_eaqf_final qfid_density_cg_eaqf_final}"
KS_VALUE="${KS:-32 96 128}"
GPUS_VALUE="${GPUS:-0 1 2 3}"
FORCE_VALUE="${FORCE:-0}"
LOG_DIR="${LOG_DIR:-logs/qfi_budget_gqa}"

read -r -a PROFILE_LIST <<< "${PROFILES_VALUE}"
read -r -a K_LIST <<< "${KS_VALUE}"
read -r -a GPU_LIST <<< "${GPUS_VALUE}"

if [ "${#PROFILE_LIST[@]}" -eq 0 ] || [ "${#K_LIST[@]}" -eq 0 ] || [ "${#GPU_LIST[@]}" -eq 0 ]; then
    echo "PROFILES, KS, and GPUS must each contain at least one value." >&2
    exit 2
fi

mkdir -p "${LOG_DIR}"
COMMAND_LOG="${LOG_DIR}/commands.sh"
if [ ! -e "${COMMAND_LOG}" ]; then
    printf '#!/usr/bin/env bash\n' > "${COMMAND_LOG}"
fi
printf '\n# %s mode=%s profiles=%q ks=%q gpus=%q\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${MODE}" "${PROFILES_VALUE}" "${KS_VALUE}" "${GPUS_VALUE}" \
    >> "${COMMAND_LOG}"

is_full_complete() {
    local log_path="$1"
    [ -f "${log_path}" ] \
        && grep -q '^Accuracy:' "${log_path}" \
        && grep -q '12578/12578' "${log_path}"
}

wait_batch() {
    local index pid description
    for index in "${!BATCH_PIDS[@]}"; do
        pid="${BATCH_PIDS[$index]}"
        description="${BATCH_DESCRIPTIONS[$index]}"
        if wait "${pid}"; then
            echo "[qfi-budget] completed ${description}"
        else
            echo "[qfi-budget] failed ${description}; inspect its log" >&2
        fi
    done
    BATCH_PIDS=()
    BATCH_DESCRIPTIONS=()
}

BATCH_PIDS=()
BATCH_DESCRIPTIONS=()
gpu_index=0

for profile in "${PROFILE_LIST[@]}"; do
    for keep_num in "${K_LIST[@]}"; do
        if ! [[ "${keep_num}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Invalid K value: ${keep_num}" >&2
            exit 2
        fi

        log_path="${LOG_DIR}/${profile}_k${keep_num}.log"
        pid_path="${LOG_DIR}/${profile}_k${keep_num}.pid"
        if [ "${FORCE_VALUE}" != "1" ] && is_full_complete "${log_path}"; then
            echo "[qfi-budget] skip completed profile=${profile} K=${keep_num} log=${log_path}"
            continue
        fi

        gpu="${GPU_LIST[$gpu_index]}"
        command=(
            env
            TRANSFORMERS_OFFLINE=1
            HF_HUB_OFFLINE=1
            HF_DATASETS_OFFLINE=1
        )
        if [ "${MODE}" = "dry" ]; then
            command+=(NUM_SAMPLES="${NUM_SAMPLES:-32}")
        fi
        command+=(bash run_gqa_profile.sh "${profile}" "${gpu}" "${keep_num}")

        {
            printf '('
            printf ' %q' "${command[@]}"
            printf ' ) > %q 2>&1\n' "${log_path}"
        } >> "${COMMAND_LOG}"

        echo "[qfi-budget] launch mode=${MODE} profile=${profile} K=${keep_num} gpu=${gpu}"
        "${command[@]}" > "${log_path}" 2>&1 &
        pid=$!
        printf '%s\n' "${pid}" > "${pid_path}"
        BATCH_PIDS+=("${pid}")
        BATCH_DESCRIPTIONS+=("profile=${profile} K=${keep_num} gpu=${gpu} pid=${pid}")

        gpu_index=$((gpu_index + 1))
        if [ "${gpu_index}" -ge "${#GPU_LIST[@]}" ]; then
            wait_batch
            gpu_index=0
        fi
    done
done

if [ "${#BATCH_PIDS[@]}" -gt 0 ]; then
    wait_batch
fi

echo "[qfi-budget] all scheduled ${MODE} jobs finished"

