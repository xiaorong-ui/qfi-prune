#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [ "${MODE}" != "dry" ] && [ "${MODE}" != "full" ]; then
    echo "Usage: bash scripts/run_qfi_budget_calib_pilot_gqa.sh dry|full" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PROFILES_VALUE="${PROFILES:-qfid_density_cg_eaqf_budget_alpha2 qfid_density_cg_eaqf_budget_alpha3 qfid_density_cg_eaqf_budget_alpha4}"
KS_VALUE="${KS:-32 64 128}"
GPUS_VALUE="${GPUS:-0 1 2 3}"
FORCE_VALUE="${FORCE:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-32}"
LOG_DIR="${LOG_DIR:-logs/qfi_budget_calib_gqa}"
DATA_DIR="${DATA_DIR:-/home/gpuadmin/txr/CDPruner_data}"

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

is_full_complete() {
    local log_path="$1"
    [ -f "${log_path}" ] && grep -q '^Accuracy:' "${log_path}" && grep -q '12578/12578' "${log_path}"
}

evaluate_dry_run() {
    local profile="$1" keep_num="$2" log_path="$3"
    local prediction_jsonl prediction_json subset_json eval_dir
    prediction_jsonl="$(sed -n 's/^\[gqa-smoke\].*predictions=//p' "${log_path}" | tail -n 1)"
    if [ -z "${prediction_jsonl}" ] || [ ! -s "${prediction_jsonl}" ]; then
        echo "[qfi-budget-calib] dry evaluation could not locate predictions" >> "${log_path}"
        return 1
    fi

    eval_dir="${LOG_DIR}/dry_eval/${profile}_k${keep_num}"
    mkdir -p "${eval_dir}"
    prediction_json="${eval_dir}/predictions.json"
    subset_json="${eval_dir}/questions.json"
    python scripts/convert_gqa_for_eval.py --src "${prediction_jsonl}" --dst "${prediction_json}" >> "${log_path}" 2>&1
    python -c 'import json,sys; p=json.load(open(sys.argv[1])); q=json.load(open(sys.argv[2])); ids={str(x["questionId"]) for x in p}; json.dump({k:v for k,v in q.items() if str(k) in ids}, open(sys.argv[3], "w"))' \
        "${prediction_json}" "${DATA_DIR}/gqa/data/questions/testdev_balanced_questions.json" "${subset_json}"
    python -u playground/data/eval/gqa/data/eval/eval.py \
        --questions "${subset_json}" \
        --predictions "${prediction_json}" \
        --tier testdev_balanced >> "${log_path}" 2>&1
}

run_one() {
    local profile="$1" keep_num="$2" gpu="$3" log_path="$4"
    local command=(
        env
        TRANSFORMERS_OFFLINE=1
        HF_HUB_OFFLINE=1
        HF_DATASETS_OFFLINE=1
    )
    if [ "${MODE}" = "dry" ]; then
        command+=(NUM_SAMPLES="${NUM_SAMPLES_VALUE}")
    fi
    command+=(bash run_gqa_profile.sh "${profile}" "${gpu}" "${keep_num}")

    "${command[@]}" > "${log_path}" 2>&1
    if [ "${MODE}" = "dry" ]; then
        evaluate_dry_run "${profile}" "${keep_num}" "${log_path}"
    fi
}

wait_batch() {
    local index pid description
    for index in "${!BATCH_PIDS[@]}"; do
        pid="${BATCH_PIDS[$index]}"
        description="${BATCH_DESCRIPTIONS[$index]}"
        if wait "${pid}"; then
            echo "[qfi-budget-calib] completed ${description}"
        else
            echo "[qfi-budget-calib] failed ${description}; inspect its log" >&2
        fi
    done
    BATCH_PIDS=()
    BATCH_DESCRIPTIONS=()
}

printf '\n# %s mode=%s profiles=%q ks=%q gpus=%q\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${MODE}" "${PROFILES_VALUE}" "${KS_VALUE}" "${GPUS_VALUE}" \
    >> "${COMMAND_LOG}"

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
        if [ "${MODE}" = "full" ] && [ "${FORCE_VALUE}" != "1" ] && is_full_complete "${log_path}"; then
            echo "[qfi-budget-calib] skip completed profile=${profile} K=${keep_num}"
            continue
        fi

        gpu="${GPU_LIST[$gpu_index]}"
        printf 'PROFILES=%q KS=%q GPUS=%q bash scripts/run_qfi_budget_calib_pilot_gqa.sh %q\n' \
            "${profile}" "${keep_num}" "${gpu}" "${MODE}" >> "${COMMAND_LOG}"
        echo "[qfi-budget-calib] launch mode=${MODE} profile=${profile} K=${keep_num} gpu=${gpu}"
        run_one "${profile}" "${keep_num}" "${gpu}" "${log_path}" &
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

echo "[qfi-budget-calib] all scheduled ${MODE} jobs finished"
