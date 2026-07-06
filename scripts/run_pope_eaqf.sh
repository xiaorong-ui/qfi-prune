#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "${REPO_ROOT}"

TOKEN=64
BASE_LOG_DIR="logs/pope_eaqf"
DRY_RUN="${DRY_RUN:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
FORCE="${FORCE:-0}"
LOG_DIR="${BASE_LOG_DIR}"
[ "${DRY_RUN}" = "1" ] && LOG_DIR="logs/pope_eaqf_dryrun"

CKPT_PATH="/home/gpuadmin/txr/CDPruner/checkpoints/llava-v1.5-7b"
POPE_SCRIPT="scripts/v1_5/eval/pope.sh"
METRIC_SCRIPT="llava/eval/eval_pope.py"
MODEL_ENTRY="llava/eval/model_vqa_loader.py"
QUESTION_FILE="playground/data/eval/pope/llava_pope_test.jsonl"
DRY_QUESTION_SOURCE="playground/data/eval/pope/llava_pope_test_mini.jsonl"
IMAGE_FOLDER="/home/gpuadmin/txr/CDPruner_data/pope/val2014"
ANNOTATION_DIR="/home/gpuadmin/txr/CDPruner_data/pope/coco"

inventory_status() {
    if [ ! -f "${POPE_SCRIPT}" ] || [ ! -f "${METRIC_SCRIPT}" ] || [ ! -f "${MODEL_ENTRY}" ]; then
        echo "missing_script"
    elif [ ! -f "${QUESTION_FILE}" ] || [ ! -f "${DRY_QUESTION_SOURCE}" ] || \
         [ ! -d "${IMAGE_FOLDER}" ] || [ ! -d "${ANNOTATION_DIR}" ] || [ ! -d "${CKPT_PATH}" ]; then
        echo "missing_data"
    else
        echo "available"
    fi
}

write_inventory() {
    local output="$1" status
    status="$(inventory_status)"
    mkdir -p "$(dirname "${output}")"
    cat > "${output}" <<EOF
benchmark_name: POPE
eval_script: ${POPE_SCRIPT}
model_entry: ${MODEL_ENTRY}
question_file: ${QUESTION_FILE}
dry_run_question_source: ${DRY_QUESTION_SOURCE}
image_folder: ${IMAGE_FOLDER}
annotation_dir: ${ANNOTATION_DIR}
checkpoint: ${CKPT_PATH}
answer_path_pattern: <log_dir>/answers/{pure_qf,eaqf_final}_k64.jsonl
metric_script: ${METRIC_SCRIPT}
dry_run_support: available; balanced mini question file truncated to NUM_SAMPLES
status: ${status}
EOF
}

write_inventory "${BASE_LOG_DIR}/inventory.txt"
[ "${1:-}" = "--inventory-only" ] && exit 0
if [ "$(inventory_status)" != "available" ]; then
    echo "[pope-eaqf] unavailable; see ${BASE_LOG_DIR}/inventory.txt"
    exit 0
fi

mkdir -p "${LOG_DIR}/answers"
write_inventory "${LOG_DIR}/inventory.txt"
MASTER_LOG="${LOG_DIR}/master.log"
STATUS_FILE="${LOG_DIR}/status.tsv"
PID_FILE="${LOG_DIR}/runner.pid"

if [ "${1:-}" != "--worker" ] && [ "${FOREGROUND:-0}" != "1" ]; then
    if [ -f "${PID_FILE}" ]; then
        old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[pope-eaqf] already running: pid=${old_pid}"
            exit 1
        fi
    fi
    printf '\n===== launch %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${MASTER_LOG}"
    nohup bash "${SCRIPT_PATH}" --worker >> "${MASTER_LOG}" 2>&1 < /dev/null &
    pid=$!; echo "${pid}" > "${PID_FILE}"
    echo "[pope-eaqf] started: pid=${pid}; monitor: tail -f ${MASTER_LOG}"
    exit 0
fi

trap 'rm -f "${PID_FILE}"' EXIT INT TERM
source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
conda activate cdpruner

RUN_QUESTION_FILE="${QUESTION_FILE}"
if [ "${DRY_RUN}" = "1" ]; then
    RUN_QUESTION_FILE="${LOG_DIR}/questions_${NUM_SAMPLES}.jsonl"
    head -n "${NUM_SAMPLES}" "${DRY_QUESTION_SOURCE}" > "${RUN_QUESTION_FILE}"
fi

if [ -n "${GPUS:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
else
    GPU_LIST=("${GPU:-auto}")
fi
pick_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -n 1 | cut -d, -f1 | tr -d ' '
}
for index in "${!GPU_LIST[@]}"; do
    GPU_LIST[$index]="${GPU_LIST[$index]//[[:space:]]/}"
    [ "${GPU_LIST[$index]}" = "auto" ] && GPU_LIST[$index]="$(pick_gpu)"
done

run_method() {
    local method="$1" profile="$2" gpu="$3" log="$4" answer="$5"
    echo "[pope-eaqf] start method=${method} profile=${profile} gpu=${gpu}"
    {
        echo "[run_pope_eaqf] method=${method}"
        echo "[run_pope_eaqf] profile=${profile}"
        echo "[run_pope_eaqf] K=${TOKEN}"
        common_env=(
            "TRANSFORMERS_OFFLINE=1" "HF_HUB_OFFLINE=1" "HF_DATASETS_OFFLINE=1"
            "CUDA_VISIBLE_DEVICES=${gpu}" "PRUNE_METHOD=ec_pruner" "EC_SCORE_SOURCE=qfid"
            "EC_QFID_SELECT_MODE=qf" "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "QUESTION_FILE=${RUN_QUESTION_FILE}" "IMAGE_FOLDER=${IMAGE_FOLDER}"
            "ANNOTATION_DIR=${ANNOTATION_DIR}" "ANSWER_FILE=${answer}"
        )
        if [ "${method}" = "pure_qf" ]; then
            env "${common_env[@]}" EC_QFID_PROB_SOURCE=semantic bash "${POPE_SCRIPT}" "${TOKEN}"
        else
            env "${common_env[@]}" EC_QFID_FINAL_PROFILE=1 EC_QFID_PROB_SOURCE=clsmix \
                EC_QFID_CLS_MIX_MODE=linear EC_QFID_CLS_MIX_BETA=0.105 \
                EC_QFID_CLS_ATTN_LAYER=-2 EC_QFID_CLS_HEAD_REDUCE=mean \
                bash "${POPE_SCRIPT}" "${TOKEN}"
        fi
    } > "${log}" 2>&1
}

record_result() {
    local method="$1" profile="$2" code="$3" log="$4" metric status
    metric="$(grep '^Average F1 score:' "${log}" 2>/dev/null | tail -n 1 | awk '{print $4}' || true)"
    if [ "${code}" -eq 0 ] && [ -n "${metric}" ]; then status="completed"; else status="failed"; metric="${metric:-NA}"; fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${method}" "${profile}" "${status}" "${code}" "${metric}" "${log}" >> "${STATUS_FILE}"
    echo "[pope-eaqf] ${status} method=${method} main_metric=${metric}"
}

methods=(pure_qf eaqf_final)
profiles=(qfid_density_depol qfid_density_eaqf_final)
printf 'method\tprofile\tstatus\texit_code\tmain_metric\tlog_path\n' > "${STATUS_FILE}"
declare -a PIDS=() JOB_METHODS=() JOB_PROFILES=() JOB_LOGS=()
parallel=0; [ "${#GPU_LIST[@]}" -ge 2 ] && parallel=1

for index in 0 1; do
    method="${methods[$index]}"; profile="${profiles[$index]}"
    gpu="${GPU_LIST[$((index % ${#GPU_LIST[@]}))]}"
    log="${LOG_DIR}/${method}_k${TOKEN}.log"
    answer="${LOG_DIR}/answers/${method}_k${TOKEN}.jsonl"

    if [ "${FORCE}" != "1" ] && [ -f "${log}" ] && grep -q '^Average F1 score:' "${log}"; then
        record_result "${method}" "${profile}" 0 "${log}"
    elif [ "${parallel}" -eq 1 ]; then
        run_method "${method}" "${profile}" "${gpu}" "${log}" "${answer}" &
        PIDS+=("$!"); JOB_METHODS+=("${method}"); JOB_PROFILES+=("${profile}"); JOB_LOGS+=("${log}")
    else
        run_method "${method}" "${profile}" "${gpu}" "${log}" "${answer}"
        code=$?; record_result "${method}" "${profile}" "${code}" "${log}"
    fi
done

for index in "${!PIDS[@]}"; do
    wait "${PIDS[$index]}"; code=$?
    record_result "${JOB_METHODS[$index]}" "${JOB_PROFILES[$index]}" "${code}" "${JOB_LOGS[$index]}"
done

echo "[pope-eaqf] finished"
column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || cat "${STATUS_FILE}"
