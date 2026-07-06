#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TOKEN="${TOKEN:-64}"
GPU_ARG="${GPU:-auto}"
LOG_DIR="${LOG_DIR:-logs/qfid_cls_studies_k${TOKEN}}"
MASTER_LOG="${LOG_DIR}/master.log"
PID_FILE="${LOG_DIR}/runner.pid"
STATUS_FILE="${LOG_DIR}/status.tsv"
FORCE="${FORCE:-0}"

mkdir -p "${LOG_DIR}"

if [ "${1:-}" != "--worker" ]; then
    if [ -f "${PID_FILE}" ]; then
        old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
        if [ -n "${old_pid}" ] && kill -0 "${old_pid}" 2>/dev/null; then
            echo "[qfid-cls-studies] already running: pid=${old_pid}"
            echo "[qfid-cls-studies] monitor: tail -f ${MASTER_LOG}"
            exit 1
        fi
    fi

    printf '\n===== launch %s =====\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "${MASTER_LOG}"
    nohup bash "$0" --worker >> "${MASTER_LOG}" 2>&1 < /dev/null &
    runner_pid=$!
    echo "${runner_pid}" > "${PID_FILE}"
    echo "[qfid-cls-studies] started in background: pid=${runner_pid}"
    echo "[qfid-cls-studies] master log: ${SCRIPT_DIR}/${MASTER_LOG}"
    echo "[qfid-cls-studies] monitor: tail -f ${MASTER_LOG}"
    exit 0
fi

cleanup() {
    rm -f "${PID_FILE}"
}
trap cleanup EXIT INT TERM

source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
conda activate cdpruner

profiles=(
    qfid_cls_topk
    qfid_density_cls_only_qf
    qfid_density_clsmix_layer_m2
    qfid_density_clsmix_layer_m4
    qfid_density_clsmix_layer_last4mean
    qfid_density_clsmix_head_entropy
)

labels=(
    cls_topk
    cls_only_qf
    clsmix_layer_m2
    clsmix_layer_m4
    clsmix_layer_last4mean
    clsmix_head_entropy
)

printf 'profile\tstatus\texit_code\taccuracy\tlog\n' > "${STATUS_FILE}"
echo "[qfid-cls-studies] worker pid=$$ token=${TOKEN} gpu=${GPU_ARG} force=${FORCE}"
echo "[qfid-cls-studies] order: defensive ablations -> layer sweep -> head entropy"

for index in "${!profiles[@]}"; do
    profile="${profiles[$index]}"
    label="${labels[$index]}"
    log_file="${LOG_DIR}/${label}_k${TOKEN}.log"

    if [ "${FORCE}" != "1" ] && [ -f "${log_file}" ] && grep -q '^Accuracy:' "${log_file}"; then
        accuracy="$(grep '^Accuracy:' "${log_file}" | tail -n 1 | awk '{print $2}')"
        echo "[qfid-cls-studies] skip completed profile=${profile} accuracy=${accuracy}"
        printf '%s\tskipped\t0\t%s\t%s\n' "${profile}" "${accuracy}" "${log_file}" >> "${STATUS_FILE}"
        continue
    fi

    echo "[qfid-cls-studies] start profile=${profile} at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    EC_QFID_DEBUG=1 \
    EC_DEBUG_LIMIT=6 \
    bash run_gqa_profile.sh "${profile}" "${GPU_ARG}" "${TOKEN}" > "${log_file}" 2>&1
    exit_code=$?

    accuracy=""
    if [ -f "${log_file}" ]; then
        accuracy="$(grep '^Accuracy:' "${log_file}" | tail -n 1 | awk '{print $2}' || true)"
    fi

    if [ "${exit_code}" -eq 0 ] && [ -n "${accuracy}" ]; then
        status="completed"
        echo "[qfid-cls-studies] completed profile=${profile} accuracy=${accuracy}"
    else
        status="failed"
        echo "[qfid-cls-studies] failed profile=${profile} exit=${exit_code}; continuing"
        tail -n 12 "${log_file}" 2>/dev/null || true
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${profile}" "${status}" "${exit_code}" "${accuracy:-NA}" "${log_file}" >> "${STATUS_FILE}"
done

echo "[qfid-cls-studies] all scheduled experiments finished at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "[qfid-cls-studies] summary:"
column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || cat "${STATUS_FILE}"

for log_file in "${LOG_DIR}"/*_k"${TOKEN}".log; do
    [ -f "${log_file}" ] || continue
    echo "===== ${log_file} ====="
    grep -E '^(Binary|Open|Accuracy|Distribution):|^  query:|^  rel:' "${log_file}" || true
done
