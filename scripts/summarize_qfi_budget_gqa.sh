#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-logs/qfi_budget_gqa}"
MAPPING_FILE="${MAPPING_FILE:-${LOG_DIR}/k64_existing_logs.tsv}"
TSV_PATH="${LOG_DIR}/qfi_budget_gqa_results.tsv"
MD_PATH="${LOG_DIR}/qfi_budget_gqa_results.md"
PROFILES_VALUE="${PROFILES:-qfid_density_depol qfid_density_eaqf_final qfid_density_cg_eaqf_final}"
KS_VALUE="${KS:-32 64 96 128}"

mkdir -p "${LOG_DIR}"
read -r -a PROFILE_LIST <<< "${PROFILES_VALUE}"
read -r -a K_LIST <<< "${KS_VALUE}"

declare -A MAPPED_LOGS
if [ -f "${MAPPING_FILE}" ]; then
    while IFS=$'\t' read -r method profile keep_num log_path; do
        [ "${profile}" = "profile" ] && continue
        [ -z "${profile}" ] && continue
        MAPPED_LOGS["${profile}:${keep_num}"]="${log_path}"
    done < "${MAPPING_FILE}"
fi

method_name() {
    case "$1" in
        qfid_density_depol) echo "Pure QF" ;;
        qfid_density_eaqf_final) echo "EA-QF" ;;
        qfid_density_cg_eaqf_final) echo "QFi-Pruner" ;;
        *) echo "$1" ;;
    esac
}

metric() {
    local log_path="$1" label="$2"
    if [ ! -f "${log_path}" ]; then
        echo "NA"
        return
    fi
    awk -v label="${label}:" '$1 == label {value=$2} END {print value == "" ? "NA" : value}' "${log_path}"
}

typed_metric() {
    local log_path="$1" section="$2" label="$3"
    if [ ! -f "${log_path}" ]; then
        echo "NA"
        return
    fi
    awk -v section="${section}" -v label="${label}:" '
        $0 == section {active=1; next}
        active && /^Accuracy \/ / {active=0}
        active && $1 == label {value=$2; active=0}
        END {print value == "" ? "NA" : value}
    ' "${log_path}"
}

status_for() {
    local log_path="$1" pid_path="$2"
    if [ -f "${log_path}" ] && grep -q '^Accuracy:' "${log_path}" && grep -q '100%' "${log_path}"; then
        echo "completed"
    elif [ -f "${log_path}" ] && grep -q '\[gqa-smoke\] completed' "${log_path}"; then
        echo "dry_complete"
    elif [ -f "${pid_path}" ] && kill -0 "$(cat "${pid_path}")" 2>/dev/null; then
        echo "running"
    elif [ -f "${log_path}" ]; then
        echo "incomplete"
    else
        echo "missing"
    fi
}

printf 'Method\tProfile\tK\tAccuracy\tBinary\tOpen\tQuery\tRel\tDistribution\tLog path\tStatus\n' > "${TSV_PATH}"

for profile in "${PROFILE_LIST[@]}"; do
    for keep_num in "${K_LIST[@]}"; do
        default_log="${LOG_DIR}/${profile}_k${keep_num}.log"
        pid_path="${LOG_DIR}/${profile}_k${keep_num}.pid"
        log_path="${default_log}"
        status="$(status_for "${default_log}" "${pid_path}")"

        mapping_key="${profile}:${keep_num}"
        if [ "${status}" != "completed" ] && [ -n "${MAPPED_LOGS[${mapping_key}]:-}" ]; then
            mapped_path="${MAPPED_LOGS[${mapping_key}]}"
            mapped_status="$(status_for "${mapped_path}" /dev/null)"
            if [ "${mapped_status}" = "completed" ]; then
                log_path="${mapped_path}"
                status="completed"
            fi
        fi

        accuracy="$(metric "${log_path}" Accuracy)"
        binary="$(metric "${log_path}" Binary)"
        open_metric="$(metric "${log_path}" Open)"
        query="$(typed_metric "${log_path}" 'Accuracy / structural type:' query)"
        rel="$(typed_metric "${log_path}" 'Accuracy / semantic type:' rel)"
        distribution="$(metric "${log_path}" Distribution)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$(method_name "${profile}")" "${profile}" "${keep_num}" "${accuracy}" "${binary}" \
            "${open_metric}" "${query}" "${rel}" "${distribution}" "${log_path}" "${status}" \
            >> "${TSV_PATH}"
    done
done

{
    echo '| Method | Profile | K | Accuracy | Binary | Open | Query | Rel | Distribution | Status | Log path |'
    echo '|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|'
    tail -n +2 "${TSV_PATH}" | while IFS=$'\t' read -r method profile keep_num accuracy binary open_metric query rel distribution log_path status; do
        printf '| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |\n' \
            "${method}" "${profile}" "${keep_num}" "${accuracy}" "${binary}" "${open_metric}" \
            "${query}" "${rel}" "${distribution}" "${status}" "${log_path}"
    done
} > "${MD_PATH}"

echo "Wrote ${TSV_PATH}"
echo "Wrote ${MD_PATH}"

