#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${LOG_DIR:-logs/qfi_budget_calib_gqa}"
BASELINE_TSV="${BASELINE_TSV:-logs/qfi_budget_gqa/qfi_budget_gqa_results.tsv}"
TSV_PATH="${LOG_DIR}/qfi_budget_calib_results.tsv"
MD_PATH="${LOG_DIR}/qfi_budget_calib_results.md"
PROFILES_VALUE="${PROFILES:-qfid_density_cg_eaqf_budget_alpha2 qfid_density_cg_eaqf_budget_alpha3 qfid_density_cg_eaqf_budget_alpha4}"
KS_VALUE="${KS:-32 64 128}"

mkdir -p "${LOG_DIR}"
read -r -a PROFILE_LIST <<< "${PROFILES_VALUE}"
read -r -a K_LIST <<< "${KS_VALUE}"

metric() {
    local log_path="$1" label="$2"
    [ -f "${log_path}" ] || { echo "NA"; return; }
    awk -v label="${label}:" '$1 == label {value=$2} END {print value == "" ? "NA" : value}' "${log_path}"
}

typed_metric() {
    local log_path="$1" section="$2" label="$3"
    [ -f "${log_path}" ] || { echo "NA"; return; }
    awk -v section="${section}" -v label="${label}:" '
        $0 == section {active=1; next}
        active && /^Accuracy \/ / {active=0}
        active && $1 == label {value=$2; active=0}
        END {print value == "" ? "NA" : value}
    ' "${log_path}"
}

status_for() {
    local log_path="$1"
    if [ -f "${log_path}" ] && grep -q '^Accuracy:' "${log_path}" && grep -q '12578/12578' "${log_path}"; then
        echo "completed"
    elif [ -f "${log_path}" ] && grep -q '^Accuracy:' "${log_path}" && grep -q '\[gqa-smoke\] completed' "${log_path}"; then
        echo "dry_complete"
    elif [ -f "${log_path}" ] && grep -q '^Accuracy:' "${log_path}"; then
        echo "completed"
    elif [ -f "${log_path}" ]; then
        echo "incomplete"
    else
        echo "missing"
    fi
}

alpha_for() {
    case "$1" in
        *_alpha2) echo "2.0" ;;
        *_alpha3) echo "3.0" ;;
        *_alpha4) echo "4.0" ;;
        *) echo "NA" ;;
    esac
}

write_row() {
    local method="$1" profile="$2" alpha="$3" keep_num="$4" log_path="$5" status="$6"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${method}" "${profile}" "${alpha}" "${keep_num}" \
        "$(metric "${log_path}" Accuracy)" "$(metric "${log_path}" Binary)" \
        "$(metric "${log_path}" Open)" "$(typed_metric "${log_path}" 'Accuracy / structural type:' query)" \
        "$(typed_metric "${log_path}" 'Accuracy / semantic type:' rel)" \
        "$(metric "${log_path}" Distribution)" "${status}" "${log_path}" >> "${TSV_PATH}"
}

printf 'Method\tProfile\tAlpha\tK\tAccuracy\tBinary\tOpen\tQuery\tRel\tDistribution\tStatus\tLog path\n' > "${TSV_PATH}"

for keep_num in "${K_LIST[@]}"; do
    baseline_found=0
    if [ -f "${BASELINE_TSV}" ]; then
        while IFS=$'\t' read -r method profile row_k accuracy binary open_metric query rel distribution log_path status; do
            if [ "${profile}" = "qfid_density_cg_eaqf_final" ] && [ "${row_k}" = "${keep_num}" ]; then
                write_row "QFi baseline" "${profile}" "NA" "${keep_num}" "${log_path}" "$(status_for "${log_path}")"
                baseline_found=1
            fi
        done < "${BASELINE_TSV}"
    fi
    if [ "${baseline_found}" -eq 0 ]; then
        printf 'QFi baseline\tqfid_density_cg_eaqf_final\tNA\t%s\tNA\tNA\tNA\tNA\tNA\tNA\tmissing\t%s\n' \
            "${keep_num}" "${BASELINE_TSV}" >> "${TSV_PATH}"
    fi
done

for profile in "${PROFILE_LIST[@]}"; do
    alpha="$(alpha_for "${profile}")"
    for keep_num in "${K_LIST[@]}"; do
        log_path="${LOG_DIR}/${profile}_k${keep_num}.log"
        write_row "Budget alpha=${alpha}" "${profile}" "${alpha}" "${keep_num}" "${log_path}" "$(status_for "${log_path}")"
    done
done

{
    echo '| Method | Profile | Alpha | K | Accuracy | Binary | Open | Query | Rel | Distribution | Status | Log path |'
    echo '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|'
    tail -n +2 "${TSV_PATH}" | while IFS=$'\t' read -r method profile alpha keep_num accuracy binary open_metric query rel distribution status log_path; do
        printf '| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |\n' \
            "${method}" "${profile}" "${alpha}" "${keep_num}" "${accuracy}" "${binary}" \
            "${open_metric}" "${query}" "${rel}" "${distribution}" "${status}" "${log_path}"
    done
} > "${MD_PATH}"

echo "Wrote ${TSV_PATH}"
echo "Wrote ${MD_PATH}"
