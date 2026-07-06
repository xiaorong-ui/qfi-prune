#!/usr/bin/env bash
set -uo pipefail

markdown=0
inputs=()
for arg in "$@"; do
    if [ "${arg}" = "--markdown" ]; then
        markdown=1
    else
        inputs+=("${arg}")
    fi
done

if [ "${#inputs[@]}" -eq 0 ]; then
    inputs=(
        logs/final_results
        logs/qfid_clsmix_beta_micro
        logs/qfid_clsmix_layer_sweep
        logs/qfid_defense_ablation
        logs/qfid_clsmix_geometric
        logs/qfid_clsmix_tau_depol_refine
        logs/qfid_cls_studies_k64
    )
fi

logs=()
for input in "${inputs[@]}"; do
    if [ -f "${input}" ]; then
        if [[ "${input}" = *.log ]]; then
            logs+=("${input}")
        else
            echo "warning: ignoring non-log file: ${input}" >&2
        fi
    elif [ -d "${input}" ]; then
        while IFS= read -r -d '' log; do
            logs+=("${log}")
        done < <(find "${input}" -type f -name '*.log' -print0)
    else
        echo "warning: path not found: ${input}" >&2
    fi
done

if [ "${#logs[@]}" -eq 0 ]; then
    echo "warning: no log files found" >&2
    exit 0
fi
mapfile -t logs < <(printf '%s\n' "${logs[@]}" | sort -u)

emit_rows() {
    local log
    for log in "${logs[@]}"; do
        awk -v path="${log}" '
            BEGIN {
                profile = experiment = binary = open = accuracy = distribution = query = rel = "NA"
            }
            /^\[run_gqa_profile\] profile=/ {
                profile = $0
                sub(/^\[run_gqa_profile\] profile=/, "", profile)
            }
            /^\[experiment\][[:space:]]*/ {
                experiment = $0
                sub(/^\[experiment\][[:space:]]*/, "", experiment)
            }
            /^Binary:/ { binary = $2 }
            /^Open:/ { open = $2 }
            /^Accuracy:/ { accuracy = $2 }
            /^Distribution:/ { distribution = $2 }
            /^[[:space:]]+query:/ { query = $2 }
            /^[[:space:]]+rel:/ { rel = $2 }
            END {
                gsub(/[\t\r\n]/, " ", path)
                gsub(/[\t\r\n]/, " ", profile)
                gsub(/[\t\r\n]/, " ", experiment)
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                    path, profile, experiment, binary, open, accuracy, distribution, query, rel
            }
        ' "${log}"
    done
}

if [ "${markdown}" -eq 0 ]; then
    printf 'log_path\tprofile\texperiment_line\tBinary\tOpen\tAccuracy\tDistribution\tquery\trel\n'
    emit_rows
    exit 0
fi

echo '| Method/Profile | Accuracy | Binary | Open | Query | Rel | Distribution | Log |'
echo '|---|---:|---:|---:|---:|---:|---:|---|'
while IFS=$'\t' read -r log profile experiment binary open accuracy distribution query rel; do
    method="${profile}"
    if [ "${method}" = "NA" ]; then
        method="${experiment}"
    fi
    if [ "${method}" = "NA" ]; then
        method="$(basename "${log}" .log)"
    fi
    method="${method//|/\\|}"
    log="${log//|/\\|}"
    printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "${method}" "${accuracy}" "${binary}" "${open}" "${query}" "${rel}" "${distribution}" "${log}"
done < <(emit_rows)
