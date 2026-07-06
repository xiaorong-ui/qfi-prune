#!/usr/bin/env bash
set -uo pipefail

markdown=0
[ "${1:-}" = "--markdown" ] && markdown=1
dirs=(logs/pope_eaqf logs/pope_eaqf_dryrun)
logs=()
for dir in "${dirs[@]}"; do
    [ -d "${dir}" ] || continue
    while IFS= read -r -d '' log; do logs+=("${log}"); done \
        < <(find "${dir}" -maxdepth 1 -type f -name '*_k64.log' -print0)
done

if [ "${#logs[@]}" -eq 0 ]; then
    echo "warning: no POPE EA-QF logs found" >&2
    exit 0
fi
mapfile -t logs < <(printf '%s\n' "${logs[@]}" | sort -u)

emit_rows() {
    local log
    for log in "${logs[@]}"; do
        awk -v path="${log}" '
            BEGIN { method = profile = "NA"; k = 64; af1 = "NA" }
            /^\[run_pope_eaqf\] method=/ { method=$0; sub(/^.*method=/, "", method) }
            /^\[run_pope_eaqf\] profile=/ { profile=$0; sub(/^.*profile=/, "", profile) }
            /^\[run_pope_eaqf\] K=/ { k=$0; sub(/^.*K=/, "", k) }
            /^Accuracy:/ { sa += $2; ca++ }
            /^Precision:/ { sp += $2; cp++ }
            /^Recall:/ { sr += $2; cr++ }
            /^F1 score:/ { sf += $3; cf++ }
            /^Yes ratio:/ { sy += $3; cy++ }
            /^Average F1 score:/ { af1 = $4 }
            END {
                accuracy = ca ? sa/ca : "NA"
                precision = cp ? sp/cp : "NA"
                recall = cr ? sr/cr : "NA"
                f1 = af1 != "NA" ? af1 : (cf ? sf/cf : "NA")
                yes_ratio = cy ? sy/cy : "NA"
                printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
                    method, profile, k, accuracy, precision, recall, f1, yes_ratio, path
            }
        ' "${log}"
    done
}

if [ "${markdown}" -eq 0 ]; then
    printf 'method\tprofile\tK\taccuracy\tprecision\trecall\tf1\tyes_ratio\tlog_path\n'
    emit_rows
    exit 0
fi

echo '| Method | Profile | K | Accuracy | Precision | Recall | F1 | Yes Ratio | Log |'
echo '|---|---|---:|---:|---:|---:|---:|---:|---|'
while IFS=$'\t' read -r method profile k accuracy precision recall f1 yes_ratio log; do
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "${method}" "${profile}" "${k}" "${accuracy}" "${precision}" \
        "${recall}" "${f1}" "${yes_ratio}" "${log}"
done < <(emit_rows)
