#!/usr/bin/env bash
set -uo pipefail
markdown=0; [ "${1:-}" = --markdown ] && markdown=1
files=()
for f in logs/cdpruner_table1_eaqf/status.tsv logs/cdpruner_table1_eaqf_dryrun/status.tsv; do
    [ -f "${f}" ] && files+=("${f}")
done
if [ "${#files[@]}" -eq 0 ]; then echo 'warning: no Table-1 status files found' >&2; exit 0; fi

declare -A pure
rows=()
for file in "${files[@]}"; do
    while IFS=$'\t' read -r benchmark method profile status code metric_name metric_value cdscore delta log; do
        [ "${benchmark}" = benchmark ] && continue
        rows+=("${file}"$'\t'"${benchmark}"$'\t'"${method}"$'\t'"${profile}"$'\t'"${metric_name}"$'\t'"${metric_value}"$'\t'"${cdscore}"$'\t'"${delta}"$'\t'"${log}")
        [ "${method}" = pure_qf ] && pure["${file}:${benchmark}"]="${metric_value}"
    done < "${file}"
done

emit() {
    local row file benchmark method profile metric_name metric_value cdscore delta log p d
    for row in "${rows[@]}"; do
        IFS=$'\t' read -r file benchmark method profile metric_name metric_value cdscore delta log <<< "${row}"
        p="${pure[${file}:${benchmark}]:-NA}"; d=NA
        if [ "${method}" != pure_qf ] && [[ "${metric_value}" =~ ^[0-9.]+%?$ ]] && [[ "${p}" =~ ^[0-9.]+%?$ ]]; then
            d="$(awk -v a="${metric_value%%%}" -v b="${p%%%}" 'BEGIN{printf "%.4f",a-b}')"
        fi
        printf '%s\t%s\t%s\t64\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${benchmark}" "${method}" "${profile}" "${metric_name}" "${metric_value}" "${cdscore}" "${delta}" "${p}" "${d}" "${log}"
    done
}

if [ "${markdown}" -eq 0 ]; then
    printf 'benchmark\tmethod\tprofile\tK\tmain_metric_name\tmain_metric_value\tcdpruner_k64_score\tdelta_vs_cdpruner\tpure_qf_value\tdelta_vs_pure_qf\tlog_path\n'
    emit; exit 0
fi
echo '| Benchmark | Method | Profile | K | Metric | Value | CDPruner K64 | Delta CDPruner | Pure QF | Delta vs Pure | Log |'
echo '|---|---|---|---:|---|---:|---:|---:|---:|---:|---|'
while IFS=$'\t' read -r b m p k mn mv cd d pq dd log; do
    printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' "$b" "$m" "$p" "$k" "$mn" "$mv" "$cd" "$d" "$pq" "$dd" "$log"
done < <(emit)
