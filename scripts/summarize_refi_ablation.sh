#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

find_log() {
    local candidate
    for candidate in "$@"; do
        if [ -f "${candidate}" ] && grep -q '^Accuracy:' "${candidate}"; then
            echo "${candidate}"
            return
        fi
    done
    echo "NA"
}

metric() {
    local log="$1" pattern="$2" field="$3"
    [ "${log}" = NA ] && { echo NA; return; }
    grep -E "${pattern}" "${log}" | tail -n 1 | awk -v field="${field}" '{print $field}' || true
}

typed_metric() {
    local log="$1" section="$2" label="$3"
    [ "${log}" = NA ] && { echo NA; return; }
    awk -v section="${section}" -v label="${label}" '
        $0 == section {active=1; next}
        active && /^Accuracy \/ / {active=0}
        active && $1 == label ":" {print $2; exit}
    ' "${log}"
}

emit_row() {
    local method="$1" profile="$2" principle="$3" semantic="$4" cls="$5" qf="$6" gate="$7" log="$8"
    local accuracy binary open query rel distribution
    accuracy="$(metric "${log}" '^Accuracy:' 2)"; accuracy="${accuracy:-NA}"
    binary="$(metric "${log}" '^Binary:' 2)"; binary="${binary:-NA}"
    open="$(metric "${log}" '^Open:' 2)"; open="${open:-NA}"
    distribution="$(metric "${log}" '^Distribution:' 2)"; distribution="${distribution:-NA}"
    query="$(typed_metric "${log}" 'Accuracy / structural type:' query)"; query="${query:-NA}"
    rel="$(typed_metric "${log}" 'Accuracy / semantic type:' rel)"; rel="${rel:-NA}"
    printf '| %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | `%s` |\n' \
        "${method}" "${profile}" "${principle}" "${semantic}" "${cls}" "${qf}" "${gate}" \
        "${accuracy}" "${binary}" "${open}" "${query}" "${rel}" "${distribution}" "${log}"
}

semantic_topk="$(find_log logs/refi_ablation_gqa/semantic_topk_k64.log)"
cls_topk="$(find_log logs/refi_ablation_gqa/cls_topk_k64.log logs/final_results/cls_topk_acc54.53.log logs/qfid_cls_studies_k64/cls_topk_k64.log)"
kcenter="$(find_log logs/refi_ablation_gqa/visual_kcenter_k64.log)"
uniform_qf="$(find_log logs/refi_ablation_gqa/uniform_qf_k64.log)"
cls_only_qf="$(find_log logs/refi_ablation_gqa/cls_only_qf_k64.log logs/final_results/cls_only_qf_acc56.62.log logs/qfid_cls_studies_k64/cls_only_qf_k64.log)"
pure_qf="$(find_log logs/refi_ablation_gqa/pure_qf_k64.log logs/qfid_depol_sweep_rerun_20260617/qfid_density_depol0.15.log)"
eaqf="$(find_log logs/refi_ablation_gqa/eaqf_k64.log logs/final_results/eaqf_final_layer_m2_acc58.67.log logs/qfid_cls_studies_k64/clsmix_layer_m2_k64.log)"
refi="$(find_log logs/refi_ablation_gqa/refi_k64.log logs/final_results/CG_EAQF_gate_agreement_k64_acc5871.log logs/qfid_eaqf_gate/eaqf_gate_agreement_k64.log)"

echo '| Method | Profile | Selection principle | Semantic prior | CLS prior | QF residual | Gate | Accuracy | Binary | Open | Query | Rel | Distribution | Log path |'
echo '|---|---|---|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---|'
emit_row 'Semantic Top-K' 'qfid_ablate_semantic_topk' 'semantic probability ranking' yes no no no "${semantic_topk}"
emit_row 'CLS Top-K' 'qfid_ablate_cls_topk' 'CLS attention ranking' no yes no no "${cls_topk}"
emit_row 'Visual K-Center' 'qfid_ablate_visual_kcenter' 'cosine farthest-point diversity' no no no no "${kcenter}"
emit_row 'Uniform QF' 'qfid_ablate_uniform_qf' 'uniform density-state fidelity' no no yes no "${uniform_qf}"
emit_row 'CLS-only QF' 'qfid_ablate_cls_only_qf' 'CLS-conditioned density-state fidelity' no yes yes no "${cls_only_qf}"
emit_row 'Pure QF' 'qfid_density_depol' 'semantic density-state fidelity' yes no yes no "${pure_qf}"
emit_row 'EA-QF' 'qfid_density_eaqf_final' 'fixed semantic/CLS fusion + QF' yes yes yes no "${eaqf}"
emit_row 'ReFi' 'qfid_density_cg_eaqf_final' 'reliability-gated fusion + QF' yes yes yes yes "${refi}"
