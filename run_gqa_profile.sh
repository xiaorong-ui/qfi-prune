#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PROFILE="${1:-qfid}"
GPU_ARG="${2:-auto}"
TOKEN="${3:-64}"
ACTION="${4:-run}"

pick_gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0"
        return
    fi

    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr \
        | head -n 1 \
        | cut -d, -f1 \
        | tr -d ' '
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU="${CUDA_VISIBLE_DEVICES}"
elif [ "${GPU_ARG}" = "auto" ]; then
    GPU="$(pick_gpu)"
else
    GPU="${GPU_ARG}"
fi

TRANSFORMERS_OFFLINE_DEFAULT="${TRANSFORMERS_OFFLINE:-1}"
HF_HUB_OFFLINE_DEFAULT="${HF_HUB_OFFLINE:-1}"
HF_DATASETS_OFFLINE_DEFAULT="${HF_DATASETS_OFFLINE:-1}"
PYTORCH_CUDA_ALLOC_CONF_DEFAULT="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

COMMON_ENV=(
    "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE_DEFAULT}"
    "HF_HUB_OFFLINE=${HF_HUB_OFFLINE_DEFAULT}"
    "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE_DEFAULT}"
    "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_DEFAULT}"
    "PRUNE_METHOD=${PRUNE_METHOD:-ec_pruner}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
)

PROFILE_ENV=()

case "${PROFILE}" in
    qfid)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-amplitude}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_uniform)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-uniform}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-amplitude}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_depol)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_clsmix005)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-clsmix}"
            "EC_QFID_CLS_MIX_MODE=${EC_QFID_CLS_MIX_MODE:-linear}"
            "EC_QFID_CLS_MIX_BETA=${EC_QFID_CLS_MIX_BETA:-0.05}"
            "EC_QFID_CLS_ATTN_LAYER=${EC_QFID_CLS_ATTN_LAYER:-last}"
            "EC_QFID_CLS_HEAD_REDUCE=${EC_QFID_CLS_HEAD_REDUCE:-mean}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-0}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_clsmix010)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-clsmix}"
            "EC_QFID_CLS_MIX_MODE=${EC_QFID_CLS_MIX_MODE:-linear}"
            "EC_QFID_CLS_MIX_BETA=${EC_QFID_CLS_MIX_BETA:-0.10}"
            "EC_QFID_CLS_ATTN_LAYER=${EC_QFID_CLS_ATTN_LAYER:-last}"
            "EC_QFID_CLS_HEAD_REDUCE=${EC_QFID_CLS_HEAD_REDUCE:-mean}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-0}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_clsmix020)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-clsmix}"
            "EC_QFID_CLS_MIX_MODE=${EC_QFID_CLS_MIX_MODE:-linear}"
            "EC_QFID_CLS_MIX_BETA=${EC_QFID_CLS_MIX_BETA:-0.20}"
            "EC_QFID_CLS_ATTN_LAYER=${EC_QFID_CLS_ATTN_LAYER:-last}"
            "EC_QFID_CLS_HEAD_REDUCE=${EC_QFID_CLS_HEAD_REDUCE:-mean}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-0}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_cls_only_qf|qfid_ablate_cls_only_qf)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}" "EC_QFID_SELECT_MODE=${EC_QFID_SELECT_MODE:-qf}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-cls_only_qf}"
            "EC_QFID_CLS_ATTN_LAYER=${EC_QFID_CLS_ATTN_LAYER:-last}"
            "EC_QFID_CLS_HEAD_REDUCE=${EC_QFID_CLS_HEAD_REDUCE:-mean}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}" "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_DEPOLARIZE_MODE=${EC_QFID_DEPOLARIZE_MODE:-fixed}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_cls_topk|qfid_ablate_cls_topk)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=cls_topk" "EC_QFID_PROB_SOURCE=cls_only_qf"
            "EC_QFID_CLS_ATTN_LAYER=${EC_QFID_CLS_ATTN_LAYER:-last}"
            "EC_QFID_CLS_HEAD_REDUCE=${EC_QFID_CLS_HEAD_REDUCE:-mean}"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_ablate_semantic_topk)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=semantic_topk" "EC_QFID_PROB_SOURCE=semantic"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_QFID_CLS_GATE=0" "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_ablate_uniform_qf)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf" "EC_QFID_PROB_SOURCE=uniform"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_QFID_CLS_GATE=0" "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_ablate_visual_kcenter)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=visual_kcenter" "EC_QFID_PROB_SOURCE=uniform"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_QFID_CLS_GATE=0" "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_clsmix_layer_m2)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf" "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0" "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_eaqf_final)
        PROFILE_ENV=(
            "EC_QFID_FINAL_PROFILE=1"
            "EC_QFID_GATE_PROFILE=0"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_GATE=0"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=qfi_residual"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfi_evidence_recover_final)
        PROFILE_ENV=(
            "EC_QFID_EVIDENCE_PROFILE=1"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=evidence_recover"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_eaqf_gate_agreement)
        PROFILE_ENV=(
            "EC_QFID_GATE_PROFILE=1"
            "EC_QFID_FINAL_PROFILE=0"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_cg_eaqf_final)
        PROFILE_ENV=(
            "EC_QFID_CG_FINAL_PROFILE=1"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=qfi_residual"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfi_core_recover_r050|qfi_core_recover_r075|qfi_core_recover_r08125|qfi_core_recover_r0875)
        case "${PROFILE}" in
            qfi_core_recover_r050) CORE_RATIO="0.5"; CORE_PROFILE="r050" ;;
            qfi_core_recover_r075) CORE_RATIO="0.75"; CORE_PROFILE="r075" ;;
            qfi_core_recover_r08125) CORE_RATIO="0.8125"; CORE_PROFILE="r08125" ;;
            qfi_core_recover_r0875) CORE_RATIO="0.875"; CORE_PROFILE="r0875" ;;
        esac
        PROFILE_ENV=(
            "EC_QFID_CORE_RECOVER_PROFILE=${CORE_PROFILE}"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=core_then_recover" "EC_QFID_CORE_RATIO=${CORE_RATIO}"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfi_adaptive_recover_budget|qfi_adaptive_recover_prior|qfi_adaptive_recover_final)
        case "${PROFILE}" in
            qfi_adaptive_recover_budget)
                ADAPT_PROFILE="budget"
                PRIOR_ANCHOR="0"
                CAND_POOL="0"
                ;;
            qfi_adaptive_recover_prior)
                ADAPT_PROFILE="prior"
                PRIOR_ANCHOR="1"
                CAND_POOL="0"
                ;;
            qfi_adaptive_recover_final)
                ADAPT_PROFILE="final"
                PRIOR_ANCHOR="1"
                CAND_POOL="1"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=${ADAPT_PROFILE}"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=adaptive_core_recover"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=${PRIOR_ANCHOR}"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=${CAND_POOL}"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qficr_ablate_full|qficr_ablate_only_reduction|qficr_ablate_random_restoration|qficr_ablate_fixed_ratio|qficr_ablate_no_spatial_local_prior|qficr_ablate_no_spatial|qficr_ablate_no_local|qficr_ablate_sem_only|qficr_ablate_cls_only|qficr_ablate_fixed_fusion)
        QFICR_ABLATION="${PROFILE#qficr_ablate_}"
        QFICR_RESTORATION_MODE="full"
        QFICR_FIXED_RATIO="0.2"
        QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
        QFICR_PRIOR_ABLATION="full"
        QFICR_OBSERVATION_MODE="full"
        QFICR_RANDOM_SEED="${EC_QFICR_RANDOM_SEED:-42}"
        case "${PROFILE}" in
            qficr_ablate_only_reduction)
                QFICR_RESTORATION_MODE="none"
                ;;
            qficr_ablate_random_restoration)
                QFICR_RESTORATION_MODE="random"
                ;;
            qficr_ablate_fixed_ratio)
                QFICR_RESTORATION_MODE="fixed_ratio"
                QFICR_FIXED_RATIO="0.2"
                ;;
            qficr_ablate_no_spatial_local_prior)
                QFICR_USE_SPATIAL_LOCAL_PRIOR="0"
                QFICR_PRIOR_ABLATION="none"
                ;;
            qficr_ablate_no_spatial)
                QFICR_PRIOR_ABLATION="no_spatial"
                ;;
            qficr_ablate_no_local)
                QFICR_PRIOR_ABLATION="no_local"
                ;;
            qficr_ablate_sem_only)
                QFICR_OBSERVATION_MODE="sem_only"
                ;;
            qficr_ablate_cls_only)
                QFICR_OBSERVATION_MODE="cls_only"
                ;;
            qficr_ablate_fixed_fusion)
                QFICR_OBSERVATION_MODE="fixed_fusion"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QFICR_ABLATION_PROFILE=${QFICR_ABLATION}"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_OVERLAP_KERNEL=${EC_QFID_OVERLAP_KERNEL:-relu_square}"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=adaptive_core_recover"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFICR_RESTORATION_MODE=${QFICR_RESTORATION_MODE}"
            "EC_QFICR_FIXED_RESTORATION_RATIO=${QFICR_FIXED_RATIO}"
            "EC_QFICR_USE_SPATIAL_LOCAL_PRIOR=${QFICR_USE_SPATIAL_LOCAL_PRIOR}"
            "EC_QFICR_PRIOR_ABLATION=${QFICR_PRIOR_ABLATION}"
            "EC_QFICR_OBSERVATION_MODE=${QFICR_OBSERVATION_MODE}"
            "EC_QFICR_RANDOM_SEED=${QFICR_RANDOM_SEED}"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qficr_tune_baseline|qficr_tune_alpha_050|qficr_tune_alpha_075|qficr_tune_alpha_125|qficr_tune_prior_lam_010|qficr_tune_prior_lam_025|qficr_tune_prior_lam_050|qficr_tune_alpha050_prior010|qficr_tune_alpha075_prior010|qficr_tune_rho_max_down|qficr_tune_rho_max_up|qficr_tune_cap_down|qficr_tune_cap_up|qficr_tune_debug_prior_stats|qficr_ksens_stable|qficr_ksens_only_reduction|qficr_micro_baseline|qficr_micro_beta_075|qficr_micro_beta_125|qficr_micro_beta_150|qficr_micro_tau_085|qficr_micro_tau_115|qficr_micro_tau_130|qficr_micro_div_010|qficr_micro_yn_gate_05)
        if [[ "${PROFILE}" == qficr_ksens_* ]]; then
            QFICR_TUNING="ksens_${PROFILE#qficr_ksens_}"
        elif [[ "${PROFILE}" == qficr_micro_* ]]; then
            QFICR_TUNING="micro_${PROFILE#qficr_micro_}"
        else
            QFICR_TUNING="${PROFILE#qficr_tune_}"
        fi
        QFICR_ANCHOR_ALPHA="0.5"
        QFICR_PRIOR_LAMBDA="0.0"
        QFICR_USE_SPATIAL_LOCAL_PRIOR="0"
        QFICR_PRIOR_ABLATION="none"
        QFICR_RHO_MAX="0.25"
        QFICR_CAP="16"
        QFICR_RESTORATION_MODE="full"
        QFICR_DEBUG_PRIOR_JSONL=""
        QFICR_REST_BETA="1.0"
        QFICR_OBS_TEMPERATURE="1.0"
        QFICR_RESTORE_DIV_LAMBDA="0.0"
        QFICR_RESTORE_CANDIDATE_FACTOR="2.0"
        QFICR_YN_BUDGET_GATE="1.0"
        case "${PROFILE}" in
            qficr_ksens_only_reduction)
                QFICR_RESTORATION_MODE="none"
                ;;
            qficr_micro_beta_075)
                QFICR_REST_BETA="0.75"
                ;;
            qficr_micro_beta_125)
                QFICR_REST_BETA="1.25"
                ;;
            qficr_micro_beta_150)
                QFICR_REST_BETA="1.50"
                ;;
            qficr_micro_tau_085)
                QFICR_OBS_TEMPERATURE="0.85"
                ;;
            qficr_micro_tau_115)
                QFICR_OBS_TEMPERATURE="1.15"
                ;;
            qficr_micro_tau_130)
                QFICR_OBS_TEMPERATURE="1.30"
                ;;
            qficr_micro_div_010)
                QFICR_RESTORE_DIV_LAMBDA="0.10"
                QFICR_RESTORE_CANDIDATE_FACTOR="2.0"
                ;;
            qficr_micro_yn_gate_05)
                QFICR_YN_BUDGET_GATE="0.5"
                ;;
            qficr_tune_alpha_075)
                QFICR_ANCHOR_ALPHA="0.75"
                ;;
            qficr_tune_alpha_125)
                QFICR_ANCHOR_ALPHA="1.25"
                ;;
            qficr_tune_prior_lam_010)
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="0.10"
                ;;
            qficr_tune_prior_lam_025)
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="0.25"
                ;;
            qficr_tune_prior_lam_050)
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="0.50"
                ;;
            qficr_tune_alpha050_prior010)
                QFICR_ANCHOR_ALPHA="0.5"
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="0.10"
                ;;
            qficr_tune_alpha075_prior010)
                QFICR_ANCHOR_ALPHA="0.75"
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="0.10"
                ;;
            qficr_tune_rho_max_down)
                QFICR_RHO_MAX="0.20"
                ;;
            qficr_tune_rho_max_up)
                QFICR_RHO_MAX="0.30"
                ;;
            qficr_tune_cap_down)
                QFICR_CAP="12"
                ;;
            qficr_tune_cap_up)
                QFICR_CAP="20"
                ;;
            qficr_tune_debug_prior_stats)
                QFICR_USE_SPATIAL_LOCAL_PRIOR="1"
                QFICR_PRIOR_ABLATION="full"
                QFICR_PRIOR_LAMBDA="1.0"
                QFICR_DEBUG_PRIOR_JSONL="outputs/qficr_tuning/debug_prior_stats.jsonl"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QFICR_TUNING_PROFILE=${QFICR_TUNING}"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_OVERLAP_KERNEL=${EC_QFID_OVERLAP_KERNEL:-relu_square}"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=adaptive_core_recover"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=${QFICR_RHO_MAX}"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=${QFICR_CAP}"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFICR_ANCHOR_ALPHA=${QFICR_ANCHOR_ALPHA}"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFICR_RESTORATION_MODE=${QFICR_RESTORATION_MODE}"
            "EC_QFICR_FIXED_RESTORATION_RATIO=0.2"
            "EC_QFICR_USE_SPATIAL_LOCAL_PRIOR=${QFICR_USE_SPATIAL_LOCAL_PRIOR}"
            "EC_QFICR_PRIOR_ABLATION=${QFICR_PRIOR_ABLATION}"
            "EC_QFICR_PRIOR_LAMBDA=${QFICR_PRIOR_LAMBDA}"
            "EC_QFICR_OBSERVATION_MODE=full"
            "EC_QFICR_REST_BETA=${QFICR_REST_BETA}"
            "EC_QFICR_OBS_TEMPERATURE=${QFICR_OBS_TEMPERATURE}"
            "EC_QFICR_RESTORE_DIV_LAMBDA=${QFICR_RESTORE_DIV_LAMBDA}"
            "EC_QFICR_RESTORE_CANDIDATE_FACTOR=${QFICR_RESTORE_CANDIDATE_FACTOR}"
            "EC_QFICR_YN_BUDGET_GATE=${QFICR_YN_BUDGET_GATE}"
            "EC_QFICR_RANDOM_SEED=${EC_QFICR_RANDOM_SEED:-42}"
            "EC_QFICR_DEBUG_PRIOR_STATS_JSONL=${QFICR_DEBUG_PRIOR_JSONL}"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfi_zeno_stabilized|qfi_zeno_stabilized_b005|qfi_zeno_stabilized_b015|qfi_spatial_buffer_recovery|qfi_spatial_buffer_recovery_b08|qfi_spatial_buffer_recovery_b16|qfi_progressive_128_to_64|qfi_progressive_192_to_64|qfi_coarse_to_fine_pruning|qfi_coarse_to_fine_256)
        QFI_CARD_PROFILE="${PROFILE}"
        QFI_CARD_SELECTOR="${PROFILE}"
        QFI_ZENO_BETA="0.10"
        QFI_BUFFER_MAX="16"
        QFI_PROGRESSIVE_MID_K="128"
        QFI_COARSE_K="192"
        case "${PROFILE}" in
            qfi_zeno_stabilized_b005)
                QFI_CARD_SELECTOR="qfi_zeno_stabilized"
                QFI_ZENO_BETA="0.05"
                ;;
            qfi_zeno_stabilized_b015)
                QFI_CARD_SELECTOR="qfi_zeno_stabilized"
                QFI_ZENO_BETA="0.15"
                ;;
            qfi_spatial_buffer_recovery_b08)
                QFI_CARD_SELECTOR="qfi_spatial_buffer_recovery"
                QFI_BUFFER_MAX="8"
                ;;
            qfi_spatial_buffer_recovery_b16)
                QFI_CARD_SELECTOR="qfi_spatial_buffer_recovery"
                QFI_BUFFER_MAX="16"
                ;;
            qfi_progressive_192_to_64)
                QFI_CARD_SELECTOR="qfi_progressive_128_to_64"
                QFI_PROGRESSIVE_MID_K="192"
                ;;
            qfi_coarse_to_fine_256)
                QFI_CARD_SELECTOR="qfi_coarse_to_fine_pruning"
                QFI_COARSE_K="256"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QFI_CARD_PROFILE=${QFI_CARD_PROFILE}"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=${QFI_CARD_SELECTOR}"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
            "EC_QFI_ZENO_BETA=${QFI_ZENO_BETA}"
            "EC_QFI_BUFFER_MODE=8neighbor"
            "EC_QFI_BUFFER_RATIO=0.15"
            "EC_QFI_BUFFER_MAX=${QFI_BUFFER_MAX}"
            "EC_QFI_PROGRESSIVE_ENABLE=1"
            "EC_QFI_PROGRESSIVE_MID_K=${QFI_PROGRESSIVE_MID_K}"
            "EC_QFI_PROGRESSIVE_FINAL_K=${TOKEN}"
            "EC_QFI_PROGRESSIVE_STAGE_A_LAYER=2"
            "EC_QFI_PROGRESSIVE_STAGE_B_LAYER=8"
            "EC_QFI_COARSE_TO_FINE_ENABLE=1"
            "EC_QFI_COARSE_K=${QFI_COARSE_K}"
            "EC_QFI_FINE_K=${TOKEN}"
            "EC_QFI_CARDS_DUMP_DIAG=1"
            "EC_QFI_CARDS_DIAG_PATH=logs/${QFI_CARD_PROFILE}_k${TOKEN}_cards_diag.jsonl"
        )
        ;;
    qmo_cr_adaptive_recover_prior|qmo_cr_adaptive_recover_prior_no_purity|qmo_cr_adaptive_recover_prior_no_coupling|qmo_cr_adaptive_recover_prior_ultra_conservative)
        QMO_PROFILE="${PROFILE}"
        QMO_USE_PURITY="1"
        QMO_W_H="0.2"
        QMO_LAMBDA_J="0.02"
        QMO_D_ETA="0.1"
        case "${PROFILE}" in
            qmo_cr_adaptive_recover_prior_no_purity)
                QMO_USE_PURITY="0"
                ;;
            qmo_cr_adaptive_recover_prior_no_coupling)
                QMO_LAMBDA_J="0.0"
                ;;
            qmo_cr_adaptive_recover_prior_ultra_conservative)
                QMO_W_H="0.1"
                QMO_LAMBDA_J="0.01"
                QMO_D_ETA="0.05"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QMO_PROFILE=${QMO_PROFILE}"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=qmo_cr"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
            "EC_QMO_USE_PURITY=${QMO_USE_PURITY}"
            "EC_QMO_W_COV=1.0"
            "EC_QMO_W_H=${QMO_W_H}"
            "EC_QMO_LAMBDA_J=${QMO_LAMBDA_J}"
            "EC_QMO_COUPLING_REDUCE=mean"
            "EC_QMO_LAMBDA_H=0.5"
            "EC_QMO_LAMBDA_P=0.5"
            "EC_QMO_D_TAU=0.75"
            "EC_QMO_D_ETA=${QMO_D_ETA}"
            "EC_QMO_RECOVER_MIN_RATIO=0.00"
            "EC_QMO_RECOVER_MAX_RATIO=0.25"
            "EC_QMO_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QMO_RECOVER_BINARY_RATIO=0.0625"
            "EC_QMO_RECOVER_ATTR_RATIO=0.125"
            "EC_QMO_RECOVER_REL_RATIO=0.25"
            "EC_QMO_RECOVER_COMPARE_RATIO=0.25"
            "EC_QMO_RECOVER_OPEN_RATIO=0.25"
            "EC_QMO_RECOVER_OCR_RATIO=0.25"
            "EC_QMO_RECOVER_CAP=16"
            "EC_QMO_RECOVER_ALPHA_PRIOR=0.5"
            "EC_QMO_RECOVER_GAMMA=0.5"
            "EC_QMO_DUMP_INDICES=logs/${QMO_PROFILE}_gqa_k${TOKEN}_indices.jsonl"
            "EC_QMO_DEBUG=0"
        )
        ;;
    qmo_gated_adaptive_recover_prior|qmo_gated_adaptive_recover_prior_full|qmo_gated_adaptive_recover_prior_cmpcat|qmo_gated_adaptive_recover_prior_b003|qmo_gated_adaptive_recover_prior_b005)
        QMO_PROFILE="${PROFILE}"
        QMO_GATE_MODE="recovery_only"
        QMO_GATE_TYPES="compare,category,object"
        QMO_GATE_BETA="0.1"
        if [ "${PROFILE}" = "qmo_gated_adaptive_recover_prior_full" ]; then
            QMO_GATE_MODE="full"
        fi
        if [ "${PROFILE}" = "qmo_gated_adaptive_recover_prior_cmpcat" ]; then
            QMO_GATE_TYPES="compare,category"
        fi
        if [ "${PROFILE}" = "qmo_gated_adaptive_recover_prior_b003" ]; then
            QMO_GATE_BETA="0.03"
        fi
        if [ "${PROFILE}" = "qmo_gated_adaptive_recover_prior_b005" ]; then
            QMO_GATE_BETA="0.05"
        fi
        PROFILE_ENV=(
            "EC_QMO_PROFILE=${QMO_PROFILE}"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=qmo_gated"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
            "EC_QMO_GATE_MODE=${QMO_GATE_MODE}"
            "EC_QMO_GATE_TYPES=${QMO_GATE_TYPES}"
            "EC_QMO_GATE_BETA=${QMO_GATE_BETA}"
            "EC_QMO_W_COV=1.0"
            "EC_QMO_W_H=0.1"
            "EC_QMO_LAMBDA_J=0.01"
            "EC_QMO_COUPLING_REDUCE=mean"
            "EC_QMO_USE_PURITY=0"
            "EC_QMO_LAMBDA_H=0.5"
            "EC_QMO_LAMBDA_P=0.5"
            "EC_QMO_D_TAU=0.75"
            "EC_QMO_D_ETA=0.1"
            "EC_QMO_RECOVER_MIN_RATIO=0.00"
            "EC_QMO_RECOVER_MAX_RATIO=0.25"
            "EC_QMO_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QMO_RECOVER_BINARY_RATIO=0.0625"
            "EC_QMO_RECOVER_ATTR_RATIO=0.125"
            "EC_QMO_RECOVER_REL_RATIO=0.25"
            "EC_QMO_RECOVER_COMPARE_RATIO=0.25"
            "EC_QMO_RECOVER_OPEN_RATIO=0.25"
            "EC_QMO_RECOVER_OCR_RATIO=0.25"
            "EC_QMO_RECOVER_CAP=16"
            "EC_QMO_RECOVER_ALPHA_PRIOR=0.5"
            "EC_QMO_RECOVER_GAMMA=0.5"
            "EC_QMO_DUMP_INDICES=logs/${QMO_PROFILE}_k${TOKEN}_indices.jsonl"
            "EC_QMO_DEBUG=0"
        )
        ;;
    qsp_cr_density_projective|density_projective_core_recover|qsp_cr_density_projective_top_p|qsp_cr_density_projective_pairwise|qsp_cr_density_projective_no_recovery|qsp_cr_density_projective_prior_recovery|qsp_cr_density_projective_spatial_prior|qsp_density_projective_task_anchor|qsp_density_projective_task_anchor_a05|qsp_density_projective_complement_bonus)
        QSP_PROFILE="${PROFILE}"
        QSP_CORE_SELECTION="density_projection"
        QSP_RECOVERY="complement_projected"
        QSP_USE_SPATIAL_LOCAL_PRIOR="0"
        QSP_SELECTOR="qsp_cr_density_projective"
        QSP_TASK_ANCHOR_ALPHA="0.0"
        QSP_COMPLEMENT_BONUS_ETA="0.15"
        case "${PROFILE}" in
            density_projective_core_recover)
                QSP_PROFILE="density_projective_core_recover"
                ;;
            qsp_cr_density_projective_top_p)
                QSP_CORE_SELECTION="top_p"
                ;;
            qsp_cr_density_projective_pairwise)
                QSP_CORE_SELECTION="pairwise_fidelity"
                ;;
            qsp_cr_density_projective_no_recovery)
                QSP_RECOVERY="none"
                ;;
            qsp_cr_density_projective_prior_recovery)
                QSP_RECOVERY="prior_recovery"
                ;;
            qsp_cr_density_projective_spatial_prior)
                QSP_USE_SPATIAL_LOCAL_PRIOR="1"
                ;;
            qsp_density_projective_task_anchor)
                QSP_SELECTOR="qsp_density_projective_task_anchor"
                QSP_TASK_ANCHOR_ALPHA="0.3"
                ;;
            qsp_density_projective_task_anchor_a05)
                QSP_SELECTOR="qsp_density_projective_task_anchor_a05"
                QSP_TASK_ANCHOR_ALPHA="0.5"
                ;;
            qsp_density_projective_complement_bonus)
                QSP_SELECTOR="qsp_density_projective_complement_bonus"
                QSP_RECOVERY="complement_bonus"
                QSP_COMPLEMENT_BONUS_ETA="0.15"
                ;;
        esac
        PROFILE_ENV=(
            "EC_QSP_PROFILE=${QSP_PROFILE}"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=${QSP_SELECTOR}"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
            "EC_QSP_USE_DENSITY_PROJ=1"
            "EC_QSP_USE_COMPLEMENT_RECOVERY=1"
            "EC_QSP_CORE_SELECTION=${QSP_CORE_SELECTION}"
            "EC_QSP_RECOVERY=${QSP_RECOVERY}"
            "EC_QSP_USE_SPATIAL_LOCAL_PRIOR=${QSP_USE_SPATIAL_LOCAL_PRIOR}"
            "EC_QSP_LAMBDA_B=0.1"
            "EC_QSP_CANDIDATE_TOPM=0"
            "EC_QSP_CORE_RATIO=0.75"
            "EC_QSP_RECOVER_CAP=16"
            "EC_QSP_TASK_ANCHOR_ALPHA=${QSP_TASK_ANCHOR_ALPHA}"
            "EC_QSP_COMPLEMENT_BONUS_ETA=${QSP_COMPLEMENT_BONUS_ETA}"
            "EC_QSP_DUMP_INDICES=logs/${QSP_PROFILE}_k${TOKEN}_indices.jsonl"
            "EC_QSP_DIAG_PATH=logs/${QSP_PROFILE}_k${TOKEN}_diag.jsonl"
        )
        ;;
    qfi_core_complement_recovery)
        PROFILE_ENV=(
            "EC_QSP_PROFILE=qfi_core_complement_recovery"
            "EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SELECTOR=qfi_core_complement_recovery"
            "EC_QFID_ADAPT_MODE=entropy_question"
            "EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00"
            "EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625"
            "EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25"
            "EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125"
            "EC_QFID_ADAPT_RECOVER_CAP=16"
            "EC_QFID_ADAPT_ENTROPY_GATE=1"
            "EC_QFID_ADAPT_ENTROPY_LOW=0.70"
            "EC_QFID_ADAPT_ENTROPY_HIGH=0.95"
            "EC_QFID_RECOVER_PRIOR_ANCHOR=1"
            "EC_QFID_RECOVER_PRIOR_GAMMA=0.5"
            "EC_QFID_RECOVER_CAND_POOL=0"
            "EC_QFID_RECOVER_CAND_MULT=3.0"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=0"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
            "EC_QSP_USE_SPATIAL_LOCAL_PRIOR=0"
            "EC_QSP_LAMBDA_B=0.1"
            "EC_QSP_DUMP_INDICES=logs/qfi_core_complement_recovery_k${TOKEN}_indices.jsonl"
            "EC_QSP_DIAG_PATH=logs/qfi_core_complement_recovery_k${TOKEN}_diag.jsonl"
        )
        ;;
    qfid_density_cg_eaqf_spectral_g05|qfid_density_cg_eaqf_spectral_g075|qfid_density_cg_eaqf_spectral_g125|qfid_density_cg_eaqf_spectral_g150)
        case "${PROFILE}" in
            *_g05) SPECTRAL_GAMMA="0.5"; SPECTRAL_PROFILE="g05" ;;
            *_g075) SPECTRAL_GAMMA="0.75"; SPECTRAL_PROFILE="g075" ;;
            *_g125) SPECTRAL_GAMMA="1.25"; SPECTRAL_PROFILE="g125" ;;
            *_g150) SPECTRAL_GAMMA="1.5"; SPECTRAL_PROFILE="g150" ;;
        esac
        PROFILE_ENV=(
            "EC_QFID_SPECTRAL_PROFILE=${SPECTRAL_PROFILE}"
            "EC_QFID_CG_FINAL_PROFILE=0"
            "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_BUDGET_CALIB=0"
            "EC_QFID_SPECTRAL_FILTER=1" "EC_QFID_SPECTRAL_GAMMA=${SPECTRAL_GAMMA}"
            "EC_QFID_SPECTRAL_EPS=1e-12" "EC_QFID_SPECTRAL_TRACE_NORM=1"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_cg_eaqf_budget_alpha2|qfid_density_cg_eaqf_budget_alpha3|qfid_density_cg_eaqf_budget_alpha4)
        case "${PROFILE}" in
            *_alpha2) BUDGET_ALPHA="2.0"; BUDGET_PROFILE="alpha2" ;;
            *_alpha3) BUDGET_ALPHA="3.0"; BUDGET_PROFILE="alpha3" ;;
            *_alpha4) BUDGET_ALPHA="4.0"; BUDGET_PROFILE="alpha4" ;;
        esac
        PROFILE_ENV=(
            "EC_QFID_BUDGET_PROFILE=${BUDGET_PROFILE}"
            "EC_QFID_CG_FINAL_PROFILE=0" "EC_QFID_FINAL_PROFILE=0" "EC_QFID_GATE_PROFILE=0"
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf"
            "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement"
            "EC_QFID_CLS_GATE_BETA_BASE=0.105"
            "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_EPS=1e-6"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_BUDGET_CALIB=1" "EC_QFID_BUDGET_ALPHA=${BUDGET_ALPHA}"
            "EC_QFID_BUDGET_GAMMA_MIN=0.05" "EC_QFID_BUDGET_GAMMA_MAX=20.0"
            "EC_QFID_BUDGET_ITERS=30" "EC_QFID_BUDGET_EPS=1e-12"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
            "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_clsmix_layer_m4)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf" "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=-4" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0" "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_clsmix_layer_last4mean)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf" "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=last4mean" "EC_QFID_CLS_HEAD_REDUCE=mean"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0" "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_clsmix_head_entropy)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=qfid" "EC_QFID_SELECT_MODE=qf" "EC_QFID_PROB_SOURCE=clsmix"
            "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105"
            "EC_QFID_CLS_ATTN_LAYER=last" "EC_QFID_CLS_HEAD_REDUCE=entropy_weighted"
            "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50" "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0" "EC_SOLVER=greedy"
        )
        ;;
    qfid_density_depol010)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.10}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_depol020)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.20}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_depol025)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.25}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_adaptive_depol)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE_MODE=${EC_QFID_DEPOLARIZE_MODE:-adaptive}"
            "EC_QFID_DEPOLARIZE_MIN=${EC_QFID_DEPOLARIZE_MIN:-0.05}"
            "EC_QFID_DEPOLARIZE_MAX=${EC_QFID_DEPOLARIZE_MAX:-0.25}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_anchor8)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-measurement}"
            "EC_QFID_MEASURE_ANCHOR_RATIO=${EC_QFID_MEASURE_ANCHOR_RATIO:-0.125}"
            "EC_QFID_MEASURE_ANCHOR_MIN=${EC_QFID_MEASURE_ANCHOR_MIN:-0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_anchor16)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-measurement}"
            "EC_QFID_MEASURE_ANCHOR_RATIO=${EC_QFID_MEASURE_ANCHOR_RATIO:-0.25}"
            "EC_QFID_MEASURE_ANCHOR_MIN=${EC_QFID_MEASURE_ANCHOR_MIN:-0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_anchor24)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-measurement}"
            "EC_QFID_MEASURE_ANCHOR_RATIO=${EC_QFID_MEASURE_ANCHOR_RATIO:-0.375}"
            "EC_QFID_MEASURE_ANCHOR_MIN=${EC_QFID_MEASURE_ANCHOR_MIN:-0}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_softprior005)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-soft}"
            "EC_QFID_MEASURE_PRIOR_LAMBDA=${EC_QFID_MEASURE_PRIOR_LAMBDA:-0.05}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_softprior010)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-soft}"
            "EC_QFID_MEASURE_PRIOR_LAMBDA=${EC_QFID_MEASURE_PRIOR_LAMBDA:-0.10}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_softprior020)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-soft}"
            "EC_QFID_MEASURE_PRIOR_LAMBDA=${EC_QFID_MEASURE_PRIOR_LAMBDA:-0.20}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_measurement)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-measurement}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_measurement_soft020)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-measurement}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-soft}"
            "EC_QFID_MEASURE_PRIOR_LAMBDA=${EC_QFID_MEASURE_PRIOR_LAMBDA:-0.20}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_spatial010)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-1}"
            "EC_QFID_SPATIAL_LAMBDA=${EC_QFID_SPATIAL_LAMBDA:-0.10}"
            "EC_QFID_SPATIAL_SIGMA=${EC_QFID_SPATIAL_SIGMA:-0.20}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_spatial020)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-1}"
            "EC_QFID_SPATIAL_LAMBDA=${EC_QFID_SPATIAL_LAMBDA:-0.20}"
            "EC_QFID_SPATIAL_SIGMA=${EC_QFID_SPATIAL_SIGMA:-0.20}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qfid_density_spatial030)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qfid}"
            "EC_QFID_TAU=${EC_QFID_TAU:-0.50}"
            "EC_QFID_EPS=${EC_QFID_EPS:-1e-6}"
            "EC_QFID_PROB_SOURCE=${EC_QFID_PROB_SOURCE:-semantic}"
            "EC_QFID_KERNEL=${EC_QFID_KERNEL:-density}"
            "EC_QFID_DEPOLARIZE=${EC_QFID_DEPOLARIZE:-0.15}"
            "EC_QFID_SPATIAL_STATE=${EC_QFID_SPATIAL_STATE:-1}"
            "EC_QFID_SPATIAL_LAMBDA=${EC_QFID_SPATIAL_LAMBDA:-0.30}"
            "EC_QFID_SPATIAL_SIGMA=${EC_QFID_SPATIAL_SIGMA:-0.20}"
            "EC_QFID_MEASURE_PRIOR_MODE=${EC_QFID_MEASURE_PRIOR_MODE:-none}"
            "EC_QFID_ANCHOR_MODE=${EC_QFID_ANCHOR_MODE:-none}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-0}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-0}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qmo_full)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qmo}"
            "EC_W_REL=${EC_W_REL:-0.40}"
            "EC_W_SEM=${EC_W_SEM:-0.25}"
            "EC_W_SPATIAL=${EC_W_SPATIAL:-0.15}"
            "EC_W_CHAIN=${EC_W_CHAIN:-0.15}"
            "EC_W_ENT=${EC_W_ENT:-0.05}"
            "EC_QMO_INIT_SOURCE=${EC_QMO_INIT_SOURCE:-semantic}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-1}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-1}"
            "EC_SPATIAL_EXCLUDE_GLOBAL=${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
            "EC_GLOBAL_CANDIDATE_RATIO=${EC_GLOBAL_CANDIDATE_RATIO:-0.625}"
            "EC_CANDIDATE_RATIO=${EC_CANDIDATE_RATIO:-2.0}"
            "EC_BETA_A=${EC_BETA_A:-0.5}"
            "EC_BETA_SEM=${EC_BETA_SEM:-0.3}"
            "EC_BETA_SPATIAL=${EC_BETA_SPATIAL:-0.2}"
            "EC_PER_UNIT_TOPL=${EC_PER_UNIT_TOPL:-8}"
            "EC_SPATIAL_GRID_SIZE=${EC_SPATIAL_GRID_SIZE:-4}"
            "EC_SPATIAL_TOPL=${EC_SPATIAL_TOPL:-1}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-1}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-1}"
            "EC_USE_PHI=${EC_USE_PHI:-1}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_LAMBDA=${EC_LAMBDA:-0.1}"
            "EC_DELTA=${EC_DELTA:-0.1}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qmo_candidate_only)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-qmo}"
            "EC_W_REL=${EC_W_REL:-0.40}"
            "EC_W_SEM=${EC_W_SEM:-0.25}"
            "EC_W_SPATIAL=${EC_W_SPATIAL:-0.15}"
            "EC_W_CHAIN=${EC_W_CHAIN:-0.15}"
            "EC_W_ENT=${EC_W_ENT:-0.05}"
            "EC_QMO_INIT_SOURCE=${EC_QMO_INIT_SOURCE:-semantic}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-1}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-1}"
            "EC_SPATIAL_EXCLUDE_GLOBAL=${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
            "EC_GLOBAL_CANDIDATE_RATIO=${EC_GLOBAL_CANDIDATE_RATIO:-0.625}"
            "EC_CANDIDATE_RATIO=${EC_CANDIDATE_RATIO:-2.0}"
            "EC_BETA_A=${EC_BETA_A:-0.5}"
            "EC_BETA_SEM=${EC_BETA_SEM:-0.3}"
            "EC_BETA_SPATIAL=${EC_BETA_SPATIAL:-0.2}"
            "EC_PER_UNIT_TOPL=${EC_PER_UNIT_TOPL:-8}"
            "EC_SPATIAL_GRID_SIZE=${EC_SPATIAL_GRID_SIZE:-4}"
            "EC_SPATIAL_TOPL=${EC_SPATIAL_TOPL:-1}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    semantic_full)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-semantic}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-1}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-1}"
            "EC_SPATIAL_EXCLUDE_GLOBAL=${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
            "EC_GLOBAL_CANDIDATE_RATIO=${EC_GLOBAL_CANDIDATE_RATIO:-0.625}"
            "EC_CANDIDATE_RATIO=${EC_CANDIDATE_RATIO:-2.0}"
            "EC_BETA_A=${EC_BETA_A:-0.5}"
            "EC_BETA_SEM=${EC_BETA_SEM:-0.3}"
            "EC_BETA_SPATIAL=${EC_BETA_SPATIAL:-0.2}"
            "EC_PER_UNIT_TOPL=${EC_PER_UNIT_TOPL:-8}"
            "EC_SPATIAL_GRID_SIZE=${EC_SPATIAL_GRID_SIZE:-4}"
            "EC_SPATIAL_TOPL=${EC_SPATIAL_TOPL:-1}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-1}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-1}"
            "EC_USE_PHI=${EC_USE_PHI:-1}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_LAMBDA=${EC_LAMBDA:-0.1}"
            "EC_DELTA=${EC_DELTA:-0.1}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    semantic_candidate_only)
        PROFILE_ENV=(
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-semantic}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-1}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-1}"
            "EC_SPATIAL_EXCLUDE_GLOBAL=${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
            "EC_GLOBAL_CANDIDATE_RATIO=${EC_GLOBAL_CANDIDATE_RATIO:-0.625}"
            "EC_CANDIDATE_RATIO=${EC_CANDIDATE_RATIO:-2.0}"
            "EC_BETA_A=${EC_BETA_A:-0.5}"
            "EC_BETA_SEM=${EC_BETA_SEM:-0.3}"
            "EC_BETA_SPATIAL=${EC_BETA_SPATIAL:-0.2}"
            "EC_PER_UNIT_TOPL=${EC_PER_UNIT_TOPL:-8}"
            "EC_SPATIAL_GRID_SIZE=${EC_SPATIAL_GRID_SIZE:-4}"
            "EC_SPATIAL_TOPL=${EC_SPATIAL_TOPL:-1}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-0}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-0}"
            "EC_USE_PHI=${EC_USE_PHI:-0}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_SOLVER=${EC_SOLVER:-greedy}"
        )
        ;;
    qanneal_debug)
        PROFILE_ENV=(
            "EC_DEBUG=${EC_DEBUG:-1}"
            "EC_DEBUG_LIMIT=${EC_DEBUG_LIMIT:-30}"
            "EC_SCORE_SOURCE=${EC_SCORE_SOURCE:-semantic}"
            "EC_USE_SEMANTIC_CANDIDATE=${EC_USE_SEMANTIC_CANDIDATE:-1}"
            "EC_USE_SPATIAL_CANDIDATE=${EC_USE_SPATIAL_CANDIDATE:-1}"
            "EC_SPATIAL_EXCLUDE_GLOBAL=${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
            "EC_GLOBAL_CANDIDATE_RATIO=${EC_GLOBAL_CANDIDATE_RATIO:-0.625}"
            "EC_CANDIDATE_RATIO=${EC_CANDIDATE_RATIO:-2.0}"
            "EC_BETA_A=${EC_BETA_A:-0.5}"
            "EC_BETA_SEM=${EC_BETA_SEM:-0.3}"
            "EC_BETA_SPATIAL=${EC_BETA_SPATIAL:-0.2}"
            "EC_PER_UNIT_TOPL=${EC_PER_UNIT_TOPL:-8}"
            "EC_SPATIAL_GRID_SIZE=${EC_SPATIAL_GRID_SIZE:-4}"
            "EC_SPATIAL_TOPL=${EC_SPATIAL_TOPL:-1}"
            "EC_USE_REPULSION=${EC_USE_REPULSION:-1}"
            "EC_USE_COMPLEMENT=${EC_USE_COMPLEMENT:-1}"
            "EC_USE_PHI=${EC_USE_PHI:-1}"
            "EC_USE_CHAIN_COVERAGE=${EC_USE_CHAIN_COVERAGE:-0}"
            "EC_LAMBDA=${EC_LAMBDA:-0.1}"
            "EC_DELTA=${EC_DELTA:-0.1}"
            "EC_SOLVER=${EC_SOLVER:-qanneal}"
            "EC_QA_STEPS=${EC_QA_STEPS:-50}"
            "EC_QA_T0=${EC_QA_T0:-1.0}"
            "EC_QA_TEND=${EC_QA_TEND:-0.01}"
            "EC_QA_MAX_SWAP_RATIO=${EC_QA_MAX_SWAP_RATIO:-0.15}"
            "EC_QA_SEED=${EC_QA_SEED:-42}"
        )
        ;;
    *)
        echo "Unknown profile: ${PROFILE}" >&2
        echo "Available profiles: qfid, qfid_uniform, qfid_density, qfid_density_depol, qfid_density_clsmix005, qfid_density_clsmix010, qfid_density_clsmix020, qfid_density_cls_only_qf, qfid_cls_topk, qfid_ablate_cls_topk, qfid_ablate_cls_only_qf, qfid_ablate_semantic_topk, qfid_ablate_uniform_qf, qfid_ablate_visual_kcenter, qfid_density_clsmix_layer_m2, qfid_density_eaqf_final, qfid_density_eaqf_gate_agreement, qfid_density_cg_eaqf_final, qfi_evidence_recover_final, qfi_core_recover_r050, qfi_core_recover_r075, qfi_core_recover_r08125, qfi_core_recover_r0875, qfi_adaptive_recover_budget, qfi_adaptive_recover_prior, qfi_adaptive_recover_final, qfi_zeno_stabilized, qfi_zeno_stabilized_b005, qfi_zeno_stabilized_b015, qfi_spatial_buffer_recovery, qfi_spatial_buffer_recovery_b08, qfi_spatial_buffer_recovery_b16, qfi_progressive_128_to_64, qfi_progressive_192_to_64, qfi_coarse_to_fine_pruning, qfi_coarse_to_fine_256, qmo_cr_adaptive_recover_prior, qmo_cr_adaptive_recover_prior_no_purity, qmo_cr_adaptive_recover_prior_no_coupling, qmo_cr_adaptive_recover_prior_ultra_conservative, qmo_gated_adaptive_recover_prior, qmo_gated_adaptive_recover_prior_full, qmo_gated_adaptive_recover_prior_cmpcat, qmo_gated_adaptive_recover_prior_b003, qmo_gated_adaptive_recover_prior_b005, qsp_cr_density_projective, density_projective_core_recover, qsp_cr_density_projective_top_p, qsp_cr_density_projective_pairwise, qsp_cr_density_projective_no_recovery, qsp_cr_density_projective_prior_recovery, qsp_cr_density_projective_spatial_prior, qsp_density_projective_task_anchor, qsp_density_projective_task_anchor_a05, qsp_density_projective_complement_bonus, qfi_core_complement_recovery, qfid_density_cg_eaqf_budget_alpha2, qfid_density_cg_eaqf_budget_alpha3, qfid_density_cg_eaqf_budget_alpha4, qfid_density_cg_eaqf_spectral_g05, qfid_density_cg_eaqf_spectral_g075, qfid_density_cg_eaqf_spectral_g125, qfid_density_cg_eaqf_spectral_g150, qfid_density_clsmix_layer_m4, qfid_density_clsmix_layer_last4mean, qfid_density_clsmix_head_entropy, qfid_density_depol010, qfid_density_depol020, qfid_density_depol025, qfid_density_adaptive_depol, qfid_density_anchor8, qfid_density_anchor16, qfid_density_anchor24, qfid_density_softprior005, qfid_density_softprior010, qfid_density_softprior020, qfid_density_measurement, qfid_density_measurement_soft020, qfid_density_spatial010, qfid_density_spatial020, qfid_density_spatial030, qmo_full, qmo_candidate_only, semantic_full, semantic_candidate_only, qanneal_debug" >&2
        exit 1
        ;;
esac

if [[ "${PROFILE}" == qfid_ablate_* ]]; then
    PROFILE_ENV+=("EC_QFID_ABLATION_PROFILE=${PROFILE#qfid_ablate_}")
fi

PROFILE_ENV+=("EC_QFID_OVERLAP_KERNEL=${EC_QFID_OVERLAP_KERNEL:-relu_square}")

if [ -n "${EC_QFID_DEBUG_STATS_JSONL:-}" ]; then
    mkdir -p "$(dirname "${EC_QFID_DEBUG_STATS_JSONL}")"
    : > "${EC_QFID_DEBUG_STATS_JSONL}"
fi

profile_env_value() {
    local key="$1"
    local item
    for item in "${PROFILE_ENV[@]}"; do
        if [[ "${item}" == "${key}="* ]]; then
            printf '%s\n' "${item#*=}"
            return 0
        fi
    done
    printf 'unset\n'
}

echo "[run_gqa_profile] profile=${PROFILE}" >&2
echo "[run_gqa_profile] gpu=${GPU}" >&2
echo "[run_gqa_profile] token=${TOKEN}" >&2
echo "[run_gqa_profile] qfid_selector=$(profile_env_value EC_QFID_SELECTOR)" >&2
echo "[run_gqa_profile] qfid_overlap_kernel=$(profile_env_value EC_QFID_OVERLAP_KERNEL)" >&2
echo "[run_gqa_profile] qfid_core_ratio=$(profile_env_value EC_QFID_CORE_RATIO)" >&2
echo "[run_gqa_profile] qfid_adapt_mode=$(profile_env_value EC_QFID_ADAPT_MODE)" >&2
echo "[run_gqa_profile] qfid_adapt_recover_cap=$(profile_env_value EC_QFID_ADAPT_RECOVER_CAP)" >&2
echo "[run_gqa_profile] qfid_recover_prior_anchor=$(profile_env_value EC_QFID_RECOVER_PRIOR_ANCHOR)" >&2
echo "[run_gqa_profile] qfid_recover_prior_gamma=$(profile_env_value EC_QFID_RECOVER_PRIOR_GAMMA)" >&2
echo "[run_gqa_profile] qfid_recover_cand_pool=$(profile_env_value EC_QFID_RECOVER_CAND_POOL)" >&2
echo "[run_gqa_profile] qfid_recover_cand_mult=$(profile_env_value EC_QFID_RECOVER_CAND_MULT)" >&2
echo "[run_gqa_profile] qfid_budget_calib=$(profile_env_value EC_QFID_BUDGET_CALIB)" >&2
echo "[run_gqa_profile] qfid_spectral_filter=$(profile_env_value EC_QFID_SPECTRAL_FILTER)" >&2
echo "[run_gqa_profile] conda_env=cdpruner" >&2

if [ "${ACTION}" = "--print-env" ] || [ "${ACTION}" = "print-env" ]; then
    printf '%s\n' "${COMMON_ENV[@]}" "${PROFILE_ENV[@]}"
    exit 0
fi

source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
conda activate cdpruner

env "${COMMON_ENV[@]}" "${PROFILE_ENV[@]}" bash scripts/v1_5/eval/gqa.sh "${TOKEN}"
