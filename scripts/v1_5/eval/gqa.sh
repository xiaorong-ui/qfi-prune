#!/bin/bash
set -euo pipefail

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="${CKPT_DIR:-/home/gpuadmin/txr/CDPruner/checkpoints}"
DATA_DIR="${DATA_DIR:-/home/gpuadmin/txr/CDPruner_data}"

CKPT="llava-v1.5-7b"
SPLIT="llava_gqa_testdev_balanced"

TOKEN=${1}
PRUNE_METHOD_NAME="${PRUNE_METHOD:-cdpruner}"
EC_SCORE_SOURCE_NAME="${EC_SCORE_SOURCE:-norm}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/gqa/${SPLIT}.jsonl}"
NUM_SAMPLES_NAME="${NUM_SAMPLES:-0}"
SMOKE_RUN=0
SMOKE_QUESTION_FILE=""
if [[ "${NUM_SAMPLES_NAME}" =~ ^[1-9][0-9]*$ ]]; then
    SMOKE_RUN=1
    mkdir -p logs/gqa_smoke
    SMOKE_QUESTION_FILE="logs/gqa_smoke/questions_${NUM_SAMPLES_NAME}_$$.jsonl"
    head -n "${NUM_SAMPLES_NAME}" "${QUESTION_FILE}" > "${SMOKE_QUESTION_FILE}"
    QUESTION_FILE="${SMOKE_QUESTION_FILE}"
    trap 'rm -f "${SMOKE_QUESTION_FILE}"' EXIT
fi
IMAGE_DIR="${DATA_DIR}/gqa/data/images"
GQA_QUESTION_DIR="${DATA_DIR}/gqa/data/questions"
GQA_DIR="$(realpath ./playground/data/eval/gqa/data)"
GQA_QUESTION_FILE="${GQA_QUESTION_DIR}/testdev_balanced_questions.json"

if [ -n "${EC_QFID_DEBUG_STATS_JSONL:-}" ] && [ "${CHUNKS}" -ne 1 ]; then
    echo "EC_QFID_DEBUG_STATS_JSONL requires a single GPU to preserve row alignment." >&2
    exit 1
fi

for required_path in \
    "${CKPT_DIR}/${CKPT}" \
    "${QUESTION_FILE}" \
    "${IMAGE_DIR}" \
    "${GQA_QUESTION_FILE}" \
    "${GQA_DIR}/eval/eval.py"; do
    if [ ! -e "${required_path}" ]; then
        echo "Missing required GQA path: ${required_path}" >&2
        exit 1
    fi
done

if [ "${PRUNE_METHOD_NAME}" = "cdpruner" ]; then
    PARAM="vtn_${TOKEN}"
else
    EC_USE_SEMANTIC_CANDIDATE_NAME="${EC_USE_SEMANTIC_CANDIDATE:-1}"
    EC_USE_SPATIAL_CANDIDATE_NAME="${EC_USE_SPATIAL_CANDIDATE:-0}"
    EC_SPATIAL_EXCLUDE_GLOBAL_NAME="${EC_SPATIAL_EXCLUDE_GLOBAL:-1}"
    EC_SOLVER_NAME="${EC_SOLVER:-greedy}"
    EC_QA_STEPS_NAME="${EC_QA_STEPS:-50}"
    EC_QA_MAX_SWAP_RATIO_NAME="${EC_QA_MAX_SWAP_RATIO:-0.15}"
    EC_GLOBAL_CANDIDATE_RATIO_NAME="${EC_GLOBAL_CANDIDATE_RATIO:-1.0}"
    EC_CANDIDATE_RATIO_NAME="${EC_CANDIDATE_RATIO:-2.0}"
    EC_USE_REPULSION_NAME="${EC_USE_REPULSION:-1}"
    EC_USE_COMPLEMENT_NAME="${EC_USE_COMPLEMENT:-1}"
    EC_USE_PHI_NAME="${EC_USE_PHI:-1}"
    EC_USE_CHAIN_COVERAGE_NAME="${EC_USE_CHAIN_COVERAGE:-1}"
    EC_LAMBDA_NAME="${EC_LAMBDA:-0.1}"
    EC_DELTA_NAME="${EC_DELTA:-0.1}"
    EC_GAMMA_CHAIN_NAME="${EC_GAMMA_CHAIN:-0.1}"
    EC_QMO_SCORE_TAG=""
    EC_QFID_SCORE_TAG=""

    EC_QA_MAX_SWAP_RATIO_TAG="${EC_QA_MAX_SWAP_RATIO_NAME//./}"
    EC_GLOBAL_CANDIDATE_RATIO_TAG="${EC_GLOBAL_CANDIDATE_RATIO_NAME//./}"
    EC_CANDIDATE_RATIO_TAG="${EC_CANDIDATE_RATIO_NAME//./}"
    EC_LAMBDA_TAG="${EC_LAMBDA_NAME//./}"
    EC_DELTA_TAG="${EC_DELTA_NAME//./}"
    EC_GAMMA_CHAIN_TAG="${EC_GAMMA_CHAIN_NAME//./}"

    if [ "${EC_SCORE_SOURCE_NAME}" = "qmo" ]; then
        EC_W_REL_NAME="${EC_W_REL:-0.40}"
        EC_W_SEM_NAME="${EC_W_SEM:-0.25}"
        EC_W_SPATIAL_NAME="${EC_W_SPATIAL:-0.15}"
        EC_W_CHAIN_NAME="${EC_W_CHAIN:-0.15}"
        EC_W_ENT_NAME="${EC_W_ENT:-0.05}"
        EC_QMO_INIT_SOURCE_NAME="${EC_QMO_INIT_SOURCE:-semantic}"
        EC_QMO_SCORE_TAG="_scoreqmo_wrel${EC_W_REL_NAME}_wsem${EC_W_SEM_NAME}_wspa${EC_W_SPATIAL_NAME}_wchain${EC_W_CHAIN_NAME}_went${EC_W_ENT_NAME}_qinit${EC_QMO_INIT_SOURCE_NAME}"
    elif [ "${EC_SCORE_SOURCE_NAME}" = "qfid" ]; then
        EC_QFID_TAU_NAME="${EC_QFID_TAU:-0.50}"
        EC_QFID_PROB_SOURCE_NAME="${EC_QFID_PROB_SOURCE:-semantic}"
        EC_QFID_SELECT_MODE_NAME="${EC_QFID_SELECT_MODE:-qf}"
        EC_QFID_KERNEL_NAME="${EC_QFID_KERNEL:-amplitude}"
        EC_QFID_DEPOLARIZE_NAME="${EC_QFID_DEPOLARIZE:-0.0}"
        EC_QFID_DEPOLARIZE_MODE_NAME="${EC_QFID_DEPOLARIZE_MODE:-fixed}"
        EC_QFID_SPATIAL_STATE_NAME="${EC_QFID_SPATIAL_STATE:-0}"
        EC_QFID_SPATIAL_LAMBDA_NAME="${EC_QFID_SPATIAL_LAMBDA:-0.10}"
        EC_QFID_SPATIAL_SIGMA_NAME="${EC_QFID_SPATIAL_SIGMA:-0.20}"
        EC_QFID_MEASURE_PRIOR_MODE_NAME="${EC_QFID_MEASURE_PRIOR_MODE:-none}"
        EC_QFID_MEASURE_PRIOR_LAMBDA_NAME="${EC_QFID_MEASURE_PRIOR_LAMBDA:-0.10}"
        EC_QFID_ANCHOR_MODE_NAME="${EC_QFID_ANCHOR_MODE:-none}"
        EC_QFID_MEASURE_ANCHOR_RATIO_NAME="${EC_QFID_MEASURE_ANCHOR_RATIO:-0.125}"
        EC_QFID_MEASURE_ANCHOR_MIN_NAME="${EC_QFID_MEASURE_ANCHOR_MIN:-0}"
        EC_QFID_CLS_MIX_MODE_NAME="${EC_QFID_CLS_MIX_MODE:-linear}"
        EC_QFID_CLS_MIX_BETA_NAME="${EC_QFID_CLS_MIX_BETA:-0.05}"
        EC_QFID_CLS_GATE_NAME="${EC_QFID_CLS_GATE:-0}"
        EC_QFID_CLS_GATE_MODE_NAME="${EC_QFID_CLS_GATE_MODE:-agreement}"
        EC_QFID_CLS_GATE_BETA_BASE_NAME="${EC_QFID_CLS_GATE_BETA_BASE:-0.105}"
        EC_QFID_CLS_GATE_MIN_NAME="${EC_QFID_CLS_GATE_MIN:-0.5}"
        EC_QFID_CLS_GATE_MAX_NAME="${EC_QFID_CLS_GATE_MAX:-1.5}"
        EC_QFID_CLS_ATTN_LAYER_NAME="${EC_QFID_CLS_ATTN_LAYER:-last}"
        EC_QFID_CLS_HEAD_REDUCE_NAME="${EC_QFID_CLS_HEAD_REDUCE:-mean}"
        EC_QFID_BUDGET_CALIB_NAME="${EC_QFID_BUDGET_CALIB:-0}"
        EC_QFID_BUDGET_ALPHA_NAME="${EC_QFID_BUDGET_ALPHA:-3.0}"
        EC_QFID_SPECTRAL_FILTER_NAME="${EC_QFID_SPECTRAL_FILTER:-0}"
        EC_QFID_SPECTRAL_GAMMA_NAME="${EC_QFID_SPECTRAL_GAMMA:-1.0}"
        EC_QFID_SPECTRAL_EPS_NAME="${EC_QFID_SPECTRAL_EPS:-1e-12}"
        EC_QFID_SPECTRAL_TRACE_NORM_NAME="${EC_QFID_SPECTRAL_TRACE_NORM:-1}"
        EC_QFID_CORE_RATIO_NAME="${EC_QFID_CORE_RATIO:-0.5}"
        EC_QFID_EXTRA_TAG=""
        if [ "${EC_QFID_DEPOLARIZE_MODE_NAME}" = "adaptive" ]; then
            EC_QFID_DEPOLARIZE_MIN_NAME="${EC_QFID_DEPOLARIZE_MIN:-0.05}"
            EC_QFID_DEPOLARIZE_MAX_NAME="${EC_QFID_DEPOLARIZE_MAX:-0.25}"
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfiddmodeadaptive_qfiddmin${EC_QFID_DEPOLARIZE_MIN_NAME}_qfiddmax${EC_QFID_DEPOLARIZE_MAX_NAME}"
        fi
        if [ "${EC_QFID_SPATIAL_STATE_NAME}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfidsp1_qfidsl${EC_QFID_SPATIAL_LAMBDA_NAME}_qfidss${EC_QFID_SPATIAL_SIGMA_NAME}"
        fi
        if [ "${EC_QFID_ANCHOR_MODE_NAME}" != "none" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfidanchor${EC_QFID_ANCHOR_MODE_NAME}_qfidar${EC_QFID_MEASURE_ANCHOR_RATIO_NAME}_qfidamin${EC_QFID_MEASURE_ANCHOR_MIN_NAME}"
        fi
        if [ "${EC_QFID_MEASURE_PRIOR_MODE_NAME}" != "none" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfidpmode${EC_QFID_MEASURE_PRIOR_MODE_NAME}_qfidpl${EC_QFID_MEASURE_PRIOR_LAMBDA_NAME}"
        fi
        if [ "${EC_QFID_PROB_SOURCE_NAME}" = "clsmix" ] || [ "${EC_QFID_PROB_SOURCE_NAME}" = "cls_only_qf" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_clslayer${EC_QFID_CLS_ATTN_LAYER_NAME}_clshead${EC_QFID_CLS_HEAD_REDUCE_NAME}"
        fi
        if [ "${EC_QFID_PROB_SOURCE_NAME}" = "clsmix" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_clsbeta${EC_QFID_CLS_MIX_BETA_NAME}"
            if [ "${EC_QFID_CLS_MIX_MODE_NAME}" != "linear" ]; then
                EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_clsmode${EC_QFID_CLS_MIX_MODE_NAME}"
            fi
        fi
        if [ "${EC_QFID_CLS_GATE_NAME}" = "1" ] && [ "${EC_QFID_PROB_SOURCE_NAME}" = "clsmix" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_clsgate${EC_QFID_CLS_GATE_MODE_NAME}_clsgatebase${EC_QFID_CLS_GATE_BETA_BASE_NAME}_clsgatemin${EC_QFID_CLS_GATE_MIN_NAME}_clsgatemax${EC_QFID_CLS_GATE_MAX_NAME}"
        fi
        if [ "${EC_QFID_BUDGET_CALIB_NAME}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfidbudget1_qfidalpha${EC_QFID_BUDGET_ALPHA_NAME}"
        fi
        if [ "${EC_QFID_SPECTRAL_FILTER_NAME}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_qfidsf1_qfidsg${EC_QFID_SPECTRAL_GAMMA_NAME}_qfidseps${EC_QFID_SPECTRAL_EPS_NAME}_qfidstrace${EC_QFID_SPECTRAL_TRACE_NORM_NAME}"
        fi
        if [ "${EC_QFID_SELECT_MODE_NAME}" = "cls_topk" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_clstopk_K${TOKEN}"
        elif [ "${EC_QFID_PROB_SOURCE_NAME}" = "cls_only_qf" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_cls_only_qf_K${TOKEN}"
        fi
        if [ "${EC_QFID_FINAL_PROFILE:-0}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_eaqf_clsmix_linear_beta0105_layer_m2_headmean_density_tau050_depol015_K${TOKEN}"
        fi
        if [ "${EC_QFID_GATE_PROFILE:-0}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_eaqf_gate_agreement_beta_base0105_layer_m2_K${TOKEN}"
        fi
        if [ "${EC_QFID_CG_FINAL_PROFILE:-0}" = "1" ]; then
            EC_QFID_EXTRA_TAG="${EC_QFID_EXTRA_TAG}_cg_eaqf_gate_agreement_beta_base0105_layer_m2_density_depol015_K${TOKEN}"
        fi
        EC_QFID_SCORE_TAG="_scoreqfid_qfidtau${EC_QFID_TAU_NAME}_qfidprob${EC_QFID_PROB_SOURCE_NAME}_qfidkernel${EC_QFID_KERNEL_NAME}_qfiddep${EC_QFID_DEPOLARIZE_NAME}${EC_QFID_EXTRA_TAG}"
    fi

    if [ -n "${EC_QSP_PROFILE:-}" ]; then
        PARAM="${EC_QSP_PROFILE}_k${TOKEN}"
    elif [ -n "${EC_QMO_PROFILE:-}" ]; then
        PARAM="${EC_QMO_PROFILE}_k${TOKEN}"
    elif [ -n "${EC_QFID_BUDGET_PROFILE:-}" ]; then
        PARAM="ec_pruner_qfid_cg_eaqf_budget_${EC_QFID_BUDGET_PROFILE}_K${TOKEN}"
    elif [ -n "${EC_QFID_SPECTRAL_PROFILE:-}" ]; then
        PARAM="ec_pruner_qfid_cg_eaqf_spectral_${EC_QFID_SPECTRAL_PROFILE}_K${TOKEN}"
    elif [ -n "${EC_QFI_CARD_PROFILE:-}" ]; then
        PARAM="${EC_QFI_CARD_PROFILE}_k${TOKEN}"
    elif [ "${EC_QFID_CORE_RECOVER_PROFILE:-}" = "r050" ]; then
        PARAM="qfi_cr50_k${TOKEN}"
    elif [ "${EC_QFID_CORE_RECOVER_PROFILE:-}" = "r075" ]; then
        PARAM="qfi_cr75_k${TOKEN}"
    elif [ "${EC_QFID_CORE_RECOVER_PROFILE:-}" = "r08125" ]; then
        PARAM="qfi_cr8125_k${TOKEN}"
    elif [ "${EC_QFID_CORE_RECOVER_PROFILE:-}" = "r0875" ]; then
        PARAM="qfi_cr875_k${TOKEN}"
    elif [ "${EC_QFID_ADAPTIVE_RECOVER_PROFILE:-}" = "budget" ]; then
        PARAM="qfi_adapt_bgt_k${TOKEN}"
    elif [ "${EC_QFID_ADAPTIVE_RECOVER_PROFILE:-}" = "prior" ]; then
        PARAM="qfi_adapt_prior_k${TOKEN}"
    elif [ "${EC_QFID_ADAPTIVE_RECOVER_PROFILE:-}" = "final" ]; then
        PARAM="qfi_adapt_final_k${TOKEN}"
    elif [ "${EC_QFID_EVIDENCE_PROFILE:-0}" = "1" ]; then
        PARAM="qfi_er_k${TOKEN}"
    elif [ "${EC_QFID_CG_FINAL_PROFILE:-0}" = "1" ]; then
        PARAM="ec_pruner_qfid_cg_eaqf_gate_agreement_beta_base0105_layer_m2_density_depol015_K${TOKEN}"
    elif [ "${EC_QFID_GATE_PROFILE:-0}" = "1" ]; then
        PARAM="ec_pruner_qfid_eaqf_gate_agreement_beta_base0105_layer_m2_headmean_density_tau050_depol015_K${TOKEN}"
    elif [ -n "${EC_QFID_ABLATION_PROFILE:-}" ]; then
        PARAM="ec_pruner_qfid_ablate_${EC_QFID_ABLATION_PROFILE}_K${TOKEN}"
    else
        PARAM="${PRUNE_METHOD_NAME}_${EC_SCORE_SOURCE_NAME}${EC_QMO_SCORE_TAG}${EC_QFID_SCORE_TAG}_solver${EC_SOLVER_NAME}_steps${EC_QA_STEPS_NAME}_swap${EC_QA_MAX_SWAP_RATIO_TAG}_g${EC_GLOBAL_CANDIDATE_RATIO_TAG}_cand${EC_CANDIDATE_RATIO_TAG}_sem${EC_USE_SEMANTIC_CANDIDATE_NAME}_spatial${EC_USE_SPATIAL_CANDIDATE_NAME}_spxg${EC_SPATIAL_EXCLUDE_GLOBAL_NAME}_r${EC_USE_REPULSION_NAME}_c${EC_USE_COMPLEMENT_NAME}_phi${EC_USE_PHI_NAME}_chain${EC_USE_CHAIN_COVERAGE_NAME}_gc${EC_GAMMA_CHAIN_TAG}_l${EC_LAMBDA_TAG}_d${EC_DELTA_TAG}_vtn${TOKEN}"
    fi
fi

if [ -n "${EC_QFID_OVERLAP_KERNEL:-}" ]; then
    EC_QFID_OVERLAP_KERNEL_TAG="$(printf '%s' "${EC_QFID_OVERLAP_KERNEL}" | tr -cd '[:alnum:]_')"
    PARAM="${PARAM}_ov${EC_QFID_OVERLAP_KERNEL_TAG}"
fi

if [ -n "${EC_QFICR_ABLATION_PROFILE:-}" ]; then
    EC_QFICR_ABLATION_TAG="$(printf '%s' "${EC_QFICR_ABLATION_PROFILE}" | tr -cd '[:alnum:]_')"
    PARAM="${PARAM}_qficr${EC_QFICR_ABLATION_TAG}"
fi

if [ -n "${EC_QFICR_TUNING_PROFILE:-}" ]; then
    EC_QFICR_TUNING_TAG="$(printf '%s' "${EC_QFICR_TUNING_PROFILE}" | tr -cd '[:alnum:]_')"
    PARAM="${PARAM}_qficrtune${EC_QFICR_TUNING_TAG}"
fi

if [ "${SMOKE_RUN}" = "1" ]; then
    PARAM="${PARAM}_smoke${NUM_SAMPLES_NAME}"
fi

answers_dir="./playground/data/eval/gqa/answers/${SPLIT}/${CKPT}/${PARAM}"
mkdir -p "${answers_dir}"

PIDS=()
ANSWER_FILES=()

for IDX in $(seq 0 $((CHUNKS-1))); do
    answer_file="${answers_dir}/${CHUNKS}_${IDX}.jsonl"
    ANSWER_FILES+=("${answer_file}")
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -u -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ${QUESTION_FILE} \
        --image-folder ${IMAGE_DIR} \
        --answers-file "${answer_file}" \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
    PIDS+=($!)
done

FAILED_PIDS=()
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAILED_PIDS+=("${pid}")
    fi
done

if [ "${#FAILED_PIDS[@]}" -gt 0 ]; then
    echo "GQA worker failed: pids=${FAILED_PIDS[*]}" >&2
    exit 1
fi

output_file="${answers_dir}/merge.jsonl"

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    answer_file="${ANSWER_FILES[$IDX]}"
    if [ ! -s "${answer_file}" ]; then
        echo "Missing or empty answer shard: ${answer_file}" >&2
        exit 1
    fi
    cat "${answer_file}" >> "$output_file"
done

if [ ! -s "$output_file" ]; then
    echo "Merged GQA prediction file is empty: ${output_file}" >&2
    exit 1
fi

if [ "${SMOKE_RUN}" = "1" ]; then
    echo "[gqa-smoke] completed samples=${NUM_SAMPLES_NAME} predictions=${output_file}"
    exit 0
fi

# The official evaluator requires a fixed prediction filename, so serialize only
# the short conversion/evaluation step when CDPruner and EC-Pruner run together.
(
    flock 9
    python -u scripts/convert_gqa_for_eval.py \
        --src "$output_file" \
        --dst "${GQA_DIR}/testdev_balanced_predictions.json"

    cd "${GQA_DIR}"
    python -u eval/eval.py \
        --questions "${GQA_QUESTION_DIR}/{tier}_questions.json" \
        --predictions "${GQA_DIR}/{tier}_predictions.json" \
        --tier testdev_balanced

    rm -f testdev_balanced_predictions.json
) 9>"${GQA_DIR}/.evaluation.lock"
