#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-logs/qficr_general_tuning}"
OUT_ROOT="${OUT_ROOT:-outputs/qficr_general_tuning}"
POPE_SCRIPT="scripts/v1_5/eval/pope.sh"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/pope/llava_pope_test.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/gpuadmin/txr/CDPruner_data/pope/val2014}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/gpuadmin/txr/CDPruner_data/pope/coco}"
CKPT_DIR="${CKPT_DIR:-/home/gpuadmin/txr/CDPruner/checkpoints}"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

if [ -f /home/gpuadmin/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
    conda activate cdpruner
fi

setting_to_env() {
    local setting="$1"
    ENTROPY_ANCHOR_RATIO="0.0"
    RESIDUAL_BUDGET_GAMMA="-1.0"
    CLS_PRIOR_LAYERS="last2"
    CLS_ATTN_LAYER="-2"
    case "${setting}" in
        baseline_last2) ;;
        entropy_anchor_005) ENTROPY_ANCHOR_RATIO="0.05" ;;
        entropy_anchor_010) ENTROPY_ANCHOR_RATIO="0.10" ;;
        residual_budget_050) RESIDUAL_BUDGET_GAMMA="0.50" ;;
        residual_budget_075) RESIDUAL_BUDGET_GAMMA="0.75" ;;
        cls_last4_avg) CLS_PRIOR_LAYERS="last4"; CLS_ATTN_LAYER="last4mean" ;;
        cls_last1_avg) CLS_PRIOR_LAYERS="last1"; CLS_ATTN_LAYER="last" ;;
        *) echo "[general] unknown setting=${setting}" >&2; return 2 ;;
    esac
}

write_deferred_commands() {
    local file="${LOG_ROOT}/deferred_commands.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "cd ${REPO_ROOT}"
        for k in 32 64; do
            echo "bash run_gqa_profile.sh qficr_general_cls_last1_avg 0 ${k} 2>&1 | tee ${LOG_ROOT}/gqa_k${k}_cls_last1_avg.log"
            echo "LOG_ROOT=${LOG_ROOT} OUT_ROOT=${OUT_ROOT} bash ${BASH_SOURCE[0]} run_one pope ${k} cls_last1_avg 1"
        done
    } > "${file}"
    chmod +x "${file}"
}

run_gqa() {
    local k="$1"
    local setting="$2"
    local gpu="$3"
    local exp="gqa_k${k}_${setting}"
    local profile="qficr_general_${setting}"
    local log="${LOG_ROOT}/${exp}.log"
    local out_dir="${OUT_ROOT}/${exp}"
    mkdir -p "${out_dir}"
    setting_to_env "${setting}" || return $?
    {
        echo "[general] experiment=${exp}"
        echo "[general] setting=${setting}"
        echo "[general] dataset=GQA"
        echo "[general] K=${k}"
        echo "[general] gpu=${gpu}"
        echo "[general] output_dir=${out_dir}"
        echo "[general] profile=${profile}"
        echo "[general] entropy_anchor_ratio=${ENTROPY_ANCHOR_RATIO}"
        echo "[general] residual_budget_gamma=${RESIDUAL_BUDGET_GAMMA}"
        echo "[general] cls_prior_layers=${CLS_PRIOR_LAYERS}"
        echo "[general] cls_attn_layer=${CLS_ATTN_LAYER}"
        echo "[general] started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        bash run_gqa_profile.sh "${profile}" "${gpu}" "${k}"
        code=$?
        echo "[general] finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "[general] exit=${code}"
        exit "${code}"
    } > "${log}" 2>&1
}

run_pope() {
    local k="$1"
    local setting="$2"
    local gpu="$3"
    local exp="pope_k${k}_${setting}"
    local log="${LOG_ROOT}/${exp}.log"
    local out_dir="${OUT_ROOT}/${exp}"
    local answer="${out_dir}/answers.jsonl"
    mkdir -p "${out_dir}"
    setting_to_env "${setting}" || return $?
    {
        echo "[general] experiment=${exp}"
        echo "[general] setting=${setting}"
        echo "[general] dataset=POPE"
        echo "[general] K=${k}"
        echo "[general] gpu=${gpu}"
        echo "[general] output_dir=${out_dir}"
        echo "[general] answer_file=${answer}"
        echo "[general] entropy_anchor_ratio=${ENTROPY_ANCHOR_RATIO}"
        echo "[general] residual_budget_gamma=${RESIDUAL_BUDGET_GAMMA}"
        echo "[general] cls_prior_layers=${CLS_PRIOR_LAYERS}"
        echo "[general] cls_attn_layer=${CLS_ATTN_LAYER}"
        echo "[general] started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        env \
            TRANSFORMERS_OFFLINE=1 \
            HF_HUB_OFFLINE=1 \
            HF_DATASETS_OFFLINE=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            CUDA_VISIBLE_DEVICES="${gpu}" \
            QUESTION_FILE="${QUESTION_FILE}" \
            IMAGE_FOLDER="${IMAGE_FOLDER}" \
            ANNOTATION_DIR="${ANNOTATION_DIR}" \
            ANSWER_FILE="${answer}" \
            CKPT_DIR="${CKPT_DIR}" \
            PRUNE_METHOD=ec_pruner \
            EC_QFICR_TUNING_PROFILE="general_${setting}" \
            EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior \
            EC_QFID_CG_FINAL_PROFILE=0 \
            EC_QFID_FINAL_PROFILE=0 \
            EC_QFID_GATE_PROFILE=0 \
            EC_SCORE_SOURCE=qfid \
            EC_QFID_SELECT_MODE=qf \
            EC_QFID_PROB_SOURCE=clsmix \
            EC_QFID_CLS_MIX_MODE=linear \
            EC_QFID_CLS_MIX_BETA=0.105 \
            EC_QFID_CLS_ATTN_LAYER="${CLS_ATTN_LAYER}" \
            EC_QFICR_CLS_PRIOR_LAYERS="${CLS_PRIOR_LAYERS}" \
            EC_QFID_CLS_HEAD_REDUCE=mean \
            EC_QFID_CLS_GATE=1 \
            EC_QFID_CLS_GATE_MODE=agreement \
            EC_QFID_CLS_GATE_BETA_BASE=0.105 \
            EC_QFID_CLS_GATE_MIN=0.5 \
            EC_QFID_CLS_GATE_MAX=1.5 \
            EC_QFID_KERNEL=density \
            EC_QFID_TAU=0.50 \
            EC_QFID_EPS=1e-6 \
            EC_QFID_OVERLAP_KERNEL=relu_square \
            EC_QFID_DEPOLARIZE_MODE=fixed \
            EC_QFID_DEPOLARIZE=0.15 \
            EC_QFID_SELECTOR=adaptive_core_recover \
            EC_QFID_ADAPT_MODE=entropy_question \
            EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00 \
            EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25 \
            EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125 \
            EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625 \
            EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25 \
            EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25 \
            EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25 \
            EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125 \
            EC_QFID_ADAPT_RECOVER_CAP=16 \
            EC_QFID_ADAPT_ENTROPY_GATE=1 \
            EC_QFID_ADAPT_ENTROPY_LOW=0.70 \
            EC_QFID_ADAPT_ENTROPY_HIGH=0.95 \
            EC_QFID_RECOVER_PRIOR_ANCHOR=1 \
            EC_QFID_RECOVER_PRIOR_GAMMA=0.5 \
            EC_QFICR_ANCHOR_ALPHA=0.5 \
            EC_QFID_RECOVER_CAND_POOL=0 \
            EC_QFID_RECOVER_CAND_MULT=3.0 \
            EC_QFICR_RESTORATION_MODE=full \
            EC_QFICR_FIXED_RESTORATION_RATIO=0.2 \
            EC_QFICR_USE_SPATIAL_LOCAL_PRIOR=0 \
            EC_QFICR_PRIOR_ABLATION=none \
            EC_QFICR_PRIOR_LAMBDA=0.0 \
            EC_QFICR_OBSERVATION_MODE=full \
            EC_QFICR_REST_BETA=1.0 \
            EC_QFICR_OBS_TEMPERATURE=1.0 \
            EC_QFICR_RESTORE_DIV_LAMBDA=0.0 \
            EC_QFICR_RESTORE_CANDIDATE_FACTOR=2.0 \
            EC_QFICR_YN_BUDGET_GATE=1.0 \
            EC_QFICR_ENTROPY_ANCHOR_RATIO="${ENTROPY_ANCHOR_RATIO}" \
            EC_QFICR_RESIDUAL_BUDGET_GAMMA="${RESIDUAL_BUDGET_GAMMA}" \
            EC_QFICR_RANDOM_SEED=42 \
            EC_QFID_BUDGET_CALIB=0 \
            EC_QFID_SPECTRAL_FILTER=0 \
            EC_QFID_SPATIAL_STATE=0 \
            EC_QFID_MEASURE_PRIOR_MODE=none \
            EC_QFID_ANCHOR_MODE=none \
            EC_USE_SEMANTIC_CANDIDATE=0 \
            EC_USE_SPATIAL_CANDIDATE=0 \
            EC_USE_REPULSION=0 \
            EC_USE_COMPLEMENT=0 \
            EC_USE_PHI=0 \
            EC_USE_CHAIN_COVERAGE=0 \
            EC_SOLVER=greedy \
            bash "${POPE_SCRIPT}" "${k}"
        code=$?
        echo "[general] finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "[general] exit=${code}"
        exit "${code}"
    } > "${log}" 2>&1
}

wait_batch() {
    local failures=0
    for pid in "$@"; do
        if ! wait "${pid}"; then
            failures=$((failures + 1))
        fi
    done
    return "${failures}"
}

run_setting_batch() {
    local setting="$1"
    echo "[general] launching setting=${setting}"
    run_gqa 32 "${setting}" 0 & p0=$!
    run_gqa 64 "${setting}" 1 & p1=$!
    run_pope 32 "${setting}" 2 & p2=$!
    run_pope 64 "${setting}" 3 & p3=$!
    wait_batch "${p0}" "${p1}" "${p2}" "${p3}"
}

if [ "${1:-}" = "run_one" ]; then
    run_"${2}" "${3}" "${4}" "${5}"
    exit $?
fi

write_deferred_commands

echo "[general] launching 24 runs: 6 settings x GQA/POPE K=32/64"
failures=0
for setting in baseline_last2 entropy_anchor_005 entropy_anchor_010 residual_budget_050 residual_budget_075 cls_last4_avg; do
    run_setting_batch "${setting}" || failures=$((failures + $?))
done

python scripts/collect_qficr_general_tuning_results.py

echo "[general] failures=${failures}"
echo "[general] summary=${OUT_ROOT}/summary.csv"
echo "[general] compare=${OUT_ROOT}/compare_against_baseline.csv"
echo "[general] deferred=${LOG_ROOT}/deferred_commands.sh"
exit "${failures}"
