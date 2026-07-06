#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-logs/qficr_micro_tuning_k64}"
OUT_ROOT="${OUT_ROOT:-outputs/qficr_micro_tuning_k64}"
POPE_SCRIPT="scripts/v1_5/eval/pope.sh"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/pope/llava_pope_test.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/gpuadmin/txr/CDPruner_data/pope/val2014}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/gpuadmin/txr/CDPruner_data/pope/coco}"
CKPT_DIR="${CKPT_DIR:-/home/gpuadmin/txr/CDPruner/checkpoints}"
TOKEN=64

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

if [ -f /home/gpuadmin/anaconda3/etc/profile.d/conda.sh ]; then
    source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
    conda activate cdpruner
fi

setting_to_env() {
    local setting="$1"
    REST_BETA="1.0"
    OBS_TEMPERATURE="1.0"
    RESTORE_DIV_LAMBDA="0.0"
    RESTORE_CANDIDATE_FACTOR="2.0"
    YN_BUDGET_GATE="1.0"
    case "${setting}" in
        baseline) ;;
        beta_075) REST_BETA="0.75" ;;
        beta_125) REST_BETA="1.25" ;;
        beta_150) REST_BETA="1.50" ;;
        tau_085) OBS_TEMPERATURE="0.85" ;;
        tau_115) OBS_TEMPERATURE="1.15" ;;
        tau_130) OBS_TEMPERATURE="1.30" ;;
        div_010) RESTORE_DIV_LAMBDA="0.10"; RESTORE_CANDIDATE_FACTOR="2.0" ;;
        yn_gate_05) YN_BUDGET_GATE="0.5" ;;
        *) echo "[micro] unknown setting=${setting}" >&2; return 2 ;;
    esac
}

write_deferred_commands() {
    local file="${LOG_ROOT}/deferred_commands.sh"
    {
        echo "#!/usr/bin/env bash"
        echo "cd ${REPO_ROOT}"
        for setting in beta_075 beta_150 tau_130 yn_gate_05; do
            local profile="qficr_micro_${setting}"
            echo "bash run_gqa_profile.sh ${profile} 0 64 2>&1 | tee ${LOG_ROOT}/gqa_${setting}.log"
            echo "LOG_ROOT=${LOG_ROOT} OUT_ROOT=${OUT_ROOT} bash ${BASH_SOURCE[0]} run_one pope ${setting} 1"
        done
    } > "${file}"
    chmod +x "${file}"
}

run_gqa() {
    local setting="$1"
    local gpu="$2"
    local exp="gqa_${setting}"
    local log="${LOG_ROOT}/${exp}.log"
    local out_dir="${OUT_ROOT}/${exp}"
    mkdir -p "${out_dir}"
    {
        echo "[micro] experiment=${exp}"
        echo "[micro] dataset=GQA"
        echo "[micro] K=${TOKEN}"
        echo "[micro] gpu=${gpu}"
        echo "[micro] output_dir=${out_dir}"
        echo "[micro] profile=qficr_micro_${setting}"
        echo "[micro] started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        bash run_gqa_profile.sh "qficr_micro_${setting}" "${gpu}" "${TOKEN}"
        code=$?
        echo "[micro] finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "[micro] exit=${code}"
        exit "${code}"
    } > "${log}" 2>&1
}

run_pope() {
    local setting="$1"
    local gpu="$2"
    local exp="pope_${setting}"
    local log="${LOG_ROOT}/${exp}.log"
    local out_dir="${OUT_ROOT}/${exp}"
    local answer="${out_dir}/answers.jsonl"
    mkdir -p "${out_dir}"
    setting_to_env "${setting}" || return $?
    {
        echo "[micro] experiment=${exp}"
        echo "[micro] dataset=POPE"
        echo "[micro] K=${TOKEN}"
        echo "[micro] gpu=${gpu}"
        echo "[micro] output_dir=${out_dir}"
        echo "[micro] answer_file=${answer}"
        echo "[micro] rest_beta=${REST_BETA}"
        echo "[micro] obs_temperature=${OBS_TEMPERATURE}"
        echo "[micro] restore_div_lambda=${RESTORE_DIV_LAMBDA}"
        echo "[micro] restore_candidate_factor=${RESTORE_CANDIDATE_FACTOR}"
        echo "[micro] yn_budget_gate=${YN_BUDGET_GATE}"
        echo "[micro] started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
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
            EC_QFICR_TUNING_PROFILE="micro_${setting}" \
            EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior \
            EC_QFID_CG_FINAL_PROFILE=0 \
            EC_QFID_FINAL_PROFILE=0 \
            EC_QFID_GATE_PROFILE=0 \
            EC_SCORE_SOURCE=qfid \
            EC_QFID_SELECT_MODE=qf \
            EC_QFID_PROB_SOURCE=clsmix \
            EC_QFID_CLS_MIX_MODE=linear \
            EC_QFID_CLS_MIX_BETA=0.105 \
            EC_QFID_CLS_ATTN_LAYER=-2 \
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
            EC_QFICR_REST_BETA="${REST_BETA}" \
            EC_QFICR_OBS_TEMPERATURE="${OBS_TEMPERATURE}" \
            EC_QFICR_RESTORE_DIV_LAMBDA="${RESTORE_DIV_LAMBDA}" \
            EC_QFICR_RESTORE_CANDIDATE_FACTOR="${RESTORE_CANDIDATE_FACTOR}" \
            EC_QFICR_YN_BUDGET_GATE="${YN_BUDGET_GATE}" \
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
            bash "${POPE_SCRIPT}" "${TOKEN}"
        code=$?
        echo "[micro] finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "[micro] exit=${code}"
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

if [ "${1:-}" = "run_one" ]; then
    run_"${2}" "${3}" "${4}"
    exit $?
fi

write_deferred_commands

echo "[micro] launching selected 10 runs: baseline beta_125 tau_085 tau_115 div_010 on GQA+POPE K=64"

failures=0

run_gqa baseline 0 & p0=$!
run_gqa beta_125 1 & p1=$!
run_gqa tau_085 2 & p2=$!
run_gqa tau_115 3 & p3=$!
wait_batch "${p0}" "${p1}" "${p2}" "${p3}" || failures=$((failures + $?))

run_gqa div_010 0 & p0=$!
run_pope baseline 1 & p1=$!
run_pope beta_125 2 & p2=$!
run_pope tau_085 3 & p3=$!
wait_batch "${p0}" "${p1}" "${p2}" "${p3}" || failures=$((failures + $?))

run_pope tau_115 0 & p0=$!
run_pope div_010 1 & p1=$!
wait_batch "${p0}" "${p1}" || failures=$((failures + $?))

python scripts/collect_qficr_micro_tuning_k64_results.py

echo "[micro] failures=${failures}"
echo "[micro] summary=${OUT_ROOT}/summary.csv"
echo "[micro] compare=${OUT_ROOT}/compare_against_baseline.csv"
echo "[micro] deferred=${LOG_ROOT}/deferred_commands.sh"
exit "${failures}"
