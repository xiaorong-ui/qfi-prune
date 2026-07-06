#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOG_ROOT="${LOG_ROOT:-logs/pope_qficr_cdpruner_k}"
OUT_ROOT="${OUT_ROOT:-outputs/pope_qficr_cdpruner_k}"
POPE_SCRIPT="scripts/v1_5/eval/pope.sh"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/pope/llava_pope_test.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/gpuadmin/txr/CDPruner_data/pope/val2014}"
ANNOTATION_DIR="${ANNOTATION_DIR:-/home/gpuadmin/txr/CDPruner_data/pope/coco}"
CKPT_DIR="${CKPT_DIR:-/home/gpuadmin/txr/CDPruner/checkpoints}"
CKPT="llava-v1.5-7b"

mkdir -p "${LOG_ROOT}" "${OUT_ROOT}"

if [ -f /home/gpuadmin/anaconda3/etc/profile.d/conda.sh ]; then
    # Keep the same environment used by the existing project scripts.
    source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
    conda activate cdpruner
fi

check_required_paths() {
    local missing=0
    for path in \
        "${POPE_SCRIPT}" \
        "${QUESTION_FILE}" \
        "${IMAGE_FOLDER}" \
        "${ANNOTATION_DIR}" \
        "${CKPT_DIR}/${CKPT}" \
        "llava/eval/model_vqa_loader.py" \
        "llava/eval/eval_pope.py"; do
        if [ ! -e "${path}" ]; then
            echo "[pope-k] missing required path: ${path}" >&2
            missing=1
        fi
    done
    return "${missing}"
}

status_for_log() {
    local log="$1"
    if [ ! -f "${log}" ]; then
        echo "missing"
    elif grep -qiE "CUDA out of memory|outofmemory|Killed" "${log}"; then
        echo "failed_oom_or_killed"
    elif grep -q "^Average F1 score:" "${log}"; then
        echo "completed"
    else
        echo "failed_or_incomplete"
    fi
}

run_one() {
    local exp="$1"
    local method="$2"
    local token="$3"
    local gpu="$4"
    local log="${LOG_ROOT}/${exp}.log"
    local out_dir="${OUT_ROOT}/${exp}"
    local answer="${out_dir}/answers.jsonl"

    mkdir -p "${out_dir}"
    {
        echo "[pope-k] experiment=${exp}"
        echo "[pope-k] dataset=POPE"
        echo "[pope-k] method=${method}"
        echo "[pope-k] K=${token}"
        echo "[pope-k] gpu=${gpu}"
        echo "[pope-k] question_file=${QUESTION_FILE}"
        echo "[pope-k] image_folder=${IMAGE_FOLDER}"
        echo "[pope-k] annotation_dir=${ANNOTATION_DIR}"
        echo "[pope-k] answer_file=${answer}"
        echo "[pope-k] started_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

        common_env=(
            "TRANSFORMERS_OFFLINE=1"
            "HF_HUB_OFFLINE=1"
            "HF_DATASETS_OFFLINE=1"
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            "CUDA_VISIBLE_DEVICES=${gpu}"
            "QUESTION_FILE=${QUESTION_FILE}"
            "IMAGE_FOLDER=${IMAGE_FOLDER}"
            "ANNOTATION_DIR=${ANNOTATION_DIR}"
            "ANSWER_FILE=${answer}"
            "CKPT_DIR=${CKPT_DIR}"
        )

        if [ "${method}" = "cdpruner" ]; then
            echo "[pope-k] prune_branch=cdpruner"
            env "${common_env[@]}" \
                PRUNE_METHOD=cdpruner \
                bash "${POPE_SCRIPT}" "${token}"
        else
            echo "[pope-k] prune_branch=ec_pruner_qficr_stable"
            echo "[pope-k] qficr_stable=overlap_relu_square restoration_full observation_full_clsmix_agreement anchor_alpha_0.5 no_spatial_local_prior rho_max_0.25"
            env "${common_env[@]}" \
                PRUNE_METHOD=ec_pruner \
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
                EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior \
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
                bash "${POPE_SCRIPT}" "${token}"
        fi
        code=$?
        echo "[pope-k] finished_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "[pope-k] exit=${code}"
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

write_status() {
    local status_file="${LOG_ROOT}/status.tsv"
    printf 'experiment\tmethod\tK\tstatus\tlog_path\toutput_dir\n' > "${status_file}"
    for spec in \
        "cdpruner_pope_k32 cdpruner 32" \
        "cdpruner_pope_k64 cdpruner 64" \
        "cdpruner_pope_k128 cdpruner 128" \
        "qficr_pope_k32 qficr 32" \
        "qficr_pope_k64 qficr 64" \
        "qficr_pope_k128 qficr 128"; do
        set -- ${spec}
        local exp="$1" method="$2" token="$3"
        local log="${LOG_ROOT}/${exp}.log"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${exp}" "${method}" "${token}" "$(status_for_log "${log}")" \
            "${log}" "${OUT_ROOT}/${exp}" >> "${status_file}"
    done
    cat "${status_file}"
}

if ! check_required_paths; then
    exit 1
fi

echo "[pope-k] launching batch 1"
run_one cdpruner_pope_k32 cdpruner 32 0 &
pid0=$!
run_one cdpruner_pope_k64 cdpruner 64 1 &
pid1=$!
run_one cdpruner_pope_k128 cdpruner 128 2 &
pid2=$!
run_one qficr_pope_k32 qficr 32 3 &
pid3=$!
wait_batch "${pid0}" "${pid1}" "${pid2}" "${pid3}"
batch1_failures=$?
echo "[pope-k] batch 1 failures=${batch1_failures}"

echo "[pope-k] launching batch 2"
run_one qficr_pope_k64 qficr 64 0 &
pid4=$!
run_one qficr_pope_k128 qficr 128 1 &
pid5=$!
wait_batch "${pid4}" "${pid5}"
batch2_failures=$?
echo "[pope-k] batch 2 failures=${batch2_failures}"

echo "[pope-k] collecting results"
python scripts/collect_pope_qficr_cdpruner_k_results.py
echo "[pope-k] status"
write_status
echo "[pope-k] summary=${OUT_ROOT}/summary.csv"
echo "[pope-k] compare=${OUT_ROOT}/compare_qficr_vs_cdpruner_pope.csv"
exit $((batch1_failures + batch2_failures))
