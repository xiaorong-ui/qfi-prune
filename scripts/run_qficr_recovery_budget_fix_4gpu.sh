#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/llava-v1.5-7b}"
OUT_ROOT="${ROOT}/outputs/qficr_recovery_budget_fix"
LOG_ROOT="${ROOT}/logs/qficr_recovery_budget_fix"
COLLECTOR="${ROOT}/scripts/collect_qficr_recovery_budget_fix_results.py"
STATUS_TSV="${LOG_ROOT}/run_status.tsv"
MATRIX_MODE="${MATRIX_MODE:-minimal}"
GPU_LIST="${GPU_LIST:-0 1 2 3}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-30000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/playground/data/eval/MME/answers"

if [ -f /home/gpuadmin/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
  conda activate cdpruner 2>/dev/null || true
fi

echo "[INFO] runner_pid=$$"
echo "[INFO] root=${ROOT}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] out_root=${OUT_ROOT}"
echo "[INFO] log_root=${LOG_ROOT}"
echo "[INFO] matrix_mode=${MATRIX_MODE}"
echo "[INFO] gpu_list=${GPU_LIST}"
echo "[INFO] gpu_free_threshold_mib=${GPU_FREE_THRESHOLD_MIB}"

python "${COLLECTOR}" || true
printf "run_id\tvariant\tbenchmark\tK\tgpu\tstatus\texit_code\tlog_path\toutput_dir\tnotes\n" > "${STATUS_TSV}"

common_env() {
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
}

apply_qficr_stable_env() {
  export PRUNE_METHOD=ec_pruner
  export EC_SCORE_SOURCE=qfid
  export EC_QFID_SELECT_MODE=qf
  export EC_QFID_PROB_SOURCE=clsmix
  export EC_QFID_CLS_MIX_MODE=linear
  export EC_QFID_CLS_MIX_BETA=0.105
  export EC_QFID_CLS_ATTN_LAYER=-2
  export EC_QFICR_CLS_PRIOR_LAYERS=last2
  export EC_QFID_CLS_HEAD_REDUCE=mean
  export EC_QFID_KERNEL=density
  export EC_QFID_TAU=0.50
  export EC_QFID_EPS=1e-6
  export EC_QFID_OVERLAP_KERNEL=relu_square
  export EC_QFID_DEPOLARIZE_MODE=fixed
  export EC_QFID_DEPOLARIZE=0.15
  export EC_QFID_SELECTOR=adaptive_core_recover
  export EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior
  export EC_QFID_ADAPT_MODE=entropy_question
  export EC_QFID_ADAPT_RECOVER_MIN_RATIO=0.00
  export EC_QFID_ADAPT_RECOVER_MAX_RATIO=0.25
  export EC_QFID_ADAPT_RECOVER_DEFAULT_RATIO=0.125
  export EC_QFID_ADAPT_RECOVER_BINARY_RATIO=0.0625
  export EC_QFID_ADAPT_RECOVER_OPEN_RATIO=0.25
  export EC_QFID_ADAPT_RECOVER_REL_RATIO=0.25
  export EC_QFID_ADAPT_RECOVER_COMPARE_RATIO=0.25
  export EC_QFID_ADAPT_RECOVER_ATTR_RATIO=0.125
  export EC_QFID_ADAPT_RECOVER_CAP=16
  export EC_QFID_ADAPT_ENTROPY_GATE=1
  export EC_QFID_ADAPT_ENTROPY_LOW=0.70
  export EC_QFID_ADAPT_ENTROPY_HIGH=0.95
  export EC_QFID_RECOVER_PRIOR_ANCHOR=1
  export EC_QFID_RECOVER_PRIOR_GAMMA=0.5
  export EC_QFICR_ANCHOR_ALPHA=0.5
  export EC_QFID_RECOVER_CAND_POOL=0
  export EC_QFID_RECOVER_CAND_MULT=3.0
  export EC_QFICR_RESTORATION_MODE=full
  export EC_QFICR_USE_SPATIAL_LOCAL_PRIOR=0
  export EC_QFICR_PRIOR_ABLATION=none
  export EC_QFICR_PRIOR_LAMBDA=0.0
  export EC_QFICR_OBSERVATION_MODE=full
  export EC_QFICR_RANDOM_SEED=42
  export EC_QFICR_RESTORE_EXTRA_PRIOR=none
  export EC_QFICR_RESTORE_EXTRA_LAMBDA=0.0
  export EC_QFICR_RESTORE_DIV_LAMBDA=0.0
  export EC_QFICR_REST_BETA=1.0
  export EC_QFICR_OBS_TEMPERATURE=1.0
  export EC_QFICR_ENTROPY_ANCHOR_RATIO=0.0
  export EC_QFICR_RESIDUAL_BUDGET_GAMMA=none
  export EC_QFICR_YN_BUDGET_GATE=0
  export EC_QFID_BUDGET_CALIB=0
  export EC_QFID_SPECTRAL_FILTER=0
  export EC_QFID_SPATIAL_STATE=0
  export EC_QFID_MEASURE_PRIOR_MODE=none
  export EC_QFID_ANCHOR_MODE=none
  export EC_USE_SEMANTIC_CANDIDATE=0
  export EC_USE_SPATIAL_CANDIDATE=0
  export EC_USE_REPULSION=0
  export EC_USE_COMPLEMENT=0
  export EC_USE_PHI=0
  export EC_USE_CHAIN_COVERAGE=0
  export EC_SOLVER=greedy
}

apply_variant_env() {
  local variant="$1"
  export EC_QFID_DEBUG_STATS_JSONL=1
  export EC_QFICR_RECOVER_ABS_CAP=16
  export EC_QFICR_RECOVER_RATIO_CAP=0.25
  export EC_QFICR_RECOVER_ENTROPY_GAMMA=1.0
  export EC_QFICR_SUPPORT_BETA=0.0
  export EC_QFICR_SELF_DISCOUNT=0.0
  case "${variant}" in
    v0_abs_cap_baseline_unified_gate)
      export EC_QFICR_RECOVER_BUDGET_MODE=abs_cap
      export EC_QFICR_RECOVER_ABS_CAP=16
      ;;
    v1_ratio_only)
      export EC_QFICR_RECOVER_BUDGET_MODE=ratio_only
      ;;
    v3_ratio_cap_020)
      export EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap
      export EC_QFICR_RECOVER_RATIO_CAP=0.20
      ;;
    v5_ratio_cap_025_support_self)
      export EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap
      export EC_QFICR_RECOVER_RATIO_CAP=0.25
      export EC_QFICR_SUPPORT_BETA=0.25
      export EC_QFICR_SELF_DISCOUNT=0.25
      ;;
    v7_ratio_cap_025_gamma2)
      export EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap
      export EC_QFICR_RECOVER_RATIO_CAP=0.25
      export EC_QFICR_RECOVER_ENTROPY_GAMMA=2.0
      ;;
    *)
      echo "[ERROR] unsupported variant=${variant}"
      return 2
      ;;
  esac
}

record_status() {
  local run_id="$1" variant="$2" bench="$3" k="$4" gpu="$5" status="$6" code="$7" log="$8" out_dir="$9" notes="${10}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_id}" "${variant}" "${bench}" "${k}" "${gpu}" "${status}" "${code}" \
    "${log#${ROOT}/}" "${out_dir#${ROOT}/}" "${notes}" >> "${STATUS_TSV}"
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v gpu="${gpu}" '$1 == gpu { print $2 + 0; found = 1 } END { if (!found) exit 1 }'
}

wait_for_gpu() {
  local gpu="$1" free_mib
  while true; do
    free_mib="$(gpu_free_mib "${gpu}" 2>/dev/null || echo 0)"
    if [ "${free_mib}" -ge "${GPU_FREE_THRESHOLD_MIB}" ]; then
      echo "[GPU_READY] gpu=${gpu} free_mib=${free_mib}"
      return 0
    fi
    echo "[GPU_WAIT] gpu=${gpu} free_mib=${free_mib} threshold=${GPU_FREE_THRESHOLD_MIB} sleep=${GPU_WAIT_SECONDS}"
    sleep "${GPU_WAIT_SECONDS}"
  done
}

safe_paths() {
  local variant="$1" bench="$2" k="$3"
  local out_dir="${OUT_ROOT}/${variant}/${bench}_k${k}"
  local log="${LOG_ROOT}/${variant}_${bench}_k${k}.log"
  if [ -s "${out_dir}/answers.jsonl" ] && [ -s "${log}" ] && rg -q "\[RUN_DONE\]" "${log}"; then
    printf "%s\t%s\t%s\n" "${log}" "${out_dir}" "reuse"
    return 0
  fi
  if [ -e "${out_dir}" ] || [ -e "${log}" ]; then
    local stamp
    stamp="$(date -u +%Y%m%d_%H%M%S)"
    out_dir="${OUT_ROOT}/${variant}/${bench}_k${k}_${stamp}"
    log="${LOG_ROOT}/${variant}_${bench}_k${k}_${stamp}.log"
  fi
  printf "%s\t%s\t%s\n" "${log}" "${out_dir}" "run"
}

run_generation() {
  local bench="$1" k="$2" answer="$3"
  local module="" qfile="" image_folder="" extra_args=()
  case "${bench}" in
    textvqa)
      module="llava.eval.model_vqa_loader"
      qfile="${ROOT}/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
      image_folder="${DATA_ROOT}/textvqa/train_images"
      ;;
    pope)
      module="llava.eval.model_vqa_loader"
      qfile="${ROOT}/playground/data/eval/pope/llava_pope_test.jsonl"
      image_folder="${DATA_ROOT}/pope/val2014"
      ;;
    mme)
      module="llava.eval.model_vqa_loader"
      qfile="${ROOT}/playground/data/eval/MME/llava_mme.jsonl"
      image_folder="${DATA_ROOT}/MME/MME_Benchmark_release_version"
      ;;
    *)
      echo "[ERROR] unsupported benchmark=${bench}"
      return 2
      ;;
  esac
  [ -e "${qfile}" ] || { echo "[ERROR] missing question file: ${qfile}"; return 3; }
  [ -e "${image_folder}" ] || { echo "[ERROR] missing image folder: ${image_folder}"; return 3; }
  local cmd=(python -m "${module}" --model-path "${MODEL_PATH}" --question-file "${qfile}" --answers-file "${answer}" --visual_token_num "${k}" --temperature 0 --conv-mode vicuna_v1 --image-folder "${image_folder}")
  cmd+=("${extra_args[@]}")
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

run_eval() {
  local bench="$1" variant="$2" k="$3" out_dir="$4" answer="$5"
  case "${bench}" in
    textvqa)
      python -m llava.eval.eval_textvqa \
        --annotation-file "${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "${answer}"
      ;;
    pope)
      python -m llava.eval.eval_pope \
        --annotation-dir "${DATA_ROOT}/pope/coco" \
        --question-file "${ROOT}/playground/data/eval/pope/llava_pope_test.jsonl" \
        --result-file "${answer}"
      ;;
    mme)
      local mme_exp="recovery_${variant}_${bench}_k${k}"
      mkdir -p "${ROOT}/playground/data/eval/MME/answers"
      cp "${answer}" "${ROOT}/playground/data/eval/MME/answers/${mme_exp}.jsonl"
      if [ ! -e "${ROOT}/playground/data/eval/MME/MME_Benchmark_release_version" ]; then
        ln -s "${DATA_ROOT}/MME/MME_Benchmark_release_version" "${ROOT}/playground/data/eval/MME/MME_Benchmark_release_version"
      fi
      (
        cd "${ROOT}/playground/data/eval/MME" &&
          python convert_answer_to_mme.py --experiment "${mme_exp}" &&
          python eval_tool/calculation.py --results_dir "eval_tool/answers/${mme_exp}"
      )
      ;;
  esac
}

run_one() {
  local gpu="$1" variant="$2" bench="$3" k="$4"
  local run_id="${variant}_${bench}_k${k}"
  local path_pair log out_dir action answer code status
  path_pair="$(safe_paths "${variant}" "${bench}" "${k}")"
  log="${path_pair%%$'\t'*}"
  path_pair="${path_pair#*$'\t'}"
  out_dir="${path_pair%%$'\t'*}"
  action="${path_pair#*$'\t'}"
  if [ "${action}" = "reuse" ]; then
    echo "[SKIP] ${run_id} completed"
    record_status "${run_id}" "${variant}" "${bench}" "${k}" "${gpu}" "reused" "0" "${log}" "${out_dir}" "existing completed run"
    return 0
  fi
  mkdir -p "${out_dir}"
  answer="${out_dir}/answers.jsonl"
  {
    echo "[RUN_START] run_id=${run_id}"
    echo "[RUN_START] variant=${variant}"
    echo "[RUN_START] benchmark=${bench}"
    echo "[RUN_START] K=${k}"
    echo "[RUN_START] gpu=${gpu}"
    echo "[RUN_START] output_dir=${out_dir}"
    echo "[RUN_START] answer_file=${answer}"
    wait_for_gpu "${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export EC_QFID_DEBUG_BENCHMARK="${bench}"
    common_env
    apply_qficr_stable_env
    apply_variant_env "${variant}"
    echo "[CONFIG] method=qficr_stable budget_mode=${EC_QFICR_RECOVER_BUDGET_MODE} ratio_cap=${EC_QFICR_RECOVER_RATIO_CAP} abs_cap=${EC_QFICR_RECOVER_ABS_CAP} entropy_gamma=${EC_QFICR_RECOVER_ENTROPY_GAMMA} support_beta=${EC_QFICR_SUPPORT_BETA} self_discount=${EC_QFICR_SELF_DISCOUNT} yn_budget_gate=${EC_QFICR_YN_BUDGET_GATE}"
    run_generation "${bench}" "${k}" "${answer}"
    run_eval "${bench}" "${variant}" "${k}" "${out_dir}" "${answer}"
    echo "[RUN_DONE] run_id=${run_id}"
  } > "${log}" 2>&1
  code=$?
  status="failed"
  if [ "${code}" -eq 0 ] && [ -s "${answer}" ]; then
    status="completed"
  fi
  record_status "${run_id}" "${variant}" "${bench}" "${k}" "${gpu}" "${status}" "${code}" "${log}" "${out_dir}" ""
  echo "[STATUS] ${run_id} ${status} exit=${code} log=${log#${ROOT}/}"
  return 0
}

run_batch() {
  local -a specs=("$@")
  local -a pids=()
  local spec gpu variant bench k pid
  for spec in "${specs[@]}"; do
    IFS=: read -r gpu variant bench k <<< "${spec}"
    run_one "${gpu}" "${variant}" "${bench}" "${k}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
}

build_specs() {
  local -a variants benches ks specs=()
  if [ "${MATRIX_MODE}" = "phase1" ]; then
    variants=(v0_abs_cap_baseline_unified_gate v1_ratio_only v3_ratio_cap_020 v5_ratio_cap_025_support_self v7_ratio_cap_025_gamma2)
    ks=(32 64 128)
  else
    variants=(v0_abs_cap_baseline_unified_gate v1_ratio_only v3_ratio_cap_020 v5_ratio_cap_025_support_self v7_ratio_cap_025_gamma2)
    ks=(32 128)
  fi
  benches=(textvqa pope mme)
  local -a gpus=()
  read -r -a gpus <<< "${GPU_LIST}"
  if [ "${#gpus[@]}" -eq 0 ]; then
    gpus=(0)
  fi
  local variant bench k gpu_index=0 gpu
  for variant in "${variants[@]}"; do
    for bench in "${benches[@]}"; do
      for k in "${ks[@]}"; do
        gpu="${gpus[$gpu_index]}"
        specs+=("${gpu}:${variant}:${bench}:${k}")
        gpu_index=$(((gpu_index + 1) % ${#gpus[@]}))
      done
    done
  done
  printf "%s\n" "${specs[@]}"
}

mapfile -t ALL_SPECS < <(build_specs)
read -r -a ACTIVE_GPUS <<< "${GPU_LIST}"
BATCH_SIZE="${#ACTIVE_GPUS[@]}"
if [ "${BATCH_SIZE}" -le 0 ]; then
  BATCH_SIZE=1
fi
echo "[INFO] total_runs=${#ALL_SPECS[@]}"
echo "[INFO] batch_size=${BATCH_SIZE}"
for ((i = 0; i < ${#ALL_SPECS[@]}; i += BATCH_SIZE)); do
  batch=("${ALL_SPECS[@]:i:BATCH_SIZE}")
  echo "[INFO] batch_start index=${i} size=${#batch[@]}"
  run_batch "${batch[@]}"
done

echo "[INFO] collecting results"
python "${COLLECTOR}" || true
echo "[INFO] done"
