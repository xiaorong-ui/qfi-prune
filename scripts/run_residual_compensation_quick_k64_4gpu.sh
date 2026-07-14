#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/llava-v1.5-7b}"
OUT_ROOT="${ROOT}/outputs/qficr_residual_compensation_quick_k64"
LOG_ROOT="${ROOT}/logs/qficr_residual_compensation_quick_k64"
COLLECTOR="${ROOT}/scripts/collect_residual_compensation_quick_k64_results.py"
STATUS_TSV="${LOG_ROOT}/run_status.tsv"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-30000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPUS=(0 1 2 3)
VARIANTS=(current_anchor weak_anchor no_anchor marginal_gain_weak_gate near_boundary_fill)
BENCHMARKS=(textvqa mme)
K=64

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
echo "[INFO] gpu_list=${GPUS[*]}"
echo "[INFO] gpu_free_threshold_mib=${GPU_FREE_THRESHOLD_MIB}"
echo "[INFO] variants=${VARIANTS[*]}"
echo "[INFO] benchmarks=${BENCHMARKS[*]}"
echo "[INFO] K=${K}"
echo "[INFO] excluded=VQAv2 VizWiz MM-Vet POPE MMBench GQA SQA-IMG K32 K128 CDPruner ratio-cap"

printf "run_id\tvariant\tbenchmark\tK\tgpu\tstatus\texit_code\tlog_path\toutput_dir\tnotes\n" > "${STATUS_TSV}"
python "${COLLECTOR}" || true

record_status() {
  local run_id="$1" variant="$2" bench="$3" k="$4" gpu="$5" status="$6" code="$7" log="$8" out_dir="$9" notes="${10:-}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_id}" "${variant}" "${bench}" "${k}" "${gpu}" "${status}" "${code}" \
    "${log#${ROOT}/}" "${out_dir#${ROOT}/}" "${notes}" >> "${STATUS_TSV}"
}

gpu_free_mib() {
  local gpu="$1"
  local attempt out
  for attempt in 1 2 3 4 5; do
    out="$(
      nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null |
        awk -F',' -v gpu="${gpu}" '{ gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if ($1 == gpu) { print ($3 - $2); found = 1 } } END { if (!found) exit 1 }'
    )"
    if [ -n "${out}" ]; then
      printf "%s\n" "${out}"
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_for_gpu_memory() {
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

common_env() {
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export DATA_DIR="${DATA_ROOT}"
  export CKPT_DIR="${ROOT}/checkpoints"
}

apply_qficr_env() {
  local variant="$1" debug_path="$2"
  common_env
  export PRUNE_METHOD=ec_pruner
  export EC_QFICR_TUNING_PROFILE="residual_compensation_quick_${variant}"
  export EC_SCORE_SOURCE=qfid
  export EC_QFID_SELECT_MODE=qf
  export EC_QFID_PROB_SOURCE=clsmix
  export EC_QFID_CLS_MIX_MODE=linear
  export EC_QFID_CLS_MIX_BETA=0.105
  export EC_QFID_CLS_ATTN_LAYER=-2
  export EC_QFICR_CLS_PRIOR_LAYERS=last2
  export EC_QFID_CLS_HEAD_REDUCE=mean
  export EC_QFID_CLS_GATE=1
  export EC_QFID_CLS_GATE_MODE=agreement
  export EC_QFID_CLS_GATE_BETA_BASE=0.105
  export EC_QFID_CLS_GATE_MIN=0.5
  export EC_QFID_CLS_GATE_MAX=1.5
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
  export EC_QFICR_COMP_SCORE_MODE="${variant}"
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
  export EC_QFICR_RECOVER_BUDGET_MODE=abs_cap
  export EC_QFICR_RECOVER_ABS_CAP=16
  unset EC_QFICR_RECOVER_RATIO_CAP
  export EC_QFICR_RECOVER_ENTROPY_GAMMA=1.0
  export EC_QFICR_SUPPORT_BETA=0.0
  export EC_QFICR_SELF_DISCOUNT=0.0
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
  export EC_QFID_DEBUG_STATS_JSONL="${debug_path}"
}

write_generation_config() {
  local out_dir="$1" variant="$2" bench="$3"
  python - "$out_dir" "$variant" "$bench" "$K" <<'PY'
import json
import os
import sys
out_dir, variant, bench, k = sys.argv[1:5]
keys = [
    "PRUNE_METHOD", "EC_QFICR_COMP_SCORE_MODE", "EC_QFICR_RECOVER_BUDGET_MODE",
    "EC_QFICR_RECOVER_ABS_CAP", "EC_QFICR_RECOVER_RATIO_CAP",
    "EC_QFID_OVERLAP_KERNEL", "EC_QFICR_RESTORATION_MODE",
    "EC_QFICR_OBSERVATION_MODE", "EC_QFICR_CLS_PRIOR_LAYERS",
    "EC_QFICR_ANCHOR_ALPHA", "EC_QFID_RECOVER_PRIOR_GAMMA",
    "EC_QFICR_SUPPORT_BETA", "EC_QFICR_SELF_DISCOUNT",
    "EC_QFICR_YN_BUDGET_GATE", "EC_QFID_DEBUG_STATS_JSONL",
]
payload = {
    "variant": variant,
    "benchmark": bench,
    "K": int(k),
    "model": "LLaVA-1.5-7B",
    "env": {key: os.environ.get(key, "") for key in keys},
}
with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

benchmark_paths() {
  local bench="$1"
  case "${bench}" in
    textvqa)
      printf "%s\t%s\t%s\n" \
        "llava.eval.model_vqa_loader" \
        "${ROOT}/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl" \
        "${DATA_ROOT}/textvqa/train_images"
      ;;
    mme)
      printf "%s\t%s\t%s\n" \
        "llava.eval.model_vqa_loader" \
        "${ROOT}/playground/data/eval/MME/llava_mme.jsonl" \
        "${DATA_ROOT}/MME/MME_Benchmark_release_version"
      ;;
    *)
      return 2
      ;;
  esac
}

is_complete() {
  local out_dir="$1" log="$2" bench="$3"
  local expected actual
  [ -s "${out_dir}/answers.jsonl" ] || return 1
  [ -s "${out_dir}/eval_summary.txt" ] || return 1
  [ -s "${out_dir}/generation_config.json" ] || return 1
  [ -s "${out_dir}/debug_qficr_stats.jsonl" ] || return 1
  [ -s "${log}" ] || return 1
  rg -q "\[RUN_DONE\]" "${log}" || return 1
  case "${bench}" in
    textvqa) expected=5000 ;;
    mme) expected=2374 ;;
    *) expected=0 ;;
  esac
  if [ "${expected}" -gt 0 ]; then
    actual="$(wc -l < "${out_dir}/answers.jsonl" | tr -d ' ')"
    [ "${actual}" = "${expected}" ] || return 1
  fi
  return 0
}

safe_paths() {
  local variant="$1" bench="$2"
  local out_dir="${OUT_ROOT}/${variant}/${bench}_k${K}"
  local log="${LOG_ROOT}/${variant}_${bench}_k${K}.log"
  if is_complete "${out_dir}" "${log}" "${bench}"; then
    printf "%s\t%s\t%s\n" "${log}" "${out_dir}" "reuse"
    return 0
  fi
  printf "%s\t%s\t%s\n" "${log}" "${out_dir}" "run"
}

run_generation() {
  local bench="$1" answer="$2"
  local paths module qfile image_folder
  paths="$(benchmark_paths "${bench}")"
  module="${paths%%$'\t'*}"
  paths="${paths#*$'\t'}"
  qfile="${paths%%$'\t'*}"
  image_folder="${paths#*$'\t'}"
  [ -e "${qfile}" ] || { echo "[ERROR] missing question file: ${qfile}"; return 3; }
  [ -e "${image_folder}" ] || { echo "[ERROR] missing image folder: ${image_folder}"; return 3; }
  local cmd=(python -m "${module}" --model-path "${MODEL_PATH}" --question-file "${qfile}" --answers-file "${answer}" --visual_token_num "${K}" --temperature 0 --conv-mode vicuna_v1 --image-folder "${image_folder}")
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

run_eval() {
  local bench="$1" variant="$2" out_dir="$3" answer="$4"
  case "${bench}" in
    textvqa)
      python -m llava.eval.eval_textvqa \
        --annotation-file "${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "${answer}" > "${out_dir}/eval_summary.txt" 2>&1
      cat "${out_dir}/eval_summary.txt"
      ;;
    mme)
      local mme_exp="rescomp_${variant}_${bench}_k${K}"
      mkdir -p "${ROOT}/playground/data/eval/MME/answers"
      cp "${answer}" "${ROOT}/playground/data/eval/MME/answers/${mme_exp}.jsonl"
      if [ ! -e "${ROOT}/playground/data/eval/MME/MME_Benchmark_release_version" ]; then
        ln -s "${DATA_ROOT}/MME/MME_Benchmark_release_version" "${ROOT}/playground/data/eval/MME/MME_Benchmark_release_version"
      fi
      (
        cd "${ROOT}/playground/data/eval/MME" &&
          python convert_answer_to_mme.py --experiment "${mme_exp}" &&
          python eval_tool/calculation.py --results_dir "eval_tool/answers/${mme_exp}"
      ) > "${out_dir}/eval_summary.txt" 2>&1
      cat "${out_dir}/eval_summary.txt"
      ;;
  esac
}

run_one() {
  local gpu="$1" variant="$2" bench="$3"
  local run_id="${variant}_${bench}_k${K}"
  local path_pair log out_dir action answer debug_path code status
  path_pair="$(safe_paths "${variant}" "${bench}")"
  log="${path_pair%%$'\t'*}"
  path_pair="${path_pair#*$'\t'}"
  out_dir="${path_pair%%$'\t'*}"
  action="${path_pair#*$'\t'}"
  if [ "${action}" = "reuse" ]; then
    echo "[SKIP] ${run_id} completed"
    record_status "${run_id}" "${variant}" "${bench}" "${K}" "${gpu}" "reused" "0" "${log}" "${out_dir}" "existing completed run"
    return 0
  fi
  mkdir -p "${out_dir}"
  answer="${out_dir}/answers.jsonl"
  debug_path="${out_dir}/debug_qficr_stats.jsonl"
  {
    echo "[RUN_START] run_id=${run_id}"
    echo "[RUN_START] variant=${variant}"
    echo "[RUN_START] benchmark=${bench}"
    echo "[RUN_START] K=${K}"
    echo "[RUN_START] gpu=${gpu}"
    echo "[RUN_START] output_dir=${out_dir}"
    echo "[RUN_START] answer_file=${answer}"
    echo "[RUN_START] debug_file=${debug_path}"
    wait_for_gpu_memory "${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export EC_QFID_DEBUG_BENCHMARK="${bench}"
    apply_qficr_env "${variant}" "${debug_path}"
    write_generation_config "${out_dir}" "${variant}" "${bench}"
    echo "[CONFIG] method=QFi-CR comp_score_mode=${EC_QFICR_COMP_SCORE_MODE} budget_mode=${EC_QFICR_RECOVER_BUDGET_MODE} abs_cap=${EC_QFICR_RECOVER_ABS_CAP} ratio_cap=disabled K=${K}"
    run_generation "${bench}" "${answer}"
    run_eval "${bench}" "${variant}" "${out_dir}" "${answer}"
    echo "[RUN_DONE] run_id=${run_id}"
  } > "${log}" 2>&1
  code=$?
  status="failed"
  if [ "${code}" -eq 0 ] && is_complete "${out_dir}" "${log}" "${bench}"; then
    status="completed"
  fi
  record_status "${run_id}" "${variant}" "${bench}" "${K}" "${gpu}" "${status}" "${code}" "${log}" "${out_dir}" ""
  echo "[STATUS] ${run_id} ${status} exit=${code} log=${log#${ROOT}/}"
  python "${COLLECTOR}" || true
  return 0
}

run_batch() {
  local -a specs=("$@")
  local -a pids=()
  local spec gpu variant bench pid
  for spec in "${specs[@]}"; do
    IFS=: read -r gpu variant bench <<< "${spec}"
    run_one "${gpu}" "${variant}" "${bench}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || true
  done
}

build_specs() {
  local -a specs=()
  local gpu_index=0 gpu variant bench
  for variant in "${VARIANTS[@]}"; do
    for bench in "${BENCHMARKS[@]}"; do
      gpu="${GPUS[$gpu_index]}"
      specs+=("${gpu}:${variant}:${bench}")
      gpu_index=$(((gpu_index + 1) % ${#GPUS[@]}))
    done
  done
  printf "%s\n" "${specs[@]}"
}

mapfile -t ALL_SPECS < <(build_specs)
BATCH_SIZE="${#GPUS[@]}"
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
