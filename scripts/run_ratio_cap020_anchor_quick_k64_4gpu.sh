#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/llava-v1.5-7b}"
SCIENCEQA_BASE_DIR="${SCIENCEQA_BASE_DIR:-${DATA_ROOT}/scienceqa}"
OUT_ROOT="${ROOT}/outputs/qficr_ratio_cap020_anchor_quick_k64"
LOG_ROOT="${ROOT}/logs/qficr_ratio_cap020_anchor_quick_k64"
COLLECTOR="${ROOT}/scripts/collect_ratio_cap020_anchor_quick_k64_results.py"
STATUS_TSV="${LOG_ROOT}/run_status.tsv"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-30000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPUS=(0 1 2 3)
VARIANTS=(ratio_cap020_current_anchor ratio_cap020_weak_anchor ratio_cap020_no_anchor)
STAGE1_BENCHES=(textvqa mme)
STAGE2_BENCHES=(gqa sqa_img)
RUN_STAGE2="${RUN_STAGE2:-0}"
K=64
RATIO_CAP=0.20

mkdir -p "${OUT_ROOT}" "${LOG_ROOT}" "${ROOT}/playground/data/eval/MME/answers"

if [ -f /home/gpuadmin/anaconda3/etc/profile.d/conda.sh ]; then
  # shellcheck source=/dev/null
  source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
  conda activate cdpruner 2>/dev/null || true
fi

echo "[INFO] runner_pid=$$"
echo "[INFO] root=${ROOT}"
echo "[INFO] data_root=${DATA_ROOT}"
echo "[INFO] scienceqa_base_dir=${SCIENCEQA_BASE_DIR}"
echo "[INFO] model_path=${MODEL_PATH}"
echo "[INFO] out_root=${OUT_ROOT}"
echo "[INFO] log_root=${LOG_ROOT}"
echo "[INFO] gpu_list=${GPUS[*]}"
echo "[INFO] gpu_free_threshold_mib=${GPU_FREE_THRESHOLD_MIB}"
echo "[INFO] variants=${VARIANTS[*]}"
echo "[INFO] stage1=${STAGE1_BENCHES[*]}"
echo "[INFO] stage2=${STAGE2_BENCHES[*]}"
echo "[INFO] run_stage2=${RUN_STAGE2}"
echo "[INFO] K=${K}"
echo "[INFO] ratio_cap=${RATIO_CAP}"
echo "[INFO] excluded=VQAv2 VizWiz MM-Vet POPE MMBench K32 K128 CDPruner"

printf "run_id\tvariant\tbenchmark\tK\tgpu\tstatus\texit_code\tlog_path\toutput_dir\tnotes\n" > "${STATUS_TSV}"
python "${COLLECTOR}" || true

variant_to_mode() {
  case "$1" in
    ratio_cap020_current_anchor) printf "current_anchor\n" ;;
    ratio_cap020_weak_anchor) printf "weak_anchor\n" ;;
    ratio_cap020_no_anchor) printf "no_anchor\n" ;;
    *) printf "current_anchor\n" ;;
  esac
}

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
  local variant="$1" bench="$2" debug_path="$3"
  local mode
  mode="$(variant_to_mode "${variant}")"
  common_env
  export PRUNE_METHOD=ec_pruner
  export EC_QFICR_TUNING_PROFILE="${variant}"
  export EC_QFID_ADAPTIVE_RECOVER_PROFILE=prior
  export EC_QFID_CG_FINAL_PROFILE=0
  export EC_QFID_FINAL_PROFILE=0
  export EC_QFID_GATE_PROFILE=0
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
  export EC_QFICR_COMP_SCORE_MODE="${mode}"
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
  export EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap
  export EC_QFICR_RECOVER_RATIO_CAP="${RATIO_CAP}"
  export EC_QFICR_RECOVER_ENTROPY_GAMMA=1.0
  export EC_QFICR_SUPPORT_BETA=0.0
  export EC_QFICR_SELF_DISCOUNT=0.0
  export EC_QFICR_YN_BUDGET_GATE=0
  export EC_QFICR_RECOVER_ABS_CAP=16
  export EC_QFID_DEBUG_STATS_JSONL="${debug_path}"
  export EC_QFID_DEBUG_BENCHMARK="${bench}"
  export EC_QFID_DEBUG_K="${K}"
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

write_generation_config() {
  local out_dir="$1" variant="$2" bench="$3"
  python - "$out_dir" "$variant" "$bench" "$K" <<'PY'
import json
import os
import sys
out_dir, variant, bench, k = sys.argv[1:5]
keys = [
    "PRUNE_METHOD", "EC_QFICR_TUNING_PROFILE", "EC_QFICR_COMP_SCORE_MODE",
    "EC_QFICR_RECOVER_BUDGET_MODE", "EC_QFICR_RECOVER_RATIO_CAP",
    "EC_QFID_OVERLAP_KERNEL", "EC_QFICR_RESTORATION_MODE",
    "EC_QFICR_OBSERVATION_MODE", "EC_QFICR_CLS_PRIOR_LAYERS",
    "EC_QFICR_ANCHOR_ALPHA", "EC_QFICR_REST_BETA", "EC_QFICR_OBS_TEMPERATURE",
    "EC_QFICR_RESTORE_EXTRA_PRIOR", "EC_QFICR_RESTORE_EXTRA_LAMBDA",
    "EC_QFICR_ENTROPY_ANCHOR_RATIO", "EC_QFICR_RESIDUAL_BUDGET_GAMMA",
    "EC_QFICR_PRIOR_LAMBDA", "EC_QFICR_USE_SPATIAL_LOCAL_PRIOR",
    "EC_QFICR_RECOVER_ENTROPY_GAMMA", "EC_QFICR_SUPPORT_BETA",
    "EC_QFICR_SELF_DISCOUNT", "EC_QFICR_YN_BUDGET_GATE",
    "EC_QFID_DEBUG_STATS_JSONL",
]
payload = {
    "variant": variant,
    "benchmark": bench,
    "K": int(k),
    "model": "LLaVA-1.5-7B",
    "ratio_cap": 0.20,
    "env": {key: os.environ.get(key, "") for key in keys},
}
with open(os.path.join(out_dir, "generation_config.json"), "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)
PY
}

expected_count() {
  case "$1" in
    textvqa) printf "5000\n" ;;
    mme) printf "2374\n" ;;
    gqa) printf "12578\n" ;;
    sqa_img) printf "2017\n" ;;
    *) printf "0\n" ;;
  esac
}

is_complete() {
  local bench="$1" out_dir="$2" log="$3"
  local expected actual
  [ -s "${out_dir}/answers.jsonl" ] || return 1
  [ -s "${out_dir}/eval_summary.txt" ] || return 1
  [ -s "${out_dir}/generation_config.json" ] || return 1
  [ -s "${out_dir}/debug_qficr_stats.jsonl" ] || return 1
  [ -s "${log}" ] || return 1
  rg -q "\[RUN_DONE\]" "${log}" || return 1
  expected="$(expected_count "${bench}")"
  if [ "${expected}" -gt 0 ]; then
    actual="$(wc -l < "${out_dir}/answers.jsonl" | tr -d ' ')"
    [ "${actual}" = "${expected}" ] || return 1
  fi
}

run_generation() {
  local bench="$1" answer="$2"
  local module="" qfile="" image_folder="" extra_args=()
  case "${bench}" in
    gqa)
      bash scripts/v1_5/eval/gqa.sh "${K}"
      return $?
      ;;
    sqa_img)
      module="llava.eval.model_vqa_science"
      qfile="${ROOT}/playground/data/eval/scienceqa/llava_test_CQM-I.json"
      image_folder="${DATA_ROOT}/scienceqa/images/test"
      extra_args+=(--single-pred-prompt)
      ;;
    textvqa)
      module="llava.eval.model_vqa_loader"
      qfile="${ROOT}/playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl"
      image_folder="${DATA_ROOT}/textvqa/train_images"
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
  local cmd=(python -m "${module}" --model-path "${MODEL_PATH}" --question-file "${qfile}" --answers-file "${answer}" --image-folder "${image_folder}" --visual_token_num "${K}" --temperature 0 --conv-mode vicuna_v1)
  cmd+=("${extra_args[@]}")
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

run_eval() {
  local bench="$1" variant="$2" out_dir="$3" answer="$4"
  case "${bench}" in
    gqa)
      local internal_dir="${ROOT}/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b/qfi_adapt_prior_k${K}_ovrelu_square_qficrtune${variant}"
      if [ -s "${internal_dir}/merge.jsonl" ]; then
        cp "${internal_dir}/merge.jsonl" "${answer}"
      else
        echo "[ERROR] missing internal GQA merge file: ${internal_dir}/merge.jsonl"
        return 4
      fi
      ;;
    sqa_img)
      python -m llava.eval.eval_science_qa \
        --base-dir "${SCIENCEQA_BASE_DIR}" \
        --result-file "${answer}" \
        --output-file "${out_dir}/answers_output.jsonl" \
        --output-result "${out_dir}/result.json"
      ;;
    textvqa)
      python -m llava.eval.eval_textvqa \
        --annotation-file "${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json" \
        --result-file "${answer}"
      ;;
    mme)
      local mme_exp="${variant}_mme_k${K}"
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
    *)
      echo "[ERROR] unsupported eval benchmark=${bench}"
      return 2
      ;;
  esac
}

write_eval_summary_from_log() {
  local bench="$1" log="$2" out_dir="$3"
  case "${bench}" in
    gqa) rg "^(Binary|Open|Accuracy|Distribution):" "${log}" > "${out_dir}/eval_summary.txt" || true ;;
    textvqa) rg "^Accuracy:" "${log}" > "${out_dir}/eval_summary.txt" || true ;;
    mme) rg "total score:|Perception|Cognition" "${log}" > "${out_dir}/eval_summary.txt" || true ;;
    sqa_img) rg "IMG-Accuracy:|Accuracy:|Total:" "${log}" > "${out_dir}/eval_summary.txt" || true ;;
  esac
}

run_one() {
  local gpu="$1" variant="$2" bench="$3"
  local run_id="${variant}_${bench}_k${K}"
  local out_dir="${OUT_ROOT}/${variant}/${bench}_k${K}"
  local log="${LOG_ROOT}/${variant}_${bench}_k${K}.log"
  local answer="${out_dir}/answers.jsonl"
  local debug_path="${out_dir}/debug_qficr_stats.jsonl"
  local code status
  if is_complete "${bench}" "${out_dir}" "${log}"; then
    echo "[SKIP] ${run_id} existing completed run"
    record_status "${run_id}" "${variant}" "${bench}" "${K}" "${gpu}" "skipped_existing" "0" "${log}" "${out_dir}" "existing completed output reused"
    return 0
  fi
  mkdir -p "${out_dir}"
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
    apply_qficr_env "${variant}" "${bench}" "${debug_path}"
    write_generation_config "${out_dir}" "${variant}" "${bench}"
    echo "[CONFIG] method=QFi-CR ratio_cap=${EC_QFICR_RECOVER_RATIO_CAP} mode=${EC_QFICR_COMP_SCORE_MODE} budget_mode=${EC_QFICR_RECOVER_BUDGET_MODE} K=${K}"
    run_generation "${bench}" "${answer}"
    run_eval "${bench}" "${variant}" "${out_dir}" "${answer}"
    write_eval_summary_from_log "${bench}" "${log}" "${out_dir}"
    echo "[RUN_DONE] run_id=${run_id}"
  } > "${log}" 2>&1
  code=$?
  cp "${log}" "${out_dir}/run.log" 2>/dev/null || true
  status="failed"
  if [ "${code}" -eq 0 ] && is_complete "${bench}" "${out_dir}" "${log}"; then
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

run_matrix() {
  local -a benches=("$@")
  local -a specs=()
  local gpu_idx=0
  local variant bench gpu
  for bench in "${benches[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
      specs+=("${gpu}:${variant}:${bench}")
      gpu_idx=$((gpu_idx + 1))
      if [ "${#specs[@]}" -eq "${#GPUS[@]}" ]; then
        run_batch "${specs[@]}"
        specs=()
      fi
    done
  done
  if [ "${#specs[@]}" -gt 0 ]; then
    run_batch "${specs[@]}"
  fi
}

should_run_stage2() {
  python - "${OUT_ROOT}/average_by_variant_ratio_cap020_anchor_quick_k64.csv" <<'PY'
import csv, sys
path = sys.argv[1]
rows = {}
try:
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["variant"]] = row
except FileNotFoundError:
    sys.exit(1)
def score(v):
    try:
        return float(rows[v]["avg_score"])
    except Exception:
        return None
cur = score("ratio_cap020_current_anchor")
weak = score("ratio_cap020_weak_anchor")
no = score("ratio_cap020_no_anchor")
if cur is None or weak is None or no is None:
    sys.exit(1)
sys.exit(0 if max(weak, no) >= cur else 1)
PY
}

echo "[INFO] stage1_start"
run_matrix "${STAGE1_BENCHES[@]}"
echo "[INFO] stage1_collect"
python "${COLLECTOR}" || true

if [ "${RUN_STAGE2}" = "1" ] && should_run_stage2; then
  echo "[INFO] stage2_condition=met"
  run_matrix "${STAGE2_BENCHES[@]}"
  echo "[INFO] final_collect"
  python "${COLLECTOR}" || true
else
  if [ "${RUN_STAGE2}" = "1" ]; then
    echo "[INFO] stage2_condition=not_met"
  else
    echo "[INFO] stage2_disabled=1"
  fi
  echo "[INFO] final_collect"
  python "${COLLECTOR}" || true
fi

echo "[INFO] done"
