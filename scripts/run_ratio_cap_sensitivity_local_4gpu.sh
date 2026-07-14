#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/llava-v1.5-7b}"
SCIENCEQA_BASE_DIR="${SCIENCEQA_BASE_DIR:-${DATA_ROOT}/scienceqa}"
OUT_ROOT="${ROOT}/outputs/qficr_ratio_cap_sensitivity_local"
LOG_ROOT="${ROOT}/logs/qficr_ratio_cap_sensitivity_local"
COLLECTOR="${ROOT}/scripts/collect_ratio_cap_sensitivity_local_results.py"
STATUS_TSV="${LOG_ROOT}/run_status.tsv"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-30000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-60}"
GPUS=(0 1 2 3)
RATIOS=(0.15 0.18 0.22 0.25)
KS=(32 64 128)
BENCHMARKS_STAGE1=(textvqa mme)
BENCHMARKS_STAGE2=(gqa sqa_img)

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
echo "[INFO] ratios=${RATIOS[*]}"
echo "[INFO] K_list=${KS[*]}"
echo "[INFO] benchmarks=TextVQA MME GQA SQA-IMG"
echo "[INFO] excluded=VQAv2 VizWiz MM-Vet POPE MMBench-EN MMBench-CN ratio_cap020 CDPruner Old-QFi-CR"

printf "run_id\tratio\tbenchmark\tK\tgpu\tstatus\texit_code\tlog_path\toutput_dir\tnotes\n" > "${STATUS_TSV}"
python "${COLLECTOR}" || true

ratio_tag() {
  local ratio="$1"
  printf "ratio_cap%s" "$(printf "%s" "${ratio}" | tr -d '.')"
}

record_status() {
  local run_id="$1" ratio="$2" bench="$3" k="$4" gpu="$5" status="$6" code="$7" log="$8" out_dir="$9" notes="${10:-}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${run_id}" "${ratio}" "${bench}" "${k}" "${gpu}" "${status}" "${code}" \
    "${log#${ROOT}/}" "${out_dir#${ROOT}/}" "${notes}" >> "${STATUS_TSV}"
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v gpu="${gpu}" '$1 == gpu { print $2 + 0; found = 1 } END { if (!found) exit 1 }'
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

apply_qficr_ratio_cap_env() {
  local ratio="$1" debug_path="$2" bench="$3" k="$4" tag="$5"
  common_env
  export PRUNE_METHOD=ec_pruner
  export EC_QFICR_TUNING_PROFILE="${tag}"
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
  export EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap
  export EC_QFICR_RECOVER_RATIO_CAP="${ratio}"
  export EC_QFICR_RECOVER_ENTROPY_GAMMA=1.0
  export EC_QFICR_SUPPORT_BETA=0.0
  export EC_QFICR_SELF_DISCOUNT=0.0
  export EC_QFICR_RECOVER_ABS_CAP=16
  export EC_QFID_DEBUG_STATS_JSONL="${debug_path}"
  export EC_QFID_DEBUG_BENCHMARK="${bench}"
  export EC_QFID_DEBUG_K="${k}"
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
  local out_dir="$1" ratio="$2" k="$3" bench="$4"
  cat > "${out_dir}/generation_config.json" <<EOF
{
  "method": "QFi-CR",
  "ratio_cap": ${ratio},
  "benchmark": "${bench}",
  "K": ${k},
  "base_model": "LLaVA-1.5-7B",
  "overlap_kernel": "relu_square",
  "restoration_mode": "full",
  "observation_mode": "full",
  "cls_prior_layers": "last2",
  "anchor_alpha": 0.5,
  "rest_beta": 1.0,
  "obs_temperature": 1.0,
  "restore_extra_prior": "none",
  "restore_extra_lambda": 0.0,
  "entropy_anchor_ratio": 0.0,
  "residual_budget_gamma": null,
  "prior_lambda": 0.0,
  "use_spatial_local_prior": false,
  "rho_max": 0.25,
  "support_beta": 0.0,
  "self_discount": 0.0,
  "entropy_gamma": 1.0,
  "yn_budget_gate": 0,
  "debug_stats_jsonl": true
}
EOF
}

is_completed() {
  local bench="$1" out_dir="$2" log="$3"
  [ -s "${out_dir}/answers.jsonl" ] || return 1
  [ -s "${out_dir}/generation_config.json" ] || return 1
  [ -s "${log}" ] || return 1
  rg -q '\[RUN_DONE\]' "${log}" || return 1
  case "${bench}" in
    gqa|textvqa|mme|sqa_img) [ -s "${out_dir}/eval_summary.txt" ] || return 1 ;;
  esac
}

run_generation() {
  local bench="$1" k="$2" answer="$3"
  local module="" qfile="" image_folder="" extra_args=()
  case "${bench}" in
    gqa)
      bash scripts/v1_5/eval/gqa.sh "${k}"
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
  if [ "${bench}" = "gqa" ]; then
    return 0
  fi
  [ -e "${qfile}" ] || { echo "[ERROR] missing question file: ${qfile}"; return 3; }
  [ -e "${image_folder}" ] || { echo "[ERROR] missing image folder: ${image_folder}"; return 3; }
  local cmd=(python -m "${module}" --model-path "${MODEL_PATH}" --question-file "${qfile}" --answers-file "${answer}" --image-folder "${image_folder}" --visual_token_num "${k}" --temperature 0 --conv-mode vicuna_v1)
  cmd+=("${extra_args[@]}")
  echo "[RUN] ${cmd[*]}"
  "${cmd[@]}"
}

run_eval() {
  local bench="$1" k="$2" out_dir="$3" answer="$4" tag="$5"
  case "${bench}" in
    gqa)
      local internal_dir="${ROOT}/playground/data/eval/gqa/answers/llava_gqa_testdev_balanced/llava-v1.5-7b/qfi_adapt_prior_k${k}_ovrelu_square_qficrtune${tag}"
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
      local mme_exp="${tag}_mme_k${k}"
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
    gqa)
      rg "^(Binary|Open|Accuracy|Distribution):" "${log}" > "${out_dir}/eval_summary.txt" || true
      ;;
    textvqa)
      rg "^Accuracy:" "${log}" > "${out_dir}/eval_summary.txt" || true
      ;;
    mme)
      rg "total score:|Perception|Cognition" "${log}" > "${out_dir}/eval_summary.txt" || true
      ;;
    sqa_img)
      rg "IMG-Accuracy:|Accuracy:|Total:" "${log}" > "${out_dir}/eval_summary.txt" || true
      ;;
  esac
}

run_one() {
  local ratio="$1" bench="$2" k="$3" gpu="$4"
  local tag run_id out_dir log answer debug_path code status notes
  tag="$(ratio_tag "${ratio}")"
  run_id="${tag}_${bench}_k${k}"
  out_dir="${OUT_ROOT}/${tag}/${bench}_k${k}"
  log="${LOG_ROOT}/${tag}_${bench}_k${k}.log"
  answer="${out_dir}/answers.jsonl"
  debug_path="${out_dir}/debug_qficr_stats.jsonl"

  if is_completed "${bench}" "${out_dir}" "${log}"; then
    echo "[SKIP] ${run_id} existing completed run"
    record_status "${run_id}" "${ratio}" "${bench}" "${k}" "${gpu}" "skipped_existing" "0" "${log}" "${out_dir}" "existing completed output reused"
    return 0
  fi

  mkdir -p "${out_dir}"
  write_generation_config "${out_dir}" "${ratio}" "${k}" "${bench}"
  {
    echo "[RUN_START] run_id=${run_id}"
    echo "[RUN_START] ratio=${ratio}"
    echo "[RUN_START] benchmark=${bench}"
    echo "[RUN_START] K=${k}"
    echo "[RUN_START] gpu=${gpu}"
    echo "[RUN_START] output_dir=${out_dir}"
    echo "[RUN_START] answer_file=${answer}"
    echo "[RUN_START] debug_path=${debug_path}"
    wait_for_gpu_memory "${gpu}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    apply_qficr_ratio_cap_env "${ratio}" "${debug_path}" "${bench}" "${k}" "${tag}"
    echo "[CONFIG] method=qficr budget_mode=${EC_QFICR_RECOVER_BUDGET_MODE} ratio_cap=${EC_QFICR_RECOVER_RATIO_CAP} entropy_gamma=${EC_QFICR_RECOVER_ENTROPY_GAMMA} support_beta=${EC_QFICR_SUPPORT_BETA} self_discount=${EC_QFICR_SELF_DISCOUNT} yn_budget_gate=${EC_QFICR_YN_BUDGET_GATE}"
    echo "[CONFIG] stable=1 overlap_kernel=relu_square restoration_mode=full observation_mode=full cls_prior_layers=last2 anchor_alpha=0.5 rest_beta=1.0 obs_temperature=1.0 restore_extra_prior=none restore_extra_lambda=0.0 entropy_anchor_ratio=0.0 residual_budget_gamma=none prior_lambda=0 use_spatial_local_prior=false rho_max=0.25"
    echo "[CONFIG] disabled=spatial_local_prior,replacement_prior,entropy_anchor,residual_budget_gating,diversity_rerank,random_restoration,cdpruner,old_qficr,k_aware_tuning"
    run_generation "${bench}" "${k}" "${answer}"
    run_eval "${bench}" "${k}" "${out_dir}" "${answer}" "${tag}"
    echo "[RUN_DONE] run_id=${run_id}"
  } > "${log}" 2>&1
  code=$?
  write_eval_summary_from_log "${bench}" "${log}" "${out_dir}"
  status="failed"
  notes=""
  if [ "${code}" -eq 0 ] && [ -s "${answer}" ] && [ -s "${out_dir}/eval_summary.txt" ]; then
    status="completed"
  elif [ "${code}" -eq 0 ]; then
    notes="missing answer or score summary"
  fi
  record_status "${run_id}" "${ratio}" "${bench}" "${k}" "${gpu}" "${status}" "${code}" "${log}" "${out_dir}" "${notes}"
  echo "[STATUS] ${run_id} ${status} exit=${code} log=${log#${ROOT}/}"
  return 0
}

declare -A GPU_PID
declare -A GPU_LABEL
for gpu in "${GPUS[@]}"; do
  GPU_PID["${gpu}"]=""
  GPU_LABEL["${gpu}"]=""
done

reap_finished() {
  local gpu pid label
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[${gpu}]}"
    if [ -n "${pid}" ] && ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" || true
      label="${GPU_LABEL[${gpu}]}"
      echo "[SLOT_FREE] gpu=${gpu} finished=${label}"
      GPU_PID["${gpu}"]=""
      GPU_LABEL["${gpu}"]=""
    fi
  done
}

SELECTED_GPU=""

next_free_gpu() {
  local gpu
  while true; do
    reap_finished
    for gpu in "${GPUS[@]}"; do
      if [ -z "${GPU_PID[${gpu}]}" ]; then
        SELECTED_GPU="${gpu}"
        return 0
      fi
    done
    wait -n || true
  done
}

wait_all_slots() {
  local gpu pid
  for gpu in "${GPUS[@]}"; do
    pid="${GPU_PID[${gpu}]}"
    if [ -n "${pid}" ]; then
      wait "${pid}" || true
      echo "[SLOT_FREE] gpu=${gpu} finished=${GPU_LABEL[${gpu}]}"
      GPU_PID["${gpu}"]=""
      GPU_LABEL["${gpu}"]=""
    fi
  done
}

run_benchmark() {
  local bench="$1" ratio k gpu tag label
  echo "[BENCH_START] ${bench}"
  for ratio in "${RATIOS[@]}"; do
    tag="$(ratio_tag "${ratio}")"
    for k in "${KS[@]}"; do
      next_free_gpu
      gpu="${SELECTED_GPU}"
      label="${tag}_${bench}_k${k}"
      echo "[DISPATCH] ${label} gpu=${gpu}"
      run_one "${ratio}" "${bench}" "${k}" "${gpu}" &
      GPU_PID["${gpu}"]=$!
      GPU_LABEL["${gpu}"]="${label}"
    done
  done
  wait_all_slots
  echo "[BENCH_DONE] ${bench}"
  python "${COLLECTOR}" || true
}

for bench in "${BENCHMARKS_STAGE1[@]}"; do
  run_benchmark "${bench}"
done

for bench in "${BENCHMARKS_STAGE2[@]}"; do
  run_benchmark "${bench}"
done

python "${COLLECTOR}" || true
echo "[ALL_DONE] run_all complete"
