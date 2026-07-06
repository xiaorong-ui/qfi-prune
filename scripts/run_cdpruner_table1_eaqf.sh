#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
cd "${ROOT}"

TOKEN="${TOKEN:-64}"
DRY_RUN="${DRY_RUN:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-32}"
FORCE="${FORCE:-0}"
DATA_ROOT="${DATA_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
DEFAULT_LOG_ROOT="logs/cdpruner_table1_eaqf"
[ "${DRY_RUN}" = "1" ] && DEFAULT_LOG_ROOT="logs/cdpruner_table1_eaqf_dryrun"
LOG_ROOT="${LOG_ROOT_OVERRIDE:-${DEFAULT_LOG_ROOT}}"
INVENTORY="logs/cdpruner_table1_eaqf/inventory.txt"
mkdir -p "$(dirname "${INVENTORY}")"

benchmarks=(vqav2 gqa vizwiz scienceqa textvqa pope mme mmbench_en mmbench_cn mmvet)
declare -A scores scripts questions images metrics answers notes
scores=([vqav2]=75.4 [gqa]=58.6 [vizwiz]=53.4 [scienceqa]=68.1 [textvqa]=55.3 [pope]=87.5 [mme]=1415.1 [mmbench_en]=61.1 [mmbench_cn]=53.2 [mmvet]=30.5)
scripts=([vqav2]=scripts/v1_5/eval/vqav2.sh [gqa]=scripts/v1_5/eval/gqa.sh [vizwiz]=scripts/v1_5/eval/vizwiz.sh [scienceqa]=scripts/v1_5/eval/sqa.sh [textvqa]=scripts/v1_5/eval/textvqa.sh [pope]=scripts/v1_5/eval/pope.sh [mme]=scripts/v1_5/eval/mme.sh [mmbench_en]=scripts/v1_5/eval/mmbench.sh [mmbench_cn]=scripts/v1_5/eval/mmbench_cn.sh [mmvet]=scripts/v1_5/eval/mmvet.sh)
questions=([vqav2]=playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl [gqa]=playground/data/eval/gqa/llava_gqa_testdev_balanced.jsonl [vizwiz]=playground/data/eval/vizwiz/llava_test.jsonl [scienceqa]=playground/data/eval/scienceqa/llava_test_CQM-I.json [textvqa]=playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl [pope]=playground/data/eval/pope/llava_pope_test.jsonl [mme]=playground/data/eval/MME/llava_mme.jsonl [mmbench_en]="${DATA_ROOT}/mmbench/mmbench_dev_20230712.tsv" [mmbench_cn]="${DATA_ROOT}/mmbench/mmbench_dev_cn_20231003.tsv" [mmvet]=playground/data/eval/mm-vet/llava-mm-vet.jsonl)
images=([vqav2]="${DATA_ROOT}/vqav2/test2015" [gqa]="${DATA_ROOT}/gqa/images" [vizwiz]="${DATA_ROOT}/vizwiz/test" [scienceqa]="${DATA_ROOT}/scienceqa/images/test" [textvqa]="${DATA_ROOT}/textvqa/train_images" [pope]="${DATA_ROOT}/pope/val2014" [mme]="${DATA_ROOT}/MME/MME_Benchmark_release_version" [mmbench_en]="embedded_in_tsv" [mmbench_cn]="embedded_in_tsv" [mmvet]="${DATA_ROOT}/mm-vet/images")
metrics=([vqav2]="official EvalAI submission" [gqa]=playground/data/eval/gqa/data/eval.py [vizwiz]="official EvalAI submission" [scienceqa]=llava/eval/eval_science_qa.py [textvqa]=llava/eval/eval_textvqa.py [pope]=llava/eval/eval_pope.py [mme]=playground/data/eval/MME/eval_tool/calculation.py [mmbench_en]="official submission server" [mmbench_cn]="official submission server" [mmvet]="external GPT evaluator")
answers=([vqav2]='<log_root>/vqav2/{method}_k64.jsonl' [gqa]='logs/final_results/*.log' [vizwiz]='<log_root>/vizwiz/{method}_k64.jsonl' [scienceqa]='<log_root>/scienceqa/{method}_k64.jsonl' [textvqa]='<log_root>/textvqa/{method}_k64.jsonl' [pope]='<log_root>/pope/{method}_k64.jsonl' [mme]='<log_root>/mme/{method}_k64.jsonl' [mmbench_en]='<log_root>/mmbench_en/{method}_k64.jsonl' [mmbench_cn]='<log_root>/mmbench_cn/{method}_k64.jsonl' [mmvet]='<log_root>/mmvet/{method}_k64.jsonl')
notes=([vqav2]="test-dev score requires EvalAI" [gqa]="already completed" [vizwiz]="test score requires EvalAI" [scienceqa]="SQA-IMG; requires CQM-I questions and ScienceQA image data" [textvqa]="local evaluator requires val annotations" [pope]="local evaluator reports average F1" [mme]="requires official eval_tool" [mmbench_en]="dev TSV embeds images; official score may require submission" [mmbench_cn]="CN dev TSV embeds images; official score may require submission" [mmvet]="score requires external GPT evaluator")

benchmark_status() {
    local b="$1"
    [ "${b}" = gqa ] && { echo already_done; return; }
    [ -f "${scripts[$b]}" ] || { echo missing_script; return; }
    [ -f "${questions[$b]}" ] || { echo missing_data; return; }
    if [ "${images[$b]}" != embedded_in_tsv ] && [ ! -d "${images[$b]}" ]; then echo missing_data; return; fi
    case "${b}" in
        scienceqa) [ -f "${DATA_ROOT}/scienceqa/problems.json" ] && [ -f "${DATA_ROOT}/scienceqa/pid_splits.json" ] || { echo missing_data; return; } ;;
        textvqa) [ -f "${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json" ] || { echo missing_data; return; } ;;
        pope) [ -d "${DATA_ROOT}/pope/coco" ] && [ -f "${metrics[$b]}" ] || { echo missing_data; return; } ;;
        mme) [ -f "${metrics[$b]}" ] || { echo missing_data; return; } ;;
    esac
    echo available
}

write_inventory() {
    : > "${INVENTORY}"
    for b in "${benchmarks[@]}"; do
        {
            echo "benchmark_name: ${b}"
            echo "cdpruner_k64_score: ${scores[$b]}"
            echo "eval_script: ${scripts[$b]}"
            echo "question_file_or_data_path: ${questions[$b]}"
            echo "image_folder: ${images[$b]}"
            echo "answer_path_pattern: ${answers[$b]}"
            echo "metric_script: ${metrics[$b]}"
            echo "status: $(benchmark_status "${b}")"
            echo "notes: ${notes[$b]}"
            echo
        } >> "${INVENTORY}"
    done
}

write_inventory
[ "${1:-}" = --inventory-only ] && exit 0

mkdir -p "${LOG_ROOT}"
cp "${INVENTORY}" "${LOG_ROOT}/inventory.txt" 2>/dev/null || true
MASTER_LOG="${LOG_ROOT}/master.log"; STATUS_FILE="${LOG_ROOT}/status.tsv"; PID_FILE="${LOG_ROOT}/runner.pid"
if [ "${1:-}" != --worker ] && [ "${FOREGROUND:-0}" != 1 ]; then
    nohup bash "${SELF}" --worker >> "${MASTER_LOG}" 2>&1 < /dev/null &
    echo $! > "${PID_FILE}"
    echo "[table1-eaqf] started pid=$!; monitor: tail -f ${MASTER_LOG}"
    exit 0
fi
trap 'rm -f "${PID_FILE}"' EXIT INT TERM
source /home/gpuadmin/anaconda3/etc/profile.d/conda.sh
conda activate cdpruner

requested="${BENCHMARKS:-scienceqa mme pope}"
read -r -a RUN_BENCHMARKS <<< "${requested}"
if [ -n "${GPUS:-}" ]; then IFS=',' read -r -a GPU_LIST <<< "${GPUS}"; else GPU_LIST=("${GPU:-0}"); fi
declare -A METHOD_PROFILES=(
    [pure_qf]=qfid_density_depol
    [eaqf_final]=qfid_density_eaqf_final
    [cg_eaqf_final]=qfid_density_cg_eaqf_final
)
read -r -a RUN_METHODS <<< "${METHODS:-pure_qf eaqf_final cg_eaqf_final}"
for method in "${RUN_METHODS[@]}"; do
    if [ -z "${METHOD_PROFILES[$method]+x}" ]; then
        echo "[table1-eaqf] unknown method: ${method}" >&2
        exit 2
    fi
done
printf 'benchmark\tmethod\tprofile\tstatus\texit_code\tmain_metric_name\tmain_metric_value\tcdpruner_k64_score\tdelta_vs_cdpruner\tlog_path\n' > "${STATUS_FILE}"

prepare_question_file() {
    local b="$1" src="$2" dst="$3"
    if [ "${DRY_RUN}" != 1 ]; then echo "${src}"; return; fi
    case "${b}" in
        mmbench_en|mmbench_cn)
            python -c 'import pandas as pd,sys; pd.read_table(sys.argv[1]).head(int(sys.argv[3])).to_csv(sys.argv[2],sep="\t",index=False)' \
                "${src}" "${dst}" "${NUM_SAMPLES}"
            ;;
        scienceqa) python -c 'import json,sys; d=json.load(open(sys.argv[1])); json.dump(d[:int(sys.argv[3])],open(sys.argv[2],"w"))' "${src}" "${dst}" "${NUM_SAMPLES}" ;;
        *) head -n "${NUM_SAMPLES}" "${src}" > "${dst}" ;;
    esac
    echo "${dst}"
}

run_task() {
    local b="$1" method="$2" profile="$3" gpu="$4" log="$5" answer="$6" qfile="$7"
    local module extra=()
    case "${b}" in
        scienceqa) module=llava.eval.model_vqa_science; extra=(--image-folder "${images[$b]}" --single-pred-prompt) ;;
        mmbench_en) module=llava.eval.model_vqa_mmbench; extra=(--single-pred-prompt) ;;
        mmbench_cn) module=llava.eval.model_vqa_mmbench; extra=(--lang cn --single-pred-prompt) ;;
        mmvet) module=llava.eval.model_vqa; extra=(--image-folder "${images[$b]}") ;;
        *) module=llava.eval.model_vqa_loader; extra=(--image-folder "${images[$b]}") ;;
    esac
    {
        echo "[table1-eaqf] benchmark=${b} method=${method} profile=${profile} K=${TOKEN}"
        env_args=(
            "TRANSFORMERS_OFFLINE=1" "HF_HUB_OFFLINE=1" "HF_DATASETS_OFFLINE=1"
            "CUDA_VISIBLE_DEVICES=${gpu}" "PRUNE_METHOD=ec_pruner" "EC_SCORE_SOURCE=qfid"
            "EC_QFID_SELECT_MODE=qf" "EC_QFID_KERNEL=density" "EC_QFID_TAU=0.50"
            "EC_QFID_DEPOLARIZE_MODE=fixed" "EC_QFID_DEPOLARIZE=0.15"
            "EC_QFID_SPATIAL_STATE=0" "EC_QFID_MEASURE_PRIOR_MODE=none" "EC_QFID_ANCHOR_MODE=none"
            "EC_USE_SEMANTIC_CANDIDATE=0" "EC_USE_SPATIAL_CANDIDATE=0"
            "EC_USE_REPULSION=0" "EC_USE_COMPLEMENT=0" "EC_USE_PHI=0" "EC_USE_CHAIN_COVERAGE=0"
        )
        case "${method}" in
            pure_qf)
                env_args+=("EC_QFID_PROB_SOURCE=semantic" "EC_QFID_CLS_GATE=0")
                ;;
            eaqf_final)
                env_args+=("EC_QFID_FINAL_PROFILE=1" "EC_QFID_PROB_SOURCE=clsmix" "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105" "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean" "EC_QFID_CLS_GATE=0")
                ;;
            cg_eaqf_final)
                env_args+=("EC_QFID_CG_FINAL_PROFILE=1" "EC_QFID_PROB_SOURCE=clsmix" "EC_QFID_CLS_MIX_MODE=linear" "EC_QFID_CLS_MIX_BETA=0.105" "EC_QFID_CLS_ATTN_LAYER=-2" "EC_QFID_CLS_HEAD_REDUCE=mean" "EC_QFID_CLS_GATE=1" "EC_QFID_CLS_GATE_MODE=agreement" "EC_QFID_CLS_GATE_BETA_BASE=0.105" "EC_QFID_CLS_GATE_MIN=0.5" "EC_QFID_CLS_GATE_MAX=1.5")
                ;;
        esac
        env "${env_args[@]}" python -m "${module}" \
            --model-path /home/gpuadmin/txr/CDPruner/checkpoints/llava-v1.5-7b \
            --question-file "${qfile}" --answers-file "${answer}" \
            --visual_token_num "${TOKEN}" --temperature 0 --conv-mode vicuna_v1 "${extra[@]}"
        case "${b}" in
            textvqa)
                python -m llava.eval.eval_textvqa \
                    --annotation-file "${DATA_ROOT}/textvqa/TextVQA_0.5.1_val.json" \
                    --result-file "${answer}"
                ;;
            mmbench_en|mmbench_cn)
                python scripts/eval_mmbench_local.py \
                    --annotation-file "${qfile}" --result-file "${answer}"
                if [ "${DRY_RUN}" != 1 ]; then
                    mkdir -p "$(dirname "${answer}")/submission"
                    converter_python=python
                    if ! python -c 'import openpyxl' >/dev/null 2>&1; then
                        converter_python=/home/gpuadmin/anaconda3/bin/python
                    fi
                    "${converter_python}" scripts/convert_mmbench_for_submission.py \
                        --annotation-file "${questions[$b]}" \
                        --result-dir "$(dirname "${answer}")" \
                        --upload-dir "$(dirname "${answer}")/submission" \
                        --experiment "$(basename "${answer%.jsonl}")"
                fi
                ;;
            *)
                if [ "${DRY_RUN}" != 1 ]; then
                    case "${b}" in
                vqav2)
                    python scripts/convert_vqav2_for_submission.py \
                        --dir "${ROOT}/playground/data/eval/vqav2" \
                        --src "${ROOT}/${answer}" \
                        --dst "${ROOT}/${answer%.jsonl}_submission.json"
                    ;;
                vizwiz)
                    python scripts/convert_vizwiz_for_submission.py \
                        --annotation-file "${questions[$b]}" --result-file "${answer}" \
                        --result-upload-file "${answer%.jsonl}_submission.json"
                    ;;
                scienceqa) python -m llava.eval.eval_science_qa --base-dir "${DATA_ROOT}/scienceqa" --result-file "${answer}" --output-file "${answer%.jsonl}_output.json" --output-result "${answer%.jsonl}_result.json" ;;
                pope) python -m llava.eval.eval_pope --annotation-dir "${DATA_ROOT}/pope/coco" --question-file "${qfile}" --result-file "${answer}" ;;
                mme)
                    mme_experiment="table1_${method}_k${TOKEN}"
                    if [ ! -e playground/data/eval/MME/MME_Benchmark_release_version ]; then
                        ln -s "${DATA_ROOT}/MME/MME_Benchmark_release_version" \
                            playground/data/eval/MME/MME_Benchmark_release_version
                    fi
                    cp "${answer}" "playground/data/eval/MME/answers/${mme_experiment}.jsonl"
                    (
                        cd playground/data/eval/MME
                        python convert_answer_to_mme.py --experiment "${mme_experiment}"
                        python eval_tool/calculation.py --results_dir "eval_tool/answers/${mme_experiment}"
                    )
                    ;;
                mmvet)
                    python scripts/convert_mmvet_for_eval.py \
                        --src "${answer}" --dst "${answer%.jsonl}_submission.json"
                    ;;
                    esac
                fi
                ;;
        esac
    } > "${log}" 2>&1
}

parse_metric() {
    local b="$1" log="$2" name value
    name=NA; value=NA
    case "${b}" in
        scienceqa) name=IMG-Accuracy; value="$(grep -o 'IMG-Accuracy: [0-9.]*%' "${log}" | tail -n1 | awk '{print $2}' || true)" ;;
        pope) name=F1; value="$(grep '^Average F1 score:' "${log}" | tail -n1 | awk '{print $4}' || true)" ;;
        textvqa|vizwiz|vqav2) name=Accuracy; value="$(grep '^Accuracy:' "${log}" | tail -n1 | awk '{print $2}' || true)" ;;
        mme)
            # CDPruner Table 1 reports the MME Perception total, which is the
            # first total emitted by the official calculation.py scorer.
            name=MME-Perception
            value="$(grep -Ei 'total score:' "${log}" | head -n1 | awk '{printf "%.4f", $NF}' || true)"
            ;;
        mmbench_en|mmbench_cn)
            name=Circular-Accuracy
            value="$(grep -Ei '^Circular-Accuracy:' "${log}" | tail -n1 | awk '{print $2}' || true)"
            [ -n "${value}" ] || value="$(grep -Ei '^Accuracy:' "${log}" | tail -n1 | awk '{print $2}' || true)"
            ;;
        mmvet) name=official_score ;;
    esac
    echo "${name}"$'\t'"${value:-NA}"
}

gpu_index=0
for b in "${RUN_BENCHMARKS[@]}"; do
    if [ -z "${scores[$b]+x}" ]; then echo "[table1-eaqf] unknown benchmark: ${b}"; continue; fi
    availability="$(benchmark_status "${b}")"
    if [ "${availability}" != available ]; then
        echo "[table1-eaqf] skip ${b}: ${availability}"
        for method in "${RUN_METHODS[@]}"; do
            printf '%s\t%s\t%s\t%s\t0\tNA\tNA\t%s\tNA\tNA\n' "${b}" "${method}" "${METHOD_PROFILES[$method]}" "${availability}" "${scores[$b]}" >> "${STATUS_FILE}"
        done
        continue
    fi
    mkdir -p "${LOG_ROOT}/${b}"
    source_question="${questions[$b]}"
    qfile="${source_question}"
    if [ "${DRY_RUN}" = 1 ]; then
        suffix=jsonl; [[ "${source_question}" = *.json ]] && suffix=json; [[ "${source_question}" = *.tsv ]] && suffix=tsv
        qfile="$(prepare_question_file "${b}" "${source_question}" "${LOG_ROOT}/${b}/questions_${NUM_SAMPLES}.${suffix}")"
    fi
    for method in "${RUN_METHODS[@]}"; do
        profile="${METHOD_PROFILES[$method]}"
        gpu="${GPU_LIST[$((gpu_index % ${#GPU_LIST[@]}))]}"; gpu_index=$((gpu_index + 1))
        log="${LOG_ROOT}/${b}/${method}_k${TOKEN}.log"; answer="${LOG_ROOT}/${b}/${method}_k${TOKEN}.jsonl"
        if [ "${FORCE}" != 1 ] && grep -q '^\[table1-eaqf\] completed$' "${log}" 2>/dev/null; then code=0
        else
            run_task "${b}" "${method}" "${profile}" "${gpu}" "${log}" "${answer}" "${qfile}"; code=$?
            [ "${code}" -eq 0 ] && [ -s "${answer}" ] && echo '[table1-eaqf] completed' >> "${log}"
        fi
        IFS=$'\t' read -r metric_name metric_value <<< "$(parse_metric "${b}" "${log}")"
        run_status=failed; [ "${code}" -eq 0 ] && [ -s "${answer}" ] && run_status=completed
        delta=NA
        if [[ "${metric_value}" =~ ^[0-9.]+%?$ ]]; then
            delta="$(awk -v a="${metric_value%%%}" -v c="${scores[$b]}" 'BEGIN{printf "%.4f",a-c}')"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "${b}" "${method}" "${profile}" "${run_status}" "${code}" "${metric_name}" "${metric_value}" "${scores[$b]}" "${delta}" "${log}" >> "${STATUS_FILE}"
        echo "[table1-eaqf] ${b}/${method}: ${run_status} ${metric_name}=${metric_value}"
    done
done
echo '[table1-eaqf] finished'
column -t -s $'\t' "${STATUS_FILE}" 2>/dev/null || cat "${STATUS_FILE}"
