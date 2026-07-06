#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/home/gpuadmin/txr/CDPruner/checkpoints}"
DATA_DIR="${DATA_DIR:-/home/gpuadmin/txr/CDPruner_data}"

CKPT="llava-v1.5-7b"
SPLIT="llava_pope_test"

TOKEN="${1:-64}"
PRUNE_METHOD_NAME="${PRUNE_METHOD:-cdpruner}"
EC_SCORE_SOURCE_NAME="${EC_SCORE_SOURCE:-norm}"
QUESTION_FILE="${QUESTION_FILE:-./playground/data/eval/pope/${SPLIT}.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${DATA_DIR}/pope/val2014}"
ANNOTATION_DIR="${ANNOTATION_DIR:-${DATA_DIR}/pope/coco}"

if [ "${PRUNE_METHOD_NAME}" = "cdpruner" ]; then
    PARAM="vtn_${TOKEN}"
elif [ "${EC_SCORE_SOURCE_NAME}" = "semantic" ]; then
    EC_USE_SEMANTIC_CANDIDATE_NAME="${EC_USE_SEMANTIC_CANDIDATE:-1}"
    EC_USE_REPULSION_NAME="${EC_USE_REPULSION:-1}"
    EC_USE_COMPLEMENT_NAME="${EC_USE_COMPLEMENT:-1}"
    EC_USE_PHI_NAME="${EC_USE_PHI:-1}"
    PARAM="${PRUNE_METHOD_NAME}_${EC_SCORE_SOURCE_NAME}_candidate${EC_USE_SEMANTIC_CANDIDATE_NAME}_r${EC_USE_REPULSION_NAME}_c${EC_USE_COMPLEMENT_NAME}_phi${EC_USE_PHI_NAME}_vtn_${TOKEN}"
else
    EC_USE_SEMANTIC_CANDIDATE_NAME="${EC_USE_SEMANTIC_CANDIDATE:-1}"
    EC_USE_REPULSION_NAME="${EC_USE_REPULSION:-1}"
    EC_USE_COMPLEMENT_NAME="${EC_USE_COMPLEMENT:-1}"
    EC_USE_PHI_NAME="${EC_USE_PHI:-1}"
    PARAM="${PRUNE_METHOD_NAME}_${EC_SCORE_SOURCE_NAME}_candidate${EC_USE_SEMANTIC_CANDIDATE_NAME}_r${EC_USE_REPULSION_NAME}_c${EC_USE_COMPLEMENT_NAME}_phi${EC_USE_PHI_NAME}_vtn_${TOKEN}"
fi

ANSWER_FILE="${ANSWER_FILE:-./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${PARAM}.jsonl}"
mkdir -p "$(dirname "${ANSWER_FILE}")"

python -m llava.eval.model_vqa_loader \
    --model-path "${CKPT_DIR}/${CKPT}" \
    --question-file "${QUESTION_FILE}" \
    --image-folder "${IMAGE_FOLDER}" \
    --answers-file "${ANSWER_FILE}" \
    --visual_token_num "${TOKEN}" \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir "${ANNOTATION_DIR}" \
    --question-file "${QUESTION_FILE}" \
    --result-file "${ANSWER_FILE}"
