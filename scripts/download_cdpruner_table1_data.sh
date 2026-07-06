#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_ROOT="${DEST_ROOT:-/data1/gpuadmin/txr_datasets/CDPruner_table1}"
LINK_ROOT="${LINK_ROOT:-/home/gpuadmin/txr/CDPruner_data}"
DOWNLOAD_ROOT="${DEST_ROOT}/.downloads"
DATASETS="${DATASETS:-scienceqa mmbench mmvet vqav2 vizwiz textvqa mme}"

mkdir -p "${DEST_ROOT}" "${DOWNLOAD_ROOT}" "${LINK_ROOT}"

download() {
    local url="$1" output="$2" partial
    partial="${output}.part"
    if [ -s "${output}" ]; then
        if [[ "${output}" != *.zip ]] || unzip -tq "${output}" >/dev/null 2>&1; then
            echo "[download] verified existing file: ${output}"
            return
        fi
        if [ ! -e "${partial}" ]; then
            mv "${output}" "${partial}"
        fi
    fi
    echo "[download] ${url}"
    wget -c --tries=5 --timeout=60 --progress=dot:giga -O "${partial}" "${url}"
    mv "${partial}" "${output}"
}

link_dataset() {
    local name="$1" target="$2" link
    link="${LINK_ROOT}/${name}"
    if [ -L "${link}" ]; then
        ln -sfn "${target}" "${link}"
    elif [ -e "${link}" ]; then
        echo "[download] warning: ${link} already exists and is not a symlink; leaving it unchanged"
    else
        ln -s "${target}" "${link}"
    fi
}

extract_once() {
    local archive="$1" destination="$2" marker="$3"
    if [ -e "${marker}" ]; then
        echo "[download] extracted data already present: ${marker}"
        return
    fi
    mkdir -p "${destination}"
    unzip -q "${archive}" -d "${destination}"
}

download_scienceqa() {
    local dst="${DEST_ROOT}/scienceqa" archive="${DOWNLOAD_ROOT}/scienceqa_test.zip"
    mkdir -p "${dst}/images"
    if [ ! -f "${dst}/problems.json" ] || [ ! -f "${dst}/pid_splits.json" ]; then
        local source="${LINK_ROOT}/ScienceQA_source/data/scienceqa"
        [ -f "${source}/problems.json" ] || {
            echo "[download] error: ScienceQA metadata clone is missing at ${source}" >&2
            return 1
        }
        cp -n "${source}/problems.json" "${source}/pid_splits.json" "${dst}/"
    fi
    download "https://scienceqa.s3.us-west-1.amazonaws.com/images/test.zip" "${archive}"
    extract_once "${archive}" "${dst}/images" "${dst}/images/test"
    link_dataset scienceqa "${dst}"

    local source_questions="${ROOT}/playground/data/eval/scienceqa/llava_test_CQM-A.json"
    local image_questions="${ROOT}/playground/data/eval/scienceqa/llava_test_CQM-I.json"
    if [ -f "${source_questions}" ] && [ ! -f "${image_questions}" ]; then
        python -c 'import json,sys; d=json.load(open(sys.argv[1])); out=[x for x in d if x.get("image")]; json.dump(out,open(sys.argv[2],"w"),indent=2); print(f"[download] wrote {len(out)} image questions to {sys.argv[2]}")' "${source_questions}" "${image_questions}"
    fi
}

download_mmbench() {
    local dst="${DEST_ROOT}/mmbench"
    mkdir -p "${dst}"
    download "https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv" "${dst}/mmbench_dev_20230712.tsv"
    download "https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv" "${dst}/mmbench_dev_cn_20231003.tsv"
    link_dataset mmbench "${dst}"
}

download_mmvet() {
    local dst="${DEST_ROOT}/mm-vet" archive="${DOWNLOAD_ROOT}/mm-vet.zip"
    download "https://github.com/yuweihao/MM-Vet/releases/download/v1/mm-vet.zip" "${archive}"
    extract_once "${archive}" "${dst}" "${dst}/mm-vet/images"
    if [ -d "${dst}/mm-vet/images" ]; then
        link_dataset mm-vet "${dst}/mm-vet"
    else
        link_dataset mm-vet "${dst}"
    fi
}

download_vqav2() {
    local dst="${DEST_ROOT}/vqav2" archive="${DOWNLOAD_ROOT}/vqav2_test2015.zip"
    download "http://images.cocodataset.org/zips/test2015.zip" "${archive}"
    extract_once "${archive}" "${dst}" "${dst}/test2015"
    link_dataset vqav2 "${dst}"
}

download_vizwiz() {
    local dst="${DEST_ROOT}/vizwiz"
    mkdir -p "${dst}"
    download "https://vizwiz.cs.colorado.edu/VizWiz_final/images/test.zip" "${DOWNLOAD_ROOT}/vizwiz_test.zip"
    extract_once "${DOWNLOAD_ROOT}/vizwiz_test.zip" "${dst}" "${dst}/test"
    download "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/Annotations.zip" "${DOWNLOAD_ROOT}/vizwiz_annotations.zip"
    extract_once "${DOWNLOAD_ROOT}/vizwiz_annotations.zip" "${dst}" "${dst}/Annotations"
    link_dataset vizwiz "${dst}"
}

download_textvqa() {
    local dst="${DEST_ROOT}/textvqa" archive="${DOWNLOAD_ROOT}/textvqa_train_val_images.zip"
    mkdir -p "${dst}"
    download "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json" "${dst}/TextVQA_0.5.1_val.json"
    download "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip" "${archive}"
    extract_once "${archive}" "${dst}" "${dst}/train_images"
    link_dataset textvqa "${dst}"
}

download_mme() {
    local dst="${DEST_ROOT}/MME"
    local data_archive="${DOWNLOAD_ROOT}/MME_Benchmark_release_version.zip"
    local eval_archive="${DOWNLOAD_ROOT}/MME_eval_tool.zip"
    mkdir -p "${dst}"
    download "https://huggingface.co/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip" "${data_archive}"
    extract_once "${data_archive}" "${dst}" "${dst}/MME_Benchmark_release_version"
    local release_root="${dst}/MME_Benchmark_release_version"
    if [ -d "${release_root}/MME_Benchmark" ]; then
        local item name
        for item in "${release_root}/MME_Benchmark"/*; do
            name="$(basename "${item}")"
            [ -e "${release_root}/${name}" ] || ln -s "${item}" "${release_root}/${name}"
        done
    fi
    download "https://raw.githubusercontent.com/BradyFU/Awesome-Multimodal-Large-Language-Models/Evaluation/tools/eval_tool.zip" "${eval_archive}"
    extract_once "${eval_archive}" "${dst}" "${dst}/eval_tool"
    link_dataset MME "${dst}"
    if [ -d "${dst}/eval_tool" ] && [ ! -e "${ROOT}/playground/data/eval/MME/eval_tool" ]; then
        ln -s "${dst}/eval_tool" "${ROOT}/playground/data/eval/MME/eval_tool"
    fi
}

for dataset in ${DATASETS}; do
    case "${dataset}" in
        scienceqa) download_scienceqa ;;
        mmbench) download_mmbench ;;
        mmvet|mm-vet) download_mmvet ;;
        vqav2) download_vqav2 ;;
        vizwiz) download_vizwiz ;;
        textvqa) download_textvqa ;;
        mme) download_mme ;;
        *) echo "[download] error: unknown dataset ${dataset}" >&2; exit 2 ;;
    esac
done

echo "[download] datasets stored under ${DEST_ROOT}"
echo "[download] project links available under ${LINK_ROOT}"
