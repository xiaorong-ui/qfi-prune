#!/usr/bin/env python3
import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = Path("/home/gpuadmin/txr/CDPruner_data/gqa/data/questions/testdev_balanced_questions.json")
DEFAULT_EVALUATOR = Path("/home/gpuadmin/txr/CDPruner_data/gqa/data/eval/eval.py")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values):
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def find_predictions(expected_count):
    candidates = []
    answer_root = REPO_ROOT / "playground/data/eval/gqa/answers"
    for path in answer_root.glob("**/merge.jsonl"):
        if "cg_eaqf" not in str(path) and "eaqf_gate" not in str(path):
            continue
        try:
            line_count = sum(1 for line in path.open() if line.strip())
        except OSError:
            continue
        if line_count == expected_count:
            candidates.append(path)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def parse_metric(output, label):
    match = re.search(rf"^{re.escape(label)}:\s+([0-9.]+)%", output, re.MULTILINE)
    return match.group(1) if match else "NA"


def parse_typed_metric(output, section, label):
    marker = f"Accuracy / {section} type:"
    if marker not in output:
        return "NA"
    block = output.split(marker, 1)[1].split("Accuracy /", 1)[0]
    match = re.search(rf"^\s+{re.escape(label)}:\s+([0-9.]+)%", block, re.MULTILINE)
    return match.group(1) if match else "NA"


def evaluate_group(name, indices, records, predictions, questions, output_dir, evaluator):
    group_stats = [records[index] for index in indices]
    raw_predictions = [predictions[index] for index in indices]
    question_ids = [str(prediction["question_id"]) for prediction in raw_predictions]
    subset_questions = {qid: questions[qid] for qid in question_ids}
    evaluator_predictions = [
        {
            "questionId": str(prediction["question_id"]),
            "prediction": str(prediction["text"]).rstrip(".").lower(),
        }
        for prediction in raw_predictions
    ]

    raw_path = output_dir / f"agreement_{name}_predictions.jsonl"
    question_path = output_dir / f"agreement_{name}_questions.json"
    evaluator_prediction_path = output_dir / f"agreement_{name}_evaluator_predictions.json"
    with raw_path.open("w") as output_file:
        for prediction in raw_predictions:
            output_file.write(json.dumps(prediction) + "\n")
    question_path.write_text(json.dumps(subset_questions))
    evaluator_prediction_path.write_text(json.dumps(evaluator_predictions))

    process = subprocess.run(
        [
            sys.executable,
            str(evaluator),
            "--questions",
            str(question_path),
            "--predictions",
            str(evaluator_prediction_path),
            "--tier",
            f"agreement_{name}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (output_dir / f"agreement_{name}_evaluator.log").write_text(process.stdout)
    return {
        "group": name,
        "num_samples": len(indices),
        "agreement_mean": statistics.fmean(item["agreement"] for item in group_stats),
        "beta_eff_mean": statistics.fmean(item["beta_eff"] for item in group_stats),
        "accuracy": parse_metric(process.stdout, "Accuracy"),
        "binary": parse_metric(process.stdout, "Binary"),
        "open": parse_metric(process.stdout, "Open"),
        "query": parse_typed_metric(process.stdout, "structural", "query"),
        "rel": parse_typed_metric(process.stdout, "semantic", "rel"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", required=True)
    parser.add_argument("--predictions")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--evaluator", default=str(DEFAULT_EVALUATOR))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in Path(args.stats).open() if line.strip()]
    if not records:
        raise ValueError("Gate stats file is empty")

    fields = ("beta_eff", "agreement", "gate", "p_sem_entropy", "p_cls_entropy")
    summary_lines = [f"samples: {len(records)}"]
    for field in fields:
        stats = describe([float(record[field]) for record in records])
        summary_lines.append(
            f"{field}: mean={stats['mean']:.8f} std={stats['std']:.8f} "
            f"min={stats['min']:.8f} max={stats['max']:.8f}"
        )
    beta_values = [float(record["beta_eff"]) for record in records]
    summary_lines.append(
        "beta_eff_percentiles: "
        + " ".join(
            f"p{percentile_value}={percentile(beta_values, percentile_value / 100):.8f}"
            for percentile_value in (10, 25, 50, 75, 90)
        )
    )
    summary_path = output_dir / "gate_stats_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    prediction_path = Path(args.predictions) if args.predictions else find_predictions(len(records))
    if prediction_path is None:
        print(f"Wrote {summary_path}; no aligned ReFi prediction file found, group evaluation skipped.")
        return
    predictions = [json.loads(line) for line in prediction_path.open() if line.strip()]
    if len(predictions) != len(records):
        raise ValueError(f"Stats/prediction count mismatch: {len(records)} != {len(predictions)}")

    questions = json.loads(Path(args.questions).read_text())
    ordered_indices = sorted(range(len(records)), key=lambda index: float(records[index]["agreement"]))
    group_size = max(1, int(len(records) * 0.30))
    groups = {
        "low": ordered_indices[:group_size],
        "high": ordered_indices[-group_size:],
    }
    results = [
        evaluate_group(
            name,
            indices,
            records,
            predictions,
            questions,
            output_dir,
            Path(args.evaluator),
        )
        for name, indices in groups.items()
    ]
    columns = ("group", "num_samples", "agreement_mean", "beta_eff_mean", "accuracy", "binary", "open", "query", "rel")
    with (output_dir / "agreement_group_accuracy.tsv").open("w") as output_file:
        output_file.write("\t".join(columns) + "\n")
        for result in results:
            output_file.write(
                "\t".join(
                    f"{result[column]:.8f}" if isinstance(result[column], float) else str(result[column])
                    for column in columns
                )
                + "\n"
            )
    print(f"Wrote {summary_path} and {output_dir / 'agreement_group_accuracy.tsv'}")


if __name__ == "__main__":
    main()
