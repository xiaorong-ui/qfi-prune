#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path


EXPERIMENTS = [
    ("cdpruner_pope_k32", "CDPruner", 32),
    ("cdpruner_pope_k64", "CDPruner", 64),
    ("cdpruner_pope_k128", "CDPruner", 128),
    ("qficr_pope_k32", "QFi-CR", 32),
    ("qficr_pope_k64", "QFi-CR", 64),
    ("qficr_pope_k128", "QFi-CR", 128),
]

METRIC_RE = {
    "accuracy": re.compile(r"^(?:Accuracy|Acc|accuracy)\s*:\s*([0-9.]+)%?", re.I | re.M),
    "precision": re.compile(r"^(?:Precision|precision)\s*:\s*([0-9.]+)%?", re.I | re.M),
    "recall": re.compile(r"^(?:Recall|recall)\s*:\s*([0-9.]+)%?", re.I | re.M),
    "f1": re.compile(r"^(?:F1 score|F1|f1)\s*:\s*([0-9.]+)%?", re.I | re.M),
    "yes_ratio": re.compile(r"^(?:Yes ratio|yes_ratio|Yes)\s*:\s*([0-9.]+)%?", re.I | re.M),
}
AVG_F1_RE = re.compile(r"^Average F1 score:\s*([0-9.]+)", re.I | re.M)
CATEGORY_RE = re.compile(r"^Category:\s*([^,\n]+)", re.I | re.M)


def fmt(value):
    return "" if value is None else f"{float(value):.6f}"


def parse_log(text):
    categories = {}
    current = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        cat_match = re.match(r"Category:\s*([^,]+)", line, re.I)
        if cat_match:
            current = cat_match.group(1).strip().lower()
            categories.setdefault(current, {})
            continue
        if current is None:
            continue
        for key, pattern in METRIC_RE.items():
            match = pattern.match(line)
            if match:
                categories[current][key] = float(match.group(1))

    average = {}
    for key in ("accuracy", "precision", "recall", "f1", "yes_ratio"):
        values = [metrics[key] for metrics in categories.values() if key in metrics]
        average[key] = sum(values) / len(values) if values else None

    avg_f1 = AVG_F1_RE.search(text)
    if avg_f1:
        average["f1"] = float(avg_f1.group(1))

    flat = dict(average)
    for category in ("random", "popular", "adversarial"):
        metrics = categories.get(category, {})
        for key in ("accuracy", "precision", "recall", "f1"):
            flat[f"{category}_{key}"] = metrics.get(key)
    return flat


def read_json_metrics(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def main():
    repo = Path(__file__).resolve().parents[1]
    log_root = repo / "logs" / "pope_qficr_cdpruner_k"
    out_root = repo / "outputs" / "pope_qficr_cdpruner_k"
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    by_method_k = {}
    for exp, method, token in EXPERIMENTS:
        log_path = log_root / f"{exp}.log"
        output_dir = out_root / exp
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        metrics = parse_log(text)
        metrics.update(read_json_metrics(output_dir / "metrics.json"))

        row = {
            "experiment": exp,
            "dataset": "POPE",
            "method": method,
            "K": token,
            "accuracy": fmt(metrics.get("accuracy")),
            "precision": fmt(metrics.get("precision")),
            "recall": fmt(metrics.get("recall")),
            "f1": fmt(metrics.get("f1")),
            "yes_ratio": fmt(metrics.get("yes_ratio")),
            "random_accuracy": fmt(metrics.get("random_accuracy")),
            "random_precision": fmt(metrics.get("random_precision")),
            "random_recall": fmt(metrics.get("random_recall")),
            "random_f1": fmt(metrics.get("random_f1")),
            "popular_accuracy": fmt(metrics.get("popular_accuracy")),
            "popular_precision": fmt(metrics.get("popular_precision")),
            "popular_recall": fmt(metrics.get("popular_recall")),
            "popular_f1": fmt(metrics.get("popular_f1")),
            "adversarial_accuracy": fmt(metrics.get("adversarial_accuracy")),
            "adversarial_precision": fmt(metrics.get("adversarial_precision")),
            "adversarial_recall": fmt(metrics.get("adversarial_recall")),
            "adversarial_f1": fmt(metrics.get("adversarial_f1")),
            "log_path": str(log_path),
            "output_dir": str(output_dir),
        }
        rows.append(row)
        by_method_k[(method, token)] = row

    summary_path = out_root / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    compare_rows = []
    for token in (32, 64, 128):
        cd = by_method_k.get(("CDPruner", token), {})
        qf = by_method_k.get(("QFi-CR", token), {})

        def num(row, key):
            value = row.get(key, "")
            return None if value == "" else float(value)

        cd_acc, qf_acc = num(cd, "accuracy"), num(qf, "accuracy")
        cd_f1, qf_f1 = num(cd, "f1"), num(qf, "f1")
        compare_rows.append({
            "K": token,
            "cdpruner_accuracy": cd.get("accuracy", ""),
            "qficr_accuracy": qf.get("accuracy", ""),
            "delta_qficr_minus_cdpruner_accuracy": fmt(None if cd_acc is None or qf_acc is None else qf_acc - cd_acc),
            "cdpruner_f1": cd.get("f1", ""),
            "qficr_f1": qf.get("f1", ""),
            "delta_qficr_minus_cdpruner_f1": fmt(None if cd_f1 is None or qf_f1 is None else qf_f1 - cd_f1),
            "cdpruner_precision": cd.get("precision", ""),
            "qficr_precision": qf.get("precision", ""),
            "cdpruner_recall": cd.get("recall", ""),
            "qficr_recall": qf.get("recall", ""),
            "cdpruner_yes_ratio": cd.get("yes_ratio", ""),
            "qficr_yes_ratio": qf.get("yes_ratio", ""),
            "cdpruner_log_path": cd.get("log_path", ""),
            "qficr_log_path": qf.get("log_path", ""),
        })

    compare_path = out_root / "compare_qficr_vs_cdpruner_pope.csv"
    with compare_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    print(summary_path)
    print(compare_path)


if __name__ == "__main__":
    main()
