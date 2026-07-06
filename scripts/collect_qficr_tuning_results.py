#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path


EXPERIMENTS = [
    ("baseline", 0.5, 0.0, False, "none", 0.00, 0.25, 16, 0),
    ("alpha_050", 0.5, 0.0, False, "none", 0.00, 0.25, 16, 0),
    ("alpha_075", 0.75, 0.0, False, "none", 0.00, 0.25, 16, 0),
    ("alpha_125", 1.25, 0.0, False, "none", 0.00, 0.25, 16, 0),
    ("prior_lam_010", 0.5, 0.10, True, "full", 0.00, 0.25, 16, 0),
    ("prior_lam_025", 0.5, 0.25, True, "full", 0.00, 0.25, 16, 0),
    ("prior_lam_050", 0.5, 0.50, True, "full", 0.00, 0.25, 16, 0),
    ("alpha050_prior010", 0.5, 0.10, True, "full", 0.00, 0.25, 16, 0),
    ("alpha075_prior010", 0.75, 0.10, True, "full", 0.00, 0.25, 16, 0),
    ("rho_max_down", 0.5, 0.0, False, "none", 0.00, 0.20, 16, 0),
    ("rho_max_up", 0.5, 0.0, False, "none", 0.00, 0.30, 16, 0),
    ("cap_down", 0.5, 0.0, False, "none", 0.00, 0.25, 12, 0),
    ("cap_up", 0.5, 0.0, False, "none", 0.00, 0.25, 20, 0),
    ("debug_prior_stats", 0.5, 1.0, True, "full", 0.00, 0.25, 16, 0),
]

PATTERNS = {
    "accuracy": re.compile(r"^(?:Accuracy|acc|overall):\s*([0-9.]+)%?", re.IGNORECASE | re.MULTILINE),
    "binary": re.compile(r"^Binary:\s*([0-9.]+)%?", re.IGNORECASE | re.MULTILINE),
    "open": re.compile(r"^Open:\s*([0-9.]+)%?", re.IGNORECASE | re.MULTILINE),
    "distribution": re.compile(r"^(?:Distribution|Dist):\s*([0-9.]+)", re.IGNORECASE | re.MULTILINE),
}


def parse_metric(text, key):
    match = PATTERNS[key].search(text)
    return float(match.group(1)) if match else None


def average_nested(records):
    totals = {}
    counts = {}

    def visit(prefix, value):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(f"{prefix}.{k}" if prefix else k, v)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            totals[prefix] = totals.get(prefix, 0.0) + float(value)
            counts[prefix] = counts.get(prefix, 0) + 1

    for record in records:
        visit("", record)
    return {key: totals[key] / counts[key] for key in sorted(totals) if counts[key] > 0}


def collect_debug_stats(repo):
    jsonl_path = repo / "outputs" / "qficr_tuning" / "debug_prior_stats.jsonl"
    json_path = repo / "outputs" / "qficr_tuning" / "debug_prior_stats.json"
    if not jsonl_path.exists():
        return
    records = []
    with jsonl_path.open(errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    summary = {
        "num_records": len(records),
        "mean": average_nested(records),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def main():
    repo = Path(__file__).resolve().parents[1]
    log_root = repo / "logs" / "qficr_tuning"
    out_root = repo / "outputs" / "qficr_tuning"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    baseline_acc = None
    parsed = {}
    for exp, *_ in EXPERIMENTS:
        text = (log_root / f"{exp}.log").read_text(errors="replace") if (log_root / f"{exp}.log").exists() else ""
        parsed[exp] = {
            "accuracy": parse_metric(text, "accuracy"),
            "binary": parse_metric(text, "binary"),
            "open": parse_metric(text, "open"),
            "distribution": parse_metric(text, "distribution"),
        }
    baseline_acc = parsed.get("baseline", {}).get("accuracy")

    for exp, anchor_alpha, prior_lambda, use_prior, prior_ablation, rho_min, rho_max, cap, topm in EXPERIMENTS:
        metrics = parsed[exp]
        accuracy = metrics["accuracy"]
        delta = "" if accuracy is None or baseline_acc is None else f"{accuracy - baseline_acc:.2f}"
        rows.append({
            "experiment": exp,
            "accuracy": "" if accuracy is None else f"{accuracy:.2f}",
            "delta_vs_baseline": delta,
            "binary": "" if metrics["binary"] is None else f"{metrics['binary']:.2f}",
            "open": "" if metrics["open"] is None else f"{metrics['open']:.2f}",
            "distribution": "" if metrics["distribution"] is None else f"{metrics['distribution']:.2f}",
            "anchor_alpha": anchor_alpha,
            "prior_lambda": prior_lambda,
            "use_spatial_local_prior": int(use_prior),
            "prior_ablation": prior_ablation,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "restoration_cap": cap,
            "topM": topm,
            "log_path": str(log_root / f"{exp}.log"),
            "output_dir": str(out_root / exp),
        })

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    collect_debug_stats(repo)
    print(csv_path)


if __name__ == "__main__":
    main()
