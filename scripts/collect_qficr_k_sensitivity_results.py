#!/usr/bin/env python3
import csv
import re
from pathlib import Path


EXPERIMENTS = [
    ("qficr_k32", 32, "qficr", "full"),
    ("qficr_k128", 128, "qficr", "full"),
    ("only_reduction_k32", 32, "only_reduction", "none"),
    ("only_reduction_k128", 128, "only_reduction", "none"),
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


def fmt(value):
    return "" if value is None else f"{value:.2f}"


def main():
    repo = Path(__file__).resolve().parents[1]
    log_root = repo / "logs" / "qficr_k_sensitivity"
    out_root = repo / "outputs" / "qficr_k_sensitivity"
    out_root.mkdir(parents=True, exist_ok=True)

    parsed = {}
    for exp, _, _, _ in EXPERIMENTS:
        log_path = log_root / f"{exp}.log"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        parsed[exp] = {
            "accuracy": parse_metric(text, "accuracy"),
            "binary": parse_metric(text, "binary"),
            "open": parse_metric(text, "open"),
            "distribution": parse_metric(text, "distribution"),
        }

    qficr_by_k = {
        32: parsed["qficr_k32"]["accuracy"],
        128: parsed["qficr_k128"]["accuracy"],
    }

    rows = []
    for exp, token, method, restoration_mode in EXPERIMENTS:
        metrics = parsed[exp]
        accuracy = metrics["accuracy"]
        ref = qficr_by_k[token]
        delta = None if accuracy is None or ref is None else accuracy - ref
        rows.append({
            "experiment": exp,
            "K": token,
            "method": method,
            "restoration_mode": restoration_mode,
            "accuracy": fmt(accuracy),
            "binary": fmt(metrics["binary"]),
            "open": fmt(metrics["open"]),
            "distribution": fmt(metrics["distribution"]),
            "delta_vs_qficr_same_K": fmt(delta),
            "log_path": str(log_root / f"{exp}.log"),
            "output_dir": str(out_root / exp),
        })

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
