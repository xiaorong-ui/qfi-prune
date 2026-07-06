#!/usr/bin/env python3
import csv
import re
from pathlib import Path


EXPERIMENTS = [
    ("full", "full", "full", "full"),
    ("only_reduction", "none", "full", "full"),
    ("random_restoration", "random", "full", "full"),
    ("fixed_ratio", "fixed_ratio", "full", "full"),
    ("no_spatial_local_prior", "full", "none", "full"),
    ("no_spatial", "full", "no_spatial", "full"),
    ("no_local", "full", "no_local", "full"),
    ("sem_only", "full", "full", "sem_only"),
    ("cls_only", "full", "full", "cls_only"),
    ("fixed_fusion", "full", "full", "fixed_fusion"),
]


METRIC_PATTERNS = {
    "binary": re.compile(r"^Binary:\s*([0-9.]+)%", re.MULTILINE),
    "open": re.compile(r"^Open:\s*([0-9.]+)%", re.MULTILINE),
    "accuracy": re.compile(r"^Accuracy:\s*([0-9.]+)%", re.MULTILINE),
    "distribution": re.compile(r"^Distribution:\s*([0-9.]+)", re.MULTILINE),
}


def parse_metric(text, name):
    match = METRIC_PATTERNS[name].search(text)
    return match.group(1) if match else ""


def main():
    repo = Path(__file__).resolve().parents[1]
    log_root = repo / "logs" / "ablation_qficr"
    out_root = repo / "outputs" / "ablation_qficr"
    out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for exp, restoration_mode, prior_ablation, observation_mode in EXPERIMENTS:
        log_path = log_root / f"{exp}.log"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        rows.append(
            {
                "experiment": exp,
                "kernel": "relu_square",
                "restoration_mode": restoration_mode,
                "prior_ablation": prior_ablation,
                "observation_mode": observation_mode,
                "binary": parse_metric(text, "binary"),
                "open": parse_metric(text, "open"),
                "accuracy": parse_metric(text, "accuracy"),
                "distribution": parse_metric(text, "distribution"),
                "log_path": str(log_path),
                "output_dir": str(out_root / exp),
            }
        )

    csv_path = out_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "kernel",
                "restoration_mode",
                "prior_ablation",
                "observation_mode",
                "binary",
                "open",
                "accuracy",
                "distribution",
                "log_path",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(csv_path)


if __name__ == "__main__":
    main()
