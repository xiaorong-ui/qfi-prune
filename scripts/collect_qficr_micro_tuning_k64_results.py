#!/usr/bin/env python3
import csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "logs" / "qficr_micro_tuning_k64"
OUT_ROOT = REPO / "outputs" / "qficr_micro_tuning_k64"
SETTINGS = {
    "baseline": (1.0, 1.0, 0.0, 2.0, 1.0),
    "beta_075": (0.75, 1.0, 0.0, 2.0, 1.0),
    "beta_125": (1.25, 1.0, 0.0, 2.0, 1.0),
    "beta_150": (1.50, 1.0, 0.0, 2.0, 1.0),
    "tau_085": (1.0, 0.85, 0.0, 2.0, 1.0),
    "tau_115": (1.0, 1.15, 0.0, 2.0, 1.0),
    "tau_130": (1.0, 1.30, 0.0, 2.0, 1.0),
    "div_010": (1.0, 1.0, 0.10, 2.0, 1.0),
    "yn_gate_05": (1.0, 1.0, 0.0, 2.0, 0.5),
}
RUN_SETTINGS = ["baseline", "beta_125", "tau_085", "tau_115", "div_010"]


def last_float(patterns, text):
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.I | re.M)
        if matches:
            try:
                return float(matches[-1])
            except (TypeError, ValueError):
                pass
    return ""


def parse_log(dataset, log_path):
    if not log_path.exists():
        return {}, "missing"
    text = log_path.read_text(errors="replace")
    failed = re.search(r"CUDA out of memory|outofmemory|Traceback|Exception|Killed", text, re.I)
    metrics = {}
    if dataset == "gqa":
        metrics["accuracy"] = last_float([r"^Accuracy:\s*([0-9.]+)%?"], text)
        metrics["binary"] = last_float([r"^Binary:\s*([0-9.]+)%?"], text)
        metrics["open"] = last_float([r"^Open:\s*([0-9.]+)%?"], text)
        metrics["distribution"] = last_float([r"^Distribution:\s*([0-9.]+)"], text)
        complete = metrics["accuracy"] != ""
    else:
        metrics["accuracy"] = last_float([r"^Accuracy:\s*([0-9.]+)%?", r"accuracy[:=]\s*([0-9.]+)"], text)
        metrics["precision"] = last_float([r"^Precision:\s*([0-9.]+)%?", r"precision[:=]\s*([0-9.]+)"], text)
        metrics["recall"] = last_float([r"^Recall:\s*([0-9.]+)%?", r"recall[:=]\s*([0-9.]+)"], text)
        metrics["f1"] = last_float([r"^F1 score:\s*([0-9.]+)%?", r"^Average F1 score:\s*([0-9.]+)%?", r"\bf1[:=]\s*([0-9.]+)"], text)
        metrics["yes_ratio"] = last_float([r"^Yes ratio:\s*([0-9.]+)%?", r"yes ratio[:=]\s*([0-9.]+)"], text)
        complete = metrics["accuracy"] != "" or metrics["f1"] != ""
    status = "completed" if complete and not failed else ("failed_or_incomplete" if failed else "incomplete")
    return metrics, status


def fmt_delta(a, b):
    if a == "" or b == "":
        return ""
    return f"{float(a) - float(b):.4f}"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in ("gqa", "pope"):
        for setting in RUN_SETTINGS:
            exp = f"{dataset}_{setting}"
            log_path = LOG_ROOT / f"{exp}.log"
            out_dir = OUT_ROOT / exp
            rest_beta, obs_temp, div_lam, cand_factor, yn_gate = SETTINGS[setting]
            metrics, status = parse_log(dataset, log_path)
            rows.append({
                "experiment": exp,
                "dataset": dataset.upper(),
                "K": 64,
                "accuracy": metrics.get("accuracy", ""),
                "binary": metrics.get("binary", ""),
                "open": metrics.get("open", ""),
                "precision": metrics.get("precision", ""),
                "recall": metrics.get("recall", ""),
                "f1": metrics.get("f1", ""),
                "yes_ratio": metrics.get("yes_ratio", ""),
                "distribution": metrics.get("distribution", ""),
                "rest_beta": rest_beta,
                "anchor_alpha": 0.5,
                "obs_temperature": obs_temp,
                "restore_div_lambda": div_lam,
                "restore_candidate_factor": cand_factor,
                "yn_budget_gate": yn_gate,
                "rho_max": 0.25,
                "use_spatial_local_prior": False,
                "prior_lambda": 0.0,
                "log_path": str(log_path.relative_to(REPO)),
                "output_dir": str(out_dir.relative_to(REPO)),
                "status": status,
            })

    summary_path = OUT_ROOT / "summary.csv"
    fieldnames = [
        "experiment", "dataset", "K", "accuracy", "binary", "open",
        "precision", "recall", "f1", "yes_ratio", "distribution",
        "rest_beta", "anchor_alpha", "obs_temperature",
        "restore_div_lambda", "restore_candidate_factor", "yn_budget_gate",
        "rho_max", "use_spatial_local_prior", "prior_lambda",
        "log_path", "output_dir", "status",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_exp = {row["experiment"]: row for row in rows}
    compare_rows = []
    for dataset in ("gqa", "pope"):
        base = by_exp.get(f"{dataset}_baseline", {})
        for setting in RUN_SETTINGS:
            exp = by_exp.get(f"{dataset}_{setting}", {})
            compare_rows.append({
                "dataset": dataset.upper(),
                "experiment": f"{dataset}_{setting}",
                "baseline_accuracy": base.get("accuracy", ""),
                "experiment_accuracy": exp.get("accuracy", ""),
                "delta_accuracy": fmt_delta(exp.get("accuracy", ""), base.get("accuracy", "")),
                "baseline_open": base.get("open", ""),
                "experiment_open": exp.get("open", ""),
                "delta_open": fmt_delta(exp.get("open", ""), base.get("open", "")),
                "baseline_f1": base.get("f1", ""),
                "experiment_f1": exp.get("f1", ""),
                "delta_f1": fmt_delta(exp.get("f1", ""), base.get("f1", "")),
                "baseline_yes_ratio": base.get("yes_ratio", ""),
                "experiment_yes_ratio": exp.get("yes_ratio", ""),
                "delta_yes_ratio": fmt_delta(exp.get("yes_ratio", ""), base.get("yes_ratio", "")),
                "notes": "baseline" if setting == "baseline" else "",
            })
    compare_path = OUT_ROOT / "compare_against_baseline.csv"
    with compare_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"summary={summary_path}")
    print(f"compare={compare_path}")


if __name__ == "__main__":
    main()
