#!/usr/bin/env python3
import csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
LOG_ROOT = REPO / "logs" / "qficr_general_tuning"
OUT_ROOT = REPO / "outputs" / "qficr_general_tuning"
SETTINGS = {
    "baseline_last2": ("0.0", "-1.0", "last2"),
    "entropy_anchor_005": ("0.05", "-1.0", "last2"),
    "entropy_anchor_010": ("0.10", "-1.0", "last2"),
    "residual_budget_050": ("0.0", "0.50", "last2"),
    "residual_budget_075": ("0.0", "0.75", "last2"),
    "cls_last4_avg": ("0.0", "-1.0", "last4"),
}
DATASETS = ("gqa", "pope")
KS = (32, 64)


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
        metrics["accuracy"] = last_float([r"^Accuracy:\s*([0-9.]+)%?"], text)
        metrics["precision"] = last_float([r"^Precision:\s*([0-9.]+)%?"], text)
        metrics["recall"] = last_float([r"^Recall:\s*([0-9.]+)%?"], text)
        metrics["f1"] = last_float([r"^Average F1 score:\s*([0-9.]+)%?", r"^F1 score:\s*([0-9.]+)%?"], text)
        metrics["yes_ratio"] = last_float([r"^Yes ratio:\s*([0-9.]+)%?"], text)
        complete = metrics["accuracy"] != "" or metrics["f1"] != ""
    status = "completed" if complete and not failed else ("failed_or_incomplete" if failed else "incomplete")
    return metrics, status


def delta(value, base):
    if value == "" or base == "":
        return ""
    return float(value) - float(base)


def delta_str(value, base):
    d = delta(value, base)
    return "" if d == "" else f"{d:.4f}"


def metric_for_compare(row):
    if row["dataset"] == "GQA":
        return row.get("accuracy", "")
    return row.get("f1", "") or row.get("accuracy", "")


def note_for_setting(setting, compare_rows):
    if setting == "baseline_last2":
        return "baseline"
    rows = [r for r in compare_rows if r["setting"] == setting]
    valid = [r for r in rows if r.get("_main_delta") != ""]
    if len(valid) < 4:
        return "incomplete"
    tol = 1e-6
    non_drop = sum(float(r["_main_delta"]) >= -tol for r in valid)
    improves = sum(float(r["_main_delta"]) > tol for r in valid)
    gqa_hurt = any(r["dataset"] == "GQA" and float(r["_main_delta"]) < -tol for r in valid)
    k32_up = any(r["K"] == 32 and float(r["_main_delta"]) > tol for r in valid)
    k64_hurt = any(r["K"] == 64 and float(r["_main_delta"]) < -tol for r in valid)
    pope_up = any(r["dataset"] == "POPE" and float(r["_main_delta"]) > tol for r in valid)
    all_down = all(float(r["_main_delta"]) < -tol for r in valid)
    all_neutral = all(abs(float(r["_main_delta"])) <= tol for r in valid)
    if non_drop >= 3 and improves >= 1:
        return "candidate_general"
    if pope_up and gqa_hurt:
        return "pope_specific_not_default"
    if k32_up and k64_hurt:
        return "low_budget_specific_not_default"
    if all_down:
        return "reject"
    if all_neutral:
        return "neutral"
    return "mixed_not_default"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for dataset in DATASETS:
        for k in KS:
            for setting, params in SETTINGS.items():
                exp = f"{dataset}_k{k}_{setting}"
                log_path = LOG_ROOT / f"{exp}.log"
                out_dir = OUT_ROOT / exp
                entropy_anchor_ratio, residual_budget_gamma, cls_prior_layers = params
                metrics, status = parse_log(dataset, log_path)
                rows.append({
                    "experiment": exp,
                    "setting": setting,
                    "dataset": dataset.upper(),
                    "K": k,
                    "accuracy": metrics.get("accuracy", ""),
                    "binary": metrics.get("binary", ""),
                    "open": metrics.get("open", ""),
                    "precision": metrics.get("precision", ""),
                    "recall": metrics.get("recall", ""),
                    "f1": metrics.get("f1", ""),
                    "yes_ratio": metrics.get("yes_ratio", ""),
                    "distribution": metrics.get("distribution", ""),
                    "entropy_anchor_ratio": entropy_anchor_ratio,
                    "residual_budget_gamma": residual_budget_gamma,
                    "cls_prior_layers": cls_prior_layers,
                    "anchor_alpha": 0.5,
                    "rho_max": 0.25,
                    "use_spatial_local_prior": False,
                    "prior_lambda": 0.0,
                    "overlap_kernel": "relu_square",
                    "restoration_mode": "full",
                    "observation_mode": "full",
                    "log_path": str(log_path.relative_to(REPO)),
                    "output_dir": str(out_dir.relative_to(REPO)),
                    "status": status,
                })

    summary_path = OUT_ROOT / "summary.csv"
    fieldnames = [
        "experiment", "setting", "dataset", "K", "accuracy", "binary", "open",
        "precision", "recall", "f1", "yes_ratio", "distribution",
        "entropy_anchor_ratio", "residual_budget_gamma", "cls_prior_layers",
        "anchor_alpha", "rho_max", "use_spatial_local_prior", "prior_lambda",
        "overlap_kernel", "restoration_mode", "observation_mode",
        "log_path", "output_dir", "status",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    base_by_key = {(r["dataset"], r["K"]): r for r in rows if r["setting"] == "baseline_last2"}
    compare_rows = []
    for row in rows:
        base = base_by_key.get((row["dataset"], row["K"]), {})
        main_delta = delta(metric_for_compare(row), metric_for_compare(base))
        compare_rows.append({
            "dataset": row["dataset"],
            "K": row["K"],
            "setting": row["setting"],
            "baseline_accuracy": base.get("accuracy", ""),
            "setting_accuracy": row.get("accuracy", ""),
            "delta_accuracy": delta_str(row.get("accuracy", ""), base.get("accuracy", "")),
            "baseline_open": base.get("open", ""),
            "setting_open": row.get("open", ""),
            "delta_open": delta_str(row.get("open", ""), base.get("open", "")),
            "baseline_f1": base.get("f1", ""),
            "setting_f1": row.get("f1", ""),
            "delta_f1": delta_str(row.get("f1", ""), base.get("f1", "")),
            "baseline_yes_ratio": base.get("yes_ratio", ""),
            "setting_yes_ratio": row.get("yes_ratio", ""),
            "delta_yes_ratio": delta_str(row.get("yes_ratio", ""), base.get("yes_ratio", "")),
            "notes": "",
            "_main_delta": "" if main_delta == "" else f"{main_delta:.8f}",
        })

    notes = {setting: note_for_setting(setting, compare_rows) for setting in SETTINGS}
    for row in compare_rows:
        row["notes"] = notes[row["setting"]]
        row.pop("_main_delta", None)

    compare_path = OUT_ROOT / "compare_against_baseline.csv"
    compare_fields = [
        "dataset", "K", "setting",
        "baseline_accuracy", "setting_accuracy", "delta_accuracy",
        "baseline_open", "setting_open", "delta_open",
        "baseline_f1", "setting_f1", "delta_f1",
        "baseline_yes_ratio", "setting_yes_ratio", "delta_yes_ratio",
        "notes",
    ]
    with compare_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=compare_fields)
        writer.writeheader()
        writer.writerows(compare_rows)

    print(f"summary={summary_path}")
    print(f"compare={compare_path}")


if __name__ == "__main__":
    main()
