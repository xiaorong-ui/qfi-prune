#!/usr/bin/env python3
import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/qficr_recovery_budget_fix"
LOG_ROOT = ROOT / "logs/qficr_recovery_budget_fix"
OLD_SUMMARY = ROOT / "outputs/qficr_cdpruner_k32_k64_k128_combined/summary_long.csv"

VARIANTS = {
    "v0_abs_cap_baseline_unified_gate": {
        "budget_mode": "abs_cap",
        "ratio_cap": "",
        "abs_cap": "16",
        "entropy_gamma": "1.0",
        "support_beta": "0.0",
        "self_discount": "0.0",
    },
    "v1_ratio_only": {
        "budget_mode": "ratio_only",
        "ratio_cap": "",
        "abs_cap": "16",
        "entropy_gamma": "1.0",
        "support_beta": "0.0",
        "self_discount": "0.0",
    },
    "v3_ratio_cap_020": {
        "budget_mode": "ratio_cap",
        "ratio_cap": "0.20",
        "abs_cap": "16",
        "entropy_gamma": "1.0",
        "support_beta": "0.0",
        "self_discount": "0.0",
    },
    "v5_ratio_cap_025_support_self": {
        "budget_mode": "ratio_cap",
        "ratio_cap": "0.25",
        "abs_cap": "16",
        "entropy_gamma": "1.0",
        "support_beta": "0.25",
        "self_discount": "0.25",
    },
    "v7_ratio_cap_025_gamma2": {
        "budget_mode": "ratio_cap",
        "ratio_cap": "0.25",
        "abs_cap": "16",
        "entropy_gamma": "2.0",
        "support_beta": "0.0",
        "self_discount": "0.0",
    },
}

BENCHES = ["textvqa", "pope", "mme"]
KS = [32, 128]

SUMMARY_FIELDS = [
    "variant",
    "benchmark",
    "K",
    "method",
    "budget_mode",
    "ratio_cap",
    "abs_cap",
    "entropy_gamma",
    "support_beta",
    "self_discount",
    "yes_no_gate",
    "metric_name",
    "main_score",
    "accuracy",
    "binary",
    "open",
    "precision",
    "recall",
    "f1",
    "yes_ratio",
    "mme_perception",
    "mme_cognition",
    "mme_total",
    "mme_total_norm",
    "mmbench_circular_accuracy",
    "status",
    "output_dir",
    "log_path",
    "notes",
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def fnum(value):
    try:
        if value == "":
            return None
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(value, ndigits=4):
    value = fnum(value)
    return "" if value is None else f"{value:.{ndigits}f}"


def last(pattern, text):
    matches = re.findall(pattern, text, flags=re.I | re.M)
    return matches[-1] if matches else ""


def first(pattern, text):
    match = re.search(pattern, text, flags=re.I | re.M)
    return match.group(1) if match else ""


def pct(value):
    return "" if value is None else f"{value * 100.0:.4f}"


def parse_pope(text):
    avg = fnum(last(r"^Average F1 score:\s*([0-9.]+)", text))
    acc_vals = [fnum(x) for x in re.findall(r"^Accuracy:\s*([0-9.]+)", text, flags=re.I | re.M)]
    prec_vals = [fnum(x) for x in re.findall(r"^Precision:\s*([0-9.]+)", text, flags=re.I | re.M)]
    rec_vals = [fnum(x) for x in re.findall(r"^Recall:\s*([0-9.]+)", text, flags=re.I | re.M)]
    yes_vals = [fnum(x) for x in re.findall(r"^Yes ratio:\s*([0-9.]+)", text, flags=re.I | re.M)]

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "metric_name": "F1",
        "main_score": pct(avg),
        "accuracy": pct(mean(acc_vals)),
        "precision": pct(mean(prec_vals)),
        "recall": pct(mean(rec_vals)),
        "f1": pct(avg),
        "yes_ratio": pct(mean(yes_vals)),
    }


def parse_mme(text):
    perception = first(r"=+\s*Perception\s*=+[\s\S]*?total score:\s*([0-9.]+)", text)
    cognition = first(r"=+\s*Cognition\s*=+[\s\S]*?total score:\s*([0-9.]+)", text)
    p = fnum(perception)
    c = fnum(cognition)
    total = p + c if p is not None and c is not None else None
    row = {
        "metric_name": "MME-Total/20",
        "mme_perception": fmt(p),
        "mme_cognition": fmt(c),
        "mme_total": fmt(total),
        "mme_total_norm": "" if total is None else f"{total / 20.0:.4f}",
    }
    row["main_score"] = row["mme_total_norm"]
    if not row["main_score"]:
        row["notes"] = "MME total unavailable; do not treat partial score as total"
    return row


def parse_log(bench, log_path):
    text = read_text(log_path)
    failed = bool(re.search(r"Traceback|CUDA out of memory|outofmemory|Killed|Exception", text, re.I))
    if bench == "textvqa":
        row = {"metric_name": "Accuracy", "main_score": fmt(last(r"^Accuracy:\s*([0-9.]+)%?", text))}
        row["accuracy"] = row["main_score"]
    elif bench == "pope":
        row = parse_pope(text)
    elif bench == "mme":
        row = parse_mme(text)
    else:
        row = {"metric_name": "Accuracy", "main_score": ""}
    row.setdefault("notes", "")
    if row.get("main_score"):
        row["status"] = "completed" if not failed else "completed_with_log_warning"
    else:
        row["status"] = "failed_or_incomplete" if text else "missing"
    return row


def write_csv(path, fields, rows, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def old_scores():
    scores = {}
    if not OLD_SUMMARY.exists():
        return scores
    with OLD_SUMMARY.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            bench = row.get("benchmark", "")
            if bench not in BENCHES:
                continue
            method = row.get("method", "").lower().replace("-", "").replace(" ", "")
            method = "qficr" if "qficr" in method else "cdpruner" if "cdpruner" in method else method
            k = row.get("K", "")
            score = fnum(row.get("main_score"))
            if score is not None:
                scores[(bench, str(k), method)] = score
    return scores


def collect():
    rows = []
    inventory = []
    for variant, cfg in VARIANTS.items():
        for bench in BENCHES:
            for k in KS:
                out_dir = OUT_ROOT / variant / f"{bench}_k{k}"
                log_path = LOG_ROOT / f"{variant}_{bench}_k{k}.log"
                answer = out_dir / "answers.jsonl"
                row = {field: "" for field in SUMMARY_FIELDS}
                row.update(cfg)
                row.update(
                    {
                        "variant": variant,
                        "benchmark": bench,
                        "K": str(k),
                        "method": "QFi-CR",
                        "yes_no_gate": "0",
                        "output_dir": rel(out_dir),
                        "log_path": rel(log_path),
                    }
                )
                parsed = parse_log(bench, log_path)
                row.update(parsed)
                rows.append(row)
                inventory.append(
                    {
                        "variant": variant,
                        "benchmark": bench,
                        "K": str(k),
                        "output_dir": rel(out_dir),
                        "log_path": rel(log_path),
                        "answers_exists": str(answer.exists()),
                        "answers_nonempty": str(answer.exists() and answer.stat().st_size > 0),
                        "status": row["status"],
                    }
                )
    return rows, inventory


def compare(rows):
    scores = old_scores()
    out = []
    for row in rows:
        score = fnum(row.get("main_score"))
        old_qfi = scores.get((row["benchmark"], row["K"], "qficr"))
        cd = scores.get((row["benchmark"], row["K"], "cdpruner"))
        out.append(
            {
                "benchmark": row["benchmark"],
                "K": row["K"],
                "metric_name": row.get("metric_name", ""),
                "old_qficr_score": fmt(old_qfi),
                "variant": row["variant"],
                "variant_score": fmt(score),
                "delta_variant_minus_old_qficr": "" if score is None or old_qfi is None else f"{score - old_qfi:.4f}",
                "cdpruner_score": fmt(cd),
                "delta_variant_minus_cdpruner": "" if score is None or cd is None else f"{score - cd:.4f}",
                "better_than_old_qficr": "" if score is None or old_qfi is None else str(score > old_qfi),
                "better_than_cdpruner": "" if score is None or cd is None else str(score > cd),
                "status": row.get("status", ""),
                "notes": row.get("notes", ""),
            }
        )
    return out


def write_table(rows):
    lines = [
        "\\begin{tabular}{lllrr}",
        "\\toprule",
        "Variant & Benchmark & K & Metric & Score \\\\",
        "\\midrule",
    ]
    for row in rows:
        if row.get("main_score"):
            lines.append(
                f"{row['variant']} & {row['benchmark']} & {row['K']} & {row.get('metric_name','')} & {row['main_score']} \\\\"
            )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (OUT_ROOT / "table_recovery_budget_fix.tex").write_text("\n".join(lines), encoding="utf-8")


def write_report(rows, comp):
    completed = [r for r in rows if r.get("status", "").startswith("completed")]
    lines = [
        "# QFi-CR Recovery Budget Fix Report",
        "",
        f"Completed runs: {len(completed)} / {len(rows)}",
        "",
        "Key debug stats are in `debug_stats_summary.csv`.",
        "",
        "Do not average raw MME with percentage metrics; this report uses MME Total/20 when available.",
    ]
    (OUT_ROOT / "report_recovery_budget_fix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    global OUT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    args = parser.parse_args()
    OUT_ROOT = Path(args.output_root)
    rows, inventory = collect()
    comp = compare(rows)
    write_csv(OUT_ROOT / "inventory.tsv", inventory[0].keys() if inventory else [], inventory, delimiter="\t")
    write_csv(OUT_ROOT / "summary.csv", SUMMARY_FIELDS, rows)
    write_csv(
        OUT_ROOT / "compare_variants.csv",
        [
            "benchmark",
            "K",
            "metric_name",
            "old_qficr_score",
            "variant",
            "variant_score",
            "delta_variant_minus_old_qficr",
            "cdpruner_score",
            "delta_variant_minus_cdpruner",
            "better_than_old_qficr",
            "better_than_cdpruner",
            "status",
            "notes",
        ],
        comp,
    )
    debug_paths = [str(p) for p in OUT_ROOT.glob("*/*/debug_qficr_stats.jsonl")]
    if debug_paths:
        subprocess.run(
            [
                "python",
                str(ROOT / "scripts/summarize_qficr_debug_stats.py"),
                *debug_paths,
                "--output",
                str(OUT_ROOT / "debug_stats_summary.csv"),
            ],
            check=False,
        )
    else:
        write_csv(
            OUT_ROOT / "debug_stats_summary.csv",
            [
                "run_name",
                "benchmark",
                "K",
                "budget_mode",
                "num_samples",
                "mean_entropy_norm",
                "std_entropy_norm",
                "mean_rho",
                "std_rho",
                "mean_raw_k_rest",
                "mean_final_k_rest",
                "mean_final_k_red",
                "cap_hit_rate",
                "yes_no_gate_rate",
                "mean_p_core",
                "mean_p_rest",
                "mean_residual_gain_rest",
                "mean_support_rest",
                "mean_self_gain_rest",
            ],
            [],
        )
    write_table(rows)
    write_report(rows, comp)
    for path in [
        OUT_ROOT / "inventory.tsv",
        OUT_ROOT / "summary.csv",
        OUT_ROOT / "compare_variants.csv",
        OUT_ROOT / "debug_stats_summary.csv",
        OUT_ROOT / "table_recovery_budget_fix.tex",
        OUT_ROOT / "report_recovery_budget_fix.md",
    ]:
        print(path)


if __name__ == "__main__":
    main()
