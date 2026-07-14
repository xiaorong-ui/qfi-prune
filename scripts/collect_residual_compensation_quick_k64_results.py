#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/qficr_residual_compensation_quick_k64"
LOG_ROOT = ROOT / "logs/qficr_residual_compensation_quick_k64"
OLD_SUMMARY = ROOT / "outputs/qficr_cdpruner_k32_k64_k128_combined/summary_long.csv"

VARIANTS = [
    "current_anchor",
    "weak_anchor",
    "no_anchor",
    "marginal_gain_weak_gate",
    "near_boundary_fill",
]
BENCHES = ["textvqa", "mme"]
K = 64
EXPECTED_ANSWER_LINES = {
    "textvqa": 5000,
    "mme": 2374,
}

SUMMARY_FIELDS = [
    "variant",
    "benchmark",
    "K",
    "metric_name",
    "main_score",
    "status",
    "answers_path",
    "score_path",
    "log_path",
    "notes",
]

AVG_FIELDS = ["variant", "num_benchmarks", "avg_score", "rank", "notes"]

COMPARE_FIELDS = [
    "benchmark",
    "K",
    "variant",
    "score",
    "current_anchor_score",
    "delta_vs_current_anchor",
    "status",
    "notes",
]

DIAG_FIELDS = [
    "variant",
    "benchmark",
    "K",
    "num_records",
    "mean_final_k_rest",
    "mean_final_k_red",
    "mean_entropy_norm",
    "mean_rho",
    "mean_p_restored",
    "median_p_restored",
    "mean_residual_gain_restored",
    "mean_max_overlap_to_core_restored",
    "mean_first_stage_rank_restored",
    "median_first_stage_rank_restored",
    "near_boundary_hit_rate",
    "mean_p_discarded",
    "median_p_discarded",
    "mean_residual_gain_discarded",
    "mean_max_overlap_to_core_discarded",
    "mean_first_stage_rank_discarded",
    "median_first_stage_rank_discarded",
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
        if value in (None, ""):
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


def parse_mme(text):
    perception = first(r"=+\s*Perception\s*=+[\s\S]*?total score:\s*([0-9.]+)", text)
    cognition = first(r"=+\s*Cognition\s*=+[\s\S]*?total score:\s*([0-9.]+)", text)
    p = fnum(perception)
    c = fnum(cognition)
    total = p + c if p is not None and c is not None else None
    return "" if total is None else f"{total / 20.0:.4f}"


def parse_score(bench, score_path):
    text = read_text(score_path)
    if bench == "textvqa":
        return "accuracy", fmt(last(r"^Accuracy:\s*([0-9.]+)%?", text))
    if bench == "mme":
        return "total / 20", parse_mme(text)
    return "score", ""


def count_lines(path: Path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def complete_answer_count(bench, answers_path: Path):
    expected = EXPECTED_ANSWER_LINES.get(bench)
    actual = count_lines(answers_path)
    if expected is None:
        return actual > 0, actual, expected
    return actual == expected, actual, expected


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def existing_qficr_reference():
    refs = {}
    if not OLD_SUMMARY.exists():
        return refs
    with OLD_SUMMARY.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("benchmark") not in BENCHES:
                continue
            if str(row.get("K", "")) != str(K):
                continue
            method = row.get("method", "")
            if "QFi-CR" not in method:
                continue
            score = fnum(row.get("main_score"))
            if score is not None:
                refs[row["benchmark"]] = score
    return refs


def collect_summary():
    rows = []
    refs = existing_qficr_reference()
    for variant in VARIANTS:
        for bench in BENCHES:
            out_dir = OUT_ROOT / variant / f"{bench}_k{K}"
            log_path = LOG_ROOT / f"{variant}_{bench}_k{K}.log"
            answers_path = out_dir / "answers.jsonl"
            score_path = out_dir / "eval_summary.txt"
            metric_name, score = parse_score(bench, score_path)
            answers_complete, answer_count, expected_count = complete_answer_count(bench, answers_path)
            status = "completed" if score and answers_complete else "missing"
            notes = ""
            if not score and variant == "current_anchor" and bench in refs:
                score = f"{refs[bench]:.4f}"
                metric_name = "accuracy" if bench == "textvqa" else "total / 20"
                status = "reference_reused"
                notes = "reused existing Old QFi-CR K64 score only as current-anchor reference; no new debug"
            elif score and not answers_complete:
                status = "failed_or_incomplete"
                notes = f"partial answers: {answer_count}/{expected_count}; score file ignored for averages"
                score = ""
            elif not score and score_path.exists():
                status = "failed_or_incomplete"
            rows.append({
                "variant": variant,
                "benchmark": bench,
                "K": K,
                "metric_name": metric_name,
                "main_score": score,
                "status": status,
                "answers_path": rel(answers_path) if answers_path.exists() else "",
                "score_path": rel(score_path) if score_path.exists() else "",
                "log_path": rel(log_path) if log_path.exists() else "",
                "notes": notes,
            })
    return rows


def mean(values):
    vals = [fnum(v) for v in values]
    vals = [v for v in vals if v is not None]
    return None if not vals else sum(vals) / len(vals)


def average_rows(summary_rows):
    rows = []
    for variant in VARIANTS:
        vals = [
            r["main_score"]
            for r in summary_rows
            if r["variant"] == variant
            and r["status"] in {"completed", "reference_reused"}
            and fnum(r["main_score"]) is not None
        ]
        avg = mean(vals)
        rows.append({
            "variant": variant,
            "num_benchmarks": len(vals),
            "avg_score": fmt(avg),
            "notes": "averages TextVQA accuracy and MME total/20",
        })
    scored = sorted([r for r in rows if fnum(r["avg_score"]) is not None], key=lambda r: fnum(r["avg_score"]), reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    return rows


def compare_rows(summary_rows):
    current = {
        r["benchmark"]: fnum(r["main_score"])
        for r in summary_rows
        if r["variant"] == "current_anchor"
        and r["status"] in {"completed", "reference_reused"}
        and fnum(r["main_score"]) is not None
    }
    rows = []
    for row in summary_rows:
        variant = row["variant"]
        if variant == "current_anchor":
            continue
        score = fnum(row["main_score"]) if row["status"] in {"completed", "reference_reused"} else None
        ref = current.get(row["benchmark"])
        delta = None if score is None or ref is None else score - ref
        rows.append({
            "benchmark": row["benchmark"],
            "K": K,
            "variant": variant,
            "score": fmt(score),
            "current_anchor_score": fmt(ref),
            "delta_vs_current_anchor": fmt(delta),
            "status": row["status"],
            "notes": row["notes"],
        })
    return rows


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def get_num(record, *keys):
    for key in keys:
        value = fnum(record.get(key))
        if value is not None:
            return value
    return None


def diag_for(variant, bench):
    path = OUT_ROOT / variant / f"{bench}_k{K}" / "debug_qficr_stats.jsonl"
    buckets = {
        "k_rest": [], "k_red": [], "entropy": [], "rho": [],
        "p_rest": [], "median_p_rest": [], "res_rest": [], "overlap_rest": [],
        "rank_rest": [], "median_rank_rest": [], "near": [],
        "p_discarded": [], "median_p_discarded": [], "res_discarded": [],
        "overlap_discarded": [], "rank_discarded": [], "median_rank_discarded": [],
    }
    records = 0
    for rec in iter_jsonl(path) or []:
        records += 1
        mapping = {
            "k_rest": ("qficr_final_k_rest",),
            "k_red": ("qficr_final_k_red",),
            "entropy": ("qficr_entropy_norm", "adapt_entropy_norm"),
            "rho": ("qficr_rho", "adapt_recover_ratio"),
            "p_rest": ("qficr_mean_p_rest",),
            "median_p_rest": ("qficr_median_p_rest",),
            "res_rest": ("qficr_mean_residual_gain_rest",),
            "overlap_rest": ("qficr_mean_max_overlap_to_core_rest",),
            "rank_rest": ("qficr_mean_qfi_rank_rest",),
            "median_rank_rest": ("qficr_median_qfi_rank_rest",),
            "near": ("qficr_near_boundary_rank_hit_rate",),
            "p_discarded": ("qficr_mean_p_discarded",),
            "median_p_discarded": ("qficr_median_p_discarded",),
            "res_discarded": ("qficr_mean_residual_gain_discarded",),
            "overlap_discarded": ("qficr_mean_max_overlap_to_core_discarded",),
            "rank_discarded": ("qficr_mean_qfi_rank_discarded",),
            "median_rank_discarded": ("qficr_median_qfi_rank_discarded",),
        }
        for bucket, keys in mapping.items():
            val = get_num(rec, *keys)
            if val is not None:
                buckets[bucket].append(val)
    notes = ""
    if not path.exists():
        notes = "debug_qficr_stats.jsonl missing"
    elif records == 0:
        notes = "debug exists but no parseable jsonl records"
    elif not buckets["rank_rest"]:
        notes = "rank diagnostic fields missing; rerun with updated ec_pruner.py if needed"
    return {
        "variant": variant,
        "benchmark": bench,
        "K": K,
        "num_records": records,
        "mean_final_k_rest": fmt(mean(buckets["k_rest"])),
        "mean_final_k_red": fmt(mean(buckets["k_red"])),
        "mean_entropy_norm": fmt(mean(buckets["entropy"])),
        "mean_rho": fmt(mean(buckets["rho"])),
        "mean_p_restored": fmt(mean(buckets["p_rest"]), 8),
        "median_p_restored": fmt(mean(buckets["median_p_rest"]), 8),
        "mean_residual_gain_restored": fmt(mean(buckets["res_rest"]), 8),
        "mean_max_overlap_to_core_restored": fmt(mean(buckets["overlap_rest"]), 8),
        "mean_first_stage_rank_restored": fmt(mean(buckets["rank_rest"])),
        "median_first_stage_rank_restored": fmt(mean(buckets["median_rank_rest"])),
        "near_boundary_hit_rate": fmt(mean(buckets["near"])),
        "mean_p_discarded": fmt(mean(buckets["p_discarded"]), 8),
        "median_p_discarded": fmt(mean(buckets["median_p_discarded"]), 8),
        "mean_residual_gain_discarded": fmt(mean(buckets["res_discarded"]), 8),
        "mean_max_overlap_to_core_discarded": fmt(mean(buckets["overlap_discarded"]), 8),
        "mean_first_stage_rank_discarded": fmt(mean(buckets["rank_discarded"])),
        "median_first_stage_rank_discarded": fmt(mean(buckets["median_rank_discarded"])),
        "notes": notes,
    }


def diagnostic_rows():
    return [diag_for(variant, bench) for variant in VARIANTS for bench in BENCHES]


def md_table(fields, rows):
    out = []
    out.append("| " + " | ".join(fields) + " |")
    out.append("|" + "|".join(["---"] * len(fields)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def build_report(summary, averages, compare, diag):
    avg_by_variant = {r["variant"]: fnum(r["avg_score"]) for r in averages}
    best = max((r for r in averages if fnum(r["avg_score"]) is not None), key=lambda r: fnum(r["avg_score"]), default=None)
    current_avg = avg_by_variant.get("current_anchor")
    def delta(variant):
        value = avg_by_variant.get(variant)
        if value is None or current_avg is None:
            return None
        return value - current_avg

    lines = [
        "# QFi-CR Residual Compensation Score Quick Check",
        "",
        "## 1. Setup",
        "",
        "- Model: LLaVA-1.5-7B",
        "- Benchmarks: TextVQA and MME only",
        "- K: 64 only",
        "- Budget: Old QFi-CR stable budget, `EC_QFICR_RECOVER_BUDGET_MODE=abs_cap`, `EC_QFICR_RECOVER_ABS_CAP=16`",
        "- Tested score modes: `current_anchor`, `weak_anchor`, `no_anchor`, `marginal_gain_weak_gate`, `near_boundary_fill`",
        "- Excluded: VQAv2, VizWiz, MM-Vet, POPE, MMBench, GQA, SQA-IMG, K32, K128, CDPruner reruns",
        "",
        "## 2. Completeness and Scores",
        "",
        md_table(SUMMARY_FIELDS[:6] + ["notes"], summary),
        "",
        "## 3. Average by Variant",
        "",
        md_table(AVG_FIELDS, averages),
        "",
        "## 4. Comparison with current_anchor",
        "",
        md_table(COMPARE_FIELDS, compare),
        "",
        "## 5. Restored Token Diagnostics",
        "",
        md_table(DIAG_FIELDS, diag),
        "",
        "## 6. Quick Judgement",
        "",
    ]
    if best:
        lines.append(f"- Best two-benchmark average: `{best['variant']}` with avg={best['avg_score']}.")
    for variant in ["weak_anchor", "no_anchor", "marginal_gain_weak_gate", "near_boundary_fill"]:
        d = delta(variant)
        if d is None:
            lines.append(f"- `{variant}` cannot be judged yet because either it or `current_anchor` is incomplete.")
        elif abs(d) <= 0.20:
            lines.append(f"- `{variant}` is close to `current_anchor` on the quick average: delta={d:.4f}.")
        else:
            direction = "above" if d > 0 else "below"
            lines.append(f"- `{variant}` is {direction} `current_anchor` on the quick average: delta={d:.4f}.")
    lines.extend([
        "",
        "## 7. Interpretation Rules",
        "",
        "- If `weak_anchor` is close to `current_anchor`, the method is not highly sensitive to the exact p_j exponent.",
        "- If `no_anchor` drops clearly, p_j remains important as a gate even when residual gain is present.",
        "- If `marginal_gain_weak_gate` is close to or above `current_anchor`, the recovery stage can be described as marginal residual compensation rather than mainly p_j anchoring.",
        "- If `near_boundary_fill` is close to `current_anchor`, current recovery may largely behave as near-boundary Top-K completion.",
        "",
        "## 8. Recommendation",
        "",
    ])
    if current_avg is None:
        lines.append("- Recommendation pending: `current_anchor` is incomplete.")
    elif best and best["variant"] == "current_anchor":
        lines.append("- Do not replace the current recovery score based on this quick check.")
    elif best and best["variant"] == "marginal_gain_weak_gate":
        lines.append("- Consider expanding `marginal_gain_weak_gate` to GQA/SQA-IMG and K32/K128 before changing the main method.")
    elif best and best["variant"] == "near_boundary_fill":
        lines.append("- Investigate whether recovery is mainly boundary fill before strengthening the residual-compensation story.")
    else:
        lines.append("- Treat the best non-current variant as a candidate only after expanding to GQA/SQA-IMG; do not change the paper method from this two-benchmark quick check alone.")
    lines.append("- Continue to GQA/SQA-IMG only if all TextVQA/MME runs complete and a non-current variant is competitive.")
    lines.append("- Continue to K32/K128 only after the four-benchmark K64 check is non-regressive.")
    return "\n".join(lines) + "\n"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = collect_summary()
    averages = average_rows(summary)
    compare = compare_rows(summary)
    diag = diagnostic_rows()
    write_csv(OUT_ROOT / "summary_residual_compensation_quick_k64.csv", SUMMARY_FIELDS, summary)
    write_csv(OUT_ROOT / "average_by_variant_quick_k64.csv", AVG_FIELDS, averages)
    write_csv(OUT_ROOT / "compare_vs_current_anchor_quick_k64.csv", COMPARE_FIELDS, compare)
    write_csv(OUT_ROOT / "restored_token_diagnostic_quick.csv", DIAG_FIELDS, diag)
    (OUT_ROOT / "report_residual_compensation_quick_k64.md").write_text(
        build_report(summary, averages, compare, diag),
        encoding="utf-8",
    )
    print(f"[COLLECT] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
