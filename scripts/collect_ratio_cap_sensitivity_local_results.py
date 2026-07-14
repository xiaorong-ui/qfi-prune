#!/usr/bin/env python3
import csv
import json
import math
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/qficr_ratio_cap_sensitivity_local"
LOG_ROOT = ROOT / "logs/qficr_ratio_cap_sensitivity_local"
RATIOS_NEW = ["0.15", "0.18", "0.22", "0.25"]
RATIOS_ALL = ["0.15", "0.18", "0.20", "0.22", "0.25"]
KS = [32, 64, 128]
BENCHES = ["gqa", "textvqa", "mme", "sqa_img"]
BENCH_DISPLAY = {
    "gqa": "GQA",
    "textvqa": "TextVQA",
    "mme": "MME",
    "sqa_img": "SQA-IMG",
}
METRIC_NAME = {
    "gqa": "accuracy",
    "textvqa": "accuracy",
    "mme": "total / 20",
    "sqa_img": "IMG-Accuracy",
}
RATIO020 = {
    ("gqa", 32): 57.12,
    ("sqa_img", 32): 68.57,
    ("textvqa", 32): 54.44,
    ("mme", 32): 83.14,
    ("gqa", 64): 58.76,
    ("sqa_img", 64): 68.62,
    ("textvqa", 64): 55.57,
    ("mme", 64): 86.72,
    ("gqa", 128): 59.72,
    ("sqa_img", 128): 68.96,
    ("textvqa", 128): 56.95,
    ("mme", 128): 88.09,
}
SUMMARY_FIELDS = [
    "ratio",
    "method",
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
COMPARE_FIELDS = [
    "benchmark",
    "K",
    "ratio",
    "score",
    "ratio020_score",
    "delta_vs_ratio020",
    "status",
    "notes",
]
AVG_K_FIELDS = ["ratio", "K", "num_benchmarks", "avg_score", "rank_within_K", "notes"]
AVG_FIELDS = ["ratio", "num_scores", "avg_score", "rank_overall", "notes"]
DEBUG_FIELDS = [
    "ratio",
    "benchmark",
    "K",
    "mean_final_k_rest",
    "mean_final_k_red",
    "mean_entropy_norm",
    "mean_rho",
    "cap_hit_rate",
    "yes_no_gate_rate",
    "notes",
]


def ratio_tag(ratio: str) -> str:
    return f"ratio_cap{ratio.replace('.', '')}"


def fnum(value):
    if value in {"", None}:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def fmt(value, digits=4):
    value = fnum(value)
    return "" if value is None else f"{value:.{digits}f}"


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def last(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.I | re.M)
    return matches[-1] if matches else ""


def first(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.M)
    return match.group(1) if match else ""


def parse_mme(text: str) -> float | None:
    perception = fnum(first(r"=+\s*Perception\s*=+[\s\S]*?total score:\s*([0-9.]+)", text))
    cognition = fnum(first(r"=+\s*Cognition\s*=+[\s\S]*?total score:\s*([0-9.]+)", text))
    if perception is not None and cognition is not None:
        return (perception + cognition) / 20.0
    totals = [fnum(x) for x in re.findall(r"total score:\s*([0-9.]+)", text, flags=re.I)]
    totals = [x for x in totals if x is not None]
    if len(totals) >= 2:
        return sum(totals[-2:]) / 20.0
    return None


def parse_score(bench: str, out_dir: Path, log_path: Path) -> tuple[float | None, str, str]:
    log_text = read_text(log_path)
    score_text = read_text(out_dir / "eval_summary.txt") or log_text
    failed = bool(re.search(r"Traceback|CUDA out of memory|outofmemory|Killed|Exception|\[ERROR\]", log_text, re.I))
    score = None
    if bench == "gqa":
        score = fnum(last(r"^Accuracy:\s*([0-9.]+)%?", score_text))
    elif bench == "textvqa":
        score = fnum(last(r"^Accuracy:\s*([0-9.]+)%?", score_text))
    elif bench == "mme":
        score = parse_mme(score_text)
    elif bench == "sqa_img":
        score = fnum(last(r"IMG-Accuracy:\s*([0-9.]+)%", score_text))
    status = "completed" if score is not None and not failed else "failed_or_incomplete" if log_text or out_dir.exists() else "missing"
    notes = ""
    if bench == "sqa_img":
        notes = "main score is IMG-Accuracy from image subset; result.json overall acc is not used"
    elif bench == "mme":
        notes = "main score is MME total / 20"
    if failed and score is not None:
        notes = (notes + "; " if notes else "") + "failure marker found in log"
    return score, status, notes


def collect_row(ratio: str, bench: str, k: int) -> dict:
    if ratio == "0.20":
        score = RATIO020[(bench, k)]
        return {
            "ratio": ratio,
            "method": "QFi-CR + ratio-cap 0.20 (existing)",
            "benchmark": BENCH_DISPLAY[bench],
            "K": str(k),
            "metric_name": METRIC_NAME[bench],
            "main_score": fmt(score),
            "status": "existing_reference",
            "answers_path": "",
            "score_path": "embedded_existing_ratio_cap_0.20_reference",
            "log_path": "",
            "notes": "not rerun; existing local result used as ratio-cap 0.20 reference",
        }
    tag = ratio_tag(ratio)
    out_dir = OUT_ROOT / tag / f"{bench}_k{k}"
    log_path = LOG_ROOT / f"{tag}_{bench}_k{k}.log"
    answers = out_dir / "answers.jsonl"
    score_path = out_dir / "eval_summary.txt"
    score, status, notes = parse_score(bench, out_dir, log_path)
    if status == "completed" and not answers.exists():
        status = "failed_or_incomplete"
        notes = (notes + "; " if notes else "") + "missing answers.jsonl"
    return {
        "ratio": ratio,
        "method": f"QFi-CR + ratio-cap {ratio}",
        "benchmark": BENCH_DISPLAY[bench],
        "K": str(k),
        "metric_name": METRIC_NAME[bench],
        "main_score": fmt(score),
        "status": status,
        "answers_path": rel(answers),
        "score_path": rel(score_path if score_path.exists() else out_dir / "result.json"),
        "log_path": rel(log_path),
        "notes": notes,
    }


def write_csv(path: Path, fields: list[str], rows: list[dict], delimiter=",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summary_rows() -> list[dict]:
    rows = []
    for ratio in RATIOS_ALL:
        for k in KS:
            for bench in BENCHES:
                rows.append(collect_row(ratio, bench, k))
    return rows


def inventory_rows(summary: list[dict]) -> list[dict]:
    rows = []
    for row in summary:
        rows.append({
            "ratio": row["ratio"],
            "benchmark": row["benchmark"],
            "K": row["K"],
            "found_existing": "yes" if row["status"] in {"completed", "existing_reference"} and row["main_score"] else "no",
            "status": row["status"],
            "score": row["main_score"],
            "answers_path": row["answers_path"],
            "score_path": row["score_path"],
            "log_path": row["log_path"],
            "notes": row["notes"],
        })
    return rows


def compare_rows(summary: list[dict]) -> list[dict]:
    by_key = {(row["ratio"], row["benchmark"], int(row["K"])): row for row in summary}
    rows = []
    for ratio in RATIOS_NEW:
        for k in KS:
            for bench in BENCHES:
                display = BENCH_DISPLAY[bench]
                row = by_key[(ratio, display, k)]
                score = fnum(row["main_score"])
                base = RATIO020[(bench, k)]
                rows.append({
                    "benchmark": display,
                    "K": str(k),
                    "ratio": ratio,
                    "score": fmt(score),
                    "ratio020_score": fmt(base),
                    "delta_vs_ratio020": "" if score is None else f"{score - base:+.4f}",
                    "status": row["status"],
                    "notes": row["notes"],
                })
    return rows


def rank_desc(rows: list[dict], score_key: str, rank_key: str) -> None:
    scored = sorted(
        [(idx, fnum(row[score_key])) for idx, row in enumerate(rows) if fnum(row[score_key]) is not None],
        key=lambda item: item[1],
        reverse=True,
    )
    last_score = None
    last_rank = 0
    for pos, (idx, score) in enumerate(scored, start=1):
        rank = last_rank if last_score is not None and abs(score - last_score) <= 1e-9 else pos
        rows[idx][rank_key] = str(rank)
        last_score = score
        last_rank = rank


def average_by_ratio_and_k(summary: list[dict]) -> list[dict]:
    rows = []
    by_key = {}
    for row in summary:
        by_key.setdefault((row["ratio"], int(row["K"])), []).append(row)
    for ratio in RATIOS_ALL:
        for k in KS:
            vals = [fnum(row["main_score"]) for row in by_key.get((ratio, k), [])]
            vals = [v for v in vals if v is not None]
            rows.append({
                "ratio": ratio,
                "K": str(k),
                "num_benchmarks": str(len(vals)),
                "avg_score": fmt(sum(vals) / len(vals) if vals else None),
                "rank_within_K": "",
                "notes": "averages available benchmarks only; MME uses total / 20",
            })
    for k in KS:
        block = [row for row in rows if row["K"] == str(k)]
        rank_desc(block, "avg_score", "rank_within_K")
        by_ratio = {row["ratio"]: row["rank_within_K"] for row in block}
        for row in rows:
            if row["K"] == str(k):
                row["rank_within_K"] = by_ratio[row["ratio"]]
    return rows


def average_by_ratio(summary: list[dict]) -> list[dict]:
    rows = []
    for ratio in RATIOS_ALL:
        vals = [fnum(row["main_score"]) for row in summary if row["ratio"] == ratio]
        vals = [v for v in vals if v is not None]
        rows.append({
            "ratio": ratio,
            "num_scores": str(len(vals)),
            "avg_score": fmt(sum(vals) / len(vals) if vals else None),
            "rank_overall": "",
            "notes": "averages available K x benchmark scores only; MME uses total / 20",
        })
    rank_desc(rows, "avg_score", "rank_overall")
    return rows


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def debug_budget_summary(summary: list[dict]) -> list[dict]:
    paths = []
    meta_by_path = {}
    for row in summary:
        if row["ratio"] == "0.20":
            continue
        answers = row["answers_path"]
        if not answers:
            continue
        debug = ROOT / answers
        debug = debug.parent / "debug_qficr_stats.jsonl"
        if debug.exists() and debug.stat().st_size > 0:
            paths.append(str(debug))
            meta_by_path[debug.resolve()] = (row["ratio"], row["benchmark"], row["K"])
    raw_path = OUT_ROOT / ".debug_budget_summary_raw.csv"
    if paths:
        subprocess.run(
            ["python", str(ROOT / "scripts/summarize_qficr_debug_stats.py"), *paths, "--output", str(raw_path)],
            check=False,
        )
    raw_rows = load_csv(raw_path)
    out = []
    used = set()
    for raw in raw_rows:
        run_name = raw.get("run_name", "")
        ratio = benchmark = k = ""
        match = re.search(r"(ratio_cap\d+)/([a-z_]+)_k(\d+)", run_name)
        if match:
            tag, bench, k = match.groups()
            ratio = f"0.{tag[-2:]}"
            benchmark = BENCH_DISPLAY.get(bench, bench)
        out.append({
            "ratio": ratio,
            "benchmark": benchmark or raw.get("benchmark", ""),
            "K": k or raw.get("K", ""),
            "mean_final_k_rest": fmt(raw.get("mean_final_k_rest")),
            "mean_final_k_red": fmt(raw.get("mean_final_k_red")),
            "mean_entropy_norm": fmt(raw.get("mean_entropy_norm")),
            "mean_rho": fmt(raw.get("mean_rho")),
            "cap_hit_rate": fmt(raw.get("cap_hit_rate")),
            "yes_no_gate_rate": fmt(raw.get("yes_no_gate_rate")),
            "notes": "" if ratio else f"could not infer metadata from run_name={run_name}",
        })
        if ratio and benchmark and k:
            used.add((ratio, benchmark, k))
    for ratio in RATIOS_NEW:
        for k in KS:
            for bench in BENCHES:
                key = (ratio, BENCH_DISPLAY[bench], str(k))
                if key not in used:
                    out.append({
                        "ratio": ratio,
                        "benchmark": BENCH_DISPLAY[bench],
                        "K": str(k),
                        "mean_final_k_rest": "",
                        "mean_final_k_red": "",
                        "mean_entropy_norm": "",
                        "mean_rho": "",
                        "cap_hit_rate": "",
                        "yes_no_gate_rate": "",
                        "notes": "missing debug_qficr_stats.jsonl",
                    })
    return sorted(out, key=lambda r: (fnum(r["ratio"]) or 0, int(r["K"] or 0), r["benchmark"]))


def md_num(value, digits=2):
    value = fnum(value)
    return "--" if value is None else f"{value:.{digits}f}"


def latex_num(value, best=None):
    value = fnum(value)
    if value is None:
        return "--"
    text = f"{value:.2f}"
    if best is not None and abs(value - best) <= 1e-9:
        return f"\\textbf{{{text}}}"
    return text


def write_latex(avg_k: list[dict], summary: list[dict]) -> None:
    avg_by = {(row["ratio"], row["K"]): fnum(row["avg_score"]) for row in avg_k}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Sensitivity analysis of fixed ratio-cap values on local benchmarks.}",
        "\\label{tab:ratio_cap_sensitivity_local_avg}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "\\textbf{Ratio Cap} & \\textbf{K=32} & \\textbf{K=64} & \\textbf{K=128} \\\\",
        "\\hline",
    ]
    best_by_k = {}
    for k in KS:
        vals = [avg_by.get((ratio, str(k))) for ratio in RATIOS_ALL]
        vals = [v for v in vals if v is not None]
        best_by_k[str(k)] = max(vals) if vals else None
    for ratio in RATIOS_ALL:
        cells = [latex_num(avg_by.get((ratio, str(k))), best_by_k[str(k)]) for k in KS]
        lines.append(f"{ratio} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    by = {(row["ratio"], row["benchmark"], row["K"]): row for row in summary}
    avg_for = {(row["ratio"], row["K"]): fnum(row["avg_score"]) for row in avg_k}
    lines += [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Detailed fixed ratio-cap sensitivity on local benchmarks.}",
        "\\label{tab:ratio_cap_sensitivity_local_detail}",
        "\\begin{tabular}{llccccc}",
        "\\hline",
        "\\textbf{Ratio} & \\textbf{K} & \\textbf{GQA} & \\textbf{TextVQA} & \\textbf{MME} & \\textbf{SQA-IMG} & \\textbf{Avg} \\\\",
        "\\hline",
    ]
    for k in KS:
        vals = [avg_for.get((ratio, str(k))) for ratio in RATIOS_ALL]
        vals = [v for v in vals if v is not None]
        best = max(vals) if vals else None
        for ratio in RATIOS_ALL:
            cells = [latex_num(by.get((ratio, BENCH_DISPLAY[b], str(k)), {}).get("main_score")) for b in BENCHES]
            cells.append(latex_num(avg_for.get((ratio, str(k))), best))
            lines.append(f"{ratio} & {k} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table*}", ""]
    (OUT_ROOT / "table_ratio_cap_sensitivity_local.tex").write_text("\n".join(lines), encoding="utf-8")


def completeness_status(row: dict) -> str:
    if row["status"] in {"completed", "existing_reference"} and row["main_score"]:
        return "OK"
    if row["status"] == "missing":
        return "MISSING"
    return "FAIL"


def best_ratio_for_k(avg_k: list[dict], k: int) -> str:
    block = [row for row in avg_k if row["K"] == str(k) and fnum(row["avg_score"]) is not None]
    if not block:
        return "not available"
    best = max(fnum(row["avg_score"]) for row in block)
    winners = [row["ratio"] for row in block if abs(fnum(row["avg_score"]) - best) <= 1e-9]
    return "/".join(winners)


def best_ratio_overall(avg: list[dict]) -> str:
    block = [row for row in avg if fnum(row["avg_score"]) is not None]
    if not block:
        return "not available"
    best = max(fnum(row["avg_score"]) for row in block)
    winners = [row["ratio"] for row in block if abs(fnum(row["avg_score"]) - best) <= 1e-9]
    return "/".join(winners)


def write_report(summary: list[dict], avg_k: list[dict], avg: list[dict], compare: list[dict]) -> None:
    by = {(row["ratio"], row["benchmark"], row["K"]): row for row in summary}
    lines = [
        "# Fixed Ratio-Cap Sensitivity on Local Benchmarks",
        "",
        "## 1. Setup",
        "",
        "- fixed global ratio-cap; no K-aware tuning",
        "- ratios: 0.15, 0.18, 0.20, 0.22, 0.25",
        "- K: 32, 64, 128",
        "- benchmarks: GQA, TextVQA, MME, SQA-IMG",
        "- model: LLaVA-1.5-7B",
        "- ratio-cap 0.20 is an existing reference and was not rerun by this runner",
        "",
        "## 2. Completeness",
        "",
        "| Ratio | K | GQA | TextVQA | MME | SQA-IMG | Status |",
        "|---|---:|---|---|---|---|---|",
    ]
    for ratio in RATIOS_ALL:
        for k in KS:
            statuses = [completeness_status(by[(ratio, BENCH_DISPLAY[b], str(k))]) for b in BENCHES]
            overall = "complete" if all(s == "OK" for s in statuses) else "incomplete"
            lines.append(f"| {ratio} | {k} | " + " | ".join(statuses) + f" | {overall} |")
    lines += [
        "",
        "## 3. Main Results",
        "",
        "| Ratio | K | GQA | TextVQA | MME | SQA-IMG | Avg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    avg_by = {(row["ratio"], row["K"]): row["avg_score"] for row in avg_k}
    for ratio in RATIOS_ALL:
        for k in KS:
            cells = [md_num(by[(ratio, BENCH_DISPLAY[b], str(k))]["main_score"]) for b in BENCHES]
            lines.append(f"| {ratio} | {k} | " + " | ".join(cells) + f" | {md_num(avg_by[(ratio, str(k))])} |")
    lines += [
        "",
        "## 4. Best Ratio by K",
        "",
        f"- K=32 best fixed ratio: {best_ratio_for_k(avg_k, 32)}",
        f"- K=64 best fixed ratio: {best_ratio_for_k(avg_k, 64)}",
        f"- K=128 best fixed ratio: {best_ratio_for_k(avg_k, 128)}",
        "- If best ratios differ across K, this indicates a fixed-ratio trade-off; it does not justify K-aware tuning.",
        "",
        "## 5. Overall Best Fixed Ratio",
        "",
        f"- Overall best fixed ratio by available K x benchmark average: {best_ratio_overall(avg)}",
        "",
        "## 6. Comparison with ratio-cap 0.20",
        "",
    ]
    completed_comp = [row for row in compare if row["status"] == "completed" and fnum(row["delta_vs_ratio020"]) is not None]
    if completed_comp:
        wins = {}
        for ratio in RATIOS_NEW:
            deltas = [fnum(row["delta_vs_ratio020"]) for row in completed_comp if row["ratio"] == ratio]
            deltas = [d for d in deltas if d is not None]
            wins[ratio] = (sum(1 for d in deltas if d > 0), len(deltas), sum(deltas) / len(deltas) if deltas else None)
        for ratio, (num_win, total, mean_delta) in wins.items():
            lines.append(f"- ratio {ratio}: {num_win}/{total} completed scores above 0.20; mean delta {md_num(mean_delta, 4)}.")
    else:
        lines.append("- No completed new-ratio comparison is available yet.")
    lines += [
        "- If no fixed ratio is stable above 0.20, keep 0.20 as the ablation setting.",
        "- If one fixed ratio is consistently stronger across K=32/64/128 and all four local benchmarks, treat it as the next candidate.",
        "",
        "## 7. Interpretation",
        "",
        "- Lower ratios can reduce restoration noise, but may lose useful detail compensation.",
        "- Higher ratios can restore more detail, but may add redundant or noisy tokens.",
        "- A fixed ratio must work across K=32, K=64, and K=128.",
        "- K-aware special casing is excluded from this experiment.",
        "",
        "## 8. Recommendation",
        "",
    ]
    best = best_ratio_overall(avg)
    if best == "0.20":
        lines.append("- Current recommendation: do not replace ratio-cap 0.20 based on the available aggregate.")
    elif best == "not available":
        lines.append("- Current recommendation: wait for completed local runs before changing the ratio-cap.")
    else:
        lines.append(f"- Current recommendation: ratio-cap {best} is the aggregate candidate, but only replace 0.20 if the completed matrix is full and stable.")
    lines += [
        "- Continue to POPE/MMBench only after choosing a candidate from these four local benchmarks.",
        "- If a new ratio wins overall, complete the local 7-benchmark set before replacing the current 0.20 ablation.",
        "- Old QFi-CR should remain a baseline until the fixed ratio-cap variant is non-regressive across the selected K settings.",
        "",
    ]
    (OUT_ROOT / "report_ratio_cap_sensitivity_local.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    summary = summary_rows()
    compare = compare_rows(summary)
    avg_k = average_by_ratio_and_k(summary)
    avg = average_by_ratio(summary)
    debug = debug_budget_summary(summary)
    write_csv(OUT_ROOT / "inventory.tsv", list(inventory_rows(summary)[0].keys()), inventory_rows(summary), delimiter="\t")
    write_csv(OUT_ROOT / "summary_ratio_cap_sensitivity_local.csv", SUMMARY_FIELDS, summary)
    write_csv(OUT_ROOT / "compare_ratio_cap_sensitivity_vs_ratio020.csv", COMPARE_FIELDS, compare)
    write_csv(OUT_ROOT / "average_by_ratio_and_k.csv", AVG_K_FIELDS, avg_k)
    write_csv(OUT_ROOT / "average_by_ratio.csv", AVG_FIELDS, avg)
    write_csv(OUT_ROOT / "debug_budget_summary.csv", DEBUG_FIELDS, debug)
    write_latex(avg_k, summary)
    write_report(summary, avg_k, avg, compare)
    print(OUT_ROOT / "summary_ratio_cap_sensitivity_local.csv")


if __name__ == "__main__":
    main()
