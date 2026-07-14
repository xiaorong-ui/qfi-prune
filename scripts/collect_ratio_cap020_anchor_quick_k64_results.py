#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "outputs/qficr_ratio_cap020_anchor_quick_k64"
LOG_ROOT = ROOT / "logs/qficr_ratio_cap020_anchor_quick_k64"

VARIANTS = [
    "ratio_cap020_current_anchor",
    "ratio_cap020_weak_anchor",
    "ratio_cap020_no_anchor",
]
VARIANT_LABEL = {
    "ratio_cap020_current_anchor": "current anchor",
    "ratio_cap020_weak_anchor": "weak anchor",
    "ratio_cap020_no_anchor": "no anchor",
}
BENCHES = ["textvqa", "mme", "gqa", "sqa_img"]
K = 64
EXPECTED_LINES = {"textvqa": 5000, "mme": 2374, "gqa": 12578, "sqa_img": 2017}
REFERENCE_SCORES = {
    ("ratio_cap020_current_anchor", "gqa"): {
        "metric_name": "accuracy",
        "main_score": "58.7600",
        "notes": "current_anchor GQA K64 reused from existing ratio-cap 0.20 results; not rerun in this stage",
    },
    ("ratio_cap020_current_anchor", "sqa_img"): {
        "metric_name": "IMG-Accuracy",
        "main_score": "68.6200",
        "notes": "current_anchor SQA-IMG K64 reused from existing ratio-cap 0.20 results; not rerun in this stage",
    },
}

SUMMARY_FIELDS = [
    "variant", "benchmark", "K", "metric_name", "main_score", "status",
    "answers_path", "score_path", "log_path", "notes",
]
AVG_FIELDS = ["variant", "num_benchmarks", "avg_score", "rank", "notes"]
COMPARE_FIELDS = [
    "benchmark", "K", "variant", "score", "current_anchor_score",
    "delta_vs_current_anchor", "status", "notes",
]
DEBUG_FIELDS = [
    "variant", "benchmark", "K", "mean_final_k_rest", "mean_final_k_red",
    "mean_entropy_norm", "mean_rho", "cap_hit_rate", "yes_no_gate_rate", "notes",
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


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


def first(pattern, text):
    m = re.search(pattern, text, flags=re.I | re.M)
    return m.group(1) if m else ""


def last(pattern, text):
    matches = re.findall(pattern, text, flags=re.I | re.M)
    return matches[-1] if matches else ""


def parse_mme(text):
    p = fnum(first(r"=+\s*Perception\s*=+[\s\S]*?total score:\s*([0-9.]+)", text))
    c = fnum(first(r"=+\s*Cognition\s*=+[\s\S]*?total score:\s*([0-9.]+)", text))
    return "" if p is None or c is None else fmt((p + c) / 20.0)


def parse_score(bench, score_path):
    text = read_text(score_path)
    if bench == "textvqa":
        return "accuracy", fmt(last(r"^Accuracy:\s*([0-9.]+)%?", text))
    if bench == "mme":
        return "total / 20", parse_mme(text)
    if bench == "gqa":
        return "accuracy", fmt(last(r"^Accuracy:\s*([0-9.]+)%?", text))
    if bench == "sqa_img":
        return "IMG-Accuracy", fmt(last(r"IMG-Accuracy:\s*([0-9.]+)%?", text))
    return "score", ""


def count_lines(path: Path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def config_notes(path: Path):
    if not path.exists():
        return "generation_config.json missing"
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "generation_config.json unreadable"
    env = cfg.get("env", {})
    notes = []
    if str(env.get("EC_QFICR_RECOVER_BUDGET_MODE", "")) != "ratio_cap":
        notes.append("budget_mode not ratio_cap")
    if str(env.get("EC_QFICR_RECOVER_RATIO_CAP", "")) not in {"0.20", "0.2"}:
        notes.append("ratio_cap not 0.20")
    mode = env.get("EC_QFICR_COMP_SCORE_MODE", "")
    expected = {
        "ratio_cap020_current_anchor": "current_anchor",
        "ratio_cap020_weak_anchor": "weak_anchor",
        "ratio_cap020_no_anchor": "no_anchor",
    }.get(cfg.get("variant", ""))
    if expected and mode != expected:
        notes.append(f"comp_score_mode {mode} != {expected}")
    return "; ".join(notes)


def collect_summary():
    rows = []
    for variant in VARIANTS:
        for bench in BENCHES:
            ref = REFERENCE_SCORES.get((variant, bench))
            out_dir = OUT_ROOT / variant / f"{bench}_k{K}"
            log_path = LOG_ROOT / f"{variant}_{bench}_k{K}.log"
            answers_path = out_dir / "answers.jsonl"
            score_path = out_dir / "eval_summary.txt"
            cfg_path = out_dir / "generation_config.json"
            metric, score = parse_score(bench, score_path)
            expected = EXPECTED_LINES[bench]
            actual = count_lines(answers_path)
            complete_answers = actual == expected
            notes = config_notes(cfg_path) if cfg_path.exists() else ""
            if ref and not (score and complete_answers and not notes):
                metric = ref["metric_name"]
                score = ref["main_score"]
                status = "completed"
                notes = ref["notes"]
            elif score and complete_answers and not notes:
                status = "completed"
            elif answers_path.exists() or score_path.exists() or cfg_path.exists():
                status = "failed_or_incomplete"
                if not complete_answers:
                    extra = f"answers {actual}/{expected}"
                    notes = f"{notes}; {extra}" if notes else extra
                if score and not complete_answers:
                    score = ""
            else:
                status = "missing"
            rows.append({
                "variant": variant,
                "benchmark": bench,
                "K": K,
                "metric_name": metric,
                "main_score": score if status == "completed" else "",
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


def average_rows(summary):
    rows = []
    for variant in VARIANTS:
        vals = [r["main_score"] for r in summary if r["variant"] == variant and r["status"] == "completed"]
        avg = mean(vals)
        rows.append({
            "variant": variant,
            "num_benchmarks": len([v for v in vals if fnum(v) is not None]),
            "avg_score": fmt(avg),
            "notes": "averages completed benchmarks only; MME uses total/20",
        })
    scored = sorted([r for r in rows if fnum(r["avg_score"]) is not None], key=lambda r: fnum(r["avg_score"]), reverse=True)
    for i, row in enumerate(scored, 1):
        row["rank"] = i
    return rows


def compare_rows(summary):
    cur = {
        r["benchmark"]: fnum(r["main_score"])
        for r in summary
        if r["variant"] == "ratio_cap020_current_anchor" and r["status"] == "completed"
    }
    rows = []
    for r in summary:
        if r["variant"] == "ratio_cap020_current_anchor":
            continue
        score = fnum(r["main_score"]) if r["status"] == "completed" else None
        ref = cur.get(r["benchmark"])
        delta = None if score is None or ref is None else score - ref
        rows.append({
            "benchmark": r["benchmark"],
            "K": K,
            "variant": r["variant"],
            "score": fmt(score),
            "current_anchor_score": fmt(ref),
            "delta_vs_current_anchor": fmt(delta),
            "status": r["status"],
            "notes": r["notes"],
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


def get_num(rec, *keys):
    for key in keys:
        value = fnum(rec.get(key))
        if value is not None:
            return value
    return None


def debug_for(variant, bench):
    if (variant, bench) in REFERENCE_SCORES:
        return {
            "variant": variant,
            "benchmark": bench,
            "K": K,
            "mean_final_k_rest": "",
            "mean_final_k_red": "",
            "mean_entropy_norm": "",
            "mean_rho": "",
            "cap_hit_rate": "",
            "yes_no_gate_rate": "",
            "notes": "reference score reused; debug stats not available in this output directory",
        }
    path = OUT_ROOT / variant / f"{bench}_k{K}" / "debug_qficr_stats.jsonl"
    vals = {"k_rest": [], "k_red": [], "entropy": [], "rho": [], "cap": [], "yn": []}
    records = 0
    for rec in iter_jsonl(path) or []:
        records += 1
        mapping = {
            "k_rest": ("qficr_final_k_rest",),
            "k_red": ("qficr_final_k_red",),
            "entropy": ("qficr_entropy_norm", "adapt_entropy_norm"),
            "rho": ("qficr_rho", "adapt_recover_ratio"),
            "cap": ("qficr_cap_hit", "qficr_ratio_cap_hit"),
            "yn": ("qficr_yes_no_gate", "qficr_yn_gate"),
        }
        for name, keys in mapping.items():
            value = get_num(rec, *keys)
            if value is not None:
                vals[name].append(value)
    notes = ""
    if not path.exists():
        notes = "debug_qficr_stats.jsonl missing"
    elif records == 0:
        notes = "debug exists but no parseable jsonl records"
    elif not vals["k_rest"]:
        notes = "budget debug fields missing"
    return {
        "variant": variant,
        "benchmark": bench,
        "K": K,
        "mean_final_k_rest": fmt(mean(vals["k_rest"])),
        "mean_final_k_red": fmt(mean(vals["k_red"])),
        "mean_entropy_norm": fmt(mean(vals["entropy"])),
        "mean_rho": fmt(mean(vals["rho"])),
        "cap_hit_rate": fmt(mean(vals["cap"])),
        "yes_no_gate_rate": fmt(mean(vals["yn"])),
        "notes": notes,
    }


def debug_rows():
    return [debug_for(v, b) for v in VARIANTS for b in BENCHES]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def table_tex(summary):
    by = {(r["variant"], r["benchmark"]): r for r in summary if r["status"] == "completed"}
    bench_order = ["textvqa", "mme", "gqa", "sqa_img"]
    labels = {"textvqa": "TextVQA", "mme": "MME", "gqa": "GQA", "sqa_img": "SQA$^{\\mathrm{IMG}}$"}
    col_best = {}
    for bench in bench_order:
        scores = [(v, fnum(by.get((v, bench), {}).get("main_score"))) for v in VARIANTS]
        scores = [(v, s) for v, s in scores if s is not None]
        if scores:
            col_best[bench] = max(s for _, s in scores)
    avg_best = None
    avg_by_variant = {}
    for v in VARIANTS:
        vals = [fnum(by.get((v, b), {}).get("main_score")) for b in bench_order]
        vals = [x for x in vals if x is not None]
        avg = mean(vals)
        avg_by_variant[v] = avg
    if any(v is not None for v in avg_by_variant.values()):
        avg_best = max(v for v in avg_by_variant.values() if v is not None)
    def cell(value, best=None):
        if value is None:
            return "--"
        text = f"{value:.4f}"
        return f"\\textbf{{{text}}}" if best is not None and abs(value - best) < 1e-9 else text
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Recovery-score variants under ratio-cap 0.20 at K=64.}",
        "\\label{tab:ratio_cap020_anchor_quick_k64}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "\\textbf{Variant} & \\textbf{TextVQA} & \\textbf{MME} & \\textbf{GQA} & \\textbf{SQA$^{\\mathrm{IMG}}$} & \\textbf{Avg.} \\\\",
        "\\hline",
    ]
    for v in VARIANTS:
        vals = [fnum(by.get((v, b), {}).get("main_score")) for b in bench_order]
        cells = [cell(vals[i], col_best.get(bench_order[i])) for i in range(len(bench_order))]
        avg = avg_by_variant[v]
        cells.append(cell(avg, avg_best))
        lines.append(f"{VARIANT_LABEL[v]} & " + " & ".join(cells) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def md_table(fields, rows):
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
    return "\n".join(lines)


def build_report(summary, averages, compare, debug):
    complete = [r for r in summary if r["status"] == "completed"]
    cfg_ok = all("ratio_cap not" not in r["notes"] and "budget_mode not" not in r["notes"] for r in complete)
    avg = {r["variant"]: fnum(r["avg_score"]) for r in averages}
    cur = avg.get("ratio_cap020_current_anchor")
    weak = avg.get("ratio_cap020_weak_anchor")
    no = avg.get("ratio_cap020_no_anchor")
    best = max((r for r in averages if fnum(r["avg_score"]) is not None), key=lambda r: fnum(r["avg_score"]), default=None)
    benches_done = sorted({r["benchmark"] for r in complete})
    lines = [
        "# QFi-CR ratio-cap 0.20 Anchor Quick Check",
        "",
        "## 1. Setup",
        "",
        "- Budget: `EC_QFICR_RECOVER_BUDGET_MODE=ratio_cap`, `EC_QFICR_RECOVER_RATIO_CAP=0.20`.",
        "- K: 64; model: LLaVA-1.5-7B.",
        "- Variants: current anchor, weak anchor, no anchor.",
        "- Benchmarks completed: " + (", ".join(benches_done) if benches_done else "none") + ".",
        "- `current_anchor` GQA/SQA-IMG K64 reused from existing ratio-cap 0.20 results; not rerun in this stage.",
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
        "## 5. Debug Budget Summary",
        "",
        md_table(DEBUG_FIELDS, debug),
        "",
        "## 6. Answers",
        "",
        f"- All completed runs use ratio-cap 0.20: {'yes' if cfg_ok else 'no; see notes'}.",
    ]
    def better(x):
        return x is not None and cur is not None and x >= cur
    lines.append(f"- `weak_anchor` is {'not lower than' if better(weak) else 'lower than or unavailable vs'} `current_anchor` on the completed-average.")
    lines.append(f"- `no_anchor` is {'not lower than' if better(no) else 'lower than or unavailable vs'} `current_anchor` on the completed-average.")
    tv = {r["variant"]: fnum(r["main_score"]) for r in summary if r["benchmark"] == "textvqa" and r["status"] == "completed"}
    mme = {r["variant"]: fnum(r["main_score"]) for r in summary if r["benchmark"] == "mme" and r["status"] == "completed"}
    consistent = (
        tv.get("ratio_cap020_weak_anchor") is not None and mme.get("ratio_cap020_weak_anchor") is not None
        and tv.get("ratio_cap020_no_anchor") is not None and mme.get("ratio_cap020_no_anchor") is not None
        and tv["ratio_cap020_weak_anchor"] >= tv.get("ratio_cap020_current_anchor", 1e9)
        and mme["ratio_cap020_weak_anchor"] >= mme.get("ratio_cap020_current_anchor", 1e9)
        and tv["ratio_cap020_no_anchor"] >= tv.get("ratio_cap020_current_anchor", 1e9)
        and mme["ratio_cap020_no_anchor"] >= mme.get("ratio_cap020_current_anchor", 1e9)
    )
    lines.append(f"- TextVQA and MME show a consistent weak/no-anchor non-regression trend: {'yes' if consistent else 'no or incomplete'}.")
    bench_names = {"textvqa": "TextVQA", "mme": "MME", "gqa": "GQA", "sqa_img": "SQA-IMG"}
    for bench in BENCHES:
        scored = [
            (r["variant"], fnum(r["main_score"]))
            for r in summary
            if r["benchmark"] == bench and r["status"] == "completed" and fnum(r["main_score"]) is not None
        ]
        if scored:
            top = max(score for _, score in scored)
            winners = [VARIANT_LABEL[v] for v, score in scored if abs(score - top) < 1e-9]
            lines.append(f"- Best on {bench_names[bench]}: {', '.join(winners)} ({top:.4f}).")
    if {"gqa", "sqa_img"}.issubset(set(benches_done)) and best:
        lines.append(f"- Four-benchmark best variant: `{best['variant']}` with avg={best['avg_score']}.")
    elif best:
        lines.append(f"- Completed-benchmark best variant: `{best['variant']}` with avg={best['avg_score']}.")
    lines.append("- `no_anchor` can be considered a ratio-cap 0.20 candidate only if it remains non-regressive after GQA/SQA-IMG and then K32/K128.")
    lines.append("- `weak_anchor` is the safer candidate if it is close to `no_anchor` while retaining a weak p_j gate.")
    if cur is not None and (weak is not None and weak < cur) and (no is not None and no < cur):
        lines.append("- If both weak/no are below current, ratio-cap 0.20 likely still needs p_j anchoring for denoising.")
    else:
        lines.append("- Current quick evidence does not show a need for strong p_j anchoring under ratio-cap 0.20.")
    lines.append("- Recommendation on K32/K128: run only after the K64 four-benchmark result is non-regressive.")
    lines.append("- Recommendation on direct greedy / SCOPE-style baselines: useful after choosing a K64 recovery-score candidate.")
    lines.append("- Recommendation on paper method: do not remove or weaken p_j anchor in the main formula until four benchmarks plus K32/K128 confirm the trend.")
    lines.append("- Scope caveat: even with four K64 benchmarks complete, this is still a K64-only conclusion and should not be generalized to K32/K128 without running them.")
    return "\n".join(lines) + "\n"


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = collect_summary()
    averages = average_rows(summary)
    compare = compare_rows(summary)
    debug = debug_rows()
    write_csv(OUT_ROOT / "summary_ratio_cap020_anchor_quick_k64.csv", SUMMARY_FIELDS, summary)
    write_csv(OUT_ROOT / "average_by_variant_ratio_cap020_anchor_quick_k64.csv", AVG_FIELDS, averages)
    write_csv(OUT_ROOT / "compare_vs_current_anchor_ratio_cap020_k64.csv", COMPARE_FIELDS, compare)
    write_csv(OUT_ROOT / "debug_budget_summary_ratio_cap020_anchor_quick_k64.csv", DEBUG_FIELDS, debug)
    (OUT_ROOT / "table_ratio_cap020_anchor_quick_k64.tex").write_text(table_tex(summary), encoding="utf-8")
    (OUT_ROOT / "report_ratio_cap020_anchor_quick_k64.md").write_text(
        build_report(summary, averages, compare, debug),
        encoding="utf-8",
    )
    print(f"[COLLECT] wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
