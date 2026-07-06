#!/usr/bin/env python3
import argparse
import json
import re

import pandas as pd


def parse_choice(text):
    text = str(text).strip().upper()
    match = re.search(r"(?:^|\b)([A-D])(?:\b|[.)])", text)
    return match.group(1) if match else text[:1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    annotations = pd.read_table(args.annotation_file)
    predictions = {}
    with open(args.result_file) as result_file:
        for line in result_file:
            result = json.loads(line)
            predictions[str(result["question_id"])] = parse_choice(result["text"])

    scored = annotations.copy()
    scored["prediction"] = scored["index"].astype(str).map(predictions)
    scored = scored[scored["prediction"].notna()].copy()
    scored["correct"] = scored["prediction"] == scored["answer"].astype(str).str.strip().str.upper()

    correct = int(scored["correct"].sum())
    accuracy = 100.0 * scored["correct"].mean() if len(scored) else 0.0
    scored["base_index"] = scored["index"].astype(int) % 1_000_000
    circular = scored.groupby("base_index")["correct"].all()
    circular_accuracy = 100.0 * circular.mean() if len(circular) else 0.0

    print(f"Samples: {len(scored)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Circular-Groups: {len(circular)}")
    print(f"Circular-Accuracy: {circular_accuracy:.2f}%")


if __name__ == "__main__":
    main()
