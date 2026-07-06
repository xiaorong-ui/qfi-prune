#!/usr/bin/env python3
import argparse
import json

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("vizwiz", "vqav2"), required=True)
    parser.add_argument("--annotation-file", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    with open(args.annotation_file) as annotation_file:
        source = json.load(annotation_file)

    if args.benchmark == "vizwiz":
        annotations = {
            str(index): [answer["answer"] for answer in item["answers"]]
            for index, item in enumerate(source)
        }
    else:
        annotations = {
            str(item["question_id"]): [answer["answer"] for answer in item["answers"]]
            for item in source["annotations"]
        }

    predictions = []
    with open(args.result_file) as result_file:
        for line in result_file:
            result = json.loads(line)
            key = str(result["question_id"])
            if key in annotations:
                predictions.append({"pred_answer": result["text"], "gt_answers": annotations[key]})

    evaluator = TextVQAAccuracyEvaluator()
    accuracy = 100.0 * evaluator.eval_pred_list(predictions) if predictions else 0.0
    print(f"Samples: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()
