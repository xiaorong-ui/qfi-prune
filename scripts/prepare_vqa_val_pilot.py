#!/usr/bin/env python3
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("vizwiz", "vqav2"), required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--num-samples", type=int, default=32)
    args = parser.parse_args()

    with open(args.question_file) as question_file:
        source = json.load(question_file)

    if args.benchmark == "vizwiz":
        questions = source[: args.num_samples]
        records = [
            {
                "question_id": index,
                "image": item["image"],
                "text": item["question"]
                + "\nWhen the provided information is insufficient, respond with 'Unanswerable'."
                + "\nAnswer the question using a single word or phrase.",
                "category": "default",
            }
            for index, item in enumerate(questions)
        ]
    else:
        questions = source["questions"][: args.num_samples]
        records = [
            {
                "question_id": item["question_id"],
                "image": f"COCO_val2014_{item['image_id']:012d}.jpg",
                "text": item["question"] + "\nAnswer the question using a single word or phrase.",
                "category": "default",
            }
            for item in questions
        ]

    with open(args.output_file, "w") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")

    print(f"Prepared {len(records)} {args.benchmark} validation questions: {args.output_file}")


if __name__ == "__main__":
    main()
