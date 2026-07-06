import argparse
import json


def preview_line(text, limit=160):
    preview = text.replace("\x00", "\\x00").replace("\n", "\\n")
    if len(preview) > limit:
        preview = preview[:limit] + "..."
    return preview


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid lines instead of raising an error.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    all_answers = []
    invalid_count = 0

    with open(args.src, "r", encoding="utf-8", errors="replace") as src_file:
        for line_idx, raw_line in enumerate(src_file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            if "\x00" in raw_line:
                message = (
                    f"Invalid prediction line with NUL byte in {args.src}:{line_idx}. "
                    f"Preview: {preview_line(raw_line)}"
                )
                if args.skip_invalid:
                    print(f"[convert_gqa_for_eval] skip: {message}")
                    invalid_count += 1
                    continue
                raise ValueError(message)

            try:
                res = json.loads(line)
            except json.JSONDecodeError as exc:
                message = (
                    f"Invalid JSON in {args.src}:{line_idx}: {exc}. "
                    f"Preview: {preview_line(raw_line)}"
                )
                if args.skip_invalid:
                    print(f"[convert_gqa_for_eval] skip: {message}")
                    invalid_count += 1
                    continue
                raise ValueError(message) from exc

            if "question_id" not in res or "text" not in res:
                message = (
                    f"Missing required keys in {args.src}:{line_idx}. "
                    f"Keys: {sorted(res.keys())}. Preview: {preview_line(raw_line)}"
                )
                if args.skip_invalid:
                    print(f"[convert_gqa_for_eval] skip: {message}")
                    invalid_count += 1
                    continue
                raise ValueError(message)

            question_id = res["question_id"]
            text = str(res["text"]).rstrip(".").lower()
            all_answers.append({"questionId": question_id, "prediction": text})

    with open(args.dst, "w") as dst_file:
        json.dump(all_answers, dst_file)

    if invalid_count > 0:
        print(f"[convert_gqa_for_eval] skipped invalid lines: {invalid_count}")


if __name__ == "__main__":
    main()
