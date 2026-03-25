#!/usr/bin/env python3
import os
import glob
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_records(input_dir):
    records = []
    pattern = os.path.join(input_dir, "results_rank*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def group_correctness(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["qid"]].append(bool(r["is_correct"]))
    return grouped


def compute_summary(grouped):
    summary = {}
    for qid, results in grouped.items():
        summary[qid] = {
            "total": len(results),
            "correct": sum(results),
        }
    return summary


def compute_stable(grouped):
    return [qid for qid, results in grouped.items() if all(results)]


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    records = load_records(args.input_dir)
    grouped = group_correctness(records)
    summary = compute_summary(grouped)
    stable = compute_stable(grouped)

    save_json(os.path.join(args.output_dir, "results.json"), records)
    save_json(os.path.join(args.output_dir, "qid_summary.json"), summary)
    save_json(os.path.join(args.output_dir, "stable_qids.json"), stable)

    print(f"results_rows={len(records)}")
    print(f"num_qids={len(summary)}")
    print(f"stable_qids={len(stable)}")


if __name__ == "__main__":
    main()