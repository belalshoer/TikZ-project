import os
import json
import glob
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    return parser.parse_args()


def load_entries(logs_dir):
    entries = []
    pattern = os.path.join(logs_dir, "scores_rank*.jsonl")
    for fname in sorted(glob.glob(pattern)):
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                entries.append(json.loads(line))
    return entries


def filter_entries(entries):
    return [
        e for e in entries
        if e.get("type") != "sample" and float(e.get("score", 0.0)) != 0.0
    ]


def last_scores(entries):
    last = {}
    for e in entries:
        last[e["qid"]] = float(e["score"])
    return last


def compute_stats(scores_dict):
    n = len(scores_dict)
    total = sum(scores_dict.values())
    mean = total / n if n > 0 else 0.0
    return n, total, mean


def main():
    args = parse_args()
    entries = load_entries(args.logs_dir)
    filtered = filter_entries(entries)
    scores = last_scores(filtered)
    n, total, mean = compute_stats(scores)

    print(f"num_qids={n}")
    print(f"sum_scores={total:.6f}")
    print(f"mean_score={mean:.6f}")


if __name__ == "__main__":
    main()