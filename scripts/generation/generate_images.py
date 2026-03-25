#!/usr/bin/env python3
import os
import json
import time
import random
import argparse

import torch
from datasets import load_dataset
from tqdm import tqdm
from detikzify.model import load
from detikzify.infer import DetikzifyPipeline
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-id", default="Shoir/merged-maths")
    parser.add_argument("--split", default="train")
    parser.add_argument("--qid-json", required=True)
    parser.add_argument("--simulate-seconds", type=float, default=80.0)
    parser.add_argument("--model-name", default="nllg/detikzify-ds-1.3b")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--resize-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_torch_dtype(dtype_name):
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def resize_image(pil_img, scale):
    if scale == 1.0:
        return pil_img
    w, h = pil_img.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil_img.resize((new_w, new_h), resample=Image.BICUBIC)


def log_jsonl(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def load_qids(path, rank, world):
    with open(path, "r", encoding="utf-8") as f:
        qids = sorted(set(json.load(f)))
    return qids[rank::world]


def select_subset(ds, qids):
    seen = set()
    qid_set = set(qids)
    keep_indices = []

    for i, ex in enumerate(ds):
        qid = int(ex["qid"])
        if qid in qid_set and qid not in seen:
            seen.add(qid)
            keep_indices.append(i)

    return ds.select(keep_indices)


def choose_figure(pipeline, pil, simulate_seconds, score_log_path, qid, rank):
    fig = pipeline.sample(image=pil)
    log_jsonl(
        score_log_path,
        {
            "qid": qid,
            "rank": rank,
            "type": "sample",
            "score": 0.0,
            "rasterizable": bool(fig and fig.is_rasterizable),
        },
    )

    start = time.time()
    last_compilable_fig = None

    for score, cand in pipeline.simulate(image=pil, timeout=simulate_seconds):
        log_jsonl(
            score_log_path,
            {
                "qid": qid,
                "rank": rank,
                "type": "simulate",
                "score": float(score),
                "rasterizable": bool(cand and cand.is_rasterizable),
            },
        )

        if cand and cand.is_rasterizable:
            last_compilable_fig = cand

        if time.time() - start >= simulate_seconds:
            break

    if last_compilable_fig is not None:
        return last_compilable_fig

    for score, cand in pipeline.simulate(image=pil):
        log_jsonl(
            score_log_path,
            {
                "qid": qid,
                "rank": rank,
                "type": "post_timeout",
                "score": float(score),
                "rasterizable": bool(cand and cand.is_rasterizable),
            },
        )

        if cand and cand.is_rasterizable:
            return cand

    raise RuntimeError("No rasterizable candidate found")


def main():
    args = parse_args()
    dtype = get_torch_dtype(args.dtype)

    os.makedirs(args.out_dir, exist_ok=True)

    rank = int(os.environ.get("SLURM_PROCID", 0))
    world = int(os.environ.get("SLURM_NTASKS", 1))

    print(f"[rank {rank}] world_size={world}", flush=True)

    set_seed(args.seed + rank)
    configure_torch()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)

    qids = load_qids(args.qid_json, rank, world)
    print(f"[rank {rank}] assigned_qids={len(qids)}", flush=True)

    ds = load_dataset(args.dataset_id, split=args.split)
    subset = select_subset(ds, qids)
    print(f"[rank {rank}] kept_examples={len(subset)}", flush=True)

    model, processor = load(
        model_name_or_path=args.model_name,
        device_map={"": 0},
        torch_dtype=dtype,
    )
    pipeline = DetikzifyPipeline(model, processor)

    success = 0
    failed = 0

    logs_dir = os.path.join(args.out_dir, "logs")
    score_log_path = os.path.join(logs_dir, f"scores_rank{rank}.jsonl")
    fail_log_path = os.path.join(logs_dir, f"fail_rank{rank}.txt")

    for ex in tqdm(subset, desc=f"rank{rank}", position=rank, leave=True):
        qid = int(ex["qid"])
        out_path = os.path.join(args.out_dir, f"{qid}.png")

        if os.path.exists(out_path):
            continue

        pil = ex["image"].convert("RGB")
        pil = resize_image(pil, args.resize_scale)

        try:
            with torch.inference_mode():
                chosen_fig = choose_figure(
                    pipeline=pipeline,
                    pil=pil,
                    simulate_seconds=args.simulate_seconds,
                    score_log_path=score_log_path,
                    qid=qid,
                    rank=rank,
                )

            png = chosen_fig.rasterize(format="png")
            png.save(out_path)

            success += 1
            print(f"[rank {rank}] saved qid={qid}", flush=True)

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(
                f"[rank {rank}] gpu_mem_allocated={allocated:.1f}GB gpu_mem_reserved={reserved:.1f}GB",
                flush=True,
            )

        except Exception as e:
            failed += 1
            with open(fail_log_path, "a", encoding="utf-8") as f:
                f.write(f"qid={qid} error={repr(e)}\n")

    print(f"[rank {rank}] done success={success} failed={failed}", flush=True)


if __name__ == "__main__":
    main()