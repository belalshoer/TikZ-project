#!/usr/bin/env python3
import os
import json
import re
import random
import argparse

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader


INSTRUCTION = (
    "You are a multiple-choice math solver.\n"
    "Rules:\n"
    "- Explain your reasoning.\n"
    "- SHOW steps.\n"
    "- Output exactly ONE capital letter: A, B, C, D, or E.\n"
)


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

PRED_RE = re.compile(r"\b([A-E])\b")
GT_RE = re.compile(r"\b([A-E])\b")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--dataset-id", default="Shoir/synthesized-detikzifyv2.5-40sec")
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-tiles", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--out-dir", default="reasoning_dp_outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def build_transform(image_size):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def build_question_text(question):
    return INSTRUCTION + "<image>\n" + question.strip() + "\n\nAnswer:"


def ensure_pil(img):
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")
    return img.convert("RGB")


def dynamic_preprocess(image, max_tiles, image_size):
    w, h = image.size
    aspect = w / h

    best = (1, 1)
    best_diff = float("inf")

    for i in range(1, max_tiles + 1):
        for j in range(1, max_tiles + 1):
            if i * j <= max_tiles:
                diff = abs(aspect - (i / j))
                if diff < best_diff:
                    best = (i, j)
                    best_diff = diff

    tw, th = best
    resized = image.resize((tw * image_size, th * image_size), resample=Image.BICUBIC)

    tiles = []
    for y in range(th):
        for x in range(tw):
            box = (
                x * image_size,
                y * image_size,
                (x + 1) * image_size,
                (y + 1) * image_size,
            )
            tiles.append(resized.crop(box))

    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size), resample=Image.BICUBIC))

    return tiles[:max_tiles]


def image_to_tensor(img, max_tiles, image_size, transform):
    tiles = dynamic_preprocess(img, max_tiles=max_tiles, image_size=image_size)
    return torch.stack([transform(tile) for tile in tiles], dim=0)


def extract_pred(text):
    if not text:
        return None
    matches = PRED_RE.findall(text.upper())
    return matches[-1] if matches else None


def normalize_gt(gt_text):
    if not gt_text:
        return set()
    return set(GT_RE.findall(gt_text.upper()))


def get_dist_info():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world_size, local_rank


def shard_indices(n, rank, world_size):
    return list(range(rank, n, world_size))


def make_collate_fn(max_tiles, image_size, transform):
    def collate_fn(examples):
        pixel_values_list = []
        num_patches_list = []
        questions = []
        metadata = []

        for ex in examples:
            img = ensure_pil(ex["image"])
            pixel_values = image_to_tensor(
                img,
                max_tiles=max_tiles,
                image_size=image_size,
                transform=transform,
            )
            pixel_values_list.append(pixel_values)
            num_patches_list.append(pixel_values.size(0))
            questions.append(build_question_text(ex["question"]))
            metadata.append({
                "qid": ex["qid"],
                "question": ex["question"],
                "gt_raw": ex["answer"],
                "gt_set": normalize_gt(ex["answer"]),
                "perm": ex.get("perm"),
            })

        return pixel_values_list, num_patches_list, questions, metadata

    return collate_fn


def get_dtype(local_rank):
    major, _ = torch.cuda.get_device_capability(local_rank)
    return torch.bfloat16 if major >= 8 else torch.float16


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    rank, world_size, local_rank = get_dist_info()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    set_seed(args.seed + rank)
    configure_torch()

    dtype = get_dtype(local_rank)
    transform = build_transform(args.image_size)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        use_fast=False,
    )

    think_token_ids = tokenizer.encode("<think>", add_special_tokens=False)

    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device).eval()

    ds = load_dataset(args.dataset_id, split=args.split)
    ds = ds.select(shard_indices(len(ds), rank, world_size))

    collate_fn = make_collate_fn(
        max_tiles=args.max_tiles,
        image_size=args.image_size,
        transform=transform,
    )

    loader_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(**loader_kwargs)

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"results_rank{rank}.jsonl")

    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "bad_words_ids": [think_token_ids],
    }

    with open(out_path, "w", encoding="utf-8") as f_out:
        for pixel_values_list, num_patches_list, questions, metadata in tqdm(
            loader,
            disable=(rank != 0),
        ):
            for pv_cpu, npatch, question_text, meta in zip(
                pixel_values_list,
                num_patches_list,
                questions,
                metadata,
            ):
                pv = pv_cpu.to(device=device, dtype=dtype, non_blocking=True)

                with torch.inference_mode():
                    output = model.chat(
                        tokenizer,
                        pv,
                        question_text,
                        generation_config,
                        num_patches_list=[npatch],
                        history=None,
                        return_history=False,
                    )

                pred = extract_pred(output)
                is_correct = (pred in meta["gt_set"]) if pred else False

                record = {
                    "qid": meta["qid"],
                    "question": meta["question"],
                    "predicted": pred,
                    "correct_raw": meta["gt_raw"],
                    "correct_set": sorted(meta["gt_set"]),
                    "is_correct": is_correct,
                    "model_output": output,
                    "perm": meta["perm"],
                    "rank": rank,
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

    if rank == 0:
        print("Done")


if __name__ == "__main__":
    main()