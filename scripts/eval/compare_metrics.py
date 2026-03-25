#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from dreamsim import dreamsim
from PIL import Image, ImageFile
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_PRED_DATASETS = [
    "Shoir/detikzify-ds-1.3b-40sec-last",
    "Shoir/detikzify-v2.5-8b-80sec-last-retry",
    "Shoir/detikzify-v2.5-8b-40sec-0.25res-last",
    "Shoir/detikzify-v2.5-8b-40sec-0.85res-last",
    "Shoir/detikzify-v2.5-8b-20sec-0.85res-last",
    "Shoir/detikzify-v2.5-8b-20sec-0.25res-last",
    "Shoir/detikzify-v2.5-8b-80sec-0.25res-last",
    "Shoir/detikzify-v2.5-8b-80sec-0.85res-last",
    "Shoir/detikzify-v2.5-8b-80sec-0.50res-last",
    "Shoir/detikzify-v2.5-8b-80sec-0.75res-last",
    "Shoir/detikzify-v2-8b-40sec-last",
    "Shoir/detikzify-v2.5-8b-80sec-last",
    "Shoir/detikzify-v2.5-8b-40sec-last",
    "Shoir/detikzify-v2.5-8b-40sec-75res-last",
    "Shoir/detikzify-v2.5-8b-40sec-50res-last",
    "Shoir/detikzify-v2.5-8b-20sec-last",
    "Shoir/detikzify-v2.5-8b-20sec-75res-last",
    "Shoir/detikzify-v2.5-8b-20sec-50res-last",
]


@dataclass
class Example:
    qid: int
    image: Image.Image
    perm: Optional[str] = None
    dataset: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="Shoir/merged-maths")
    parser.add_argument("--pred", action="append", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", default="hf_metrics_results")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dreamsim-model", default="ensemble")
    parser.add_argument("--max-qids", type=int, default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument(
        "--resize-resample",
        default="bicubic",
        choices=["nearest", "bilinear", "bicubic", "lanczos"],
    )
    parser.add_argument("--gt-revision", default=None)
    parser.add_argument("--pred-revision", default=None)
    parser.add_argument("--streaming", action="store_true", default=False)
    parser.add_argument("--save-json-summary", action="store_true", default=False)
    return parser.parse_args()


def infer_device(device_arg: Optional[str]) -> str:
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def sanitize_name(name: str) -> str:
    name = name.replace("/", "__")
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def get_resample(name: str):
    mapping = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
        "lanczos": Image.Resampling.LANCZOS,
    }
    return mapping[name]


def pil_to_rgb(img: Image.Image) -> Image.Image:
    if not isinstance(img, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(img)}")
    return img.convert("RGB")


def pil_to_float_np(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255.0


def resize_to_match(pred_img: Image.Image, gt_img: Image.Image, resample) -> Image.Image:
    if pred_img.size == gt_img.size:
        return pred_img
    return pred_img.resize(gt_img.size, resample=resample)


def compute_ssim_psnr(gt_img: Image.Image, pred_img: Image.Image) -> Tuple[float, float]:
    gt_np = pil_to_float_np(gt_img)
    pred_np = pil_to_float_np(pred_img)

    ssim_val = structural_similarity(
        gt_np,
        pred_np,
        channel_axis=2,
        data_range=1.0,
    )
    psnr_val = peak_signal_noise_ratio(
        gt_np,
        pred_np,
        data_range=1.0,
    )
    return float(ssim_val), float(psnr_val)


@torch.inference_mode()
def compute_dreamsim_distance(
    gt_img: Image.Image,
    pred_img: Image.Image,
    model,
    preprocess,
    device: str,
) -> float:
    gt_tensor = preprocess(gt_img).to(device)
    pred_tensor = preprocess(pred_img).to(device)
    distance = model(gt_tensor, pred_tensor)
    if isinstance(distance, torch.Tensor):
        distance = distance.detach().float().mean().item()
    return float(distance)


def hf_load(
    dataset_name: str,
    split: str,
    revision: Optional[str],
    streaming: bool,
    token: Optional[str],
):
    kwargs = {
        "path": dataset_name,
        "split": split,
        "streaming": streaming,
    }
    if revision is not None:
        kwargs["revision"] = revision
    if token is not None:
        kwargs["token"] = token

    try:
        return load_dataset(**kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load dataset {dataset_name!r} split={split!r}: {e}"
        ) from e


def build_unique_qid_index(
    dataset_name: str,
    split: str,
    revision: Optional[str],
    streaming: bool,
    token: Optional[str],
    max_qids: Optional[int] = None,
) -> Dict[int, Example]:
    ds = hf_load(dataset_name, split, revision, streaming, token)
    unique: Dict[int, Example] = OrderedDict()
    num_rows = 0

    for row in ds:
        num_rows += 1

        if "qid" not in row:
            raise KeyError(f"{dataset_name}: row does not contain 'qid'")

        qid = int(row["qid"])
        if qid in unique:
            continue

        if "image" not in row:
            raise KeyError(f"{dataset_name}: row does not contain 'image'")

        try:
            image = pil_to_rgb(row["image"])
        except Exception as e:
            print(
                f"[WARN] {dataset_name}: failed to decode image for qid={qid}: {e}",
                file=sys.stderr,
            )
            continue

        unique[qid] = Example(
            qid=qid,
            image=image,
            perm=str(row["perm"]) if row.get("perm") is not None else None,
            dataset=str(row["dataset"]) if row.get("dataset") is not None else None,
            question=str(row["question"]) if row.get("question") is not None else None,
            answer=str(row["answer"]) if row.get("answer") is not None else None,
        )

        if max_qids is not None and len(unique) >= max_qids:
            break

    print(f"[INFO] {dataset_name}: scanned_rows={num_rows} unique_qids_kept={len(unique)}")
    return unique


def summarize(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }


def save_csv(rows: List[dict], path: Path, empty_header: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow(empty_header)
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_datasets = args.pred if args.pred is not None else DEFAULT_PRED_DATASETS
    device = infer_device(args.device)
    resample = get_resample(args.resize_resample)

    print(f"[INFO] device={device}")
    print(f"[INFO] loading_dreamsim_model={args.dreamsim_model}")
    model, preprocess = dreamsim(
        pretrained=True,
        device=device,
        dreamsim_type=args.dreamsim_model,
    )
    model.eval()

    print(f"[INFO] building_gt_unique_qid_index_from={args.gt}")
    gt_map = build_unique_qid_index(
        dataset_name=args.gt,
        split=args.split,
        revision=args.gt_revision,
        streaming=args.streaming,
        token=args.hf_token,
        max_qids=args.max_qids,
    )

    summary_rows = []

    for pred_name in pred_datasets:
        print(f"[INFO] processing_prediction_dataset={pred_name}")
        pred_map = build_unique_qid_index(
            dataset_name=pred_name,
            split=args.split,
            revision=args.pred_revision,
            streaming=args.streaming,
            token=args.hf_token,
            max_qids=args.max_qids,
        )

        common_qids = sorted(set(gt_map) & set(pred_map))
        print(f"[INFO] {pred_name}: common_qids={len(common_qids)}")

        per_image_rows = []
        ssim_vals: List[float] = []
        psnr_vals: List[float] = []
        dreamsim_vals: List[float] = []

        for idx, qid in enumerate(common_qids, start=1):
            gt_ex = gt_map[qid]
            pred_ex = pred_map[qid]

            gt_img = pil_to_rgb(gt_ex.image)
            pred_img = pil_to_rgb(pred_ex.image)
            pred_for_pixel = resize_to_match(pred_img, gt_img, resample=resample)

            try:
                ssim_val, psnr_val = compute_ssim_psnr(gt_img, pred_for_pixel)
            except Exception as e:
                print(f"[WARN] qid={qid}: failed_ssim_psnr={e}", file=sys.stderr)
                ssim_val, psnr_val = float("nan"), float("nan")

            try:
                dreamsim_val = compute_dreamsim_distance(
                    gt_img=gt_img,
                    pred_img=pred_img,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                )
            except Exception as e:
                print(f"[WARN] qid={qid}: failed_dreamsim={e}", file=sys.stderr)
                dreamsim_val = float("nan")

            if not math.isnan(ssim_val):
                ssim_vals.append(ssim_val)
            if not math.isnan(psnr_val):
                psnr_vals.append(psnr_val)
            if not math.isnan(dreamsim_val):
                dreamsim_vals.append(dreamsim_val)

            per_image_rows.append(
                {
                    "qid": qid,
                    "gt_perm_used": gt_ex.perm,
                    "pred_perm_used": pred_ex.perm,
                    "gt_dataset_field": gt_ex.dataset,
                    "pred_dataset_field": pred_ex.dataset,
                    "gt_width": gt_img.width,
                    "gt_height": gt_img.height,
                    "pred_width_original": pred_img.width,
                    "pred_height_original": pred_img.height,
                    "pred_width_for_ssim_psnr": pred_for_pixel.width,
                    "pred_height_for_ssim_psnr": pred_for_pixel.height,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "dreamsim": dreamsim_val,
                }
            )

            if idx % 100 == 0 or idx == len(common_qids):
                print(f"[INFO] {pred_name}: processed={idx}/{len(common_qids)}")

        ssim_summary = summarize(ssim_vals)
        psnr_summary = summarize(psnr_vals)
        dreamsim_summary = summarize(dreamsim_vals)

        safe_name = sanitize_name(pred_name)
        per_image_csv_path = out_dir / f"{safe_name}_per_image_metrics.csv"
        save_csv(
            per_image_rows,
            per_image_csv_path,
            ["qid", "ssim", "psnr", "dreamsim"],
        )

        summary_row = {
            "dataset": pred_name,
            "num_gt_unique_qids": len(gt_map),
            "num_pred_unique_qids": len(pred_map),
            "num_common_qids": len(common_qids),
            "ssim_count": ssim_summary["count"],
            "ssim_mean": ssim_summary["mean"],
            "ssim_std": ssim_summary["std"],
            "ssim_median": ssim_summary["median"],
            "ssim_min": ssim_summary["min"],
            "ssim_max": ssim_summary["max"],
            "psnr_count": psnr_summary["count"],
            "psnr_mean": psnr_summary["mean"],
            "psnr_std": psnr_summary["std"],
            "psnr_median": psnr_summary["median"],
            "psnr_min": psnr_summary["min"],
            "psnr_max": psnr_summary["max"],
            "dreamsim_count": dreamsim_summary["count"],
            "dreamsim_mean": dreamsim_summary["mean"],
            "dreamsim_std": dreamsim_summary["std"],
            "dreamsim_median": dreamsim_summary["median"],
            "dreamsim_min": dreamsim_summary["min"],
            "dreamsim_max": dreamsim_summary["max"],
            "per_image_csv": str(per_image_csv_path),
        }
        summary_rows.append(summary_row)

        print(json.dumps(summary_row, indent=2))

    summary_csv_path = out_dir / "summary_metrics.csv"
    save_csv(summary_rows, summary_csv_path, ["dataset"])
    print(f"[INFO] wrote_summary_csv={summary_csv_path}")

    if args.save_json_summary:
        summary_json_path = out_dir / "summary_metrics.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        print(f"[INFO] wrote_summary_json={summary_json_path}")


if __name__ == "__main__":
    main()