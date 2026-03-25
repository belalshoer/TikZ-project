# TikZ-project
 
This repository contains code for generating TikZ-based images with DeTikZify and evaluating them with VLM-based reasoning and image similarity metrics.
 
## Project Structure
 
```text
scripts/
├── generation/
│   └── generate_images.py
└── eval/
    ├── run_internvl.py
    ├── merge_results.py
    ├── compute_scores.py
    └── compare_metrics.py
data/
└── stable_qids.json
```
 
---
 
## Generation
 
### `generate_images.py`
 
Generates TikZ images using DeTikZify.
 
```bash
python scripts/generation/generate_images.py \
  --out-dir outputs/generation \
  --qid-json data/stable_qids.json \
  --dataset-id Shoir/merged-maths \
  --simulate-seconds 80 \
  --resize-scale 1.0 \
  --seed 42
```
 
---
 
## Evaluation
 
### `run_internvl.py`
 
Runs InternVL for reasoning and answer prediction.
 
**Single GPU:**
```bash
python scripts/eval/run_internvl.py \
  --dataset-id Shoir/detikzify-v2.5-8b-40sec-last \
  --out-dir outputs/eval
```
 
**Multi-GPU:**
```bash
torchrun --nproc_per_node=4 scripts/eval/run_internvl.py \
  --dataset-id Shoir/detikzify-v2.5-8b-40sec-last \
  --out-dir outputs/eval
```
 
---
 
### `merge_results.py`
 
Merges distributed outputs and extracts stable QIDs.
 
```bash
python scripts/eval/merge_results.py \
  --input-dir outputs/eval \
  --output-dir outputs/eval
```
 
---
 
### `compute_scores.py`
 
Computes final scores from generation logs.
 
```bash
python scripts/eval/compute_scores.py \
  --logs-dir outputs/generation/logs
```
 
---
 
### `compare_metrics.py`
 
Computes SSIM, PSNR, and DreamSim between datasets.
 
```bash
python scripts/eval/compare_metrics.py \
  --gt Shoir/merged-maths \
  --out-dir outputs/metrics \
  --device cuda
```
 
---
 
## Notes
 
- Generation and evaluation are separated.
- Scripts support distributed execution via environment variables.
- Outputs are written incrementally for long-running jobs.
 
---
 
## Datasets
 
### Ground Truth
 
| Dataset | Link |
|---|---|
| `Shoir/merged-maths` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/merged-maths) |
 
### Generated Datasets
 
| Dataset | Link |
|---|---|
| `Shoir/detikzify-ds-1.3b-40sec-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-ds-1.3b-40sec-last) |
| `Shoir/detikzify-v2.5-8b-80sec-last-retry` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-last-retry) |
| `Shoir/detikzify-v2.5-8b-40sec-0.25res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-40sec-0.25res-last) |
| `Shoir/detikzify-v2.5-8b-40sec-0.85res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-40sec-0.85res-last) |
| `Shoir/detikzify-v2.5-8b-20sec-0.85res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-20sec-0.85res-last) |
| `Shoir/detikzify-v2.5-8b-20sec-0.25res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-20sec-0.25res-last) |
| `Shoir/detikzify-v2.5-8b-80sec-0.25res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-0.25res-last) |
| `Shoir/detikzify-v2.5-8b-80sec-0.85res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-0.85res-last) |
| `Shoir/detikzify-v2.5-8b-80sec-0.50res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-0.50res-last) |
| `Shoir/detikzify-v2.5-8b-80sec-0.75res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-0.75res-last) |
| `Shoir/detikzify-v2-8b-40sec-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2-8b-40sec-last) |
| `Shoir/detikzify-v2.5-8b-80sec-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-80sec-last) |
| `Shoir/detikzify-v2.5-8b-40sec-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-40sec-last) |
| `Shoir/detikzify-v2.5-8b-40sec-75res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-40sec-75res-last) |
| `Shoir/detikzify-v2.5-8b-40sec-50res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-40sec-50res-last) |
| `Shoir/detikzify-v2.5-8b-20sec-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-20sec-last) |
| `Shoir/detikzify-v2.5-8b-20sec-75res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-20sec-75res-last) |
| `Shoir/detikzify-v2.5-8b-20sec-50res-last` | [🤗 HuggingFace](https://huggingface.co/datasets/Shoir/detikzify-v2.5-8b-20sec-50res-last) |
