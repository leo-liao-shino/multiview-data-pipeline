# Data Processing Pipeline

## Getting Started

Run `setup.sh` to setup conda env.

## Config-Driven Runs

This repo now supports JSON config files under `config/` so you do not need to pass long CLI commands.

- `config/prompt_generating.json`
- `config/qwen_edit_test.json`

Both scripts follow this precedence:

1. CLI argument
2. Config value
3. Script default

### Folder Structure
```
/workspace/data/all_multiview_datasets
 |-LivingRoom
 |-Bedroom
 |-DiningRoom
 |-Kitchen
 |-Outdoor
 |-Balcony
```

Each category contains images named:
```
{Category}_scene_{NNNN}_{Variant}{View}.jpg
```
- `Variant`: `A` or `B` (different room layouts)
- `View`: `1` (raw/source) or `2` (furnished, used for prompt generation)

Only `*_A2.jpg` and `*_B2.jpg` (furnished views) are used as input. `*_A1.jpg` and `*_B1.jpg` belong to a separate pipeline and are ignored.

---

## Prompt Generation (`scripts/prompt_generating.py`)

Uses GPT-4o vision to generate furniture-editing prompts for each furnished room image.

Current operation set:

- `MultiEdit`
- `ColorChange`
- `MaterialChange`
- `Resize`
- `LargeElement`
- `Combination`

Operations are assigned in round-robin rotation across the image queue using the active operation list (from config or `--operations`) to keep a balanced distribution. Results are saved as JSONL.

### Requirements

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

### Usage

**Basic (process all furnished views with defaults):**
```bash
python scripts/prompt_generating.py
```

**Run with explicit config file:**
```bash
python scripts/prompt_generating.py --config config/prompt_generating.json
```

**Debug (10 random images, writes to `resume/prompts_debug.jsonl`):**
```bash
python scripts/prompt_generating.py --debug
# or sample a different number:
python scripts/prompt_generating.py --debug --debug-n 5
# reproducible run with a fixed seed:
python scripts/prompt_generating.py --debug --seed 42
```
> Note: debug mode always overwrites `resume/prompts_debug.jsonl` from scratch on each run.

**Custom options:**
```bash
python scripts/prompt_generating.py \
    --dataset-root /workspace/data/all_multiview_datasets \
    --output /workspace/multiview-data-pipeline/resume/prompts.jsonl \
    --model gpt-4o \
    --views A2,B2 \
    --operations MultiEdit,ColorChange,MaterialChange,Resize,LargeElement,Combination \
    --workers 4
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/prompt_generating.json` (if exists) | JSON config path |
| `--dataset-root` | `/workspace/data/all_multiview_datasets` | Root directory of images |
| `--output` | `/workspace/multiview-data-pipeline/resume/prompts.jsonl` | Output JSONL file path |
| `--model` | `gpt-4o` | OpenAI model to use |
| `--views` | `A2,B2` | View suffixes to include (furnished views only) |
| `--operations` | all six operation types | Comma-separated operation subset to generate |
| `--workers` | `4` | Parallel API request workers |
| `--debug` | off | Sample random images and write to `prompts_debug.jsonl` |
| `--debug-n` | `10` | Number of images to sample in debug mode |
| `--seed` | `None` | Random seed for reproducible debug sampling |

### Output Format

Each line in `prompts.jsonl` is a JSON object:
```json
{
  "image_path": "Bedroom/Bedroom_scene_0001_A2.jpg",
  "category": "Bedroom",
  "scene_id": "0001",
  "variant": "A",
  "view": "2",
  "operation": "ColorChange",
  "prompt": "Change the sofa color from grey to deep navy blue"
}
```

`operation` is one of:

- `MultiEdit`
- `ColorChange`
- `MaterialChange`
- `Resize`
- `LargeElement`
- `Combination`

### Resume Support

The script automatically skips images whose `image_path` already appears in the output JSONL. Re-running the script after an interruption will continue from where it left off, and the operation rotation resumes from the next position in sequence.

When `--operations` (or config `operations`) is provided, balancing happens over that subset only.

> Debug mode does **not** support resume — it always starts from a blank file.

## Image Editing (`scripts/qwen_edit_test.py`)

Runs Qwen-Image-Edit-2509 on a sample of image-instruction pairs and saves edited images and side-by-side comparisons.

Config is supported via `config/qwen_edit_test.json`.

### Usage

```bash
python scripts/qwen_edit_test.py [OPTIONS]
```

```bash
python scripts/qwen_edit_test.py --config config/qwen_edit_test.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | `config/qwen_edit_test.json` (if exists) | JSON config path |
| `--jsonl FILE` | `/workspace/multiview-data-pipeline/resume/prompts_debug.jsonl` | Prompts JSONL to sample from |
| `--dataset-root DIR` | `/workspace/data/all_multiview_datasets` | Dataset root |
| `--output-dir DIR` | `/workspace/data/qwen_test_outputs` | Where to save edited images |
| `--n N` | `5` | Number of pairs to process |
| `--seed N` | `42` | Random seed for sampling |
| `--steps N` | `20` | Diffusion steps (20 = good quality/speed balance) |
| `--cfg FLOAT` | `4.0` | `true_cfg_scale` |
| `--model MODEL` | `Qwen/Qwen-Image-Edit-2509` | HuggingFace model ID |
| `--width N` | `None` | Native generation width request (if pipeline supports it) |
| `--height N` | `None` | Native generation height request (if pipeline supports it) |
| `--int8` | ✓ (default) | 8-bit quantization: ~24 GB VRAM |
| `--bf16` | — | Full bfloat16: ~47 GB VRAM, faster |
| `--wandb` | off | Log compare images to Weights & Biases |
| `--wandb-project NAME` | `qwen-furniture-edit` | W&B project name |

### Examples

```bash
# Quick test, 5 pairs
python scripts/qwen_edit_test.py --n 5 --seed 42

# 10 pairs with W&B logging
python scripts/qwen_edit_test.py --n 10 --seed 42 --wandb

# Full bf16 (needs full 48 GB headroom)
python scripts/qwen_edit_test.py --n 5 --bf16

# Request native higher resolution (if supported by installed pipeline)
python scripts/qwen_edit_test.py --n 1 --width 1536 --height 1536
```

### Outputs

For each pair:
- `{stem}_edited.jpg` — edited image
- `{stem}_compare.jpg` — side-by-side before/after with operation label
- `results.jsonl` — metadata log (overwritten per run)

### Notes

- Native output resolution depends on your installed Qwen/diffusers pipeline support.
- If `--width/--height` is set and supported, the script requests that native output size.
- If unsupported, the script logs a warning and continues with default pipeline behavior.
- 8-bit quantization (`--int8`) reduces VRAM from ~47 GB to ~24 GB with minimal quality impact.
- Steps=20 was found to be visually equivalent to the model card default of 40 at half the time (~3 min/image on A6000).

---

## Dataset Structure

```
all_multiview_datasets/
  {Category}/
    {Category}_scene_{NNNN}_{Variant}{View}.jpg
```

Example: `Bedroom/Bedroom_scene_0001_A2.jpg`

- Only `A2` and `B2` views are processed (not `A1`/`B1`)
- Categories: Bedroom, LivingRoom, DiningRoom, Kitchen, Outdoor, Balcony
