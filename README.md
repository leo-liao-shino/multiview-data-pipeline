# Data Processing Pipeline

## Getting Started

Run `setup.sh` to setup conda env.

## Config-Driven Runs

This repo now supports JSON config files under `config/` so you do not need to pass long CLI commands.

- `config/prompt_generating.json`
- `config/qwen_edit.json`
- `config/qwen_edit_test.json`

These scripts follow this precedence:

1. CLI argument
2. Config value
3. Script default

### Folder Structure
```
/workspace/data/all-multiview-datasets
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
- `Move`
- `Rotation`

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
  --dataset-root /workspace/data/all-multiview-datasets \
    --output /workspace/multiview-data-pipeline/resume/prompts.jsonl \
    --model gpt-4o \
    --views A2,B2 \
  --operations MultiEdit,ColorChange,MaterialChange,Resize,LargeElement,Combination,Move,Rotation \
    --workers 4
```

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/prompt_generating.json` (if exists) | JSON config path |
| `--dataset-root` | `/workspace/data/all-multiview-datasets` | Root directory of images |
| `--output` | `/workspace/multiview-data-pipeline/resume/prompts.jsonl` | Output JSONL file path |
| `--model` | `gpt-4o` | OpenAI model to use |
| `--views` | `A2,B2` | View suffixes to include (furnished views only) |
| `--operations` | all eight operation types | Comma-separated operation subset to generate |
| `--workers` | `4` | Parallel API request workers |
| `--debug` | off | Sample random images and write to `prompts_debug.jsonl` |
| `--debug-n` | `10` | Number of images to sample in debug mode |
| `--seed` | `None` | Random seed for reproducible debug sampling |
| `--append` | off | Append mode: generate new operations for existing images in output JSONL |
| `--append-n` | all eligible | In append mode, sample this many images per operation |

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
- `Move`
- `Rotation`

### Resume Support

The script automatically skips images whose `image_path` already appears in the output JSONL. Re-running the script after an interruption will continue from where it left off, and the operation rotation resumes from the next position in sequence.

When `--operations` (or config `operations`) is provided, balancing happens over that subset only.

> Debug mode does **not** support resume — it always starts from a blank file.

## Image Editing (`scripts/qwen_edit.py`)

Runs Qwen-Image-Edit-2511 on prompt records from JSONL and saves edited images, side-by-side comparisons, and metadata logs.

Config is supported via `config/qwen_edit.json`.

### Usage

```bash
python scripts/qwen_edit.py [OPTIONS]
```

```bash
python scripts/qwen_edit.py --config config/qwen_edit.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--config FILE` | `config/qwen_edit.json` (if exists) | JSON config path |
| `--jsonl FILE` | `/workspace/multiview-data-pipeline/resume/prompts.jsonl` | Prompts JSONL |
| `--dataset-root DIR` | `/workspace/data/all-multiview-datasets` | Dataset root |
| `--output-dir DIR` | `/workspace/data/qwen-outputs` | Where to save edited images |
| `--seed N` | `42` | Base random seed |
| `--steps N` | `20` | Diffusion steps (20 = good quality/speed balance) |
| `--cfg FLOAT` | `4.0` | `true_cfg_scale` |
| `--width N` | none | Fixed output width (must be used with `--height`) |
| `--height N` | none | Fixed output height (must be used with `--width`) |
| `--match-input-resolution` | off | Use each source image resolution for generation |
| `--max-megapixels FLOAT` | none (`7` auto when matching input resolution) | Caps generation area to avoid VRAM/OOM issues |
| `--model MODEL` | `Qwen/Qwen-Image-Edit-2511` | HuggingFace model ID |
| `--preserve-untouched` | on (via config/default) | Appends instruction to keep non-target regions unchanged |
| `--preserve-suffix TEXT` | default suffix | Custom suffix for preserve-untouched behavior |
| `--precision {int8,bf16}` | `int8` | Explicit precision mode |
| `--int8` | ✓ (default) | 8-bit quantization: ~24 GB VRAM |
| `--bf16` | — | Full bfloat16: ~47 GB VRAM, faster |
| `--resume` | on (via config) | Skip records already completed in `results.jsonl` |
| `--overwrite` | off | Start a fresh run and rewrite results |
| `--limit N` | none | Process only first N pending records |
| `--wandb` | off | Log compare images to Weights & Biases |
| `--wandb-project NAME` | `qwen-furniture-edit` | W&B project name |

### Examples

```bash
# Full run from prompts.jsonl
python scripts/qwen_edit.py

# Resume run with W&B logging
python scripts/qwen_edit.py --resume --wandb

# Full bf16 (needs full 48 GB headroom)
python scripts/qwen_edit.py --bf16

# Override the model explicitly if needed
python scripts/qwen_edit.py --model Qwen/Qwen-Image-Edit-2511
```

### Outputs

For each pair:
- `{index:06d}_{stem}_{operation}_edited.jpg` — edited image
- `{index:06d}_{stem}_{operation}_compare.jpg` — side-by-side before/after with operation label
- `results.jsonl` — metadata log (append mode for resume)

## Image Editing Test Sampler (`scripts/qwen_edit_test.py`)

`qwen_edit_test.py` is available for quick smoke tests on a random sample (`--n`) and for scheduler/prompt sweeps.

Config is supported via `config/qwen_edit_test.json`.

### Usage

```bash
python scripts/qwen_edit_test.py --config config/qwen_edit_test.json
```

```bash
# quick sample run
python scripts/qwen_edit_test.py --n 5 --seed 42

# scheduler/config sweep from config
python scripts/qwen_edit_test.py --sweep
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--n N` | `5` | Number of sampled prompt records |
| `--scheduler` | `default` | One of `default`, `flowmatch_euler`, `flowmatch_euler_karras`, `flowmatch_lcm` |
| `--guidance-scale FLOAT` | `1.0` | Guidance scale |
| `--negative-prompt TEXT` | single space | Negative prompt text |
| `--prompt-prefix TEXT` | empty | Prefix added to every prompt |
| `--prompt-case {none,lower,upper}` | `none` | Prompt case transform |
| `--prompt-max-chars N` | none | Truncate prompt length |
| `--sweep` | off | Run config-driven sweep over scheduler/steps/cfg/etc. |
| `--sweep-limit N` | none | Cap number of sweep trials |

### Notes

- The default tested model is now `Qwen/Qwen-Image-Edit-2511`.
- The 2511 model card recommends using the latest `diffusers` build.
- 8-bit quantization (`--int8`) reduces VRAM from ~47 GB to ~24 GB with minimal quality impact.
- Steps=20 was found to be visually equivalent to the model card default of 40 at half the time (~3 min/image on A6000).

---

## Subsampling (`scripts/subsample.py`)

Creates a balanced subset from `prompts.jsonl` by sampling exactly `N` records per `(category, operation)` pair.

```bash
python scripts/subsample.py \
  --input /workspace/multiview-data-pipeline/resume/prompts.jsonl \
  --output /workspace/multiview-data-pipeline/resume/prompts_subset.jsonl \
  --n 20 \
  --seed 42
```

If a pair has fewer than `N` records, all available records are kept and a warning is printed.

---

## Dataset Structure

```
all-multiview-datasets/
  {Category}/
    {Category}_scene_{NNNN}_{Variant}{View}.jpg
```

Example: `Bedroom/Bedroom_scene_0001_A2.jpg`

- Only `A2` and `B2` views are processed (not `A1`/`B1`)
- Categories: Bedroom, LivingRoom, DiningRoom, Kitchen, Outdoor, Balcony


## Mask Generation

Conda env `gsam2` 

```bash
PYTHONPATH=/workspace/multiview-data-pipeline/Grounded-SAM-2 \
python ../scripts/generate_edit_masks.py \
    --jsonl /workspace/data/qwen-outputs/results.jsonl \
    --image_root /workspace/data/all-multiview-datasets \
    --output_dir /workspace/data/masks \
    --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --expand_pixels 30 \
    --device cuda \
    --limit 5 \
    --box_threshold 0.20 \
    --text_threshold 0.20
```

### 1. Create Environment
```bash
conda create -n gsam2 python=3.10 -y
conda activate gsam2
```

### 2. Install PyTorch (Blackwell GPU / sm_120 support)
```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
```

> If you are not on a Blackwell GPU (RTX 5000 series / sm_120), install the stable release instead:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 3. Clone and Install Grounded-SAM-2
```bash
git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
cd Grounded-SAM-2
pip install -e .
```

### 4. Patch and Install GroundingDINO

GroundingDINO's CUDA kernel uses a deprecated PyTorch API that must be patched before building:
```bash
# Patch deprecated value.type() API
sed -i \
    's/AT_DISPATCH_FLOATING_TYPES(value\.type(),/AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),/g' \
    grounding_dino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu

sed -i \
    's/value\.type()\.is_cuda()/value.is_cuda()/g' \
    grounding_dino/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn.h

# Patch BertModelWarper for transformers compatibility
sed -i \
    's/self.get_head_mask = bert_model.get_head_mask/self.get_head_mask = getattr(bert_model, "get_head_mask", None)/g' \
    grounding_dino/groundingdino/models/GroundingDINO/bertwarper.py

pip install --no-build-isolation -e grounding_dino
```

### 5. Install Remaining Dependencies
```bash
pip install supervision qwen-vl-utils accelerate bitsandbytes
pip install transformers==4.49.0
```

### 6. Download Weights
```bash
# SAM 2.1
cd checkpoints && bash download_ckpts.sh && cd ..

# GroundingDINO
cd gdino_checkpoints && bash download_ckpts.sh && cd ..
```

### 7. Set PYTHONPATH

Either set it permanently via conda:
```bash
conda env config vars set PYTHONPATH=/path/to/Grounded-SAM-2
conda deactivate && conda activate gsam2
```

Or set it per-run:
```bash
PYTHONPATH=/path/to/Grounded-SAM-2 python ../scripts/generate_edit_masks.py ...
```

### Usage

All commands must be run from inside the `Grounded-SAM-2` directory:
```bash
cd Grounded-SAM-2

PYTHONPATH=$(pwd) python ../scripts/generate_edit_masks.py \
    --jsonl /path/to/results.jsonl \
    --image_root /path/to/images \
    --output_dir /path/to/masks \
    --vlm_model Qwen/Qwen2.5-VL-7B-Instruct \
    --expand_pixels 30 \
    --close_pixels 40 \
    --device cuda
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--vlm_model` | `Qwen/Qwen2.5-VL-7B-Instruct` | Local VLM for grounding phrase extraction |
| `--no_vlm` | off | Skip VLM, use regex parser only |
| `--expand_pixels` | `30` | Dilate mask outward by N pixels |
| `--close_pixels` | `40` | Morphological closing to fill interior gaps |
| `--box_threshold` | `0.30` | GroundingDINO confidence threshold (lower = more detections) |
| `--diff_refine` | off | Intersect SAM mask with pixel diff to suppress false positives |
| `--limit` | None | Process only first N records (for testing) |

# Post Processing Pipelie
## Selection
Run in local machine to enbale GUI. Either mount directory with `sshfs` or download with `rclone copy`. 
```bash
python scripts/image_selections.py --images /path/to/images --meta /path/to/jsonl --out /path/to/outs
```

## Checking
Check those being accpeted. 
```bash
python scripts/show_image path/to/outs --base /path/to/local/dir --cache 8
```
Caching applied for lazy loading. 