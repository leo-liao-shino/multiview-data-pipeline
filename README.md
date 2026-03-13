# Data Processing Pipeline

## Getting Started

Run `setup.sh` to setup conda env.

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

Uses GPT-4o vision to generate furniture-editing prompts (`Add`, `Delete`, or `Replace`) for each furnished room image. Operations are assigned in round-robin rotation (`Add → Delete → Replace → …`) across the image queue to ensure a balanced 1/3 distribution. Results are saved as a JSONL file.

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
    --workers 4
```

| Argument | Default | Description |
|---|---|---|
| `--dataset-root` | `/workspace/all_multiview_datasets` | Root directory of images |
| `--output` | `resume/prompts.jsonl` | Output JSONL file path |
| `--model` | `gpt-4o` | OpenAI model to use |
| `--views` | `A2,B2` | View suffixes to include (furnished views only) |
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
  "operation": "Add",
  "prompt": "Add a queen-sized bed with a dark wooden headboard against the main wall"
}
```

`operation` is one of: `Add`, `Delete`, `Replace`.

### Resume Support

The script automatically skips images whose `image_path` already appears in the output JSONL. Re-running the script after an interruption will continue from where it left off, and the operation rotation resumes from the next position in sequence.

> Debug mode does **not** support resume — it always starts from a blank file.
