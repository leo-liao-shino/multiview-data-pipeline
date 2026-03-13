"""
Generate furniture editing prompts for furnished room images using GPT-4o vision.
Covers six challenging operation types designed to stress-test image-editing models:
  MultiEdit, ColorChange, MaterialChange, Resize, LargeElement, Combination.
Results are saved as a JSONL file with full resume support.

Usage:
    conda activate data_process
    python scripts/prompt_generating.py [--dataset-root DIR] [--output FILE]
                                [--model MODEL] [--views A2,B2]
                                [--operations MultiEdit,ColorChange,...]
                                [--config FILE]
                                [--workers N] [--debug] [--debug-n N] [--seed N]
"""

import os
import re
import json
import time
import base64
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

from config_utils import load_json_config, pick_value, normalize_csv_or_list

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = "/workspace/data/all-multiview-datasets"
DEFAULT_OUTPUT = "/workspace/multiview-data-pipeline/resume/prompts.jsonl"
DEFAULT_MODEL = "gpt-4o"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "prompt_generating.json"
# Process only these view suffixes by default (one representative per scene variant)
DEFAULT_VIEWS = {"A2", "B2"}

SYSTEM_PROMPT = """
You are an expert interior designer. You will be given a photograph of a furnished room and a required operation type.
Your task is to generate ONE concise, specific furniture-editing instruction for image-editing purposes.

Operation definitions:
- MultiEdit:      specify 2–3 simultaneous furniture edits (any mix of add/delete/replace) targeting different objects
- ColorChange:    change the color of one or more existing furniture items or surfaces
- MaterialChange: change the material or texture of one or more existing furniture items
- Resize:         make one or more existing furniture items larger or smaller; mention relative scale when helpful, try to avoid too slight changes that may be ambiguous
- LargeElement:   edit a dominant scene element such as a wall, floor, ceiling, or a large piece of furniture
- Combination:    chain 2–3 different edit types in one instruction (e.g. resize one thing, change another's material, replace a third)

Rules:
- You MUST use the exact operation type specified in the user message.
- Your instruction MUST reference objects or surfaces that are visible in the photo.
- Be specific: mention color, material, size, style, and location when relevant.
- For single-focus operations (ColorChange, MaterialChange, Resize, LargeElement): keep the prompt under 20 words.
- For multi-focus operations (MultiEdit, Combination): keep the prompt under 40 words; separate sub-instructions with semicolons.
- Respond ONLY with valid JSON — no explanation, no markdown.

Response format:
{"operation": "<required operation>", "prompt": "<instruction>"}

Examples:
{"operation": "MultiEdit", "prompt": "Add a floor lamp in the corner; remove the small coffee table; replace the armchair with a velvet accent chair"}
{"operation": "ColorChange", "prompt": "Change the sofa color from grey to deep navy blue"}
{"operation": "ColorChange", "prompt": "Change the curtains to warm beige and the accent chair to forest green"}
{"operation": "MaterialChange", "prompt": "Replace the wooden coffee table top with white marble"}
{"operation": "MaterialChange", "prompt": "Change the dining chairs to velvet upholstery and the tabletop to tempered glass"}
{"operation": "Resize", "prompt": "Scale down the oversized sectional sofa by about 20% to open up the room"}
{"operation": "Resize", "prompt": "Make the nightstand on the left taller to align with the top of the bed frame"}
{"operation": "LargeElement", "prompt": "Paint the accent wall behind the sofa a deep charcoal grey"}
{"operation": "LargeElement", "prompt": "Replace the hardwood flooring with light grey large-format stone tiles"}
{"operation": "Combination", "prompt": "Scale up the dining table slightly; change its surface to dark walnut veneer; replace the overhead pendant with a modern brass chandelier"}
{"operation": "Combination", "prompt": "Change the sofa fabric to cognac leather; resize the coffee table to be smaller; remove the floor lamp by the window"}
"""

# Addressed in this revision:
# 1. MultiEdit   — simultaneous edits on multiple objects
# 2. LargeElement — large scene elements (walls, floors, dominant furniture)
# 3. ColorChange / MaterialChange — color & material accuracy
# 4. Combination  — chained multi-type instructions
# 5. Resize       — scale up/down with consistency
# (Mask dilation is a separate pipeline concern, not prompt-level)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_filename(filename: str) -> dict:
    """Extract structured metadata from image filename.

    Expected pattern: {Category}_scene_{NNNN}_{Variant}{View}.jpg
    e.g.  Bedroom_scene_0001_A2.jpg
    """
    m = re.match(r"(.+)_scene_(\d+)_([AB])([2])\.jpe?g$", filename, re.IGNORECASE)
    if not m:
        return {}
    return {
        "category": m.group(1),
        "scene_id": m.group(2),
        "variant": m.group(3),
        "view": m.group(4),
    }


# All valid operation types for the challenging-case dataset.
# Use --operations at runtime to target a specific subset.
OPERATIONS = ["MultiEdit", "ColorChange", "MaterialChange", "Resize", "LargeElement", "Combination"]


def call_gpt(client: OpenAI, image_path: Path, model: str, operation: str, retries: int = 3) -> dict:
    b64 = encode_image(image_path)
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "text",
                                "text": f"Required operation: {operation}",
                            },
                        ],
                    },
                ],
                max_tokens=150,
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from model")
            parsed = json.loads(content)
            if "operation" not in parsed or "prompt" not in parsed:
                raise ValueError(f"Unexpected JSON keys: {parsed}")
            if parsed["operation"] not in OPERATIONS:
                raise ValueError(f"Unknown operation '{parsed['operation']}'; expected one of {OPERATIONS}")
            # Enforce the assigned operation in case model ignored it
            parsed["operation"] = operation
            return parsed
        except RateLimitError:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s
            time.sleep(wait)
        except (APIError, json.JSONDecodeError, ValueError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError(f"Failed after {retries} attempts")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate furniture-editing prompts via GPT-4o vision")
    parser.add_argument("--config", default=None, help="Path to JSON config file (default: config/prompt_generating.json)")
    parser.add_argument("--dataset-root", default=None, help="Root directory of image dataset")
    parser.add_argument("--output", default=None, help="Output JSONL file path")
    parser.add_argument("--model", default=None, help="OpenAI model to use")
    parser.add_argument(
        "--views",
        default=None,
        help="Comma-separated list of view suffixes to include (e.g. A2,B2)",
    )
    parser.add_argument(
        "--operations",
        default=None,
        help="Comma-separated list of operation types to generate (default: all six challenging types)",
    )
    parser.add_argument("--workers", type=int, default=None, help="Parallel API workers (default: 4)")
    parser.add_argument("--debug", action="store_true", default=None, help="Debug mode: sample a few random images and write to a separate JSONL")
    parser.add_argument("--debug-n", type=int, default=None, help="Number of images to sample in debug mode (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible debug sampling")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    try:
        config = load_json_config(config_path, require_exists=bool(args.config))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        parser.error(str(e))

    args.dataset_root = pick_value(args.dataset_root, config, "dataset_root", DEFAULT_DATASET_ROOT)
    args.output = pick_value(args.output, config, "output", DEFAULT_OUTPUT)
    args.model = pick_value(args.model, config, "model", DEFAULT_MODEL)
    args.views = normalize_csv_or_list(
        pick_value(args.views, config, "views", sorted(DEFAULT_VIEWS))
    )
    args.operations = normalize_csv_or_list(
        pick_value(args.operations, config, "operations", OPERATIONS)
    )
    args.workers = int(pick_value(args.workers, config, "workers", 4))
    args.debug = bool(pick_value(args.debug, config, "debug", False))
    args.debug_n = int(pick_value(args.debug_n, config, "debug_n", 10))
    args.seed = pick_value(args.seed, config, "seed", None)

    if config_path.exists():
        print(f"Using config: {config_path}")

    allowed_views = set(v.strip().upper() for v in args.views.split(","))
    active_operations = [op.strip() for op in args.operations.split(",") if op.strip()]
    invalid_ops = [op for op in active_operations if op not in OPERATIONS]
    if invalid_ops:
        parser.error(f"Unknown operation(s): {invalid_ops}. Valid choices: {OPERATIONS}")
    dataset_root = Path(args.dataset_root)
    output_path = (
        Path(args.output).with_stem(Path(args.output).stem + "_debug")
        if args.debug
        else Path(args.output)
    )

    # Collect image files filtered by view suffix
    all_images = sorted(dataset_root.rglob("*.jpg"))
    images = []
    for img in all_images:
        meta = parse_filename(img.name)
        if not meta:
            continue
        view_key = meta["variant"] + meta["view"]
        if view_key in allowed_views:
            images.append(img)

    if not images:
        print(f"No images found under {dataset_root} matching views {allowed_views}")
        return

    if args.debug:
        if args.seed is not None:
            random.seed(args.seed)
        n = min(args.debug_n, len(images))
        images = random.sample(images, n)
        # Always start fresh in debug mode
        if output_path.exists():
            output_path.unlink()
        print(f"[DEBUG] Sampled {n} random images (seed={args.seed}) → output: {output_path}")

    # Load already-processed paths for resume support
    processed: set[str] = set()
    write_lock = threading.Lock()
    if output_path.exists():
        with open(output_path, "r") as f:
            for line in f:
                try:
                    processed.add(json.loads(line)["image_path"])
                except (json.JSONDecodeError, KeyError):
                    pass

    # Assign operations in round-robin rotation for balanced distribution
    pending_all = [img for img in images if str(img.relative_to(dataset_root)) not in processed]
    pending = [(img, active_operations[i % len(active_operations)]) for i, img in enumerate(pending_all)]
    print(f"Total images matching filters : {len(images)}")
    print(f"Already processed             : {len(processed)}")
    print(f"Remaining                     : {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    client = OpenAI()
    errors: list[str] = []

    def process_one(img_path: Path, operation: str):
        rel = str(img_path.relative_to(dataset_root))
        meta = parse_filename(img_path.name)
        try:
            result = call_gpt(client, img_path, args.model, operation)
            record = {
                "image_path": rel,
                "category": meta.get("category", ""),
                "scene_id": meta.get("scene_id", ""),
                "variant": meta.get("variant", ""),
                "view": meta.get("view", ""),
                "operation": result["operation"],
                "prompt": result["prompt"],
            }
            with write_lock:
                with open(output_path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            return rel, None
        except Exception as e:
            return rel, str(e)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, img, op): img for img, op in pending}
        with tqdm(total=len(pending), unit="img", desc="Generating prompts") as pbar:
            for future in as_completed(futures):
                rel, err = future.result()
                if err:
                    errors.append(f"{rel}: {err}")
                    tqdm.write(f"[ERROR] {rel}: {err}")
                # else:
                #     tqdm.write(f"[OK] {rel}")
                pbar.update(1)

    print(f"\nDone. Output: {output_path}")
    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  {e}")


if __name__ == "__main__":
    main()
