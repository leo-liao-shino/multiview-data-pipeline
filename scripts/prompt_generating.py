"""
Generate Add/Delete/Replace furniture editing prompts for furnished room images
using GPT-4o vision. Results are saved as a JSONL file.

Usage:
    conda activate data_process
    python scripts/prompt_generating.py [--dataset-root DIR] [--output FILE]
                                [--model MODEL] [--views A2,B2]
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

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DATASET_ROOT = "/workspace/all_multiview_datasets"
DEFAULT_OUTPUT = "/workspace/data_processing/resume/prompts.jsonl"
DEFAULT_MODEL = "gpt-4o"
# Process only these view suffixes by default (one representative per scene variant)
DEFAULT_VIEWS = {"A2", "B2"}

SYSTEM_PROMPT = """
You are an expert interior designer. You will be given a photograph of a furnished room and a required operation type.
Your task is to generate ONE concise, specific furniture-editing instruction for image-editing purposes.

Operation definitions:
- Add:     introduce a new piece of furniture that is NOT currently present
- Delete:  remove an existing piece of furniture that IS currently present
- Replace: swap an existing piece of furniture for a different style or type

Rules:
- You MUST use the exact operation type specified in the user message.
- Your instruction MUST be relevant to the specific room in the photo.
- Be specific about the item (size, material, color, style when relevant).
- Be specific about location when it aids clarity (e.g. "against the left wall").
- Keep the prompt under 20 words.
- Respond ONLY with valid JSON — no explanation, no markdown.

Response format:
{"operation": "<required operation>", "prompt": "<instruction>"}

Examples:
{"operation": "Add", "prompt": "Add a queen-sized bed with a dark wooden headboard against the main wall"}
{"operation": "Delete", "prompt": "Remove the small round bedside table on the right"}
{"operation": "Replace", "prompt": "Replace the wooden dresser with a sleek white built-in wardrobe"}
"""

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


OPERATIONS = ["Add", "Delete", "Replace"]


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
                max_tokens=100,
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
                raise ValueError(f"Unknown operation: {parsed['operation']}")
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
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT, help="Root directory of image dataset")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model to use")
    parser.add_argument(
        "--views",
        default=",".join(sorted(DEFAULT_VIEWS)),
        help="Comma-separated list of view suffixes to include (e.g. A2,B2)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel API workers (default: 4)")
    parser.add_argument("--debug", action="store_true", help="Debug mode: sample a few random images and write to a separate JSONL")
    parser.add_argument("--debug-n", type=int, default=10, help="Number of images to sample in debug mode (default: 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible debug sampling")
    args = parser.parse_args()

    allowed_views = set(v.strip().upper() for v in args.views.split(","))
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
    pending = [(img, OPERATIONS[i % len(OPERATIONS)]) for i, img in enumerate(pending_all)]
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
