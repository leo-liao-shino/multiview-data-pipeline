"""
prepare_kontext_dataset.py

Unifies 3 source datasets into the Flux Kontext training format:

    kontext_unified/
    ├── target/          # after/edited images + multi-line .txt captions
    │   ├── 000001.jpg
    │   ├── 000001.txt
    │   └── ...
    ├── control/         # before/source images (matching filenames)
    │   ├── 000001.jpg
    │   └── ...
    └── masks/           # merged before+after masks (white = edit region)
        ├── 000001.png
        └── ...

Source datasets:
  1. /workspace/data/edit_new/          (1489 pairs, single prompt)
  2. /workspace/data/editing_v3/        (3 sub-datasets, multi-line captions)
  3. /workspace/data/multiview-qwen-edit/ (2026 pairs, single prompt)

For datasets with single prompts (edit_new, multiview-qwen-edit), GPT-4o is
called to expand each prompt into 12 multi-line variants matching the style
in editing_v3.

Masks are merged (union of before + after) to cover the full edit region.
Items without masks are included but won't have a mask file (uniform loss).

Usage:
    conda run -n kontext-prep python scripts/prepare_kontext_dataset.py
"""

import argparse
import asyncio
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm import tqdm

load_dotenv()

# ─── Defaults ─────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/workspace/data")
DEFAULT_OUTPUT = DATA_ROOT / "kontext_unified"

EDIT_NEW = DATA_ROOT / "edit_new"
EDITING_V3 = DATA_ROOT / "editing_v3"
MULTIVIEW = DATA_ROOT / "multiview-qwen-edit"

# GPT-4o caption expansion
MAX_CONCURRENT = 30
GPT_MODEL = "gpt-4o"

SYSTEM_PROMPT = """\
You are a caption variant generator for image editing instructions.
Given an editing instruction, generate 11 additional variants (12 total \
including the original) in 3 tiers:

Tier 1 (4 lines): Detailed rephrasings of the original. Use different verbs \
(replace, transform, modify, convert) and keep all detail.
Tier 2 (4 lines): Short versions. Same meaning but drop most descriptive \
detail. Keep the key action and object.
Tier 3 (4 lines): Very short versions. Use pronouns ("it") instead of object \
names where possible.

Rules:
- Output one variant per line, no numbering, no blank lines.
- First line must be the original prompt unchanged.
- Vary sentence structure and verb choice naturally.
- Do not add information not present in the original.
- Keep the editing intent identical across all variants."""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_image_as_jpg(src: Path, dst: Path):
    """Copy or convert an image to JPG at the destination."""
    if src.suffix.lower() in (".jpg", ".jpeg"):
        shutil.copy2(src, dst)
    else:
        img = Image.open(src).convert("RGB")
        img.save(dst, "JPEG", quality=95)


def extract_left_half(src: Path, dst: Path):
    """Extract the left half of a side-by-side compare image and save as JPG."""
    img = Image.open(src).convert("RGB")
    w, h = img.size
    left = img.crop((0, 0, w // 2, h))
    left.save(dst, "JPEG", quality=95)


def merge_masks(
    mask_before: Optional[Path],
    mask_after: Optional[Path],
    dst: Path,
):
    """Merge before and after masks via union (max) and save as PNG."""
    if mask_before and mask_before.exists() and mask_after and mask_after.exists():
        mb = np.array(Image.open(mask_before).convert("L"))
        ma = np.array(Image.open(mask_after).convert("L"))
        # Resize if shapes differ (use after mask's shape as reference)
        if mb.shape != ma.shape:
            mb_img = Image.fromarray(mb).resize(
                (ma.shape[1], ma.shape[0]), Image.NEAREST
            )
            mb = np.array(mb_img)
        merged = np.maximum(mb, ma)
    elif mask_before and mask_before.exists():
        merged = np.array(Image.open(mask_before).convert("L"))
    elif mask_after and mask_after.exists():
        merged = np.array(Image.open(mask_after).convert("L"))
    else:
        return False

    Image.fromarray(merged, mode="L").save(dst, "PNG")
    return True


# ─── Caption expansion ───────────────────────────────────────────────────────

async def expand_captions_batch(
    prompts: list[str],
    pbar: tqdm,
) -> list[str]:
    """Call GPT-4o to expand a list of prompts into multi-line variants."""
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def _expand_one(prompt: str) -> str:
        async with semaphore:
            try:
                resp = await client.chat.completions.create(
                    model=GPT_MODEL,
                    temperature=0.7,
                    max_tokens=500,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt.strip()},
                    ],
                )
                result = resp.choices[0].message.content.strip()
            except Exception as e:
                print(f"\n  GPT-4o error for '{prompt[:60]}...': {e}")
                result = prompt.strip()
            pbar.update(1)
            return result

    tasks = [_expand_one(p) for p in prompts]
    return await asyncio.gather(*tasks)



# ─── Dataset collectors ───────────────────────────────────────────────────────

def collect_editing_v3(sub_name: str) -> list[dict]:
    """Collect items from an editing_v3 sub-dataset. Captions already multi-line.

    Mask strategy:
      - partially_editing: merge before_thing_masks + after_thing_masks
      - partially_removal: before_thing_masks only (object removed, no after)
      - partially_staging: after_thing_masks only (object added, no before)
    """
    sub_dir = EDITING_V3 / sub_name
    before_dir = sub_dir / "before_editing_original_images"
    after_dir = sub_dir / "after_editing_images"
    caption_dir = sub_dir / "captions"
    before_mask_dir = sub_dir / "before_editing_thing_masks"
    after_mask_dir = sub_dir / "after_editing_thing_masks"

    items = []
    for cap_file in sorted(caption_dir.glob("*.txt")):
        if cap_file.name.startswith("._"):
            continue
        stem = cap_file.stem
        before_img = before_dir / f"{stem}.jpg"
        after_img = after_dir / f"{stem}.jpg"

        if not before_img.exists() or not after_img.exists():
            continue

        caption_text = cap_file.read_text(encoding="utf-8", errors="replace").strip()

        # Resolve mask paths
        mask_before = before_mask_dir / f"{stem}.png"
        mask_after = after_mask_dir / f"{stem}.png"
        if not mask_before.exists():
            mask_before = None
        if not mask_after.exists():
            mask_after = None

        items.append({
            "before": before_img,
            "after": after_img,
            "caption": caption_text,
            "needs_expansion": False,
            "source": f"editing_v3/{sub_name}",
            "mask_before": mask_before,
            "mask_after": mask_after,
        })
    print(f"  {sub_name}: {len(items)} pairs")
    return items


def collect_edit_new() -> list[dict]:
    """Collect items from edit_new. Single prompts need GPT-4o expansion."""
    mask_dir = EDIT_NEW / "masks"
    items = []
    records_path = EDIT_NEW / "records.jsonl"
    with open(records_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            before_img = Path(rec["before_path"])
            after_img = Path(rec["output_path"])
            if not before_img.exists() or not after_img.exists():
                continue
            # First line is the actual editing instruction
            prompt = rec["prompt"].strip().split("\n")[0].strip()
            rid = rec["id"]

            mask_before = mask_dir / f"{rid}_mask_before.png"
            mask_after = mask_dir / f"{rid}_mask_after.png"
            if not mask_before.exists():
                mask_before = None
            if not mask_after.exists():
                mask_after = None

            items.append({
                "before": before_img,
                "after": after_img,
                "caption": prompt,
                "needs_expansion": True,
                "source": "edit_new",
                "mask_before": mask_before,
                "mask_after": mask_after,
            })
    print(f"  edit_new: {len(items)} pairs")
    return items


def collect_multiview() -> list[dict]:
    """Collect items from multiview-qwen-edit. Before = left half of compare."""
    mask_dir = MULTIVIEW / "multiview-qwen-edit-masks"
    items = []
    skipped = 0
    results_path = MULTIVIEW / "results.jsonl"
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue

            # Edited image: check local dir first, then original path
            output_fname = Path(rec["output_path"]).name
            edited_img = MULTIVIEW / output_fname
            if not edited_img.exists():
                edited_img = Path(rec["output_path"])
            if not edited_img.exists():
                continue

            # Compare image (side-by-side: left=before, right=after)
            compare_fname = Path(rec["compare_path"]).name
            compare_img = MULTIVIEW / compare_fname
            if not compare_img.exists():
                compare_img = Path(rec["compare_path"])
            if not compare_img.exists():
                continue

            prompt = rec["prompt"].strip()

            # Mask filenames use the edited image name as prefix
            edited_stem = Path(output_fname).stem
            mask_before = mask_dir / f"{edited_stem}_mask_before.png"
            mask_after = mask_dir / f"{edited_stem}_mask_after.png"
            has_mask = mask_before.exists() or mask_after.exists()

            # Only include high-quality pairs that have masks
            if not has_mask:
                skipped += 1
                continue

            if not mask_before.exists():
                mask_before = None
            if not mask_after.exists():
                mask_after = None

            items.append({
                "before": compare_img,
                "before_is_compare": True,
                "after": edited_img,
                "caption": prompt,
                "needs_expansion": True,
                "source": "multiview",
                "mask_before": mask_before,
                "mask_after": mask_after,
            })
    print(f"  multiview-qwen-edit: {len(items)} pairs ({skipped} skipped, no mask)")
    return items


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(output_dir: Path):
    target_dir = output_dir / "target"
    control_dir = output_dir / "control"
    mask_out_dir = output_dir / "masks"
    target_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Flux Kontext Dataset Preparation")
    print("=" * 60)

    # ── Step 1: Collect ──
    print("\n[1/3] Collecting dataset items...")
    all_items = []
    all_items.extend(collect_editing_v3("partially_editing"))
    all_items.extend(collect_editing_v3("partially_removal"))
    all_items.extend(collect_editing_v3("partially_staging"))
    all_items.extend(collect_edit_new())
    all_items.extend(collect_multiview())
    print(f"\n  Total: {len(all_items)} pairs")

    # ── Step 2: Expand captions ──
    to_expand = [i for i in all_items if i["needs_expansion"]]
    print(f"\n[2/3] Expanding {len(to_expand)} captions with GPT-4o...")
    if to_expand:
        prompts = [item["caption"] for item in to_expand]
        with tqdm(total=len(prompts), desc="  GPT-4o") as pbar:
            expanded = await expand_captions_batch(prompts, pbar)
        for item, new_caption in zip(to_expand, expanded):
            item["caption"] = new_caption
    print(f"  Done.")

    # ── Step 3: Write unified dataset ──
    print(f"\n[3/3] Writing unified dataset to {output_dir}...")
    manifest = []
    mask_count = 0

    for idx, item in enumerate(tqdm(all_items, desc="  Writing")):
        uid = f"{idx:06d}"
        target_img_path = target_dir / f"{uid}.jpg"
        control_img_path = control_dir / f"{uid}.jpg"
        caption_path = target_dir / f"{uid}.txt"
        mask_path = mask_out_dir / f"{uid}.png"

        # Save after/target image
        save_image_as_jpg(item["after"], target_img_path)

        # Save before/control image
        if item.get("before_is_compare"):
            extract_left_half(item["before"], control_img_path)
        else:
            save_image_as_jpg(item["before"], control_img_path)

        # Save caption
        caption_path.write_text(item["caption"], encoding="utf-8")

        # Merge and save mask
        has_mask = merge_masks(
            item.get("mask_before"),
            item.get("mask_after"),
            mask_path,
        )
        if has_mask:
            mask_count += 1

        manifest.append({
            "id": uid,
            "source": item["source"],
            "has_mask": has_mask,
            "original_prompt": item["caption"].split("\n")[0],
        })

    # Save manifest for reference
    manifest_path = output_dir / "manifest.jsonl"
    with open(manifest_path, "w") as f:
        for rec in manifest:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Done! {len(all_items)} pairs written.")
    print(f"  Target images:  {target_dir}")
    print(f"  Control images: {control_dir}")
    print(f"  Masks:          {mask_out_dir} ({mask_count}/{len(all_items)} have masks)")
    print(f"  Manifest:       {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare unified Kontext dataset")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for unified dataset",
    )
    args = parser.parse_args()
    asyncio.run(main(args.output_dir))
