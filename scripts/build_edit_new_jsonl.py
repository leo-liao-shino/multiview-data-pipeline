"""
build_edit_new_jsonl.py

Scans the /workspace/data/edit_new/ folder tree and produces a JSONL file
compatible with generate_edit_masks.py.

Folder structure:
  edit_new/
    before_edit_replace/{ID}.jpg          # before images for edit task
    replace_after_edit/{ID}.png           # after images for edit task
    replace_after_edit/{ID}.txt           # prompts for edit task
    before_resize/{ID}.jpg                # before images for resize task
    before_resize/{ID}.txt                # prompts for resize task
    after_resize/{ID}_1.png              # after images for resize (base IDs)
    after_resize/{ID}_2.png              # after images for resize (_2 variant)

Usage:
    python scripts/build_edit_new_jsonl.py \
        --data_dir /workspace/data/edit_new \
        --output /workspace/data/edit_new/records.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def build_edit_records(data_dir: str) -> list[dict]:
    """Build records for the edit (replace) task."""
    before_dir = os.path.join(data_dir, "before_edit_replace")
    after_dir = os.path.join(data_dir, "replace_after_edit")
    records = []

    txt_files = sorted(Path(after_dir).glob("*.txt"))
    for txt_path in txt_files:
        stem = txt_path.stem
        before_path = os.path.join(before_dir, f"{stem}.jpg")
        after_path = os.path.join(after_dir, f"{stem}.png")

        if not os.path.exists(before_path) or not os.path.exists(after_path):
            print(f"  [SKIP] edit {stem}: missing before or after image")
            continue

        prompt = txt_path.read_text().strip().split("\n")[0]

        records.append({
            "id": stem,
            "task": "edit",
            "operation": "Replace",
            "prompt": prompt,
            "before_path": before_path,
            "output_path": after_path,
            "status": "ok",
        })

    return records


def build_resize_records(data_dir: str) -> list[dict]:
    """Build records for the resize task."""
    before_dir = os.path.join(data_dir, "before_resize")
    after_dir = os.path.join(data_dir, "after_resize")
    records = []

    txt_files = sorted(Path(before_dir).glob("*.txt"))
    for txt_path in txt_files:
        stem = txt_path.stem  # e.g. "00004" or "00004_2"
        before_path = os.path.join(before_dir, f"{stem}.jpg")

        # Determine the after filename:
        # variant IDs (e.g. 00004_2) -> {stem}.png (keep as-is)
        # base IDs: try {stem}_1.png first, fall back to {stem}.png
        if "_" in stem:
            after_path = os.path.join(after_dir, f"{stem}.png")
        else:
            after_path = os.path.join(after_dir, f"{stem}_1.png")
            if not os.path.exists(after_path):
                after_path = os.path.join(after_dir, f"{stem}.png")

        if not os.path.exists(before_path):
            print(f"  [SKIP] resize {stem}: missing before image")
            continue
        if not os.path.exists(after_path):
            print(f"  [SKIP] resize {stem}: missing after image {after_path}")
            continue

        prompt = txt_path.read_text().strip().split("\n")[0]

        records.append({
            "id": stem,
            "task": "resize",
            "operation": "Resize",
            "prompt": prompt,
            "before_path": before_path,
            "output_path": after_path,
            "status": "ok",
        })

    return records


def main():
    parser = argparse.ArgumentParser(
        description="Build JSONL from edit_new folder structure"
    )
    parser.add_argument("--data_dir", default="/workspace/data/edit_new")
    parser.add_argument("--output", default="/workspace/data/edit_new/records.jsonl")
    parser.add_argument("--task", choices=["edit", "resize", "all"], default="all",
                        help="Which task to include")
    args = parser.parse_args()

    records = []
    if args.task in ("edit", "all"):
        edit_recs = build_edit_records(args.data_dir)
        print(f"Edit records: {len(edit_recs)}")
        records.extend(edit_recs)
    if args.task in ("resize", "all"):
        resize_recs = build_resize_records(args.data_dir)
        print(f"Resize records: {len(resize_recs)}")
        records.extend(resize_recs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Total: {len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
