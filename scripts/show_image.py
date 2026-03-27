"""
Load a JSONL file, filter entries where accept=True,
and display compare images with lazy loading + LRU cache.
Left/Right arrow keys to navigate, Q to quit.
"""

import json
import sys
import argparse
from pathlib import Path
from functools import lru_cache

try:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg


def load_accepted(jsonl_path: str) -> list[dict]:
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("accept") is True:
                entries.append(record)
    return entries


def show_images(entries: list[dict], image_base: str = "", cache_size: int = 16) -> None:
    total = len(entries)
    if total == 0:
        print("No accepted entries found.")
        return

    def resolve_path(entry):
        if image_base:
            return Path(image_base) / Path(entry["compare_path"]).name
        return Path(entry["compare_path"])

    @lru_cache(maxsize=cache_size)
    def load_image(idx):
        path = resolve_path(entries[idx])
        if not path.exists():
            print(f"⚠ Not found: {path}")
            return None
        return mpimg.imread(str(path))

    state = {"idx": 0}

    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(top=0.91, bottom=0.02, left=0.02, right=0.98)
    ax.axis("off")
    title = fig.suptitle("", fontsize=11, wrap=True)
    counter = fig.text(0.5, 0.955, "", ha="center", fontsize=10, color="#555555")

    img_display = ax.imshow(load_image(0))

    def update(idx):
        img = load_image(idx)
        if img is None:
            return
        entry = entries[idx]
        img_display.set_data(img)
        img_display.set_extent([0, img.shape[1], img.shape[0], 0])
        ax.set_xlim(0, img.shape[1])
        ax.set_ylim(img.shape[0], 0)
        title.set_text(f"{entry['operation']}  |  Scene {entry['scene_id']}\n{entry['prompt']}")
        counter.set_text(f"{idx + 1} / {total}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("right", "d"):
            state["idx"] = (state["idx"] + 1) % total
        elif event.key in ("left", "a"):
            state["idx"] = (state["idx"] - 1) % total
        elif event.key in ("q", "escape"):
            plt.close()
            return
        update(state["idx"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Browse accepted compare images from a JSONL file.")
    parser.add_argument("jsonl", help="Path to the .jsonl file")
    parser.add_argument("--base", default="", help="Base directory for images (e.g. /T9/multiview-qwen-edit)")
    parser.add_argument("--cache", type=int, default=16, help="Number of images to keep in memory (default: 16)")
    args = parser.parse_args()

    entries = load_accepted(args.jsonl)
    print(f"Loaded {len(entries)} accepted entries.")
    show_images(entries, image_base=args.base, cache_size=args.cache)


if __name__ == "__main__":
    main()