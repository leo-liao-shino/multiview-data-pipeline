#!/usr/bin/env python3
"""
image_review.py — Keyboard-driven image selection tool

Usage:
    python image_review.py --images ./images --meta data.jsonl --out selected.jsonl

Controls:
    Y / Enter  → keep (writes entry to output JSONL)
    N / Space  → skip
    U          → undo last decision
    Q          → quit and save progress
    ←/→        → go back/forward without changing decision

Assumptions:
    - Your JSONL has one JSON object per line
    - Each object has a field that matches the image filename
      (configure IMAGE_KEY below — default: "file_name")
"""

import argparse
import json
import sys
import os
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
IMAGE_KEY = "compare_path"   # field in JSONL that holds the image filename
# ─────────────────────────────────────────────────────────────────────────────

try:
    from PIL import Image
    import tkinter as tk
    from tkinter import Label, Frame, StringVar
    from PIL import ImageTk
    HAS_GUI = True
except ImportError:
    HAS_GUI = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def append_entry(entry: dict, path: Path):
    """Write a single entry immediately to the output file."""
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def find_image(images_dir: Path, filename: str) -> Path | None:
    candidate = images_dir / filename
    if candidate.exists():
        return candidate
    # Try stripping directory prefix from filename
    candidate = images_dir / Path(filename).name
    if candidate.exists():
        return candidate
    return None


# ── CLI (terminal) fallback — no GUI ────────────────────────────────────────
def run_cli(entries, images_dir, out_path, start_index=0):
    """Minimal terminal mode using cv2 or PIL for display."""
    if not HAS_CV2 and not HAS_GUI:
        print("❌  Neither OpenCV nor Pillow+Tkinter found.")
        print("    Install one:  pip install pillow  OR  pip install opencv-python")
        sys.exit(1)

    history  = []   # list of (index, accepted: bool)
    reviewed = 0
    accepted = 0
    i = start_index

    print("\n🖼  Image Review  |  Y=keep  N=skip  U=undo  Q=quit\n")

    while i < len(entries):
        entry = entries[i]
        filename = entry.get(IMAGE_KEY, "")
        img_path = find_image(images_dir, filename)

        print(f"[{i+1}/{len(entries)}]  reviewed:{reviewed} accepted:{accepted}  {filename}", end="  ")

        if img_path is None:
            print("⚠️  image not found, skipping")
            i += 1
            continue

        # Display image
        if HAS_CV2:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w*scale), int(h*scale)))
                cv2.imshow("Image Review  |  Y=keep  N=skip  U=undo  Q=quit", img)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
            else:
                print("⚠️  could not load image")
                key = ord('n')
        else:
            print(f"\n    Path: {img_path}")
            key_raw = input("    Y/N/U/Q > ").strip().lower()
            key = ord(key_raw[0]) if key_raw else ord('n')

        if key in (ord('y'), ord('Y'), 13):
            print("✅  kept")
            append_entry({**entry, "accept": True}, out_path)
            history.append((i, True))
            reviewed += 1; accepted += 1; i += 1
        elif key in (ord('n'), ord('N'), 32):
            print("❌  skipped")
            append_entry({**entry, "accept": False}, out_path)
            history.append((i, False))
            reviewed += 1; i += 1
        elif key in (ord('u'), ord('U')):
            if history:
                last_i, last_acc = history.pop()
                # Truncate last line from file
                with open(out_path, "rb+") as f:
                    f.seek(0, 2)
                    pos = f.tell()
                    while pos > 0:
                        pos -= 1
                        f.seek(pos)
                        if f.read(1) == b"\n" and pos != f.tell() - 1:
                            break
                    f.truncate(pos + 1 if pos > 0 else 0)
                reviewed -= 1
                if last_acc:
                    accepted -= 1
                print(f"↩️  undid decision for [{last_i+1}]")
                i = last_i
            else:
                print("⚠️  nothing to undo")
        elif key in (ord('q'), ord('Q'), 27):
            print("\n💾  Quitting...")
            break
        else:
            print("?  (press Y, N, U, or Q)")

    print(f"\n✅  Done. {accepted} accepted, {reviewed-accepted} rejected — saved to {out_path}")
    print(f"   Reviewed {reviewed} / {len(entries)} images.")


# ── Tkinter GUI ──────────────────────────────────────────────────────────────
def run_gui(entries, images_dir, out_path, start_index=0):
    history = []   # list of (index, accepted, was_reviewed, was_accepted) for undo
    existing_count = len(load_jsonl(out_path)) if out_path.exists() else 0
    existing_accepted = sum(1 for e in (load_jsonl(out_path) if out_path.exists() else []) if e.get("accept"))
    state   = {"i": start_index, "reviewed": existing_count, "accepted": existing_accepted}

    root = tk.Tk()
    root.title("Image Review")
    root.configure(bg="#111")
    root.geometry("900x750")

    # Layout
    top = Frame(root, bg="#111")
    top.pack(fill="x", padx=20, pady=(16, 0))

    status_var = StringVar()
    status_lbl = Label(top, textvariable=status_var, bg="#111", fg="#aaa",
                       font=("Courier", 12))
    status_lbl.pack(side="left")

    hint_lbl = Label(top, text="Y=keep  N=skip  U=undo  ←/→=browse  +/-=zoom  0=reset  Q=quit",
                     bg="#111", fg="#555", font=("Courier", 11))
    hint_lbl.pack(side="right")

    img_lbl = Label(root, bg="#111")
    img_lbl.pack(expand=True, fill="both", padx=20, pady=10)

    # Meta panel: operation + prompt
    meta_frame = Frame(root, bg="#1a1a1a")
    meta_frame.pack(fill="x", padx=20, pady=(0, 4))

    op_var = StringVar()
    op_lbl = Label(meta_frame, textvariable=op_var, bg="#1a1a1a", fg="#f0a500",
                   font=("Courier", 10, "bold"), anchor="w", padx=12, pady=4)
    op_lbl.pack(fill="x")

    prompt_var = StringVar()
    prompt_lbl = Label(meta_frame, textvariable=prompt_var, bg="#1a1a1a", fg="#7ecfff",
                       font=("Courier", 10), wraplength=860, justify="left",
                       anchor="w", padx=12, pady=4)
    prompt_lbl.pack(fill="x")

    bar_frame = Frame(root, bg="#222", height=6)
    bar_frame.pack(fill="x", padx=20, pady=(0, 16))
    bar_inner = Frame(bar_frame, bg="#3ecf8e", height=6)
    bar_inner.place(x=0, y=0, relheight=1, relwidth=0)

    # Zoom state
    zoom = {"scale": 1.0, "pil": None, "pil_orig": None}
    ZOOM_STEP = 0.15
    ZOOM_MIN  = 0.2
    ZOOM_MAX  = 5.0
    CHROME_H  = 160  # pixels reserved for status bar + meta + progress

    def fit_to_window():
        """Scale image to fit current window size and reset zoom to 1.0."""
        pil_orig = zoom["pil_orig"]
        if pil_orig is None:
            return
        root.update_idletasks()
        avail_w = root.winfo_width()  - 40
        avail_h = root.winfo_height() - CHROME_H - 40
        if avail_w <= 0 or avail_h <= 0:
            return
        scale = min(avail_w / pil_orig.width, avail_h / pil_orig.height, 1.0)
        base  = pil_orig.resize((int(pil_orig.width * scale), int(pil_orig.height * scale)),
                                 Image.LANCZOS) if scale < 1.0 else pil_orig.copy()
        zoom["pil"]   = base
        zoom["scale"] = 1.0
        render_zoom()

    def render_zoom():
        pil = zoom["pil"]
        if pil is None:
            return
        w = int(pil.width  * zoom["scale"])
        h = int(pil.height * zoom["scale"])
        resized = pil.resize((w, h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)
        img_lbl.config(image=tk_img, text="")
        img_lbl._image = tk_img

    def zoom_in(event=None):
        if zoom["scale"] < ZOOM_MAX:
            zoom["scale"] = min(ZOOM_MAX, zoom["scale"] + ZOOM_STEP)
            render_zoom()

    def zoom_out(event=None):
        if zoom["scale"] > ZOOM_MIN:
            zoom["scale"] = max(ZOOM_MIN, zoom["scale"] - ZOOM_STEP)
            render_zoom()

    def zoom_reset(event=None):
        zoom["scale"] = 1.0
        render_zoom()

    # Refit image whenever window is resized (covers moving to a different monitor)
    _resize_job = [None]
    def on_resize(event):
        if event.widget is not root:
            return
        if _resize_job[0]:
            root.after_cancel(_resize_job[0])
        _resize_job[0] = root.after(150, fit_to_window)  # debounce 150ms

    root.bind("<Configure>", on_resize)

    def show_current():
        i = state["i"]
        if i >= len(entries):
            finish()
            return
        entry = entries[i]
        filename = entry.get(IMAGE_KEY, "")
        img_path = find_image(images_dir, filename)

        # Status
        status_var.set(f"[{i+1}/{len(entries)}]  reviewed: {state['reviewed']}  accepted: {state['accepted']}  —  {filename}")

        # Progress bar
        prog = (i + 1) / len(entries)
        bar_inner.place(relwidth=prog)

        # Meta: operation and prompt specifically
        op_val = entry.get("operation", "—")
        prompt_val = entry.get("prompt", "—")
        op_var.set(f"operation: {op_val}")
        prompt_var.set(f"prompt: {prompt_val}")

        # Image — fit to current window size, preserving aspect ratio
        if img_path:
            try:
                pil = Image.open(img_path)
                zoom["pil_orig"] = pil  # store original for window-resize re-renders
                zoom["scale"]    = 1.0
                fit_to_window()
            except Exception as e:
                zoom["pil"]      = None
                zoom["pil_orig"] = None
                img_lbl.config(image="", text=f"⚠️ {e}", fg="red")
        else:
            zoom["pil"]      = None
            zoom["pil_orig"] = None
            img_lbl.config(image="", text="⚠️  Image not found", fg="#ff6b6b",
                           font=("Courier", 14))

    # decisions: ordered dict of {filename -> entry_with_accept}
    # Using a list to preserve order, dict for O(1) lookup
    decisions      = {}   # filename -> entry dict
    decisions_order = []  # filenames in review order (for file write)

    # Seed from existing file (resume case)
    if out_path.exists():
        for e in load_jsonl(out_path):
            fn = e.get(IMAGE_KEY, "")
            if fn not in decisions:
                decisions_order.append(fn)
            decisions[fn] = e

    def write_decisions():
        with open(out_path, "w") as f:
            for fn in decisions_order:
                if fn in decisions:
                    f.write(json.dumps(decisions[fn]) + "\n")

    def decide(accepted: bool):
        i = state["i"]
        if i >= len(entries):
            return
        fn    = entries[i].get(IMAGE_KEY, "")
        entry = {**entries[i], "accept": accepted}
        # Track counts before update
        was_reviewed = fn in decisions
        was_accepted = decisions[fn].get("accept") if was_reviewed else None
        # Update decisions
        if not was_reviewed:
            decisions_order.append(fn)
            state["reviewed"] += 1
        # Adjust accepted count
        if not was_reviewed:
            if accepted:
                state["accepted"] += 1
        else:
            if was_accepted and not accepted:
                state["accepted"] -= 1
            elif not was_accepted and accepted:
                state["accepted"] += 1
        decisions[fn] = entry
        write_decisions()
        history.append((i, accepted, was_reviewed, was_accepted))
        state["i"] += 1
        show_current()

    def keep():
        decide(True)

    def skip():
        decide(False)

    def undo():
        if not history:
            return
        last_i, last_accepted, was_reviewed, was_accepted = history.pop()
        fn = entries[last_i].get(IMAGE_KEY, "")
        if was_reviewed:
            # Restore previous accept value
            decisions[fn] = {**entries[last_i], "accept": was_accepted}
            if last_accepted and not was_accepted:
                state["accepted"] -= 1
            elif not last_accepted and was_accepted:
                state["accepted"] += 1
        else:
            # Entry didn't exist before — remove it
            decisions.pop(fn, None)
            decisions_order.remove(fn)
            state["reviewed"] -= 1
            if last_accepted:
                state["accepted"] -= 1
        write_decisions()
        state["i"] = last_i
        show_current()

    def go_back():
        """Browse back without changing any decision."""
        if state["i"] > 0:
            state["i"] -= 1
            show_current()

    def go_forward():
        """Browse forward without changing any decision."""
        if state["i"] < len(entries) - 1:
            state["i"] += 1
            show_current()

    def finish():
        status_var.set(f"✅  Done! {state['accepted']} accepted, {state['reviewed'] - state['accepted']} rejected — saved to {out_path}")
        op_var.set("")
        prompt_var.set("")
        img_lbl.config(image="", text=f"All done!\n{state['accepted']} accepted / {state['reviewed']} reviewed.",
                       font=("Courier", 18), fg="#3ecf8e")
        root.unbind("<Key>")

    def quit_save(event=None):
        root.destroy()

    def on_key(event):
        k = event.keysym  # keep original case for Left/Right
        kl = k.lower()
        if kl in ("y", "return"):
            keep()
        elif kl in ("n", "space"):
            skip()
        elif kl == "u":
            undo()
        elif k == "Left":
            go_back()
        elif k == "Right":
            go_forward()
        elif k in ("plus", "equal", "KP_Add"):
            zoom_in()
        elif k in ("minus", "KP_Subtract"):
            zoom_out()
        elif k == "0":
            zoom_reset()
        elif kl in ("q", "escape"):
            quit_save()

    def on_scroll(event):
        # macOS: event.delta is ±1; Windows/Linux: ±120
        if event.delta > 0 or event.num == 4:
            zoom_in()
        else:
            zoom_out()

    root.bind("<Key>", on_key)
    root.bind("<MouseWheel>", on_scroll)   # macOS / Windows
    root.bind("<Button-4>",   on_scroll)   # Linux scroll up
    root.bind("<Button-5>",   on_scroll)   # Linux scroll down
    root.protocol("WM_DELETE_WINDOW", quit_save)

    show_current()
    root.mainloop()


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    global IMAGE_KEY
    parser = argparse.ArgumentParser(
        description="Keyboard-driven image review: Y=keep, N=skip → output JSONL"
    )
    parser.add_argument("--images", required=True,
                        help="Directory containing images")
    parser.add_argument("--meta",   required=True,
                        help="Input JSONL with metadata")
    parser.add_argument("--out",    required=True,
                        help="Output JSONL for selected entries")
    parser.add_argument("--start",  type=int, default=0,
                        help="Start from index N (0-based, for resuming)")
    parser.add_argument("--key",    default=IMAGE_KEY,
                        help=f"JSONL field containing filename (default: compare_path)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    meta_path  = Path(args.meta)
    out_path   = Path(args.out)

    if not images_dir.is_dir():
        print(f"❌  Images dir not found: {images_dir}"); sys.exit(1)
    if not meta_path.exists():
        print(f"❌  JSONL not found: {meta_path}"); sys.exit(1)

    IMAGE_KEY = args.key

    entries = load_jsonl(meta_path)
    print(f"📂  Loaded {len(entries)} entries from {meta_path}")

    # Resume: skip entries already in output file
    start = args.start
    if out_path.exists() and start == 0:
        existing = load_jsonl(out_path)
        accepted = sum(1 for e in existing if e.get("accept"))
        print(f"⚠️  Output file already exists: {len(existing)} reviewed ({accepted} accepted, {len(existing)-accepted} rejected).")
        ans = input("   Resume (skip already reviewed)? (y) or overwrite and start fresh? (n): ").strip().lower()
        if ans == "y":
            already_seen = {e.get(IMAGE_KEY) for e in existing}
            for idx, e in enumerate(entries):
                if e.get(IMAGE_KEY) not in already_seen:
                    start = idx
                    break
            print(f"   Resuming from index {start} ({len(already_seen)} already reviewed)...")
        else:
            out_path.unlink()  # delete old output, start fresh

    if HAS_GUI:
        run_gui(entries, images_dir, out_path, start)
    else:
        run_cli(entries, images_dir, out_path, start)


if __name__ == "__main__":
    main()