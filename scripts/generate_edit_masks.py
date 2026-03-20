"""
generate_edit_masks.py

Pipeline: Qwen2.5-VL (local) → GroundingDINO → SAM 2 → dilated binary masks

Must be run from inside the Grounded-SAM-2 repo directory:
    cd /workspace/multiview-data-pipeline/Grounded-SAM-2

    PYTHONPATH=/workspace/multiview-data-pipeline/Grounded-SAM-2 \
    python ../scripts/generate_edit_masks.py \
        --jsonl /workspace/data/qwen-output/results.jsonl \
        --image_root /workspace/data/all-multiview-datasets \
        --output_dir /workspace/data/masks \
        --vlm_model Qwen/Qwen2.5-VL-72B-Instruct \
        --device cuda \
        --expand_pixels 30 \
        --limit 5

Dependencies (on top of Grounded-SAM-2 env):
    pip install transformers==4.36.2  (already pinned)
    pip install qwen-vl-utils
    pip install accelerate
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 1. Qwen2.5-VL — local VLM for grounding phrase extraction
# ─────────────────────────────────────────────────────────────

VLM_SYSTEM = """You are a computer vision assistant helping localize edits in image pairs.
You will be given a BEFORE image, an AFTER image, and the edit instruction that was applied.

Your task: identify every object or surface that was visually changed.

Return ONLY valid JSON, no explanation, no markdown:
{
  "changed_regions": ["<short grounding phrase>", ...],
  "change_type": "local" | "global"
}

Rules:
- Each phrase must be a short, specific noun phrase suitable for open-vocabulary detection.
  Good: "rattan sofa", "wooden coffee table top", "left armchair", "pendant light"
  Bad:  "furniture", "the thing on the left", "it"
- List ALL visually affected objects, even if the prompt only names one.
  E.g. if a sofa has two cushions that changed color, list "sofa cushion" not just "sofa".
- Use 1-6 phrases maximum.
- change_type is "global" only if the change covers most of the scene (wall, floor, ceiling).
- If nothing visually changed (model failed), return {"changed_regions": [], "change_type": "local"}.
"""

VLM_USER_TMPL = 'Edit instruction: "{prompt}"\nBEFORE image is first, AFTER image is second. What changed?'


def load_vlm(model_name: str, device: str):
    """Load Qwen2.5-VL model and processor."""
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError:
        sys.exit("Install transformers >= 4.49: pip install -U transformers")

    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        sys.exit("Install qwen-vl-utils: pip install qwen-vl-utils")

    print(f"  Loading {model_name} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,   # change from float16 to bfloat16
        device_map="auto",
        attn_implementation="eager",  # disable flash attention
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def call_vlm(
    model,
    processor,
    before_path: str,
    after_path: str,
    prompt: str,
    max_pixels: int = 1280 * 720,
) -> dict:
    """
    Run Qwen2.5-VL on both images + edit prompt.
    Returns {"changed_regions": [...], "change_type": "local"|"global"}.
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {"role": "system", "content": VLM_SYSTEM},
        {
            "role": "user",
            "content": [
                {"type": "text",  "text": "BEFORE:"},
                {"type": "image", "image": f"file://{os.path.abspath(before_path)}","max_pixels": max_pixels,},
                {"type": "text",  "text": "AFTER:"},
                {"type": "image", "image": f"file://{os.path.abspath(after_path)}","max_pixels": max_pixels,},
                {"type": "text",  "text": VLM_USER_TMPL.format(prompt=prompt)},
            ],
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the newly generated tokens
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    # Strip markdown fences if present
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())

    try:
        result = json.loads(raw)
        assert "changed_regions" in result
        return result
    except Exception:
        # Fallback: return empty so we fall through to diff mask
        return {"changed_regions": [], "change_type": "local"}


# ─────────────────────────────────────────────────────────────
# 2. Regex fallback parser (used when VLM is disabled)
# ─────────────────────────────────────────────────────────────

GLOBAL_OPERATIONS = {"LargeElement"}


def extract_grounding_phrases_regex(prompt: str, operation: str) -> tuple[list[str], str]:
    if operation in GLOBAL_OPERATIONS:
        return [], "global"

    clauses = [c.strip() for c in prompt.split(";")]
    phrases = []
    for clause in clauses:
        match = re.search(
            r"(?:change|replace|remove|scale(?:\s+\w+)?|make|paint|add|convert|swap|resize)\s+"
            r"(?:the\s+|a\s+|an\s+)?"
            r"(.+?)"
            r"(?:\s+(?:to|with|from|by|in|into|of|behind|above|below|next|near|on|at)\b|$)",
            clause, re.IGNORECASE,
        )
        if match:
            phrase = match.group(1).strip().lower()
            phrase = re.sub(
                r"\b(slightly|about|oversized|dominant|existing|current|small|large|big|tall|short|left|right)\b",
                "", phrase,
            ).strip()
            phrase = re.sub(r"\s{2,}", " ", phrase)
            if phrase:
                phrases.append(phrase)

    seen: set[str] = set()
    phrases = [p for p in phrases if not (p in seen or seen.add(p))]  # type: ignore
    return phrases or [prompt.lower()], "local"


# ─────────────────────────────────────────────────────────────
# 3. GroundingDINO
# ─────────────────────────────────────────────────────────────

def load_grounding_dino(config_path: str, weights_path: str, device: str):
    try:
        from grounding_dino.groundingdino.util.inference import load_model
    except ImportError:
        sys.exit(
            "GroundingDINO not found. Run from inside Grounded-SAM-2 repo with PYTHONPATH set."
        )
    return load_model(config_path, weights_path, device=device)


def run_grounding_dino(
    model,
    image_path: str,
    phrases: list[str],
    device: str,
    box_threshold: float = 0.30,
    text_threshold: float = 0.25,
) -> list[list[int]]:
    from grounding_dino.groundingdino.util.inference import load_image, predict

    # Must be lowercased and end with a dot
    caption = ". ".join(p.lower().rstrip(".") for p in phrases) + "."

    image_source, image_tensor = load_image(image_path)
    h, w = image_source.shape[:2]

    with torch.no_grad():
        boxes, logits, labels = predict(
            model=model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

    boxes_xyxy: list[list[int]] = []
    for box in boxes:
        cx, cy, bw, bh = box.tolist()
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        boxes_xyxy.append([max(0, x1), max(0, y1), min(w, x2), min(h, y2)])

    return boxes_xyxy


# ─────────────────────────────────────────────────────────────
# 4. SAM 2
# ─────────────────────────────────────────────────────────────

def load_sam2(checkpoint_path: str, model_cfg: str, device: str):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        sys.exit("SAM 2 not found. Run: pip install -e . inside Grounded-SAM-2 repo.")
    model = build_sam2(model_cfg, checkpoint_path, device=device)
    return SAM2ImagePredictor(model)


def run_sam2(predictor, image_path: str, boxes_xyxy: list[list[int]]) -> np.ndarray:
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]

    if not boxes_xyxy:
        return np.zeros((h, w), dtype=np.uint8)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        predictor.set_image(image)
        boxes_tensor = torch.tensor(boxes_xyxy, dtype=torch.float32)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_tensor,
            multimask_output=False,
        )

    if masks.ndim == 4:
        masks = masks[:, 0]

    return np.any(masks, axis=0).astype(np.uint8) * 255


# ─────────────────────────────────────────────────────────────
# 5. Postprocessing
# ─────────────────────────────────────────────────────────────

def postprocess_mask(mask: np.ndarray, expand_pixels: int = 30, close_pixels: int = 40) -> np.ndarray:
    """
    1. Close (dilate then erode) — fills interior holes and connects nearby fragments
    2. Dilate — expands outward by expand_pixels
    3. Fill remaining holes — flood fill from border to catch any leftover interior gaps
    """
    # Step 1: Morphological closing to fill gaps and connect fragments
    if close_pixels > 0:
        k = close_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Step 2: Dilate outward
    if expand_pixels > 0:
        k = expand_pixels * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Step 3: Flood fill to remove any remaining interior holes
    filled = mask.copy()
    h, w = filled.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 255)       # fill background from corner
    interior_holes = cv2.bitwise_not(filled)         # holes = what got filled that wasn't mask
    mask = cv2.bitwise_or(mask, interior_holes)      # add holes back to mask

    return mask

def expand_mask(mask: np.ndarray, expand_pixels: int) -> np.ndarray:
    """Dilate mask outward by expand_pixels using an elliptical kernel."""
    if expand_pixels <= 0:
        return mask
    k = expand_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def compute_diff_mask(
    before_path: str,
    after_path: str,
    threshold: int = 30,
    blur_ksize: int = 21,
    dilate_iters: int = 3,
) -> np.ndarray:
    before = cv2.imread(before_path)
    after  = cv2.imread(after_path)
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    diff = cv2.absdiff(before, after)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    return cv2.dilate(mask, kernel, iterations=dilate_iters)


def compute_global_diff_mask(before_path: str, after_path: str) -> np.ndarray:
    return compute_diff_mask(before_path, after_path, threshold=20, blur_ksize=31, dilate_iters=5)


def refine_mask_with_diff(
    sam_mask: np.ndarray,
    diff_mask: np.ndarray,
    min_coverage: float = 0.10,
) -> np.ndarray:
    if diff_mask.shape != sam_mask.shape:
        diff_mask = cv2.resize(diff_mask, (sam_mask.shape[1], sam_mask.shape[0]))
    intersection = cv2.bitwise_and(sam_mask, diff_mask)
    coverage = intersection.sum() / (sam_mask.sum() + 1e-6)
    return sam_mask if coverage < min_coverage else intersection


# ─────────────────────────────────────────────────────────────
# 6. Per-record pipeline
# ─────────────────────────────────────────────────────────────

def merge_boxes(boxes_xyxy: list[list[int]], padding: int = 20) -> list[list[int]]:
    """Merge all boxes into a single bounding box enclosing all of them."""
    if not boxes_xyxy:
        return []
    x1 = max(0, min(b[0] for b in boxes_xyxy) - padding)
    y1 = max(0, min(b[1] for b in boxes_xyxy) - padding)
    x2 = max(b[2] for b in boxes_xyxy) + padding
    y2 = max(b[3] for b in boxes_xyxy) + padding
    return [[x1, y1, x2, y2]]

def process_record(
    record: dict,
    image_root: str,
    output_dir: str,
    vlm_model,
    vlm_processor,
    gdino_model,
    sam_predictor,
    device: str,
    use_vlm: bool,
    diff_refine: bool,
    expand_pixels: int,
    close_pixels: int,
    box_threshold: float,
    text_threshold: float,
) -> dict:
    before_path = os.path.join(image_root, record["image_path"])
    after_path  = record["output_path"]
    operation   = record["operation"]
    prompt      = record["prompt"]

    if not os.path.exists(before_path):
        return {**record, "mask_before": None, "mask_after": None,
                "error": f"before image not found: {before_path}"}
    if not os.path.exists(after_path):
        return {**record, "mask_before": None, "mask_after": None,
                "error": f"after image not found: {after_path}"}

    stem = Path(after_path).stem
    os.makedirs(output_dir, exist_ok=True)
    mask_before_path = os.path.join(output_dir, f"{stem}_mask_before.png")
    mask_after_path  = os.path.join(output_dir, f"{stem}_mask_after.png")

    # ── Stage 1: Get grounding phrases ──
    change_type = "local"
    if use_vlm:
        try:
            vlm_result  = call_vlm(vlm_model, vlm_processor, before_path, after_path, prompt)
            phrases     = vlm_result.get("changed_regions", [])
            change_type = vlm_result.get("change_type", "local")
            # Fall back to regex if VLM returned nothing useful
            if not phrases and operation not in GLOBAL_OPERATIONS:
                phrases, change_type = extract_grounding_phrases_regex(prompt, operation)
        except Exception as e:
            phrases, change_type = extract_grounding_phrases_regex(prompt, operation)
            print(f"\n  [VLM fallback] {record.get('image_path')}: {e}")
    else:
        phrases, change_type = extract_grounding_phrases_regex(prompt, operation)

    # ── Global fallback (LargeElement or VLM said global) ──
    if change_type == "global" or not phrases:
        diff = compute_global_diff_mask(before_path, after_path)
        diff = expand_mask(diff, expand_pixels)
        cv2.imwrite(mask_before_path, diff)
        cv2.imwrite(mask_after_path,  diff)
        return {
            **record,
            "mask_before":       mask_before_path,
            "mask_after":        mask_after_path,
            "grounding_phrases": phrases,
            "change_type":       "global",
            "n_boxes_before":    0,
            "n_boxes_after":     0,
            "error":             None,
        }

    # ── Stage 2: GroundingDINO on each image ──
    try:
        boxes_before = run_grounding_dino(
            gdino_model, before_path, phrases, device, box_threshold, text_threshold)
        boxes_after  = run_grounding_dino(
            gdino_model, after_path,  phrases, device, box_threshold, text_threshold)
    except Exception as e:
        return {**record, "mask_before": None, "mask_after": None,
                "error": f"GroundingDINO failed: {e}"}

    # Handle add/remove edge cases
    if boxes_before and not boxes_after:
        boxes_after = boxes_before
    if boxes_after and not boxes_before:
        boxes_before = boxes_after
        
    # boxes_before = merge_boxes(boxes_before)
    # boxes_after  = merge_boxes(boxes_after)

    # ── Stage 3: SAM 2 on each image ──
    try:
        mask_before = run_sam2(sam_predictor, before_path, boxes_before)
        mask_after  = run_sam2(sam_predictor, after_path,  boxes_after)
    except Exception as e:
        return {**record, "mask_before": None, "mask_after": None,
                "error": f"SAM 2 failed: {e}"}

    # ── Stage 4: Optional diff refinement ──
    if diff_refine:
        diff = compute_diff_mask(before_path, after_path)
        mask_before = refine_mask_with_diff(mask_before, diff)
        mask_after  = refine_mask_with_diff(mask_after,  diff)

    # ── Stage 5: Expand mask ──
    # mask_before = expand_mask(mask_before, expand_pixels)
    # mask_after  = expand_mask(mask_after,  expand_pixels)
    mask_before = postprocess_mask(mask_before, expand_pixels=expand_pixels, close_pixels=close_pixels)
    mask_after  = postprocess_mask(mask_after,  expand_pixels=expand_pixels, close_pixels=close_pixels)

    cv2.imwrite(mask_before_path, mask_before)
    cv2.imwrite(mask_after_path,  mask_after)

    return {
        **record,
        "mask_before":       mask_before_path,
        "mask_after":        mask_after_path,
        "grounding_phrases": phrases,
        "change_type":       change_type,
        "n_boxes_before":    len(boxes_before),
        "n_boxes_after":     len(boxes_after),
        "error":             None,
    }


# ─────────────────────────────────────────────────────────────
# 7. CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate before/after change masks for paired edited room images"
    )
    p.add_argument("--jsonl",           required=True)
    p.add_argument("--image_root",      required=True)
    p.add_argument("--output_dir",      required=True)
    p.add_argument("--sam2_checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    p.add_argument("--sam2_config",     default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--gdino_config",
                   default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    p.add_argument("--gdino_weights",   default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    p.add_argument("--vlm_model",       default="Qwen/Qwen2.5-VL-72B-Instruct",
                   help="HuggingFace model ID or local path for Qwen2.5-VL")
    p.add_argument("--no_vlm",          action="store_true",
                   help="Skip VLM and use regex parser only")
    p.add_argument("--device",          default="cuda")
    p.add_argument("--diff_refine",     action="store_true",
                   help="Intersect SAM mask with pixel diff to suppress false positives")
    p.add_argument("--expand_pixels",   type=int, default=30,
                   help="Dilate final mask outward by this many pixels (0 to disable)")
    p.add_argument("--box_threshold",   type=float, default=0.30)
    p.add_argument("--text_threshold",  type=float, default=0.25)
    p.add_argument("--limit",           type=int, default=None)
    p.add_argument("--vlm_max_pixels", type=int, default=1280*720,
               help="Max pixels per image fed to VLM (default: 720p)")
    p.add_argument("--close_pixels", type=int, default=40,
               help="Morphological closing kernel size to fill interior gaps")
    return p.parse_args()


def main():
    args = parse_args()
    use_vlm = not args.no_vlm

    # Load VLM
    vlm_model, vlm_processor = None, None
    if use_vlm:
        print("Loading Qwen2.5-VL...")
        vlm_model, vlm_processor = load_vlm(args.vlm_model, args.device)

    # Load GroundingDINO
    print("Loading GroundingDINO...")
    gdino_model = load_grounding_dino(args.gdino_config, args.gdino_weights, args.device)

    # Load SAM 2
    print("Loading SAM 2...")
    sam_predictor = load_sam2(args.sam2_checkpoint, args.sam2_config, args.device)

    # Read JSONL
    with open(args.jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]
    records = [r for r in records if r.get("status") == "ok"]
    if args.limit:
        records = records[: args.limit]

    print(f"Processing {len(records)} records (VLM={'on' if use_vlm else 'off'}, "
          f"expand={args.expand_pixels}px)...")

    output_records = []
    error_count = 0

    for record in tqdm(records):
        result = process_record(
            record=record,
            image_root=args.image_root,
            output_dir=args.output_dir,
            vlm_model=vlm_model,
            vlm_processor=vlm_processor,
            gdino_model=gdino_model,
            sam_predictor=sam_predictor,
            device=args.device,
            use_vlm=use_vlm,
            diff_refine=args.diff_refine,
            expand_pixels=args.expand_pixels,
            close_pixels=args.close_pixels,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        output_records.append(result)
        if result.get("error"):
            error_count += 1
            print(f"\n  [WARN] {record.get('image_path')}: {result['error']}")

    out_jsonl = os.path.join(args.output_dir, "masks_output.jsonl")
    with open(out_jsonl, "w") as f:
        for r in output_records:
            f.write(json.dumps(r) + "\n")

    success = len(output_records) - error_count
    print(f"\nDone. {success}/{len(output_records)} succeeded.")
    print(f"Results → {out_jsonl}")


if __name__ == "__main__":
    main()