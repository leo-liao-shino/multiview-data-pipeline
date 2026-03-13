"""
Test Qwen-Image-Edit-2511 on a sample of image-instruction pairs.

Usage:
    conda activate data_process
    python scripts/qwen_edit_test.py [--jsonl FILE] [--dataset-root DIR]
                                     [--output-dir DIR] [--n N] [--seed N]
                                     [--steps N] [--cfg FLOAT]
                                     [--config FILE]
                                     [--bf16 | --int8]

Outputs per pair:
    {output_dir}/{category}_scene_{id}_{variant}{view}_edited.jpg   — edited image
    {output_dir}/{category}_scene_{id}_{variant}{view}_compare.jpg  — side-by-side
    {output_dir}/results.jsonl                                       — metadata log
"""

import gc
import os
import json
import random
import argparse
import math
from pathlib import Path

import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
from diffusers.quantizers.pipe_quant_config import PipelineQuantizationConfig

from config_utils import load_json_config, pick_value

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_JSONL        = "/workspace/multiview-data-pipeline/resume/prompts_debug.jsonl"
DEFAULT_DATASET_ROOT = "/workspace/data/all-multiview-datasets"
DEFAULT_OUTPUT_DIR   = "/workspace/data/qwen-test-outputs"
DEFAULT_N            = 5
DEFAULT_SEED         = 42
DEFAULT_STEPS        = 20
DEFAULT_CFG          = 4.0   # true_cfg_scale recommended in model card
DEFAULT_MODEL        = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_MATCH_INPUT_MAX_MEGAPIXELS = 7  # safety cap when matching source resolution
DEFAULT_PRESERVE_UNTOUCHED = True
DEFAULT_PRESERVE_SUFFIX = (
    " Only edit the requested target. Keep all other objects, textures, lighting, "
    "geometry, and background unchanged."
)
DEFAULT_CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config" / "qwen-edit-test.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str = DEFAULT_MODEL, precision: str = "int8") -> QwenImageEditPlusPipeline:
    if precision == "int8":
        print(f"Loading {model_id} with 8-bit quantization (pure GPU) ...")
        quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
        )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config,
        )
        pipe.to("cuda")
        print("Model loaded on GPU (8-bit quantized LLM backbone, ~24 GB VRAM).")
    else:
        print(f"Loading {model_id} in full bf16 ...")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        pipe.to("cuda")
        print("Model loaded on GPU (full bf16, ~47 GB VRAM).")
    return pipe


def make_side_by_side(before: Image.Image, after: Image.Image,
                      prompt: str, operation: str) -> Image.Image:
    """Combine before / after images with a label strip."""
    w, h = before.size
    after_resized = after.resize((w, h), Image.Resampling.LANCZOS)

    label_h = 40
    canvas = Image.new("RGB", (w * 2, h + label_h), (30, 30, 30))
    canvas.paste(before, (0, label_h))
    canvas.paste(after_resized, (w, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    draw.text((4, 4), "BEFORE", fill=(200, 200, 200), font=font)
    draw.text((w + 4, 4), f"[{operation}] {prompt}", fill=(255, 220, 80), font=font)
    return canvas


def cap_resolution_to_megapixels(width: int,
                                 height: int,
                                 max_megapixels: float,
                                 multiple_of: int = 32) -> tuple[int, int, bool]:
    """Cap resolution by area while preserving aspect ratio and model divisibility."""
    area = width * height
    max_area = int(max_megapixels * 1_000_000)
    if area <= max_area:
        return width, height, False

    scale = math.sqrt(max_area / float(area))
    capped_w = max(multiple_of, int(width * scale) // multiple_of * multiple_of)
    capped_h = max(multiple_of, int(height * scale) // multiple_of * multiple_of)
    return capped_w, capped_h, True


def run_pair(pipe: QwenImageEditPlusPipeline,
             record: dict,
             dataset_root: str,
             output_dir: Path,
             steps: int,
             cfg: float,
             seed: int,
             width: int | None,
             height: int | None,
             match_input_resolution: bool,
             max_megapixels: float | None,
             preserve_untouched: bool,
             preserve_suffix: str) -> dict:
    img_path = Path(dataset_root) / record["image_path"]
    if not img_path.exists():
        return {**record, "status": "missing_image", "output_path": None}

    image = Image.open(img_path).convert("RGB")
    prompt = record["prompt"]
    if preserve_untouched:
        prompt = f"{prompt.rstrip('. ')}.{preserve_suffix}"
    operation = record["operation"]
    target_width = width
    target_height = height
    if target_width is None and target_height is None and match_input_resolution:
        target_width = image.width
        target_height = image.height

    effective_max_megapixels = max_megapixels
    if effective_max_megapixels is None and target_width is not None and target_height is not None and match_input_resolution:
        effective_max_megapixels = DEFAULT_MATCH_INPUT_MAX_MEGAPIXELS

    resolution_capped = False
    if effective_max_megapixels is not None and target_width is not None and target_height is not None:
        orig_w, orig_h = target_width, target_height
        target_width, target_height, resolution_capped = cap_resolution_to_megapixels(
            target_width,
            target_height,
            effective_max_megapixels,
        )
        if resolution_capped:
            print(
                f"  resolution capped: {orig_w}x{orig_h} -> {target_width}x{target_height} "
                f"(max {effective_max_megapixels:.2f} MP)"
            )

    inputs = {
        "image": [image],
        "prompt": prompt,
        "generator": torch.manual_seed(seed),
        "true_cfg_scale": cfg,
        "negative_prompt": " ",
        "num_inference_steps": steps,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }
    if target_width is not None and target_height is not None:
        inputs["width"] = target_width
        inputs["height"] = target_height

    oom_fallback_used = False
    try:
        with torch.inference_mode():
            output = pipe(**inputs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if "width" in inputs and "height" in inputs:
            print("  OOM at requested resolution; retrying with pipeline default resolution...")
            inputs.pop("width", None)
            inputs.pop("height", None)
            with torch.inference_mode():
                output = pipe(**inputs)
            target_width = None
            target_height = None
            oom_fallback_used = True
        else:
            raise

    edited: Image.Image = output.images[0] # type: ignore

    # Free GPU residuals before next iteration
    torch.cuda.empty_cache()
    gc.collect()

    # Build output stem from image_path
    stem = Path(record["image_path"]).stem   # e.g. Bedroom_scene_0001_A2
    edited_path  = output_dir / f"{stem}_edited.jpg"
    compare_path = output_dir / f"{stem}_compare.jpg"

    edited.save(edited_path, quality=100)
    make_side_by_side(image, edited, prompt, operation).save(compare_path, quality=100)

    return {
        **record,
        "status": "ok",
        "output_path": str(edited_path),
        "compare_path": str(compare_path),
        "requested_width": target_width,
        "requested_height": target_height,
        "output_width": edited.width,
        "output_height": edited.height,
        "resolution_capped": resolution_capped,
        "max_megapixels": effective_max_megapixels,
        "oom_fallback_used": oom_fallback_used,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test Qwen-Image-Edit-2511 on sample pairs")
    p.add_argument("--config",        default=None,                     help="Path to JSON config file (default: config/qwen-edit-test.json)")
    p.add_argument("--jsonl",         default=None,                     help="Path to prompts JSONL")
    p.add_argument("--dataset-root",  default=None,                     help="Dataset root directory")
    p.add_argument("--output-dir",    default=None,                     help="Where to save outputs")
    p.add_argument("--n",             type=int,   default=None,         help="Number of pairs to test")
    p.add_argument("--seed",          type=int,   default=None,         help="Random seed")
    p.add_argument("--steps",         type=int,   default=None,         help="Inference steps (default: 20)")
    p.add_argument("--cfg",           type=float, default=None,         help="true_cfg_scale")
    p.add_argument("--width",         type=int,   default=None,         help="Output width for generation (defaults to input image width)")
    p.add_argument("--height",        type=int,   default=None,         help="Output height for generation (defaults to input image height)")
    p.add_argument("--match-input-resolution", action="store_true", default=None,
                   help="Generate at each source image resolution (slower, more VRAM)")
    p.add_argument("--max-megapixels", type=float, default=None,
                   help="Cap generation area in megapixels (recommended with --match-input-resolution)")
    p.add_argument("--model",         default=None,                     help="Model ID")
    p.add_argument("--preserve-untouched", action="store_true", default=None,
                   help="Append instruction to preserve all non-target regions")
    p.add_argument("--preserve-suffix", default=None,
                   help="Custom suffix used when --preserve-untouched is enabled")
    p.add_argument("--precision",     choices=["int8", "bf16"], default=None,
                   help="Precision mode: int8 or bf16")
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--int8", dest="precision", action="store_const", const="int8",
                      help="8-bit quantization: ~24 GB VRAM, slower (default)")
    prec.add_argument("--bf16", dest="precision", action="store_const", const="bf16",
                      help="Full bfloat16: ~47 GB VRAM, faster")
    p.set_defaults(precision=None)
    p.add_argument("--wandb",         action="store_true", default=None,  help="Log results to Weights & Biases")
    p.add_argument("--wandb-project", default=None,                        help="W&B project name")

    args = p.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    try:
        config = load_json_config(config_path, require_exists=bool(args.config))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        p.error(str(e))

    args.jsonl = pick_value(args.jsonl, config, "jsonl", DEFAULT_JSONL)
    args.dataset_root = pick_value(args.dataset_root, config, "dataset_root", DEFAULT_DATASET_ROOT)
    args.output_dir = pick_value(args.output_dir, config, "output_dir", DEFAULT_OUTPUT_DIR)
    args.n = int(pick_value(args.n, config, "n", DEFAULT_N))
    args.seed = int(pick_value(args.seed, config, "seed", DEFAULT_SEED))
    args.steps = int(pick_value(args.steps, config, "steps", DEFAULT_STEPS))
    args.cfg = float(pick_value(args.cfg, config, "cfg", DEFAULT_CFG))
    args.width = pick_value(args.width, config, "width", None)
    args.height = pick_value(args.height, config, "height", None)
    args.match_input_resolution = bool(
        pick_value(args.match_input_resolution, config, "match_input_resolution", False)
    )
    args.max_megapixels = pick_value(args.max_megapixels, config, "max_megapixels", None)
    args.model = pick_value(args.model, config, "model", DEFAULT_MODEL)
    args.preserve_untouched = bool(
        pick_value(args.preserve_untouched, config, "preserve_untouched", DEFAULT_PRESERVE_UNTOUCHED)
    )
    args.preserve_suffix = str(
        pick_value(args.preserve_suffix, config, "preserve_suffix", DEFAULT_PRESERVE_SUFFIX)
    )
    args.precision = pick_value(args.precision, config, "precision", "int8")
    args.wandb = bool(pick_value(args.wandb, config, "wandb", False))
    args.wandb_project = pick_value(args.wandb_project, config, "wandb_project", "qwen-furniture-edit")

    if args.precision not in {"int8", "bf16"}:
        p.error("precision must be one of: int8, bf16")
    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be provided together")
    if args.width is not None:
        assert args.height is not None
        args.width = int(args.width)
        args.height = int(args.height)
        if args.width <= 0 or args.height <= 0:
            p.error("--width and --height must be positive integers")
    if args.max_megapixels is not None:
        args.max_megapixels = float(args.max_megapixels)
        if args.max_megapixels <= 0:
            p.error("--max-megapixels must be positive")

    if config_path.exists():
        print(f"Using config: {config_path}")
    print(f"Preserve untouched regions: {args.preserve_untouched}")
    if args.width is None and args.match_input_resolution:
        print("Generation resolution: match each input image (runtime detected)")
    elif args.width is None:
        print("Generation resolution: pipeline default (~1MP, faster)")
    else:
        print(f"Generation resolution: {args.width}x{args.height}")
    if args.max_megapixels is not None:
        print(f"Max megapixels cap: {args.max_megapixels:.2f} MP")
    elif args.match_input_resolution:
        print(f"Max megapixels cap: auto {DEFAULT_MATCH_INPUT_MAX_MEGAPIXELS:.2f} MP")

    return args


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and sample records
    with open(args.jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    random.seed(args.seed)
    sample = random.sample(records, min(args.n, len(records)))

    print(f"Selected {len(sample)} pairs (seed={args.seed}):")
    # for r in sample:
    #     print(f"  [{r['operation']:7s}] {r['image_path']} — {r['prompt']}")
    # print()

    pipe = load_pipeline(args.model, args.precision)

    # Init W&B run if requested
    wb_run = None
    if args.wandb:
        wb_run = wandb.init(
            project=args.wandb_project,
            config={
                "model":     args.model,
                "precision": args.precision,
                "steps":     args.steps,
                "cfg":       args.cfg,
                "n":         len(sample),
                "seed":      args.seed,
            },
        )

    results_path = output_dir / "results.jsonl"
    with open(results_path, "w") as out_f: # overwrite if exists
        for i, record in enumerate(sample, 1):
            print(f"[{i}/{len(sample)}] {record['image_path']}")
            result = run_pair(pipe, record, args.dataset_root,
                              output_dir, args.steps, args.cfg, args.seed,
                              args.width, args.height, args.match_input_resolution,
                              args.max_megapixels,
                              args.preserve_untouched, args.preserve_suffix)
            print(f"  → {result['status']}"
                  + (f"  saved: {result['output_path']}" if result["status"] == "ok" else ""))
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

            if wb_run and result["status"] == "ok":
                compare_img = Image.open(result["compare_path"])
                wb_run.log({
                    "compare": wandb.Image(
                        compare_img,
                        caption=f"[{result['operation']}] {result['prompt']}",
                    ),
                    "operation": result["operation"],
                    "scene":     result["image_path"],
                })

    print(f"\nDone. Results logged to {results_path}")
    print(f"Outputs in: {output_dir}")
    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    main()
