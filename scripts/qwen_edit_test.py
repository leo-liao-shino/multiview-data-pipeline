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
    {output_dir}/{trial_tag}/{stem}_edited.jpg   — edited image
    {output_dir}/{trial_tag}/{stem}_compare.jpg  — side-by-side
    {output_dir}/results.jsonl                   — metadata log (single run)
    {output_dir}/sweep_results.jsonl             — metadata log (sweep)
    {output_dir}/sweep_summary.json              — ranked sweep summary
"""

import gc
import os
import json
import random
import argparse
import math
import itertools
import inspect
import hashlib
from pathlib import Path

import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    FlowMatchLCMScheduler,
    UniPCMultistepScheduler,
)
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
    " Only edit the requested target. Keep all other regions of the image unchanged."
)
DEFAULT_SCHEDULER = "default"
DEFAULT_NEGATIVE_PROMPT = " "
DEFAULT_PROMPT_PREFIX = ""
DEFAULT_PROMPT_CASE = "none"
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config" / "qwen-edit-test.json"

# ---------------------------------------------------------------------------
# Scheduler / sampler registry
#
# Each entry maps a string key to:
#   cls    – the Diffusers scheduler class (None = use model default)
#   kwargs – extra kwargs passed to from_config() to select sampler variant
#
# Compatibility requirement: set_timesteps() must accept both `mu` and
# custom `sigmas` parameters (Qwen edit pipeline requirement).
#
# LCM note: FlowMatchLCMScheduler is designed for 4–8 steps. Configs that
# pair an LCM scheduler with >10 steps will be filtered out automatically.
# ---------------------------------------------------------------------------
SCHEDULER_CONFIGS: dict[str, dict] = {
    # ── Flow-match Euler ────────────────────────────────────────────────────
    "default": {
        "cls": None,
        "kwargs": {},
        "description": "Model-bundled default (FlowMatchEuler, uniform sigma spacing)",
    },
    "flowmatch_euler": {
        "cls": FlowMatchEulerDiscreteScheduler,
        "kwargs": {},
        "description": "Explicit FlowMatchEuler, uniform sigma spacing (same algorithm as default)",
    },
    "flowmatch_euler_karras": {
        "cls": FlowMatchEulerDiscreteScheduler,
        "kwargs": {"use_karras_sigmas": True},
        "description": "FlowMatchEuler with Karras sigma spacing — often sharper at same step count",
    },
    # ── LCM (few-step) ──────────────────────────────────────────────────────
    "flowmatch_lcm": {
        "cls": FlowMatchLCMScheduler,
        "kwargs": {},
        "description": "LCM consistency distillation — fast (4–8 steps), softer edits",
    },
    # ── Incompatible (kept for clear error messages) ─────────────────────
    # "unipc":  UniPCMultistepScheduler — missing mu/sigmas, do NOT add
    # "dpm++":  DPMSolverMultistepScheduler — missing mu/sigmas, do NOT add
    # "ddim":   DDIMScheduler — wrong noise space entirely
}

# Flat alias dict used by argparse choices + compatibility checks
SCHEDULER_FACTORY = {k: v["cls"] for k, v in SCHEDULER_CONFIGS.items()}

# Steps above which LCM schedulers are filtered out of the sweep
LCM_MAX_STEPS = 10


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def scheduler_is_qwen_compatible(scheduler_cls) -> bool:
    """Return True if the scheduler's set_timesteps accepts both mu and sigmas."""
    if scheduler_cls is None:
        return True
    sig = inspect.signature(scheduler_cls.set_timesteps)
    return "mu" in sig.parameters and "sigmas" in sig.parameters


def _short_hash(s: str, n: int = 6) -> str:
    """Return a short deterministic hex hash of a string."""
    return hashlib.md5(s.encode()).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Pipeline loading
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str = DEFAULT_MODEL,
                  precision: str = "int8",
                  scheduler: str = DEFAULT_SCHEDULER) -> QwenImageEditPlusPipeline:
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

    scheduler_key = scheduler.lower().strip()
    sched_cfg = SCHEDULER_CONFIGS.get(scheduler_key)
    if sched_cfg is None:
        raise ValueError(
            f"Unknown scheduler '{scheduler_key}'. "
            f"Valid options: {list(SCHEDULER_CONFIGS.keys())}"
        )

    sched_cls = sched_cfg["cls"]
    sched_kwargs = sched_cfg["kwargs"]

    if sched_cls is None:
        print(f"Scheduler: model default  [{sched_cfg['description']}]")
    else:
        if not scheduler_is_qwen_compatible(sched_cls):
            raise ValueError(
                f"Scheduler '{scheduler_key}' is incompatible with the Qwen edit pipeline "
                "(set_timesteps must support both 'mu' and custom 'sigmas')."
            )
        pipe.scheduler = sched_cls.from_config(pipe.scheduler.config, **sched_kwargs)
        print(f"Scheduler: {scheduler_key}  kwargs={sched_kwargs}  [{sched_cfg['description']}]")

    return pipe


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def make_side_by_side(before: Image.Image, after: Image.Image,
                      prompt: str, operation: str) -> Image.Image:
    """Combine before / after images with a label strip."""
    w, h = after.size
    before_resized = before.resize((w, h), Image.Resampling.LANCZOS)

    label_h = 40
    canvas = Image.new("RGB", (w * 2, h + label_h), (30, 30, 30))
    canvas.paste(before_resized, (0, label_h))
    canvas.paste(after, (w, label_h))

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


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(base_prompt: str,
                 preserve_untouched: bool,
                 preserve_suffix: str,
                 prompt_prefix: str,
                 prompt_case: str,
                 prompt_max_chars: int | None) -> str:
    prompt = base_prompt
    if preserve_untouched:
        prompt = f"{prompt.rstrip('. ')}.{preserve_suffix}"
    if prompt_prefix:
        prompt = f"{prompt_prefix.strip()} {prompt}".strip()

    case_mode = prompt_case.lower().strip()
    if case_mode == "lower":
        prompt = prompt.lower()
    elif case_mode == "upper":
        prompt = prompt.upper()

    if prompt_max_chars is not None and prompt_max_chars > 0:
        prompt = prompt[:prompt_max_chars].rstrip()
    return prompt


# ---------------------------------------------------------------------------
# Sweep construction
# ---------------------------------------------------------------------------

def _to_list(value, default):
    if value is None:
        return [default]
    if isinstance(value, list):
        return value if value else [default]
    return [value]


def _is_lcm_scheduler(scheduler_key: str) -> bool:
    return "lcm" in scheduler_key.lower()


def build_sweep_trials(args, config: dict) -> list[dict]:
    sweep_cfg = config.get("sweep", {})
    if not isinstance(sweep_cfg, dict):
        sweep_cfg = {}

    schedulers     = [str(v) for v in _to_list(sweep_cfg.get("scheduler"),        args.scheduler)]
    steps_list     = [int(v) for v in _to_list(sweep_cfg.get("steps"),            args.steps)]
    cfg_list       = [float(v) for v in _to_list(sweep_cfg.get("cfg"),            args.cfg)]
    guidance_scales= [float(v) for v in _to_list(sweep_cfg.get("guidance_scale"), args.guidance_scale)]
    negative_prompts=[str(v) for v in _to_list(sweep_cfg.get("negative_prompt"),  args.negative_prompt)]
    prompt_prefixes= [str(v) for v in _to_list(sweep_cfg.get("prompt_prefix"),    args.prompt_prefix)]
    prompt_cases   = [str(v) for v in _to_list(sweep_cfg.get("prompt_case"),      args.prompt_case)]
    prompt_max_chars_values = _to_list(sweep_cfg.get("prompt_max_chars"), args.prompt_max_chars)

    # ── Validate schedulers up-front, before building the product ──────────
    unknown = [s for s in schedulers if s not in SCHEDULER_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown schedulers in sweep: {unknown}. "
            f"Valid options: {list(SCHEDULER_CONFIGS.keys())}"
        )
    incompatible = [
        s for s in schedulers
        if not scheduler_is_qwen_compatible(SCHEDULER_CONFIGS[s]["cls"])
    ]
    if incompatible:
        raise ValueError(
            f"Schedulers {incompatible} are incompatible with the Qwen edit pipeline "
            f"(set_timesteps missing 'mu' or custom 'sigmas'). Remove them from sweep config.\n"
            f"Incompatible known schedulers: unipc, dpm++, ddim."
        )

    # ── Validate other sweep axes ──────────────────────────────────────────
    for pc in prompt_cases:
        if pc not in {"none", "lower", "upper"}:
            raise ValueError(f"Invalid prompt_case in sweep: {pc}")

    max_chars_parsed: list[int | None] = []
    for v in prompt_max_chars_values:
        if v is None:
            max_chars_parsed.append(None)
        else:
            iv = int(v)
            if iv <= 0:
                raise ValueError("prompt_max_chars values in sweep must be positive")
            max_chars_parsed.append(iv)

    # ── Build full Cartesian product ───────────────────────────────────────
    trials: list[dict] = []
    for scheduler, steps, cfg, guidance_scale, negative_prompt, prompt_prefix, prompt_case, prompt_max_chars in itertools.product(
        schedulers,
        steps_list,
        cfg_list,
        guidance_scales,
        negative_prompts,
        prompt_prefixes,
        prompt_cases,
        max_chars_parsed,
    ):
        # Filter: LCM schedulers should only run at low step counts
        if _is_lcm_scheduler(scheduler) and steps > LCM_MAX_STEPS:
            continue

        trials.append({
            "scheduler":      scheduler,
            "steps":          int(steps),
            "cfg":            float(cfg),
            "guidance_scale": float(guidance_scale),
            "negative_prompt": str(negative_prompt),
            "prompt_prefix":  str(prompt_prefix),
            "prompt_case":    str(prompt_case),
            "prompt_max_chars": prompt_max_chars,
        })

    if not trials:
        raise ValueError(
            "Sweep produced zero valid trials after filtering. "
            "Check your scheduler/steps combinations — LCM requires steps <= "
            f"{LCM_MAX_STEPS}."
        )

    random.shuffle(trials)
    if args.sweep_limit is not None:
        trials = trials[:args.sweep_limit]

    # Print a summary of what we're actually running
    scheduler_counts: dict[str, int] = {}
    for t in trials:
        scheduler_counts[t["scheduler"]] = scheduler_counts.get(t["scheduler"], 0) + 1
    print("Sweep trial breakdown by scheduler:")
    for sched, count in sorted(scheduler_counts.items()):
        desc = SCHEDULER_CONFIGS[sched]["description"]
        print(f"  {sched}: {count} trials  — {desc}")

    return trials


def _build_trial_tag(trial_idx: int, trial: dict) -> str:
    """Build a unique, human-readable directory tag for a sweep trial."""
    max_chars_tag = f"_mc{trial['prompt_max_chars']}" if trial.get("prompt_max_chars") else ""
    tag = (
        f"trial_{trial_idx:03d}"
        f"_{trial['scheduler']}"
        f"_steps{trial['steps']}"
        f"_cfg{trial['cfg']:.2f}"
        f"_gs{trial['guidance_scale']:.2f}"
        f"_{trial['prompt_case']}"
        f"_neg{_short_hash(trial['negative_prompt'])}"
        f"_pfx{_short_hash(trial['prompt_prefix'])}"
        f"{max_chars_tag}"
    )
    return tag.replace(" ", "_")


# ---------------------------------------------------------------------------
# Per-pair inference
# ---------------------------------------------------------------------------

def run_pair(pipe: QwenImageEditPlusPipeline,
             record: dict,
             dataset_root: str,
             output_dir: Path,
             steps: int,
             cfg: float,
             guidance_scale: float,
             seed: int,
             width: int | None,
             height: int | None,
             match_input_resolution: bool,
             max_megapixels: float | None,
             preserve_untouched: bool,
             preserve_suffix: str,
             negative_prompt: str,
             prompt_prefix: str,
             prompt_case: str,
             prompt_max_chars: int | None) -> dict:
    img_path = Path(dataset_root) / record["image_path"]
    if not img_path.exists():
        return {**record, "status": "missing_image", "output_path": None}

    image = Image.open(img_path).convert("RGB")
    original_image = image
    prompt = build_prompt(
        record["prompt"],
        preserve_untouched,
        preserve_suffix,
        prompt_prefix,
        prompt_case,
        prompt_max_chars,
    )
    operation = record["operation"]
    target_width = width
    target_height = height
    if target_width is None and target_height is None and match_input_resolution:
        target_width = image.width
        target_height = image.height

    effective_max_megapixels = max_megapixels
    if (effective_max_megapixels is None
            and target_width is not None
            and target_height is not None
            and match_input_resolution):
        effective_max_megapixels = DEFAULT_MATCH_INPUT_MAX_MEGAPIXELS

    resolution_capped = False
    if (effective_max_megapixels is not None
            and target_width is not None
            and target_height is not None):
        orig_w, orig_h = target_width, target_height
        target_width, target_height, resolution_capped = cap_resolution_to_megapixels(
            target_width, target_height, effective_max_megapixels,
        )
        if resolution_capped:
            print(
                f"  resolution capped: {orig_w}x{orig_h} -> {target_width}x{target_height} "
                f"(max {effective_max_megapixels:.2f} MP)"
            )

    if (target_width is not None
            and target_height is not None
            and (image.width != target_width or image.height != target_height)):
        image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        print(f"  resized input image to: {target_width}x{target_height}")

    inputs = {
        "image":               [image],
        "prompt":              prompt,
        "generator":           torch.manual_seed(seed),
        "true_cfg_scale":      cfg,
        "negative_prompt":     negative_prompt,
        "num_inference_steps": steps,
        "guidance_scale":      guidance_scale,
        "num_images_per_prompt": 1,
    }
    if target_width is not None and target_height is not None:
        inputs["width"]  = target_width
        inputs["height"] = target_height

    oom_fallback_used = False
    try:
        with torch.inference_mode():
            output = pipe(**inputs)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        if image.width * image.height > 1_000_000:
            retry_w, retry_h, _ = cap_resolution_to_megapixels(image.width, image.height, 1.0)
            print(
                f"  OOM at {image.width}x{image.height}; retrying with resized input "
                f"{retry_w}x{retry_h}..."
            )
            image = image.resize((retry_w, retry_h), Image.Resampling.LANCZOS)
            inputs["image"]  = [image]
            inputs["width"]  = retry_w
            inputs["height"] = retry_h
            with torch.inference_mode():
                output = pipe(**inputs)
            target_width  = retry_w
            target_height = retry_h
            oom_fallback_used = True
        else:
            raise

    edited: Image.Image = output.images[0]  # type: ignore

    torch.cuda.empty_cache()
    gc.collect()

    stem         = Path(record["image_path"]).stem
    edited_path  = output_dir / f"{stem}_edited.jpg"
    compare_path = output_dir / f"{stem}_compare.jpg"

    edited.save(edited_path, quality=100)
    make_side_by_side(original_image, edited, prompt, operation).save(compare_path, quality=100)

    return {
        **record,
        "status":           "ok",
        "output_path":      str(edited_path),
        "compare_path":     str(compare_path),
        "requested_width":  target_width,
        "requested_height": target_height,
        "output_width":     edited.width,
        "output_height":    edited.height,
        "resolution_capped":    resolution_capped,
        "max_megapixels":       effective_max_megapixels,
        "guidance_scale":       guidance_scale,
        "negative_prompt":      negative_prompt,
        "prompt_prefix":        prompt_prefix,
        "prompt_case":          prompt_case,
        "prompt_max_chars":     prompt_max_chars,
        "oom_fallback_used":    oom_fallback_used,
    }


# ---------------------------------------------------------------------------
# Argument parsing + config merging
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Test Qwen-Image-Edit-2511 on sample pairs")
    p.add_argument("--config",         default=None,  help="Path to JSON config file")
    p.add_argument("--jsonl",          default=None,  help="Path to prompts JSONL")
    p.add_argument("--dataset-root",   default=None,  help="Dataset root directory")
    p.add_argument("--output-dir",     default=None,  help="Where to save outputs")
    p.add_argument("--n",              type=int,   default=None, help="Number of pairs to test")
    p.add_argument("--seed",           type=int,   default=None, help="Random seed")
    p.add_argument("--steps",          type=int,   default=None, help="Inference steps")
    p.add_argument("--cfg",            type=float, default=None, help="true_cfg_scale")
    p.add_argument("--guidance-scale", type=float, default=None, help="guidance_scale")
    p.add_argument("--width",          type=int,   default=None, help="Output width")
    p.add_argument("--height",         type=int,   default=None, help="Output height")
    p.add_argument("--match-input-resolution", action="store_true", default=None,
                   help="Generate at each source image resolution")
    p.add_argument("--max-megapixels", type=float, default=None,
                   help="Cap generation area in megapixels")
    p.add_argument("--model",          default=None, help="Model ID")
    p.add_argument("--scheduler",      choices=list(SCHEDULER_CONFIGS.keys()), default=None,
                   help=(
                       "Diffusion scheduler/sampler. Options:\n" +
                       "\n".join(f"  {k}: {v['description']}" for k, v in SCHEDULER_CONFIGS.items())
                   ))
    p.add_argument("--negative-prompt",  default=None, help="Negative prompt text")
    p.add_argument("--prompt-prefix",    default=None, help="Prefix prepended to every prompt")
    p.add_argument("--prompt-case",      choices=["none", "lower", "upper"], default=None)
    p.add_argument("--prompt-max-chars", type=int, default=None,
                   help="Truncate prompt to this many characters")
    p.add_argument("--preserve-untouched", action="store_true", default=None)
    p.add_argument("--preserve-suffix",    default=None)
    p.add_argument("--precision",          choices=["int8", "bf16"], default=None)
    prec = p.add_mutually_exclusive_group()
    prec.add_argument("--int8", dest="precision", action="store_const", const="int8",
                      help="8-bit quantization: ~24 GB VRAM (default)")
    prec.add_argument("--bf16", dest="precision", action="store_const", const="bf16",
                      help="Full bfloat16: ~47 GB VRAM, faster")
    p.set_defaults(precision=None)
    p.add_argument("--wandb",         action="store_true", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--sweep",         action="store_true", default=None,
                   help="Run config-driven hyperparameter sweep")
    p.add_argument("--sweep-limit",   type=int, default=None,
                   help="Cap on number of sampled sweep trials")

    args = p.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    try:
        config = load_json_config(config_path, require_exists=bool(args.config))
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as e:
        p.error(str(e))

    # ── Merge: CLI > config file > defaults ───────────────────────────────
    args.jsonl          = pick_value(args.jsonl,          config, "jsonl",          DEFAULT_JSONL)
    args.dataset_root   = pick_value(args.dataset_root,   config, "dataset_root",   DEFAULT_DATASET_ROOT)
    args.output_dir     = pick_value(args.output_dir,     config, "output_dir",     DEFAULT_OUTPUT_DIR)
    args.n              = int(pick_value(args.n,           config, "n",              DEFAULT_N))
    args.seed           = int(pick_value(args.seed,        config, "seed",           DEFAULT_SEED))
    args.steps          = int(pick_value(args.steps,       config, "steps",          DEFAULT_STEPS))
    args.cfg            = float(pick_value(args.cfg,       config, "cfg",            DEFAULT_CFG))
    args.guidance_scale = float(pick_value(args.guidance_scale, config, "guidance_scale", DEFAULT_GUIDANCE_SCALE))
    args.width          = pick_value(args.width,           config, "width",          None)
    args.height         = pick_value(args.height,          config, "height",         None)
    args.match_input_resolution = bool(
        pick_value(args.match_input_resolution, config, "match_input_resolution", False)
    )
    args.max_megapixels = pick_value(args.max_megapixels, config, "max_megapixels", None)
    args.model          = pick_value(args.model,           config, "model",          DEFAULT_MODEL)
    args.scheduler      = str(pick_value(args.scheduler,  config, "scheduler",      DEFAULT_SCHEDULER))
    args.negative_prompt = str(pick_value(args.negative_prompt, config, "negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    args.prompt_prefix  = str(pick_value(args.prompt_prefix, config, "prompt_prefix", DEFAULT_PROMPT_PREFIX))
    args.prompt_case    = str(pick_value(args.prompt_case, config, "prompt_case",   DEFAULT_PROMPT_CASE))
    args.prompt_max_chars = pick_value(args.prompt_max_chars, config, "prompt_max_chars", None)
    args.preserve_untouched = bool(
        pick_value(args.preserve_untouched, config, "preserve_untouched", DEFAULT_PRESERVE_UNTOUCHED)
    )
    args.preserve_suffix = str(
        pick_value(args.preserve_suffix, config, "preserve_suffix", DEFAULT_PRESERVE_SUFFIX)
    )
    args.precision      = pick_value(args.precision,       config, "precision",      "int8")
    args.wandb          = bool(pick_value(args.wandb,      config, "wandb",          False))
    args.wandb_project  = pick_value(args.wandb_project,  config, "wandb_project",  "qwen-furniture-edit")
    args.sweep          = bool(
        pick_value(
            args.sweep,
            config,
            "sweep_enabled",
            bool(config.get("sweep", {}).get("enabled", False))
            if isinstance(config.get("sweep", {}), dict) else False,
        )
    )
    args.sweep_limit    = pick_value(args.sweep_limit,     config, "sweep_limit",    None)

    # ── Validation ────────────────────────────────────────────────────────
    if args.precision not in {"int8", "bf16"}:
        p.error("precision must be one of: int8, bf16")
    if args.scheduler not in SCHEDULER_CONFIGS:
        p.error(f"scheduler must be one of: {', '.join(SCHEDULER_CONFIGS.keys())}")
    if (args.width is None) != (args.height is None):
        p.error("--width and --height must be provided together")
    if args.width is not None:
        args.width  = int(args.width)
        args.height = int(args.height)
        if args.width <= 0 or args.height <= 0:
            p.error("--width and --height must be positive integers")
    if args.max_megapixels is not None:
        args.max_megapixels = float(args.max_megapixels)
        if args.max_megapixels <= 0:
            p.error("--max-megapixels must be positive")
    if args.guidance_scale <= 0:
        p.error("--guidance-scale must be positive")
    if args.prompt_case not in {"none", "lower", "upper"}:
        p.error("--prompt-case must be one of: none, lower, upper")
    if args.prompt_max_chars is not None:
        args.prompt_max_chars = int(args.prompt_max_chars)
        if args.prompt_max_chars <= 0:
            p.error("--prompt-max-chars must be positive")
    if args.sweep_limit is not None:
        args.sweep_limit = int(args.sweep_limit)
        if args.sweep_limit <= 0:
            p.error("--sweep-limit must be positive")

    # ── Print resolved config ─────────────────────────────────────────────
    if config_path.exists():
        print(f"Using config: {config_path}")
    print(f"Scheduler: {args.scheduler}  [{SCHEDULER_CONFIGS[args.scheduler]['description']}]")
    print(f"Preserve untouched regions: {args.preserve_untouched}")
    if args.width is None and args.match_input_resolution:
        print("Input preprocessing: use each image's original resolution")
    elif args.width is None:
        print("Input preprocessing: no explicit resize (pipeline decides output size)")
    else:
        print(f"Input preprocessing: resize image to {args.width}x{args.height}")
    if args.max_megapixels is not None:
        print(f"Max megapixels cap: {args.max_megapixels:.2f} MP")
    elif args.match_input_resolution:
        print(f"Max megapixels cap: auto {DEFAULT_MATCH_INPUT_MAX_MEGAPIXELS:.2f} MP")
    if args.prompt_prefix:
        print(f"Prompt prefix: {args.prompt_prefix}")
    print(f"Prompt case: {args.prompt_case}")
    if args.prompt_max_chars is not None:
        print(f"Prompt max chars: {args.prompt_max_chars}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Sweep enabled: {args.sweep}")
    if args.sweep_limit is not None:
        print(f"Sweep trial limit: {args.sweep_limit}")

    return args, config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args, config = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and sample records
    with open(args.jsonl) as f:
        records = [json.loads(line) for line in f if line.strip()]

    random.seed(args.seed)
    sample = random.sample(records, min(args.n, len(records)))

    print(f"Selected {len(sample)} pairs (seed={args.seed}).")

    # ── Build trial list ──────────────────────────────────────────────────
    if args.sweep:
        try:
            trials = build_sweep_trials(args, config)
        except ValueError as e:
            raise SystemExit(f"Invalid sweep config: {e}")
        if not trials:
            raise SystemExit("Sweep enabled but no trials were generated.")
        print(f"Running sweep with {len(trials)} trials × {len(sample)} pairs "
              f"= {len(trials) * len(sample)} inference calls")
    else:
        trials = [{
            "scheduler":       args.scheduler,
            "steps":           args.steps,
            "cfg":             args.cfg,
            "guidance_scale":  args.guidance_scale,
            "negative_prompt": args.negative_prompt,
            "prompt_prefix":   args.prompt_prefix,
            "prompt_case":     args.prompt_case,
            "prompt_max_chars": args.prompt_max_chars,
        }]

    # ── Init W&B ──────────────────────────────────────────────────────────
    wb_run = None
    if args.wandb:
        wb_run = wandb.init(
            project=args.wandb_project,
            config={
                "model":          args.model,
                "precision":      args.precision,
                "n":              len(sample),
                "seed":           args.seed,
                "sweep_enabled":  args.sweep,
                "sweep_trials":   len(trials),
                "schedulers_available": list(SCHEDULER_CONFIGS.keys()),
            },
        )

    results_path = output_dir / ("sweep_results.jsonl" if args.sweep else "results.jsonl")
    summary_path = output_dir / "sweep_summary.json"
    sweep_rows: list[dict] = []

    current_scheduler: str | None = None
    pipe: QwenImageEditPlusPipeline | None = None

    with open(results_path, "w") as out_f:
        for trial_idx, trial in enumerate(trials, 1):
            scheduler = trial["scheduler"]

            # ── Reload pipeline only when scheduler actually changes ──────
            if pipe is None or scheduler != current_scheduler:
                if pipe is not None:
                    del pipe
                    torch.cuda.empty_cache()
                    gc.collect()
                try:
                    pipe = load_pipeline(args.model, args.precision, scheduler)
                except Exception as e:
                    err_msg = f"failed to load scheduler '{scheduler}': {e}"
                    print(f"Skipping trial {trial_idx}: {err_msg}")
                    sweep_rows.append({
                        "trial_index":  trial_idx,
                        "trial_tag":    f"trial_{trial_idx:03d}_{scheduler}_load_failed",
                        "trial":        trial,
                        "ok_count":     0,
                        "total":        len(sample),
                        "success_rate": 0.0,
                        "error":        err_msg,
                    })
                    continue
                current_scheduler = scheduler

            trial_tag        = _build_trial_tag(trial_idx, trial)
            trial_output_dir = output_dir / trial_tag
            trial_output_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n=== Trial {trial_idx}/{len(trials)} | scheduler={scheduler} "
                f"[{SCHEDULER_CONFIGS[scheduler]['description']}] | "
                f"steps={trial['steps']} | cfg={trial['cfg']} | "
                f"gs={trial['guidance_scale']} | prompt_case={trial['prompt_case']} ==="
            )

            ok_count = 0
            for i, record in enumerate(sample, 1):
                print(f"[{i}/{len(sample)}] {record['image_path']}")
                try:
                    result = run_pair(
                        pipe, record, args.dataset_root,
                        trial_output_dir,
                        trial["steps"], trial["cfg"], trial["guidance_scale"],
                        args.seed,
                        args.width, args.height,
                        args.match_input_resolution, args.max_megapixels,
                        args.preserve_untouched, args.preserve_suffix,
                        trial["negative_prompt"],
                        trial["prompt_prefix"],
                        trial["prompt_case"],
                        trial["prompt_max_chars"],
                    )
                except Exception as e:
                    result = {
                        **record,
                        "status":       "failed_exception",
                        "error":        str(e),
                        "output_path":  None,
                        "compare_path": None,
                    }
                    print(f"  error: {result['error']}")

                result["trial_index"] = trial_idx
                result["trial_tag"]   = trial_tag
                result["trial"]       = trial

                if result["status"] == "ok":
                    ok_count += 1
                print(
                    f"  → {result['status']}"
                    + (f"  saved: {result['output_path']}" if result["status"] == "ok" else "")
                )
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()

                if wb_run and result["status"] == "ok":
                    compare_img = Image.open(result["compare_path"])
                    wb_run.log({
                        "trial_index": trial_idx,
                        "trial_tag":   trial_tag,
                        "compare":     wandb.Image(
                            compare_img,
                            caption=f"[{result['operation']}] {result['prompt']}",
                        ),
                        "operation": result["operation"],
                        "scene":     result["image_path"],
                    })

            success_rate = ok_count / max(1, len(sample))
            sweep_row = {
                "trial_index":  trial_idx,
                "trial_tag":    trial_tag,
                "trial":        trial,
                "ok_count":     ok_count,
                "total":        len(sample),
                "success_rate": success_rate,
            }
            sweep_rows.append(sweep_row)
            print(
                f"Trial {trial_idx} summary: {ok_count}/{len(sample)} ok "
                f"({success_rate * 100:.1f}%)"
            )

            if wb_run:
                wb_run.log({
                    "trial_index":  trial_idx,
                    "success_rate": success_rate,
                    "ok_count":     ok_count,
                    # Log all hparams for W&B sweep analysis / parallel coords
                    **{f"hparam/{k}": v for k, v in trial.items() if v is not None},
                })

    # ── Cleanup ───────────────────────────────────────────────────────────
    if pipe is not None:
        del pipe
        torch.cuda.empty_cache()
        gc.collect()

    # ── Write sweep summary (ranked by success rate) ──────────────────────
    if args.sweep:
        ranked = sorted(sweep_rows, key=lambda x: (-x["success_rate"], x["trial_index"]))
        with open(summary_path, "w") as sf:
            json.dump(ranked, sf, indent=2)
        print(f"\nSweep summary written to {summary_path}")

        # Print top-5 trials to stdout for quick inspection
        print("\nTop trials by success rate:")
        for row in ranked[:5]:
            t = row["trial"]
            print(
                f"  [{row['trial_index']:03d}] {t['scheduler']:25s} "
                f"steps={t['steps']:2d} cfg={t['cfg']:.1f} gs={t['guidance_scale']:.2f} "
                f"→ {row['ok_count']}/{row['total']} ok  "
                f"tag={row['trial_tag']}"
            )

    print(f"\nDone. Results logged to {results_path}")
    print(f"Outputs in: {output_dir}")
    if wb_run:
        wb_run.finish()


if __name__ == "__main__":
    main()