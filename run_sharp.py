"""
SHARP: remote-sensing image generation.
"""

import argparse
import gc
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT_ROOT = SCRIPT_DIR / "checkpoints"
SHARP_LABEL = "sharp"


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHARP: remote-sensing image generation"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "A majestic orthophoto of a bustling city square, centered around a large "
            "circular marble fountain with intricate water jets and visible ripples. "
            "The square is paved with a complex mosaic of gray paving stones and is "
            "lined with neatly trimmed hedges and flower beds. In the background, a "
            "grand domed church acts as a focal point, its architecture reflected in "
            "the wet surfaces around the fountain. Tall neoclassical buildings border "
            "the square, with distinct window frames and balcony details. Satellite "
            "photography style, high dynamic range, crisp textures, deep shadows, "
            "cinematic urban planning visualization."
        ),
        help="Text prompt used when --prompt_file is not provided",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Optional text file containing one prompt per line",
    )
    parser.add_argument(
        "--prompt_offset",
        type=int,
        default=0,
        help="Skip the first N prompts from --prompt_file",
    )
    parser.add_argument(
        "--prompt_limit",
        type=int,
        default=None,
        help="Only process the first N prompts after applying --prompt_offset",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height in pixels")
    parser.add_argument("--width", type=int, default=1024, help="Image width in pixels")
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="sharp",
        help="Prefix used when saving generated images",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sharp_outputs",
        help="Directory used to save generated images",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(DEFAULT_CKPT_ROOT),
        help="Path to a SHARP checkpoint directory, or to a checkpoints root for auto-discovery",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip images whose target files already exist",
    )
    return parser.parse_args()


def load_prompts(args):
    if not args.prompt_file:
        return [args.prompt]

    prompt_path = Path(args.prompt_file).expanduser().resolve()
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

    prompts = []
    with prompt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            prompt = line.strip()
            if prompt:
                prompts.append(prompt)

    if args.prompt_offset:
        prompts = prompts[args.prompt_offset:]

    if args.prompt_limit is not None:
        prompts = prompts[:args.prompt_limit]

    if not prompts:
        raise ValueError("No prompts available after applying offset/limit.")

    return prompts


def warn_if_resolution_unaligned(width, height):
    if width % 16 != 0 or height % 16 != 0:
        print(
            "Warning: width and height are ideally divisible by 16 for FLUX latent "
            "packing. The effective output size may be rounded down internally."
        )


def find_checkpoint_candidates(checkpoint_root: Path):
    candidates = []
    if not checkpoint_root.exists():
        return candidates

    if (checkpoint_root / "transformer").is_dir():
        return [checkpoint_root]

    for child in sorted(checkpoint_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if (child / "transformer").is_dir():
            candidates.append(child)

    return candidates


def resolve_checkpoint_path(ckpt_arg: str) -> Path:
    ckpt_path = Path(ckpt_arg).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    if ckpt_path.is_dir() and (ckpt_path / "transformer").is_dir():
        return ckpt_path

    if ckpt_path.is_dir():
        candidates = find_checkpoint_candidates(ckpt_path)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            names = ", ".join(candidate.name for candidate in candidates)
            raise ValueError(
                "Multiple checkpoint directories were found under "
                f"{ckpt_path}: {names}. Please pass --ckpt_path explicitly."
            )
        raise FileNotFoundError(
            "No SHARP checkpoint directory was found under "
            f"{ckpt_path}. Expected a directory containing a 'transformer' subfolder."
        )

    raise FileNotFoundError(
        f"Invalid checkpoint path: {ckpt_path}. Expected a checkpoint directory."
    )


def build_output_path(out_dir, save_prefix, seed, height, width, prompt_idx, total_prompts):
    if total_prompts == 1 and prompt_idx == 0:
        filename = f"{save_prefix}_seed_{seed}_{SHARP_LABEL}_{width}x{height}.png"
    else:
        digits = max(3, len(str(total_prompts)))
        filename = (
            f"{save_prefix}_prompt{prompt_idx + 1:0{digits}d}_seed_{seed}_"
            f"{SHARP_LABEL}_{width}x{height}.png"
        )
    return out_dir / filename


def main():
    args = parse_args()
    import torch

    from flux.pipeline_flux import FluxPipeline
    from flux.transformer_flux import FluxTransformer2DModel

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args)
    ckpt_path = resolve_checkpoint_path(args.ckpt_path)

    warn_if_resolution_unaligned(args.width, args.height)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print("=" * 72)
    print(
        f"Loading SHARP | resolution={args.width}x{args.height} | prompts={len(prompts)} | "
        f"checkpoint={ckpt_path.name}"
    )
    print("=" * 72)

    transformer = FluxTransformer2DModel.from_sharp_checkpoint(
        str(ckpt_path),
        torch_dtype=torch_dtype,
    )

    pipe = FluxPipeline.from_pretrained(
        str(ckpt_path),
        transformer=transformer,
        torch_dtype=torch_dtype,
    )
    if device == "cuda":
        try:
            pipe.enable_model_cpu_offload()
        except RuntimeError as exc:
            print(f"Warning: cpu offload unavailable, falling back to pipe.to('cuda'): {exc}")
            pipe = pipe.to(device)
    else:
        print("Warning: CUDA is unavailable, falling back to CPU inference.")
        pipe = pipe.to(device)

    for prompt_idx, prompt in enumerate(prompts):
        current_seed = args.seed + prompt_idx
        output_path = build_output_path(
            out_dir=out_dir,
            save_prefix=args.save_prefix,
            seed=current_seed,
            height=args.height,
            width=args.width,
            prompt_idx=prompt_idx,
            total_prompts=len(prompts),
        )

        if args.skip_existing and output_path.exists():
            print(f"[{prompt_idx + 1}/{len(prompts)}] Skip existing file: {output_path.name}")
            continue

        print(f"[{prompt_idx + 1}/{len(prompts)}] Generating {output_path.name}")
        generator = torch.Generator(device=device).manual_seed(current_seed)
        image = pipe(
            prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            generator=generator,
            num_inference_steps=args.steps,
        ).images[0]
        image.save(output_path)
        print(f"Saved: {output_path}")

    del pipe
    del transformer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
