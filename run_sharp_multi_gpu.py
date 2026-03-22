import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, PriorityQueue


SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = SCRIPT_DIR / "run_sharp.py"
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "rs_t2i_eval_prompts_100.txt"
DEFAULT_CKPT = SCRIPT_DIR / "checkpoints"
DEFAULT_SCALES = ["1024x1024", "1024x1536", "1764x1764", "1920x1024"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-GPU SHARP launcher for remote-sensing image generation"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=str(DEFAULT_PROMPT_FILE),
        help="Text file containing one prompt per line",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=str(DEFAULT_CKPT),
        help="Path to a SHARP checkpoint directory, or to a checkpoints root for auto-discovery",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="sharp_outputs_eval",
        help="Directory for generated images and logs",
    )
    parser.add_argument(
        "--save_prefix",
        type=str,
        default="sharp_eval",
        help="Prefix for output image names",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU ids, e.g. --gpus 0 1 2 or --gpus 0,1,2",
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        default=DEFAULT_SCALES,
        help="Target scales in widthxheight format",
    )
    parser.add_argument("--steps", type=int, default=28, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.5,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--prompt_offset",
        type=int,
        default=0,
        help="Skip the first N prompts in the prompt file",
    )
    parser.add_argument(
        "--prompt_limit",
        type=int,
        default=None,
        help="Only process the first N prompts after the offset",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip output files that already exist",
    )
    parser.add_argument(
        "--stagger_seconds",
        type=float,
        default=8.0,
        help="Delay first launch on later GPUs to reduce simultaneous model loading",
    )
    return parser.parse_args()


def parse_gpu_ids(raw_gpu_args):
    if not raw_gpu_args:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to auto-detect GPUs.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU detected. Please provide --gpus explicitly.")
        return list(range(torch.cuda.device_count()))

    gpu_ids = []
    for item in raw_gpu_args:
        for chunk in item.split(","):
            chunk = chunk.strip()
            if chunk:
                gpu_ids.append(int(chunk))

    if not gpu_ids:
        raise ValueError("No valid GPU ids were provided.")
    return gpu_ids


def parse_scale(scale_text):
    normalized = scale_text.lower().replace(" ", "").replace("×", "x")
    parts = normalized.split("x")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid scale format: {scale_text}")
    width, height = int(parts[0]), int(parts[1])
    return width, height


def build_tasks(args):
    tasks = []
    for order, scale_text in enumerate(args.scales):
        width, height = parse_scale(scale_text)
        pixels = width * height
        tasks.append(
            {
                "priority": (-pixels, order),
                "width": width,
                "height": height,
            }
        )
    return tasks


def build_command(args, task):
    command = [
        sys.executable,
        str(RUN_SCRIPT),
        "--prompt_file",
        str(Path(args.prompt_file).expanduser().resolve()),
        "--ckpt_path",
        str(Path(args.ckpt_path).expanduser().resolve()),
        "--out_dir",
        str(Path(args.out_dir).expanduser().resolve()),
        "--save_prefix",
        args.save_prefix,
        "--width",
        str(task["width"]),
        "--height",
        str(task["height"]),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--guidance_scale",
        str(args.guidance_scale),
        "--prompt_offset",
        str(args.prompt_offset),
    ]

    if args.prompt_limit is not None:
        command.extend(["--prompt_limit", str(args.prompt_limit)])
    if args.skip_existing:
        command.append("--skip_existing")
    return command


def tail_log(log_path, num_lines=20):
    if not log_path.exists():
        return ""
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()
    return "".join(lines[-num_lines:]).strip()


def worker(gpu_id, worker_rank, task_queue, args, log_dir, stop_event, failures):
    first_launch = True
    while not stop_event.is_set():
        try:
            _, task = task_queue.get_nowait()
        except Empty:
            return

        if first_launch and args.stagger_seconds > 0:
            time.sleep(worker_rank * args.stagger_seconds)
            first_launch = False

        log_path = log_dir / f"{args.save_prefix}_{task['width']}x{task['height']}.log"
        command = build_command(args, task)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(
            f"[GPU {gpu_id}] Start SHARP @ {task['width']}x{task['height']} -> {log_path}"
        )
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=str(SCRIPT_DIR),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )

        if result.returncode != 0:
            stop_event.set()
            failures.append(
                {
                    "gpu_id": gpu_id,
                    "task": task,
                    "log_path": log_path,
                    "log_tail": tail_log(log_path),
                    "returncode": result.returncode,
                }
            )
            task_queue.task_done()
            return

        print(f"[GPU {gpu_id}] Done  SHARP @ {task['width']}x{task['height']}")
        task_queue.task_done()


def main():
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)
    out_dir = Path(args.out_dir).expanduser().resolve()
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not RUN_SCRIPT.exists():
        raise FileNotFoundError(f"Runner script not found: {RUN_SCRIPT}")

    tasks = build_tasks(args)
    if not tasks:
        raise ValueError("No tasks were created. Please check your scale settings.")

    task_queue = PriorityQueue()
    for task in tasks:
        task_queue.put((task["priority"], task))

    print("=" * 72)
    print(f"Prompt file : {Path(args.prompt_file).expanduser().resolve()}")
    print(f"Checkpoint  : {Path(args.ckpt_path).expanduser().resolve()}")
    print(f"Output dir  : {out_dir}")
    print(f"GPUs        : {gpu_ids}")
    print(f"Tasks       : {len(tasks)} ({len(args.scales)} scales, SHARP only)")
    print("=" * 72)

    stop_event = threading.Event()
    failures = []
    workers = []
    for worker_rank, gpu_id in enumerate(gpu_ids):
        thread = threading.Thread(
            target=worker,
            args=(gpu_id, worker_rank, task_queue, args, log_dir, stop_event, failures),
            daemon=False,
        )
        thread.start()
        workers.append(thread)

    for thread in workers:
        thread.join()

    if failures:
        failure = failures[0]
        task = failure["task"]
        print("=" * 72)
        print(
            f"FAILED on GPU {failure['gpu_id']} | SHARP | "
            f"{task['width']}x{task['height']} | returncode={failure['returncode']}"
        )
        print(f"Log file: {failure['log_path']}")
        if failure["log_tail"]:
            print("-" * 72)
            print(failure["log_tail"])
            print("-" * 72)
        raise SystemExit(1)

    print("All multi-GPU SHARP generation tasks finished successfully.")


if __name__ == "__main__":
    main()
