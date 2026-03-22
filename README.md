# SHARP

`SHARP` is a remote-sensing image generation codebase built on top of FLUX. This public release currently includes code only.

## ◆ TODO

- ✓ SHARP code
- ○ Fine-tuned RS-FLUX weights
- ○ Training data

## ◆ Method Overview

The figure below is a placeholder for the method / structure diagram that will be added later.

![SHARP Structure Placeholder](docs/structure.png)

## ◆ Repository Layout

```text
SHARP/
├── flux/
│   ├── pipeline_flux.py
│   └── transformer_flux.py
├── docs/
│   └── structure.png
├── checkpoints/
│   └── .gitkeep
├── LICENSE
├── README.md
├── requirements.txt
├── rs_t2i_eval_prompts_100.txt
├── run_sharp.py
├── run_sharp_multi_gpu.py
├── run_sharp.sh
```

## ◆ What Is Included

- `run_sharp.py` → official single-GPU SHARP generation entry point
- `run_sharp_multi_gpu.py` → official multi-GPU SHARP batch launcher
- `run_sharp.sh` → lightweight shell entry point for SHARP inference
- `flux/pipeline_flux.py`, `flux/transformer_flux.py` → SHARP implementation used by the remote-sensing scripts
- `rs_t2i_eval_prompts_100.txt` → example prompt list for batch evaluation

## ◆ Installation

Create a Python environment and install the dependencies:

```bash
conda create -n sharp python=3.10
conda activate sharp
pip install -r requirements.txt
```

## ◆ Entry Points

- `bash run_sharp.sh` → shell entry for single-image or prompt-file generation
- `python run_sharp.py` → single-GPU generation
- `python run_sharp_multi_gpu.py` → multi-GPU batch generation

These entry points were sanity-checked with `--help`.

## ◆ Validation

You can validate the installation and CLI wiring without model weights:

```bash
bash run_sharp.sh --help
python run_sharp.py --help
python run_sharp_multi_gpu.py --help
```

## ◆ Checkpoints

This release does not include model weights.

By default, the scripts look under:

```text
checkpoints/
```

You can use either of these workflows:

- Put exactly one checkpoint directory under `checkpoints/`, then omit `--ckpt_path`
- Pass the checkpoint directory explicitly with `--ckpt_path /path/to/your_checkpoint_dir`

A valid checkpoint path should be the model directory itself, for example:

```text
checkpoints/<your_checkpoint_dir>/
```

## ◆ Quick Start

### 1. Generate from a single prompt

```bash
bash run_sharp.sh \
  --prompt "A satellite image of a rural market town with dense shop blocks, a bus station, surrounding crop fields, narrow feeder roads, and mixed residential and commercial parcels." \
  --width 1024 \
  --height 1024
```

SHARP exposes a single fixed inference path. No method toggle is needed.

### 2. Generate from a prompt file

```bash
python run_sharp.py \
  --prompt_file rs_t2i_eval_prompts_100.txt \
  --width 1024 \
  --height 1536 \
  --ckpt_path checkpoints/<your_checkpoint_dir> \
  --save_prefix sharp_eval
```

### 3. Launch multi-GPU evaluation

```bash
python run_sharp_multi_gpu.py \
  --gpus 0 1 2 \
  --prompt_file rs_t2i_eval_prompts_100.txt \
  --ckpt_path checkpoints/<your_checkpoint_dir> \
  --scales 1024x1024 1764x1764 1024x1536 1920x1024
```

## ◆ Notes

- SHARP uses a single fixed inference path.
- The main generation script uses `--save_prefix` for output naming.
- If `--ckpt_path` is omitted, SHARP auto-discovers a checkpoint only when exactly one checkpoint directory exists under `checkpoints/`.
- For FLUX latent packing, image sizes are ideally divisible by 16. If not, the effective generated size may be rounded down internally.

## ◆ Attribution

This repository retains the upstream license in `LICENSE`.
