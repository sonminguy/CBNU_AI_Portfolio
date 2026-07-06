# Setup Guide

This guide is the clean setup path for running the final-project pipeline on Ubuntu/Linux.

The project uses Stable Virtual Camera v1.1 for single-image novel-view generation, then reconstructs a sparse point cloud from the generated multi-view image set. On Windows, use Ubuntu through WSL2 and follow the same Ubuntu commands from the project folder.

## 1. Ubuntu Base Packages

Run inside Ubuntu:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git build-essential
python3 --version
nvidia-smi
```

If `nvidia-smi` is not found, install or update the NVIDIA driver first. For WSL2, this means updating the Windows NVIDIA driver with WSL CUDA support and reopening Ubuntu.

## 2. Project Folder

Native Ubuntu example:

```bash
cd ~/ImageProcessing
```

Current workspace example:

```bash
cd "/home/sims/바탕화면/ImageProcessing"
```

Windows through WSL2 example:

```bash
cd /mnt/c/Users/<USER>/Desktop/ImageProcessing
```

## 3. Install SVC Environment

From the project folder:

```bash
bash scripts/setup_svc_ubuntu.sh
```

What this script does:

- Checks whether the GPU is visible.
- Copies SVC to `~/.cache/imageprocessing-svc/stable-virtual-camera`.
- Creates the venv at `~/.cache/imageprocessing-svc/venv`.
- Records those paths in `stable-virtual-camera/.svc_repo_path` and `stable-virtual-camera/.svc_venv_path`.
- Also writes legacy `.wsl_*` path files so older commands still work.
- Installs CUDA PyTorch, SVC dependencies, `scipy`, and `pycolmap`.

The runnable copy under `~/.cache` avoids permission issues that can happen when the project lives on a Windows-mounted `/mnt/c` folder.

## 4. Hugging Face Login

Activate the SVC environment:

```bash
source "$(cat stable-virtual-camera/.svc_venv_path)/bin/activate"
python -m pip install -U huggingface-hub
huggingface-cli login
huggingface-cli whoami
```

Open these pages in a browser and accept/request access if prompted:

```text
https://huggingface.co/stabilityai/stable-virtual-camera
https://huggingface.co/Manojb/stable-diffusion-2-1-base
```

SVC expects a Diffusers `AutoencoderKL` VAE under a `vae` subfolder. This project uses `Manojb/stable-diffusion-2-1-base` as the SD 2.1-compatible VAE source.

## 5. Generate Multi-View Images

Run from the project folder in Ubuntu:

```bash
python3 src/generate_views_svc.py \
  --backend native \
  --image data/source/Jeans.png \
  --out data/raw/Jeans_svc \
  --vae-repo Manojb/stable-diffusion-2-1-base \
  --target-count 40 \
  --traj-prior orbit \
  --short-side 576
```

The wrapper automatically reads `stable-virtual-camera/.svc_repo_path` and `stable-virtual-camera/.svc_venv_path`, so `--svc-repo` and `--python` are optional after setup.

Expected output:

```text
data/raw/Jeans_svc/view_001.png
data/raw/Jeans_svc/view_002.png
...
```

For a quicker smoke test:

```bash
python3 src/generate_views_svc.py \
  --backend native \
  --image data/source/Jeans.png \
  --out data/raw/Jeans_svc_test \
  --target-count 4 \
  --traj-prior orbit \
  --short-side 576
```

## 6. Reconstruct

Run after the views are generated:

```bash
python3 src/reconstruct_object.py --images data/raw/Jeans_svc --out outputs/Jeans_svc --pair-strategy window --pair-window 3 --match-mode hybrid --pose-mode orbit
```

Main output:

```text
outputs/Jeans_svc/reconstruction.ply
```

For real captured photos where the camera path is not a clean orbit, use pairwise estimated pose instead:

```bash
python3 src/reconstruct_object.py --images data/raw/Jeans_svc --out outputs/Jeans_svc_estimated --pair-strategy window --pair-window 3 --match-mode hybrid --pose-mode estimated
```

## Troubleshooting

`huggingface-cli: command not found`:

```bash
source "$(cat stable-virtual-camera/.svc_venv_path)/bin/activate"
python -m pip install -U huggingface-hub
```

`401 Unauthorized` from Hugging Face:

```bash
source "$(cat stable-virtual-camera/.svc_venv_path)/bin/activate"
huggingface-cli login
huggingface-cli whoami
```

`ModuleNotFoundError: scipy`:

```bash
source "$(cat stable-virtual-camera/.svc_venv_path)/bin/activate"
python -m pip install scipy
```

If the patched VAE loader must be synced manually:

```bash
cp stable-virtual-camera/seva/modules/autoencoder.py "$(cat stable-virtual-camera/.svc_repo_path)/seva/modules/autoencoder.py"
```

## Notes

- Keep `--short-side 576` for an RTX 4070 Super-class GPU.
- Generate 20 to 40 views for the final project; use 4 views only for testing.
- The generated views are useful for experiments and presentation comparison, but they are AI-synthesized views, not physical ground-truth measurements.
- SVC v1.1 uses `modelv1.1.safetensors`; the wrapper default `--weight-name model.safetensors` is intentional because SVC appends `v1.1` internally.
