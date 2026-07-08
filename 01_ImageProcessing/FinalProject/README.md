# ImageProcessing Final Project: Object-Centric 3D Reconstruction

This project focuses on reconstructing the 3D shape of an object from 2D images, rather than hallucinating a plausible 3D asset from a generative model.

The core idea is a hybrid pipeline:

1. Capture multiple 2D views of one object.
2. Isolate the object using classical segmentation or optional AI segmentation.
3. Match local features across views.
4. Estimate camera motion with RANSAC and epipolar geometry.
5. Triangulate matched points into a sparse 3D point cloud.
6. Evaluate reconstruction quality and discuss where AI helps or hurts.

## Why This Fits the Lecture Topics

The project uses more than four lecture algorithms:

- Histogram/gamma preprocessing
- Edge and morphology based mask cleanup
- GrabCut or region-based segmentation
- SIFT/ORB feature detection
- Feature matching
- RANSAC
- Fundamental/essential matrix estimation
- Camera model and calibration assumptions
- Epipolar geometry
- Triangulation

## Quick Start

Create an environment and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Put object photos here:

```text
data/raw/object01/
```

Recommended capture setup:

- 20 to 40 photos around one rigid object
- fixed focal length, no zoom
- object centered and well lit
- textured background is okay, but a clean background makes segmentation easier
- avoid reflective or transparent objects for the first demo

Run the sparse reconstruction:

```bash
python src/reconstruct_object.py --images data/raw/object01 --out outputs/object01
```

## Optional: Generate Synthetic Views Locally

Use Stable Virtual Camera v1.1 on Ubuntu/Linux for local novel-view generation. For a clean setup, follow `Setup.md`. On Windows, run the same Ubuntu path inside WSL2; older WSL troubleshooting notes are in `SVC_LOCAL_SETUP.md`.

```bash
bash scripts/setup_svc_ubuntu.sh
python3 src/generate_views_svc.py --backend native --image data/source/object.png --out data/raw/object01_svc --target-count 40
python src/reconstruct_object.py --images data/raw/object01_svc --out outputs/object01_svc --pair-strategy window --pair-window 3 --match-mode hybrid --pose-mode orbit
```

These generated views are useful for checking the reconstruction pipeline and for a presentation comparison. They should not be treated as real measured geometry because the model hallucinates unobserved sides of the object.

The script writes:

- `outputs/object01/reconstruction.ply`
- `outputs/object01/matches/*.jpg`
- `outputs/object01/masks/*.png`

Open the `.ply` file in MeshLab, CloudCompare, Blender, or any point-cloud viewer.

## Project Position

Trellis-like sparse latent diffusion models are powerful, but they infer likely geometry from learned priors. This project instead tries to recover geometry supported by image measurements. AI is used as an assistant for object masking or depth priors, while the reconstruction itself is grounded in feature correspondence, camera geometry, and triangulation.
