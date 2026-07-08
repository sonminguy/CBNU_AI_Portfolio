# Image Processing Final Project Summary

## Project Title

Lecture-Based Feature and Point-Cloud Skeleton Extraction for 3D Reconstruction

## Project Goal

이 프로젝트의 목표는 영상처리 수업에서 배운 알고리즘을 조합하여 3D Reconstruction의 전 단계가 되는 특징점 기반 skeleton을 추출하는 것이다. 최종 결과물은 완성된 3D mesh가 아니라, object mask, local feature, feature correspondence, RANSAC-filtered matches, 그리고 sparse point-cloud skeleton이다.

멀티뷰 이미지는 한 장의 사물 이미지에서 얻은 입력 데이터 역할을 한다. AI view generation 모델은 데이터 확보를 위한 보조 도구일 뿐이며, 프로젝트의 핵심은 생성된 여러 view에서 영상처리 알고리즘을 통해 reconstruction에 필요한 구조적 단서를 추출하는 과정이다.

## Project Scope

본 프로젝트는 다음 질문에 답하는 것을 목표로 한다.

- 한 장의 사물 이미지에서 얻은 multi-view image set을 reconstruction 입력으로 사용할 수 있는가?
- 각 view에서 object 영역을 분리하고, 안정적인 특징점을 추출할 수 있는가?
- view 간 특징점 대응을 만들고 outlier를 제거할 수 있는가?
- 대응점이 camera geometry와 triangulation을 통해 sparse point-cloud skeleton으로 확장될 수 있는가?

따라서 발표에서는 AI 모델 자체보다 feature extraction, segmentation, matching, RANSAC, epipolar geometry, triangulation을 중심으로 설명한다.

## Motivation

3D Reconstruction은 일반적으로 이미지들 사이의 대응점에서 출발한다. 좋은 correspondence가 없으면 camera pose estimation, triangulation, dense reconstruction, mesh reconstruction 모두 불안정해진다. 그래서 이 프로젝트에서는 완성형 3D 모델을 바로 만드는 대신, reconstruction pipeline의 뼈대가 되는 특징점과 sparse point cloud를 추출하는 데 집중했다.

전체 흐름은 다음과 같다.

1. 단일 사물 이미지에서 multi-view image set을 준비한다.
2. 각 view에서 foreground object mask를 만든다.
3. CLAHE, SIFT/AKAZE/ORB, RootSIFT 등을 이용해 특징점을 추출한다.
4. Lowe ratio test와 symmetric matching으로 view 간 대응점을 만든다.
5. RANSAC과 fundamental/essential matrix estimation으로 outlier를 제거한다.
6. 검증된 대응점을 triangulation하여 point-cloud skeleton을 만든다.

## Role of AI View Generation

AI 기반 view generation은 프로젝트의 중심 알고리즘이 아니다. 실제 촬영 multi-view dataset이 없을 때 실험용 view set을 만들기 위한 입력 생성 도구로만 사용했다.

발표에서 강조할 점은 다음이다.

- AI 모델은 multi-view image를 제공하는 보조 단계이다.
- reconstruction skeleton은 OpenCV 기반 영상처리 알고리즘으로 추출한다.
- 결과물의 핵심은 generated image 자체가 아니라 mask, feature, correspondence, sparse point cloud이다.

## Relation to Lecture Topics

프로젝트에는 강의에서 다룬 여러 영상처리 및 컴퓨터 비전 개념이 직접 사용된다.

- Grayscale conversion
- CLAHE contrast enhancement
- Morphological opening and closing
- GrabCut foreground segmentation
- SIFT, AKAZE, ORB feature extraction
- Descriptor normalization
- Feature matching and Lowe ratio test
- Symmetric matching
- Dense optical flow
- RANSAC outlier rejection
- Fundamental matrix estimation
- Essential matrix estimation
- Camera intrinsic/extrinsic model
- Epipolar geometry
- Linear triangulation
- Reprojection error filtering
- PLY point-cloud export

## System Overview

전체 시스템은 AI view preparation과 lecture-based feature skeleton extraction으로 나뉜다.

### Stage 1. Multi-View Input Preparation

한 장의 사물 이미지를 기반으로 여러 시점 이미지를 준비한다. 이 단계는 reconstruction 알고리즘을 검증하기 위한 입력 데이터 생성 단계이다.

현재 사용한 입력과 산출물:

```text
source image: data/source/Jeans.png
multi-view images: data/raw/Jeans_svc/view_*.png
number of generated views: 41
```

### Stage 2. Feature and Point-Cloud Skeleton Extraction

각 view에서 object mask와 특징점을 추출하고, 이미지 쌍 사이의 대응점을 계산한다. 이후 RANSAC으로 outlier를 제거하고 triangulation으로 sparse skeleton point cloud를 만든다.

주요 산출물:

```text
object masks: outputs/Jeans_svc/masks/*.png
match visualizations: outputs/Jeans_svc/matches/*.jpg
point-cloud skeleton: outputs/Jeans_svc/reconstruction.ply
pipeline step images: outputs/Jeans_svc/pipeline_steps/*.jpg
```

## Pipeline Step Images

MD와 PPT에 사용할 파이프라인 단계 이미지는 다음 6개로 정리했다.

### 1. Multi-view Input

![Multi-view Input](outputs/Jeans_svc/pipeline_steps/01_multi_view_input.jpg)

### 2. Object Mask

![Object Mask](outputs/Jeans_svc/pipeline_steps/02_object_mask.jpg)

### 3. Feature Extraction

![Feature Extraction](outputs/Jeans_svc/pipeline_steps/03_feature_extraction.jpg)

### 4. Correspondence

![Correspondence](outputs/Jeans_svc/pipeline_steps/04_correspondence.jpg)

### 5. RANSAC Geometry

![RANSAC Geometry](outputs/Jeans_svc/pipeline_steps/05_ransac_geometry.jpg)

### 6. Triangulation

![Triangulation](outputs/Jeans_svc/pipeline_steps/06_triangulation.jpg)

## Core Algorithm Walkthrough With Code

이 섹션은 실제 특징점 추출 코드 중 영상처리 수업과 직접 연결되는 부분을 정리한다.

### 1. Grayscale Conversion and CLAHE

컬러 이미지를 grayscale로 변환하고 CLAHE를 적용해 local contrast를 높인다. 이는 조명 차이와 낮은 대비로 인해 특징점이 적게 검출되는 문제를 완화한다.

```python
def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
```

강의 연결:

- intensity transform
- histogram equalization
- local contrast enhancement
- feature detection 전처리

### 2. GrabCut Segmentation and Morphology

사물 중심 이미지를 가정하고 중앙 rectangle을 foreground 후보로 둔 뒤 GrabCut을 적용한다. 이후 morphology opening과 closing으로 작은 잡음과 구멍을 정리한다.

```python
def central_grabcut_mask(image: np.ndarray, iterations: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    labels = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, labels, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask
```

강의 연결:

- foreground/background segmentation
- graph-cut 기반 region segmentation
- morphological opening
- morphological closing
- mask 기반 object-centric processing

### 3. SIFT, AKAZE, ORB Feature Extraction

OpenCV 환경에서 사용 가능한 detector를 순차적으로 선택한다. SIFT가 가능하면 SIFT를 우선 사용하고, 불가능하면 AKAZE, ORB로 fallback한다.

```python
def create_detector():
    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=12)
    if hasattr(cv2, "AKAZE_create"):
        return "akaze", cv2.AKAZE_create()
    return "orb", cv2.ORB_create(nfeatures=5000)
```

특징점은 mask 내부에서만 검출한다.

```python
keypoints, descriptors = detector.detectAndCompute(gray, mask)
```

강의 연결:

- local feature detection
- scale-space feature
- keypoint descriptor
- object 영역 기반 feature extraction

### 4. RootSIFT Descriptor Normalization

float descriptor는 RootSIFT 방식으로 정규화한다. descriptor를 L1 normalize한 뒤 square root를 적용해 matching 안정성을 높인다.

```python
def rootsift(descriptors: np.ndarray | None) -> np.ndarray | None:
    if descriptors is None:
        return None
    descriptors = descriptors.astype(np.float32)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(descriptors)
```

강의 연결:

- feature descriptor normalization
- distance metric sensitivity
- robust matching을 위한 descriptor preprocessing

### 5. Lowe Ratio Test and Symmetric Matching

k-NN matching을 수행한 뒤 Lowe ratio test로 모호한 match를 제거한다. A to B, B to A 양방향 matching을 모두 수행하고, 서로 일치하는 match만 유지한다.

```python
def ratio_matches(knn: list, ratio: float = 0.8) -> list:
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good
```

```python
ab = ratio_matches(matcher.knnMatch(a.descriptors, b.descriptors, k=2))
ba = ratio_matches(matcher.knnMatch(b.descriptors, a.descriptors, k=2))
reverse = {(m.trainIdx, m.queryIdx) for m in ba}
good = [m for m in ab if (m.queryIdx, m.trainIdx) in reverse]
```

강의 연결:

- nearest-neighbor matching
- descriptor distance
- ambiguous match rejection
- cross-check matching

### 6. Dense Optical Flow Fallback

feature matching이 충분하지 않을 때 Farneback dense optical flow를 사용해 추가 correspondence를 만든다. forward-backward consistency를 이용해 흐름이 불안정한 point를 제거한다.

```python
flow_ab = cv2.calcOpticalFlowFarneback(a.gray, b.gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)
flow_ba = cv2.calcOpticalFlowFarneback(b.gray, a.gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)
```

```python
bdx, bdy = flow_ba[yi, xi]
if np.hypot(dx + bdx, dy + bdy) > fb_threshold:
    continue
```

강의 연결:

- optical flow
- motion correspondence
- forward-backward consistency
- dense correspondence estimation

### 7. RANSAC With Fundamental Matrix

대응점에는 outlier가 포함되므로 RANSAC으로 fundamental matrix를 추정하고 inlier correspondence만 남긴다.

```python
fmat, inlier_mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.999)
```

강의 연결:

- RANSAC
- outlier rejection
- epipolar constraint
- fundamental matrix

### 8. Essential Matrix and Pose Estimation

일반 이미지 쌍에서는 camera intrinsic matrix를 근사하고 essential matrix를 추정한다. 이후 `recoverPose`로 상대 camera pose를 얻는다.

```python
essential, inlier_mask = cv2.findEssentialMat(
    pts_a,
    pts_b,
    k,
    method=cv2.RANSAC,
    prob=0.999,
    threshold=1.0,
)
_, r, t, pose_mask = cv2.recoverPose(essential, pts_a_in, pts_b_in, k)
```

강의 연결:

- camera intrinsic matrix
- essential matrix
- relative camera pose
- epipolar geometry

### 9. Camera Intrinsic Approximation

실제 calibration 정보가 없기 때문에 이미지 크기를 바탕으로 단순 intrinsic matrix를 근사한다.

```python
def estimate_intrinsics(image: np.ndarray, focal_scale: float = 1.2) -> np.ndarray:
    h, w = image.shape[:2]
    focal = focal_scale * max(w, h)
    return np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
```

강의 연결:

- pinhole camera model
- intrinsic parameter
- focal length and principal point

### 10. Triangulation and Point-Cloud Skeleton

검증된 correspondence와 projection matrix를 이용해 2D point pair를 3D point로 triangulation한다.

```python
points_h = cv2.triangulatePoints(p1, p2, pts_a_in.T, pts_b_in.T)
points = (points_h[:3] / points_h[3]).T
```

추가로 reprojection error를 계산해 geometry와 맞지 않는 point를 제거한다.

```python
reproj_a = reproject_points(points, p1)
reproj_b = reproject_points(points, p2)
err = np.linalg.norm(reproj_a - pts_a_in, axis=1) + np.linalg.norm(reproj_b - pts_b_in, axis=1)
keep = err < max_reprojection_error
points = points[keep]
```

강의 연결:

- projective geometry
- triangulation
- reprojection error
- sparse point cloud

### 11. PLY Export

최종 3D skeleton point를 PLY 형식으로 저장한다.

```python
def write_ply(path: Path, points: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
```

강의 연결:

- 3D point representation
- sparse reconstruction output
- visualization-ready point cloud

## Outputs

이 프로젝트의 산출물은 완성된 3D mesh가 아니라 reconstruction skeleton 검증 자료이다.

```text
outputs/Jeans_svc/pipeline_steps/01_multi_view_input.jpg
outputs/Jeans_svc/pipeline_steps/02_object_mask.jpg
outputs/Jeans_svc/pipeline_steps/03_feature_extraction.jpg
outputs/Jeans_svc/pipeline_steps/04_correspondence.jpg
outputs/Jeans_svc/pipeline_steps/05_ransac_geometry.jpg
outputs/Jeans_svc/pipeline_steps/06_triangulation.jpg
outputs/Jeans_svc/masks/*.png
outputs/Jeans_svc/matches/*.jpg
outputs/Jeans_svc/reconstruction.ply
```

각 파일의 의미:

- `pipeline_steps/01_multi_view_input.jpg`: multi-view input 대표 이미지
- `pipeline_steps/02_object_mask.jpg`: object mask overlay
- `pipeline_steps/03_feature_extraction.jpg`: 검출된 local feature
- `pipeline_steps/04_correspondence.jpg`: RANSAC 적용 전 correspondence
- `pipeline_steps/05_ransac_geometry.jpg`: RANSAC geometry filtering 후 correspondence
- `pipeline_steps/06_triangulation.jpg`: triangulation 결과인 sparse point-cloud skeleton projection
- `masks/*.png`: GrabCut과 morphology로 얻은 object mask
- `matches/*.jpg`: feature/correspondence matching 결과 시각화
- `reconstruction.ply`: 대응점을 triangulation한 sparse point-cloud skeleton

## Limitations

현재 시스템은 3D Reconstruction의 최종 결과물이 아니라 특징점 추출 및 대응점 검증 backbone이므로 다음 한계가 있다.

- Multi-view input이 실제 촬영 이미지가 아니라 생성 이미지일 경우 true geometry가 아닐 수 있다.
- 가려진 영역은 view generation model의 prior에 의해 만들어질 수 있다.
- Camera intrinsic/extrinsic은 실제 calibration이 아니라 근사값이다.
- Point cloud는 특징점 기반 sparse skeleton이며 dense reconstruction이나 mesh reconstruction은 포함하지 않는다.
- Texture가 적거나 반복 패턴이 많은 object에서는 correspondence 품질이 낮아질 수 있다.
- Transparent, reflective object는 segmentation과 feature matching 모두 어렵다.

## Future Work

개선 방향은 다음과 같다.

- 실제 촬영 multi-view image와 generated multi-view image 비교
- Camera calibration 또는 COLMAP 연동
- SAM 또는 다른 AI segmentation 모델로 mask 개선
- Bundle adjustment 적용
- Dense matching 추가
- 현재 feature/correspondence backbone 위에 dense reconstruction 또는 surface reconstruction 적용
- Poisson reconstruction 또는 mesh generation
- Quantitative evaluation metric 추가

## Presentation Structure Suggestion

PPT는 다음 순서로 구성하면 좋다.

1. Project goal: 3D Reconstruction을 위한 feature skeleton 추출
2. Why feature correspondence matters in 3D Reconstruction
3. Overall pipeline: multi-view input to sparse point-cloud skeleton
4. Lecture algorithms used in the project
5. Image preprocessing: grayscale and CLAHE
6. Object segmentation: GrabCut and morphology
7. Local feature extraction: SIFT, AKAZE, ORB
8. Descriptor normalization and feature matching
9. Robust correspondence: Lowe ratio, symmetric matching, RANSAC
10. Optical flow fallback for weak feature cases
11. Camera geometry: intrinsic approximation and epipolar constraints
12. Triangulation and sparse point-cloud skeleton
13. Results: masks, match visualizations, PLY point cloud
14. Limitations and future extension to full 3D Reconstruction

## Conclusion

이 프로젝트는 AI 모델을 강조하는 프로젝트가 아니라, 영상처리 수업에서 배운 알고리즘을 실제 reconstruction pipeline의 backbone으로 연결한 프로젝트이다. 핵심 결과는 object mask, local feature, feature correspondence, RANSAC-filtered matching, 그리고 sparse point-cloud skeleton이다. 이 skeleton은 이후 camera calibration, bundle adjustment, dense reconstruction, mesh reconstruction으로 확장될 수 있는 기반 단계이다.

## Source Code 1: `src/generate_views_svc.py`

```python
from __future__ import annotations

import argparse
import os
import platform
import shutil
import shlex
import subprocess
import sys
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SVC_REPO_PATH_FILES = (".svc_repo_path", ".wsl_svc_repo_path")
SVC_VENV_PATH_FILES = (".svc_venv_path", ".wsl_svc_venv_path")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_generated_views(folder: Path) -> None:
    ensure_dir(folder)
    for pattern in ("view_*.png", "source_input.png"):
        for path in folder.glob(pattern):
            try:
                path.unlink()
            except PermissionError:
                print(f"Warning: locked file was not removed: {path}")


def prepare_single_image_input(image_path: Path, work_dir: Path) -> Path:
    ensure_dir(work_dir)
    prepared_path = work_dir / image_path.name
    image = Image.open(image_path).convert("RGB")
    image.save(prepared_path)
    return prepared_path


def is_windows() -> bool:
    return platform.system().lower() == "windows"


def wsl_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    text = str(resolved)
    drive, rest = os.path.splitdrive(text)
    if not drive:
        return text.replace("\\", "/")
    drive_letter = drive.rstrip(":").lower()
    rest = rest.replace("\\", "/").lstrip("/")
    return f"/mnt/{drive_letter}/{rest}"


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def read_first_existing_path_file(repo: Path, names: tuple[str, ...]) -> str | None:
    for name in names:
        path_file = repo / name
        if path_file.exists():
            value = path_file.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def iter_path_file_values(repo: Path, names: tuple[str, ...]):
    for name in names:
        path_file = repo / name
        if path_file.exists():
            value = path_file.read_text(encoding="utf-8").strip()
            if value:
                yield value


def python_from_venv_dir(venv_dir: str, backend: str) -> str | None:
    python_path = Path(venv_dir).expanduser() / "bin" / "python"
    if python_path.exists():
        return wsl_path(python_path) if backend == "wsl" else str(python_path)
    return None


def resolve_svc_repo(repo: Path, backend: str) -> Path:
    repo = repo.expanduser().resolve()
    recorded_repo = read_first_existing_path_file(repo, SVC_REPO_PATH_FILES)
    if recorded_repo and backend != "wsl":
        recorded_path = Path(recorded_repo).expanduser()
        if (recorded_path / "demo.py").exists():
            return recorded_path.resolve()
    return repo


def find_python(repo: Path, requested: str, backend: str, fallback_repo: Path | None = None) -> str:
    if requested != "auto":
        if backend == "wsl" and os.path.splitdrive(requested)[0]:
            return wsl_path(Path(requested))
        return requested

    for path_repo in (repo, fallback_repo):
        if path_repo is None:
            continue
        for venv_dir in iter_path_file_values(path_repo, SVC_VENV_PATH_FILES):
            python_path = python_from_venv_dir(venv_dir, backend)
            if python_path:
                return python_path
            print(f"Warning: recorded SVC Python does not exist: {Path(venv_dir).expanduser() / 'bin' / 'python'}")

    if backend == "wsl":
        candidates = [
            repo / ".venv-wsl" / "bin" / "python",
            repo / ".venv" / "bin" / "python",
            repo / "venv" / "bin" / "python",
        ]
    else:
        candidates = [
            repo / ".venv" / "Scripts" / "python.exe",
            repo / "venv" / "Scripts" / "python.exe",
            repo / ".venv" / "bin" / "python",
            repo / "venv" / "bin" / "python",
        ]
    for candidate in candidates:
        if candidate.exists():
            return wsl_path(candidate) if backend == "wsl" else str(candidate)
    if backend == "wsl":
        return "python3"
    return sys.executable


def run_command(
    command: list[str],
    cwd: Path | None = None,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def query_cuda_memory() -> tuple[int, int] | None:
    if shutil.which("nvidia-smi") is None:
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    first_line = next((line.strip() for line in result.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    try:
        free_text, total_text = [part.strip() for part in first_line.split(",", maxsplit=1)]
        return int(free_text), int(total_text)
    except ValueError:
        return None


def check_cuda_memory(min_free_gb: float) -> None:
    if min_free_gb <= 0:
        return
    memory = query_cuda_memory()
    if memory is None:
        return
    free_mib, total_mib = memory
    required_mib = int(min_free_gb * 1024)
    if free_mib < required_mib:
        raise RuntimeError(
            "Not enough free GPU memory for Stable Virtual Camera.\n"
            f"Free VRAM: {free_mib / 1024:.1f} GB / {total_mib / 1024:.1f} GB\n"
            f"Required by preflight: {min_free_gb:.1f} GB\n\n"
            "Run `nvidia-smi` and stop other GPU-heavy processes, then retry.\n"
            "If you intentionally want to try anyway, pass `--min-free-vram-gb 0`."
        )


def run_wsl_command(
    command: list[str],
    repo: Path,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    linux_repo = wsl_path(repo)
    exports = ""
    if env:
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    shell_command = f"cd {shlex.quote(linux_repo)} && {exports}{shell_join(command)}"
    return subprocess.run(
        ["wsl", "bash", "-lc", shell_command],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=check,
    )


def decode_process_output(data: bytes) -> str:
    if not data:
        return ""
    if data.count(b"\x00") > max(1, len(data) // 8):
        return data.decode("utf-16le", errors="replace").replace("\x00", "")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp949", errors="replace")


def check_wsl_available() -> None:
    if shutil.which("wsl") is None:
        raise RuntimeError("wsl.exe was not found. Install WSL first with: wsl --install -d Ubuntu-22.04")
    result = subprocess.run(
        ["wsl", "bash", "-lc", "printf ready"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        wsl_output = decode_process_output(result.stderr or result.stdout).strip()
        raise RuntimeError(
            "WSL is not ready on this computer.\n\n"
            "Install Ubuntu 22.04 from an Administrator PowerShell:\n"
            "  wsl --install -d Ubuntu-22.04\n\n"
            "After rebooting/opening Ubuntu, run:\n"
            "  nvidia-smi\n"
            "  cd /mnt/c/Users/zcz84/Desktop/ImageProcessing\n"
            "  bash scripts/setup_svc_wsl.sh\n\n"
            f"WSL output:\n{wsl_output}"
        )


def check_svc_environment(python_exe: str, repo: Path, backend: str) -> None:
    if backend != "wsl" and not shutil.which(python_exe) and not Path(python_exe).exists():
        raise RuntimeError(
            "Stable Virtual Camera Python executable was not found.\n"
            f"Python used: {python_exe}\n\n"
            "Run the Ubuntu setup again from the project folder:\n"
            "  bash scripts/setup_svc_ubuntu.sh"
        )

    check = (
        "import importlib.util, sys; "
        "print(sys.executable); "
        "missing=[m for m in ['fire','seva','torch'] if importlib.util.find_spec(m) is None]; "
        "print(','.join(missing))"
    )
    if backend == "wsl":
        check_wsl_available()
        result = run_wsl_command([python_exe, "-c", check], repo)
    else:
        result = run_command([python_exe, "-c", check], cwd=repo)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    executable = lines[0] if lines else python_exe
    missing = lines[1].split(",") if len(lines) > 1 and lines[1] else []
    if result.returncode != 0 or missing:
        missing_text = ", ".join(missing) if missing else "unknown import check failure"
        raise RuntimeError(
            "Stable Virtual Camera environment is not ready.\n"
            f"Backend: {backend}\n"
            f"Python used: {executable}\n"
            f"Missing modules: {missing_text}\n\n"
            "Recommended setup from Ubuntu/Linux:\n"
            "  bash scripts/setup_svc_ubuntu.sh\n\n"
            "If you are launching from Windows, run the same setup inside WSL Ubuntu first.\n\n"
            "Native Windows may fail later because the official Stable Virtual Camera repo recommends WSL for Windows users."
        )


def resolve_svc_hf_weight_name(weight_name: str, model_version: float) -> str:
    if model_version <= 1:
        return weight_name
    base, ext = os.path.splitext(weight_name)
    version_suffix = f"v{model_version}"
    if base.endswith(version_suffix):
        return weight_name
    return f"{base}{version_suffix}{ext}"


def check_huggingface_access(
    python_exe: str,
    repo: Path,
    backend: str,
    weight_name: str,
    vae_repo: str,
) -> None:
    check = r"""
from huggingface_hub import hf_hub_download

targets = [
    ("__VAE_REPO__", "vae/diffusion_pytorch_model.safetensors"),
    ("stabilityai/stable-virtual-camera", "__WEIGHT_NAME__"),
]

for repo_id, filename in targets:
    try:
        hf_hub_download(repo_id=repo_id, filename=filename, dry_run=True)
    except Exception as exc:
        raise SystemExit(f"{repo_id}/{filename}: {type(exc).__name__}: {exc}")
print("huggingface-access-ok")
""".replace("__VAE_REPO__", vae_repo).replace("__WEIGHT_NAME__", weight_name)

    if backend == "wsl":
        result = run_wsl_command([python_exe, "-c", check], repo)
    else:
        result = run_command([python_exe, "-c", check], cwd=repo)

    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(
            "Hugging Face access is not ready.\n\n"
            "Log in inside the same SVC environment and accept both gated model terms:\n"
            "  source \"$(cat stable-virtual-camera/.svc_venv_path 2>/dev/null || cat stable-virtual-camera/.wsl_svc_venv_path)/bin/activate\"\n"
            "  hf auth login --force\n"
            "  hf auth whoami\n\n"
            "Open and accept access if prompted:\n"
            "  https://huggingface.co/stabilityai/stable-virtual-camera\n"
            f"  https://huggingface.co/{vae_repo}\n\n"
            f"Underlying error:\n{details}"
        )


def choose_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "wsl" if is_windows() else "native"


def run_stable_virtual_camera(args: argparse.Namespace, prepared_image: Path) -> Path:
    backend = choose_backend(args.backend)
    requested_repo = args.svc_repo.expanduser().resolve()
    repo = resolve_svc_repo(requested_repo, backend)
    if not (repo / "demo.py").exists():
        raise FileNotFoundError(f"Stable Virtual Camera repo not found or incomplete: {repo}")

    python_exe = find_python(repo, args.python, backend, fallback_repo=requested_repo)
    check_svc_environment(python_exe, repo, backend)
    check_huggingface_access(
        python_exe,
        repo,
        backend,
        resolve_svc_hf_weight_name(args.weight_name, args.model_version),
        args.vae_repo,
    )
    if backend == "native":
        check_cuda_memory(args.min_free_vram_gb)

    data_path = wsl_path(prepared_image.parent) if backend == "wsl" else str(prepared_image.parent.resolve())
    svc_env = os.environ.copy()
    svc_env["SEVA_VAE_REPO"] = args.vae_repo
    svc_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command = [
        python_exe,
        "demo.py",
        "--data_path",
        data_path,
        "--task",
        "img2trajvid_s-prob",
        "--replace_or_include_input",
        "True",
        "--traj_prior",
        args.traj_prior,
        "--cfg",
        args.cfg,
        "--guider",
        args.guider,
        "--num_targets",
        str(args.target_count),
        "--L_short",
        str(args.short_side),
        "--use_traj_prior",
        "True",
        "--chunk_strategy",
        args.chunk_strategy,
        "--camera_scale",
        str(args.camera_scale),
        "--version",
        str(args.model_version),
        "--weight_name",
        args.weight_name,
        "--seed",
        str(args.seed),
        "--save_subdir",
        args.save_subdir,
        "--video_save_fps",
        str(args.fps),
    ]

    print(f"Running Stable Virtual Camera via {backend}:")
    print(shell_join(command))
    if backend == "wsl":
        result = run_wsl_command(command, repo, env={"SEVA_VAE_REPO": args.vae_repo})
        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            raise subprocess.CalledProcessError(result.returncode, ["wsl", "bash", "-lc", shell_join(command)])
    else:
        subprocess.run(command, cwd=repo, env=svc_env, check=True)

    scene_name = prepared_image.stem
    output_scene = repo / "work_dirs" / "demo" / "img2trajvid_s-prob" / args.save_subdir / scene_name
    if not output_scene.exists():
        raise FileNotFoundError(f"Stable Virtual Camera output folder was not found: {output_scene}")
    return output_scene


def collect_outputs(output_scene: Path, out_dir: Path, include_input: bool) -> None:
    clear_generated_views(out_dir)

    generated = sorted((output_scene / "samples-rgb").glob("*.png"))
    if not generated:
        raise FileNotFoundError(f"No generated PNG files found in {output_scene / 'samples-rgb'}")

    index = 1
    if include_input:
        input_images = sorted((output_scene / "input").glob("*.png"))
        if input_images:
            shutil.copyfile(input_images[0], out_dir / "source_input.png")

    for image_path in generated:
        shutil.copyfile(image_path, out_dir / f"view_{index:03d}.png")
        index += 1

    print(f"Copied {len(generated)} generated views to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate novel views locally with Stable Virtual Camera v1.1.")
    parser.add_argument("--image", type=Path, required=True, help="Input image.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder for generated views.")
    parser.add_argument(
        "--svc-repo",
        type=Path,
        default=Path("stable-virtual-camera"),
        help="Path to the cloned Stability-AI/stable-virtual-camera repository.",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "wsl", "native"],
        default="auto",
        help="Execution backend. On Windows, 'auto' uses WSL because SVC recommends WSL.",
    )
    parser.add_argument(
        "--python",
        default="auto",
        help="Python executable inside the SVC environment. 'auto' uses .venv if present.",
    )
    parser.add_argument("--target-count", type=int, default=40, help="Number of target views to generate.")
    parser.add_argument("--short-side", type=int, default=576, help="SVC L_short preprocessing size.")
    parser.add_argument("--traj-prior", default="orbit", help="SVC preset trajectory, e.g. orbit, spiral, pan-left.")
    parser.add_argument("--camera-scale", type=float, default=2.0, help="SVC camera motion scale.")
    parser.add_argument("--cfg", default="4.0,2.0", help="SVC two-pass CFG values.")
    parser.add_argument("--guider", default="1,2", help="SVC guider setting.")
    parser.add_argument("--chunk-strategy", default="interp", help="SVC chunk strategy.")
    parser.add_argument("--model-version", type=float, default=1.1, help="Stable Virtual Camera model version.")
    parser.add_argument(
        "--vae-repo",
        default="Manojb/stable-diffusion-2-1-base",
        help="Diffusers repo containing the SD 2.1-compatible VAE used by SVC.",
    )
    parser.add_argument(
        "--weight-name",
        default="model.safetensors",
        help="Base checkpoint file name. SVC appends v1.1 automatically when --model-version is 1.1.",
    )
    parser.add_argument("--seed", type=int, default=23, help="Random seed.")
    parser.add_argument("--fps", type=float, default=10.0, help="Saved video FPS.")
    parser.add_argument("--save-subdir", default="imageprocessing", help="SVC work_dirs/demo subfolder.")
    parser.add_argument("--include-input", action="store_true", help="Also copy source_input.png.")
    parser.add_argument(
        "--min-free-vram-gb",
        type=float,
        default=12.0,
        help="Minimum free GPU memory required before launching SVC. Use 0 to skip the check.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        parser.error(f"Input image does not exist: {args.image}")
    if args.image.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f"Unsupported image extension: {args.image.suffix}")
    if args.target_count < 1:
        parser.error("--target-count must be at least 1.")
    return args


def main() -> None:
    args = parse_args()
    ensure_dir(args.out)
    temp_input_dir = args.out / "_svc_input"
    prepared_image = prepare_single_image_input(args.image, temp_input_dir)
    output_scene = run_stable_virtual_camera(args, prepared_image)
    collect_outputs(output_scene, args.out, args.include_input)


if __name__ == "__main__":
    main()

```

## Source Code 2: `src/reconstruct_object.py`

```python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ViewFeatures:
    image: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    keypoints: list
    descriptors: np.ndarray | None


@dataclass
class Correspondences:
    pts_a: np.ndarray
    pts_b: np.ndarray
    matches: list | None = None
    inliers_for_draw: np.ndarray | None = None
    method: str = "features"


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    canvas = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.5, min(canvas.shape[:2]) / 700.0)
    thickness = max(1, int(round(scale * 2)))
    (text_w, text_h), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = int(8 * scale) + 4
    cv2.rectangle(canvas, (0, 0), (text_w + pad * 2, text_h + pad * 2), (0, 0, 0), -1)
    cv2.putText(canvas, label, (pad, text_h + pad), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return canvas


def resize_to_height(image: np.ndarray, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == height:
        return image
    width = max(1, int(round(w * height / h)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_image_grid(images: list[np.ndarray], labels: list[str], cell_size: tuple[int, int] = (260, 260), cols: int = 4) -> np.ndarray:
    cell_w, cell_h = cell_size
    cells = []
    for image, label in zip(images, labels):
        h, w = image.shape[:2]
        scale = min(cell_w / w, cell_h / h)
        resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        cell = np.full((cell_h, cell_w, 3), 245, dtype=np.uint8)
        y = (cell_h - resized.shape[0]) // 2
        x = (cell_w - resized.shape[1]) // 2
        cell[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        cells.append(add_label(cell, label))

    rows = []
    for start in range(0, len(cells), cols):
        row = cells[start : start + cols]
        while len(row) < cols:
            row.append(np.full((cell_h, cell_w, 3), 245, dtype=np.uint8))
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def natural_key(path: Path) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.stem)]


def load_images(folder: Path) -> list[Path]:
    excluded_name_parts = {"grid", "mask"}
    paths = [
        p
        for p in sorted(folder.iterdir(), key=natural_key)
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not any(part in p.stem.lower() for part in excluded_name_parts)
    ]
    if len(paths) < 2:
        raise ValueError(f"Need at least two images in {folder}")
    return paths


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preprocess(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def rootsift(descriptors: np.ndarray | None) -> np.ndarray | None:
    if descriptors is None:
        return None
    descriptors = descriptors.astype(np.float32)
    descriptors /= descriptors.sum(axis=1, keepdims=True) + 1e-7
    return np.sqrt(descriptors)


def central_grabcut_mask(image: np.ndarray, iterations: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
    labels = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, labels, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def create_detector():
    if hasattr(cv2, "SIFT_create"):
        return "sift", cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.01, edgeThreshold=12)
    if hasattr(cv2, "AKAZE_create"):
        return "akaze", cv2.AKAZE_create()
    return "orb", cv2.ORB_create(nfeatures=5000)


def extract_features(image_path: Path, mask_dir: Path) -> ViewFeatures:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = preprocess(image)
    mask = central_grabcut_mask(image)
    ensure_dir(mask_dir)
    cv2.imwrite(str(mask_dir / f"{image_path.stem}_mask.png"), mask)

    _, detector = create_detector()
    keypoints, descriptors = detector.detectAndCompute(gray, mask)
    if descriptors is not None and descriptors.dtype != np.uint8:
        descriptors = rootsift(descriptors)
    return ViewFeatures(image=image, gray=gray, mask=mask, keypoints=keypoints, descriptors=descriptors)


def save_stage_images(image_path: Path, features: ViewFeatures, stage_dir: Path) -> None:
    ensure_dir(stage_dir)
    stem = image_path.stem

    gray_bgr = cv2.cvtColor(features.gray, cv2.COLOR_GRAY2BGR)
    mask_bgr = cv2.cvtColor(features.mask, cv2.COLOR_GRAY2BGR)

    overlay = features.image.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    overlay = np.where(features.mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.65, green, 0.35, 0), overlay)

    keypoints_vis = cv2.drawKeypoints(
        features.image,
        features.keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    stages = [
        ("01_original", features.image, "Original"),
        ("02_clahe", gray_bgr, "CLAHE"),
        ("03_mask", mask_bgr, "GrabCut Mask"),
        ("04_mask_overlay", overlay, "Mask Overlay"),
        ("05_keypoints", keypoints_vis, "Keypoints"),
    ]

    labeled = []
    for suffix, image, label in stages:
        cv2.imwrite(str(stage_dir / f"{stem}_{suffix}.jpg"), image)
        labeled.append(add_label(image, label))

    contact_height = min(260, max(120, features.image.shape[0] // 3))
    contact = cv2.hconcat([resize_to_height(image, contact_height) for image in labeled])
    cv2.imwrite(str(stage_dir / f"{stem}_pipeline.jpg"), contact)


def mask_overlay(features: ViewFeatures) -> np.ndarray:
    overlay = features.image.copy()
    green = np.zeros_like(overlay)
    green[:, :, 1] = 255
    return np.where(features.mask[:, :, None] > 0, cv2.addWeighted(overlay, 0.65, green, 0.35, 0), overlay)


def save_initial_pipeline_steps(image_paths: list[Path], views: list[ViewFeatures], pipeline_dir: Path) -> None:
    ensure_dir(pipeline_dir)

    sample_count = min(8, len(views))
    indices = np.linspace(0, len(views) - 1, sample_count).astype(int)
    multi_view_images = [views[index].image for index in indices]
    multi_view_labels = [f"View {index + 1}" for index in indices]
    cv2.imwrite(str(pipeline_dir / "01_multi_view_input.jpg"), make_image_grid(multi_view_images, multi_view_labels))

    first_path = image_paths[0]
    first_view = views[0]
    cv2.imwrite(str(pipeline_dir / "02_object_mask.jpg"), add_label(mask_overlay(first_view), "Object Mask"))

    keypoints_vis = cv2.drawKeypoints(
        first_view.image,
        first_view.keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    cv2.imwrite(str(pipeline_dir / "03_feature_extraction.jpg"), add_label(keypoints_vis, f"Feature Extraction: {first_path.name}"))


def render_point_cloud_views(points: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    if len(points) == 0:
        return

    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return

    center = np.median(points, axis=0)
    shifted = points - center
    scale = np.percentile(np.linalg.norm(shifted, axis=1), 95) + 1e-9
    normalized = np.clip(shifted / scale, -1.0, 1.0)

    panels = [
        ("XY Projection", (0, 1), 2),
        ("XZ Projection", (0, 2), 1),
        ("YZ Projection", (1, 2), 0),
    ]
    panel_images = []
    size = 420
    for label, axes, color_axis in panels:
        canvas = np.full((size, size, 3), 250, dtype=np.uint8)
        coords = normalized[:, axes]
        depth = normalized[:, color_axis]
        pixels = ((coords + 1.0) * 0.5 * (size - 50) + 25).astype(int)
        colors = cv2.applyColorMap(((depth + 1.0) * 127.5).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
        for (x, y), color in zip(pixels, colors):
            cv2.circle(canvas, (int(x), int(size - 1 - y)), 1, tuple(int(v) for v in color), -1, cv2.LINE_AA)
        panel_images.append(add_label(canvas, label))

    cv2.imwrite(str(path), cv2.hconcat(panel_images))


def match_features(a: ViewFeatures, b: ViewFeatures) -> list:
    if a.descriptors is None or b.descriptors is None:
        return []

    detector_name, _ = create_detector()
    if a.descriptors.dtype != np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    ab = ratio_matches(matcher.knnMatch(a.descriptors, b.descriptors, k=2))
    ba = ratio_matches(matcher.knnMatch(b.descriptors, a.descriptors, k=2))
    reverse = {(m.trainIdx, m.queryIdx) for m in ba}
    good = [m for m in ab if (m.queryIdx, m.trainIdx) in reverse]
    return good


def ratio_matches(knn: list, ratio: float = 0.8) -> list:
    good = []
    for pair in knn:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def estimate_intrinsics(image: np.ndarray, focal_scale: float = 1.2) -> np.ndarray:
    h, w = image.shape[:2]
    focal = focal_scale * max(w, h)
    return np.array([[focal, 0.0, w / 2.0], [0.0, focal, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def look_at_world_to_camera(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward) + 1e-12
    up = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(forward, right)
    r = np.vstack([right, true_up, forward])
    t = -r @ eye.reshape(3, 1)
    return r, t


def orbit_projection_matrices(count: int, k: np.ndarray, radius: float) -> list[np.ndarray]:
    projections = []
    target = np.zeros(3, dtype=np.float64)
    for index in range(count):
        theta = 2.0 * np.pi * index / count
        eye = np.array([radius * np.sin(theta), 0.0, radius * np.cos(theta)], dtype=np.float64)
        r, t = look_at_world_to_camera(eye, target)
        projections.append(k @ np.hstack([r, t]))
    return projections


def triangulate_correspondences(
    correspondences: Correspondences,
    k: np.ndarray,
) -> tuple[np.ndarray, dict]:
    pts_a = np.float64(correspondences.pts_a)
    pts_b = np.float64(correspondences.pts_b)
    if len(pts_a) < 8:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": 0}

    essential, inlier_mask = cv2.findEssentialMat(
        pts_a,
        pts_b,
        k,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if essential is None or inlier_mask is None:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": 0}

    inliers = inlier_mask.ravel().astype(bool)
    pts_a_in = pts_a[inliers]
    pts_b_in = pts_b[inliers]
    if len(pts_a_in) < 8:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": int(inliers.sum())}

    _, r, t, pose_mask = cv2.recoverPose(essential, pts_a_in, pts_b_in, k)
    pose_inliers = pose_mask.ravel().astype(bool)
    pts_a_pose = pts_a_in[pose_inliers]
    pts_b_pose = pts_b_in[pose_inliers]
    if len(pts_a_pose) < 8:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": int(inliers.sum())}

    p1 = k @ np.hstack([np.eye(3), np.zeros((3, 1))])
    p2 = k @ np.hstack([r, t])
    points_h = cv2.triangulatePoints(p1, p2, pts_a_pose.T, pts_b_pose.T)
    points = (points_h[:3] / points_h[3]).T

    z1 = points[:, 2] > 0
    points_cam2 = (r @ points.T + t).T
    z2 = points_cam2[:, 2] > 0
    points = points[z1 & z2]

    correspondences.inliers_for_draw = inliers
    stats = {
        "method": correspondences.method,
        "pose": "estimated",
        "matches": len(pts_a),
        "inliers": int(inliers.sum()),
        "pose_inliers": int(pose_inliers.sum()),
        "triangulated": len(points),
    }
    return points, stats


def triangulate_with_projections(
    correspondences: Correspondences,
    p1: np.ndarray,
    p2: np.ndarray,
    max_reprojection_error: float,
) -> tuple[np.ndarray, dict]:
    pts_a = np.float64(correspondences.pts_a)
    pts_b = np.float64(correspondences.pts_b)
    if len(pts_a) < 8:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": 0}

    fmat, inlier_mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.FM_RANSAC, 2.0, 0.999)
    if fmat is None or inlier_mask is None:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": 0}

    inliers = inlier_mask.ravel().astype(bool)
    pts_a_in = pts_a[inliers]
    pts_b_in = pts_b[inliers]
    if len(pts_a_in) < 8:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": int(inliers.sum())}

    points_h = cv2.triangulatePoints(p1, p2, pts_a_in.T, pts_b_in.T)
    points = (points_h[:3] / points_h[3]).T
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    pts_a_in = pts_a_in[finite]
    pts_b_in = pts_b_in[finite]

    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64), {"matches": len(pts_a), "inliers": int(inliers.sum())}

    reproj_a = reproject_points(points, p1)
    reproj_b = reproject_points(points, p2)
    err = np.linalg.norm(reproj_a - pts_a_in, axis=1) + np.linalg.norm(reproj_b - pts_b_in, axis=1)
    keep = err < max_reprojection_error
    points = points[keep]

    draw_inliers = np.zeros(len(pts_a), dtype=bool)
    inlier_indices = np.flatnonzero(inliers)
    draw_inliers[inlier_indices[finite][keep]] = True
    correspondences.inliers_for_draw = draw_inliers

    stats = {
        "method": correspondences.method,
        "pose": "orbit",
        "matches": len(pts_a),
        "inliers": int(inliers.sum()),
        "reproj_kept": int(keep.sum()),
        "triangulated": len(points),
    }
    return points, stats


def reproject_points(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
    points_h = np.hstack([points, np.ones((len(points), 1), dtype=np.float64)])
    projected = (projection @ points_h.T).T
    return projected[:, :2] / (projected[:, 2:3] + 1e-12)


def feature_correspondences(a: ViewFeatures, b: ViewFeatures) -> Correspondences:
    matches = match_features(a, b)
    pts_a = np.float64([a.keypoints[m.queryIdx].pt for m in matches])
    pts_b = np.float64([b.keypoints[m.trainIdx].pt for m in matches])
    return Correspondences(pts_a=pts_a, pts_b=pts_b, matches=matches, method="features")


def dense_flow_correspondences(
    a: ViewFeatures,
    b: ViewFeatures,
    step: int = 12,
    fb_threshold: float = 1.5,
    max_points: int = 2500,
) -> Correspondences:
    flow_ab = cv2.calcOpticalFlowFarneback(a.gray, b.gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)
    flow_ba = cv2.calcOpticalFlowFarneback(b.gray, a.gray, None, 0.5, 5, 25, 5, 7, 1.5, 0)

    h, w = a.gray.shape
    grad_x = cv2.Sobel(a.gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(a.gray, cv2.CV_32F, 0, 1, ksize=3)
    texture = cv2.magnitude(grad_x, grad_y)
    texture_cutoff = np.percentile(texture[a.mask > 0], 55) if np.any(a.mask > 0) else np.percentile(texture, 55)

    pts_a = []
    pts_b = []
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if a.mask[y, x] == 0 or texture[y, x] < texture_cutoff:
                continue

            dx, dy = flow_ab[y, x]
            xb = x + float(dx)
            yb = y + float(dy)
            xi = int(round(xb))
            yi = int(round(yb))
            if xi < 0 or yi < 0 or xi >= w or yi >= h or b.mask[yi, xi] == 0:
                continue

            bdx, bdy = flow_ba[yi, xi]
            if np.hypot(dx + bdx, dy + bdy) > fb_threshold:
                continue
            if np.hypot(dx, dy) < 0.5:
                continue

            pts_a.append((x, y))
            pts_b.append((xb, yb))

    if len(pts_a) > max_points:
        indices = np.linspace(0, len(pts_a) - 1, max_points).astype(int)
        pts_a = [pts_a[i] for i in indices]
        pts_b = [pts_b[i] for i in indices]

    return Correspondences(
        pts_a=np.asarray(pts_a, dtype=np.float64),
        pts_b=np.asarray(pts_b, dtype=np.float64),
        method="dense-flow",
    )


def hybrid_correspondences(a: ViewFeatures, b: ViewFeatures, min_feature_inliers: int, k: np.ndarray) -> Correspondences:
    features = feature_correspondences(a, b)
    _, stats = triangulate_correspondences(features, k)
    if stats.get("inliers", 0) >= min_feature_inliers:
        return features
    return dense_flow_correspondences(a, b)


def draw_correspondences(
    a: ViewFeatures,
    b: ViewFeatures,
    correspondences: Correspondences,
    path: Path,
    label: str | None = None,
    use_inliers: bool = True,
) -> None:
    ensure_dir(path.parent)
    inliers = correspondences.inliers_for_draw if use_inliers else None
    if correspondences.matches is not None and inliers is not None:
        inlier_matches = [m for m, keep in zip(correspondences.matches, inliers) if keep]
        vis = cv2.drawMatches(a.image, a.keypoints, b.image, b.keypoints, inlier_matches[:120], None)
        if label:
            vis = add_label(vis, label)
        cv2.imwrite(str(path), vis)
        return

    vis = np.hstack([a.image, b.image])
    pts_a = correspondences.pts_a
    pts_b = correspondences.pts_b
    if inliers is not None and len(inliers) == len(pts_a):
        pts_a = pts_a[inliers]
        pts_b = pts_b[inliers]
    if len(pts_a) > 160:
        idx = np.linspace(0, len(pts_a) - 1, 160).astype(int)
        pts_a = pts_a[idx]
        pts_b = pts_b[idx]
    offset = a.image.shape[1]
    for pa, pb in zip(pts_a, pts_b):
        p1 = tuple(np.round(pa).astype(int))
        p2 = tuple(np.round(pb + np.array([offset, 0])).astype(int))
        cv2.line(vis, p1, p2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, p1, 2, (0, 0, 255), -1)
        cv2.circle(vis, p2, 2, (255, 0, 0), -1)
    if label:
        vis = add_label(vis, label)
    cv2.imwrite(str(path), vis)


def write_ply(path: Path, points: np.ndarray) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def iter_pairs(count: int, strategy: str, window: int):
    if strategy == "adjacent":
        for i in range(count - 1):
            yield i, i + 1
        return
    if strategy == "window":
        for i in range(count):
            for j in range(i + 1, min(count, i + window + 1)):
                yield i, j
        return
    for i in range(count):
        for j in range(i + 1, count):
            yield i, j


def reconstruct(
    image_dir: Path,
    out_dir: Path,
    pair_strategy: str,
    match_mode: str,
    pair_window: int,
    min_feature_inliers: int,
    pose_mode: str,
    orbit_radius: float,
    max_reprojection_error: float,
) -> None:
    ensure_dir(out_dir)
    mask_dir = out_dir / "masks"
    match_dir = out_dir / "matches"
    stage_dir = out_dir / "stages"
    pipeline_dir = out_dir / "pipeline_steps"

    image_paths = load_images(image_dir)
    views = []
    for path in image_paths:
        features = extract_features(path, mask_dir)
        save_stage_images(path, features, stage_dir)
        views.append(features)
    save_initial_pipeline_steps(image_paths, views, pipeline_dir)
    k = estimate_intrinsics(views[0].image)
    projections = orbit_projection_matrices(len(views), k, orbit_radius) if pose_mode == "orbit" else None

    all_points = []
    saved_pair_steps = False
    print("Image order:")
    for path in image_paths:
        print(f"  {path.name}")

    for i, j in iter_pairs(len(views), pair_strategy, pair_window):
        if match_mode == "features":
            correspondences = feature_correspondences(views[i], views[j])
        elif match_mode == "dense-flow":
            correspondences = dense_flow_correspondences(views[i], views[j])
        else:
            correspondences = hybrid_correspondences(views[i], views[j], min_feature_inliers, k)

        if projections is not None:
            points, stats = triangulate_with_projections(
                correspondences,
                projections[i],
                projections[j],
                max_reprojection_error,
            )
        else:
            points, stats = triangulate_correspondences(correspondences, k)
        draw_correspondences(
            views[i],
            views[j],
            correspondences,
            match_dir / f"{image_paths[i].stem}_to_{image_paths[j].stem}_{stats.get('method', match_mode)}.jpg",
        )
        if not saved_pair_steps and stats.get("triangulated", 0) > 0:
            pair_name = f"{image_paths[i].name} -> {image_paths[j].name}"
            draw_correspondences(
                views[i],
                views[j],
                correspondences,
                pipeline_dir / "04_correspondence.jpg",
                label=f"Correspondence: {pair_name}",
                use_inliers=False,
            )
            draw_correspondences(
                views[i],
                views[j],
                correspondences,
                pipeline_dir / "05_ransac_geometry.jpg",
                label=f"RANSAC Geometry: {pair_name}",
                use_inliers=True,
            )
            saved_pair_steps = True
        print(f"{image_paths[i].name} -> {image_paths[j].name}: {stats}")
        if len(points):
            all_points.append(points)

    if not all_points:
        raise RuntimeError("No 3D points were triangulated. Try more textured images or better masks.")

    cloud = np.vstack(all_points)
    finite = np.isfinite(cloud).all(axis=1)
    cloud = cloud[finite]

    distances = np.linalg.norm(cloud, axis=1)
    cutoff = np.percentile(distances, 95)
    cloud = cloud[distances <= cutoff]

    write_ply(out_dir / "reconstruction.ply", cloud)
    render_point_cloud_views(cloud, pipeline_dir / "06_triangulation.jpg")
    print(f"Wrote {len(cloud)} points to {out_dir / 'reconstruction.ply'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Object-centric sparse 3D reconstruction from multi-view images.")
    parser.add_argument("--images", type=Path, required=True, help="Folder containing input images.")
    parser.add_argument("--out", type=Path, required=True, help="Output folder.")
    parser.add_argument(
        "--pair-strategy",
        choices=["adjacent", "window", "all"],
        default="adjacent",
        help="Use adjacent pairs or all image pairs. Synthetic views often benefit from all.",
    )
    parser.add_argument(
        "--pair-window",
        type=int,
        default=3,
        help="For --pair-strategy window, match each view with this many following views.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["features", "dense-flow", "hybrid"],
        default="hybrid",
        help="Correspondence method. hybrid uses RootSIFT first, then dense optical flow if needed.",
    )
    parser.add_argument(
        "--pose-mode",
        choices=["estimated", "orbit"],
        default="estimated",
        help="Use pairwise estimated pose or a shared synthetic orbit camera rig. Use orbit for ordered SVC views.",
    )
    parser.add_argument("--orbit-radius", type=float, default=3.0, help="Synthetic orbit camera radius.")
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=8.0,
        help="Pixel error threshold for --pose-mode orbit triangulation.",
    )
    parser.add_argument(
        "--min-feature-inliers",
        type=int,
        default=40,
        help="Hybrid mode falls back to dense optical flow below this many feature RANSAC inliers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruct(
        args.images,
        args.out,
        args.pair_strategy,
        args.match_mode,
        args.pair_window,
        args.min_feature_inliers,
        args.pose_mode,
        args.orbit_radius,
        args.max_reprojection_error,
    )


if __name__ == "__main__":
    main()

```
