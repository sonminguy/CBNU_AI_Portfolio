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
