# Source Notes

`reconstruct_object.py` is a compact baseline for the final project.

It intentionally keeps the reconstruction classical:

- CLAHE preprocessing
- GrabCut object mask
- SIFT or ORB feature extraction
- Lowe ratio feature matching
- RANSAC essential matrix estimation
- relative camera pose recovery
- linear triangulation
- ASCII PLY export

The generated point cloud is sparse and scale-ambiguous. That is expected. The goal is to show measured geometry from multi-view constraints, then compare how masking and AI assistance affect the result.

