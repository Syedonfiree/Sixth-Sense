"""
depth_map.py
============
Layer 1.1 — Perception  (Depth + Point Cloud)

What this file does
-------------------
Takes a rectified stereo pair, computes disparity, converts it into a
real-world 3D point cloud, then extracts a rich data bundle that every
layer above this one can use.

This file does NOT build the occupancy map — that is occupancy_map.py.
This file does NOT make navigation decisions — that is environment_state.py.
This file's only job is:
    rectified stereo frames → clean, filtered 3D point cloud + depth stats

Output bundle keys (all distances in metres unless noted)
---------------------------------------------------------
points_3d           : (H, W, 3) float32 — full reprojected point cloud
                      channels are (X, Y, Z) in camera frame
                      X = right,  Y = up (flipped),  Z = forward
                      NOTE: cv2.reprojectImageTo3D outputs Y positive=down
                      for this camera.  We flip Y immediately after reprojection
                      so that Y positive = up throughout all downstream code.
                      Floor = most negative Y.  Ceiling = most positive Y.
valid_mask          : (H, W) bool — True where depth is usable
depth_map_m         : (H, W) float32 — per-pixel depth (Z), invalid = 0
depth_valid_ratio   : float  — fraction of pixels with valid depth  (0–1)
depth_percentiles   : dict   — p10/p25/p50/p75/p90 of valid depths
floor_y_m           : float  — estimated Y of the floor plane
                      (camera Y is positive downward, so floor Y is
                       the largest Y value in the lower image rows)
obstacle_height_m   : float  — OBSTACLE_HEIGHT_THRESH above floor
near_depth_hist     : (N,)   — histogram of depths in 0.4–1.5 m band
mid_depth_hist      : (N,)   — histogram of depths in 1.5–3.0 m band
far_depth_hist      : (N,)   — histogram of depths in 3.0–5.0 m band
disparity_raw       : (H, W) float32 — raw disparity / 16 (SGBM output)
debug_colormap      : (H, W, 3) uint8 — JET colormap for display/video
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

# =============================================================================
# Tunable constants
# =============================================================================

# Depth range we trust. Closer than MIN → stereo baseline noise.
# Farther than MAX → disparity too small to be accurate.
MIN_DEPTH_M = 0.40
MAX_DEPTH_M = 5.00

# Y-axis sanity clamp.
# After our Y flip: Y positive = up, Y negative = down.
# Floor is at roughly Y = -camera_height (negative).
# Ceiling is at roughly Y = +camera_height (positive).
# Clamp to ±2 m to reject bad reprojection artefacts.
Y_CLIP_MIN = -2.0   # below camera (floor direction)
Y_CLIP_MAX =  0.5   # above camera — tightened from 2.0 to reject ceiling artefacts
                    # most useful obstacles are below camera level anyway

# How many pixels from the bottom of the image do we sample to estimate
# the floor plane?  The floor is almost always visible in the bottom rows.
FLOOR_ROW_FRACTION = 0.25   # bottom 25 % of image rows

# We call a surface point an "obstacle" if it is this far above the floor.
OBSTACLE_HEIGHT_THRESH_M = 0.12   # 12 cm — catches kerbs, bags, chair legs

# Sub-sampling stride for the point cloud extraction loop.
# STRIDE=2 → process every other pixel → 4× fewer points, still accurate.
# STRIDE=4 → fast on Jetson Orin Nano.
STRIDE = 4

# Histogram bins for the per-zone depth distributions.
HIST_BINS = 20

# =============================================================================
# Helper: build the stereo matcher (call once, reuse every frame)
# =============================================================================

def build_stereo_matcher() -> cv2.StereoSGBM:
    """
    StereoSGBM  (Semi-Global Block Matching)
    -----------------------------------------
    This is OpenCV's best built-in stereo algorithm.
    It works by comparing small patches of the left image with the right
    image and finding the horizontal shift (disparity) that gives the
    best match.  Larger disparity = closer object.

    Key parameters explained:
    - numDisparities  : how many pixel-shift levels to search.
                        Must be divisible by 16.  96 covers ~0.4–5 m at
                        typical baseline/focal-length combos.
    - blockSize       : patch size for matching.  5 is a good balance
                        between accuracy and speed.
    - P1 / P2         : smoothness penalties.  P2 > P1 discourages large
                        sudden depth jumps (reduces noise at edges).
    - uniquenessRatio : reject a match if the second-best match is within
                        10 % of the best.  Filters ambiguous pixels.
    - speckleWindowSize / speckleRange : removes tiny isolated blobs of
                        wrong disparity (speckle noise).
    """
    return cv2.StereoSGBM_create(
        minDisparity      = 0,
        numDisparities    = 64,   # was 96 — fewer levels = ~30% faster
        blockSize         = 5,
        P1                = 8  * 3 * 5 ** 2,
        P2                = 32 * 3 * 5 ** 2,
        disp12MaxDiff     = 1,
        uniquenessRatio   = 10,
        speckleWindowSize = 100,
        speckleRange      = 32,
        mode              = cv2.STEREO_SGBM_MODE_HH4,  # was SGBM_3WAY — much faster
    )


# =============================================================================
# Floor estimation
# =============================================================================

def _estimate_floor_y(
    Y_all:  np.ndarray,
    Z_all:  np.ndarray,
    eye_h:  int,
    rows:   np.ndarray,
    cols:   np.ndarray,
    keep:   np.ndarray,
) -> float:
    """
    Estimate the Y coordinate of the floor plane.

    After Y-flip: Y positive = up, Y negative = down.
    The floor is the surface with the MOST NEGATIVE Y values.
    We look at the bottom quarter of the image (where the floor is
    almost always visible when the camera points forward), take the
    10th percentile of Y values there (most negative = most downward),
    and call that the floor level.

    Returns 0.0 if there is not enough data.
    """
    floor_row_start = int(eye_h * (1.0 - FLOOR_ROW_FRACTION))
    in_floor_region = (rows[keep] >= floor_row_start)

    if in_floor_region.sum() < 10:
        return 0.0

    floor_Y = Y_all[keep][in_floor_region]
    # 10th percentile = most negative = closest to floor
    return float(np.percentile(floor_Y, 10))


# =============================================================================
# Per-zone depth histograms
# =============================================================================

def _zone_histogram(Z_valid: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    """
    Build a depth histogram for one distance zone.

    Why histograms?
    ---------------
    A single number like "average depth in zone" loses too much detail.
    A histogram tells you the *shape* of the depth distribution:
    - A sharp peak near 1.0 m → solid wall close ahead
    - Flat / spread distribution → open space with scattered objects
    - Two peaks → near object + far wall

    The situation layer above this one can read these shapes and make
    smarter decisions than "there's something at X metres".
    """
    mask = (Z_valid >= z_min) & (Z_valid < z_max)
    if mask.sum() == 0:
        return np.zeros(HIST_BINS, dtype=np.float32)

    hist, _ = np.histogram(Z_valid[mask], bins=HIST_BINS, range=(z_min, z_max))
    # Normalise to 0–1 so the shape is comparable across frames
    total = hist.sum()
    if total > 0:
        hist = hist.astype(np.float32) / total
    return hist.astype(np.float32)


# =============================================================================
# Main entry point — called once per frame
# =============================================================================

def process_frame(
    rectL:   np.ndarray,
    rectR:   np.ndarray,
    Q:       np.ndarray,
    stereo:  cv2.StereoSGBM,
    eye_w:   int,
    eye_h:   int,
) -> dict:
    """
    Convert a rectified stereo pair into a rich depth bundle.

    Parameters
    ----------
    rectL, rectR : BGR uint8 images, already rectified (use rectification.py)
    Q            : 4x4 disparity-to-depth matrix from stereo calibration
    stereo       : StereoSGBM matcher (build once with build_stereo_matcher())
    eye_w, eye_h : per-eye image dimensions in pixels

    Returns
    -------
    dict  — see module docstring for full key list
    """

    # ------------------------------------------------------------------
    # Step 1 — Grayscale conversion
    # ------------------------------------------------------------------
    # StereoSGBM works on single-channel (grayscale) images.
    # We keep the colour images for the debug colormap but match on grey.
    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # Step 2 — Disparity computation
    # ------------------------------------------------------------------
    # stereo.compute() returns a fixed-point integer map.
    # Divide by 16 to get real disparity values in pixels.
    # Negative or zero disparity → invalid (no match found).
    disparity_raw = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

    # ------------------------------------------------------------------
    # Step 3 — Reproject disparity to 3D
    # ------------------------------------------------------------------
    # cv2.reprojectImageTo3D uses the Q matrix from stereo calibration.
    #
    # Q encodes the camera geometry:  given a pixel at (u, v) with
    # disparity d, it computes the real-world (X, Y, Z) in metres.
    #
    # Q comes from cv2.stereoRectify() during calibration.
    # Without an accurate Q, depth values are meaningless.
    #
    # Output shape: (eye_h, eye_w, 3)
    # Channel 0 = X (right is positive)
    # Channel 1 = Y (down is positive)
    # Channel 2 = Z (forward is positive)
    points_3d = cv2.reprojectImageTo3D(disparity_raw, Q)

    # ------------------------------------------------------------------
    # Y-axis flip
    # ------------------------------------------------------------------
    # cv2.reprojectImageTo3D outputs Y positive=downward for this camera.
    # We immediately flip Y so that Y positive=up everywhere downstream.
    # This means:
    #   floor points  → most negative Y  (e.g. -0.8 m below camera)
    #   ceiling points → most positive Y  (e.g. +0.5 m above camera)
    #   obstacles at waist height → slightly negative Y
    # All floor estimation and obstacle height logic depends on this convention.
    points_3d[:, :, 1] *= -1

    # ------------------------------------------------------------------
    # Step 4 — Sub-sampled validity mask
    # ------------------------------------------------------------------
    # We don't need every single pixel — STRIDE=4 gives us one sample
    # per 4×4 block.  This is plenty for a 5 cm occupancy grid.
    rows_idx = np.arange(0, eye_h, STRIDE)
    cols_idx = np.arange(0, eye_w, STRIDE)
    rr, cc   = np.meshgrid(rows_idx, cols_idx, indexing="ij")

    X_s = points_3d[rr, cc, 0]
    Y_s = points_3d[rr, cc, 1]
    Z_s = points_3d[rr, cc, 2]
    d_s = disparity_raw[rr, cc]

    # A pixel is valid if:
    #   - disparity > 0            (matcher found a real match)
    #   - X, Y, Z are finite       (no divide-by-zero artefacts)
    #   - Z in [MIN, MAX]          (within our trusted depth range)
    #   - Y in [Y_CLIP_MIN, MAX]   (not a sky or ground outlier)
    keep = (
        (d_s > 0)            &
        np.isfinite(X_s)     &
        np.isfinite(Y_s)     &
        np.isfinite(Z_s)     &
        (Z_s >= MIN_DEPTH_M) &
        (Z_s <= MAX_DEPTH_M) &
        (Y_s >= Y_CLIP_MIN)  &
        (Y_s <= Y_CLIP_MAX)
    )

    X_v = X_s[keep]
    Y_v = Y_s[keep]
    Z_v = Z_s[keep]

    # ------------------------------------------------------------------
    # Step 5 — Full-resolution valid mask and depth map
    # ------------------------------------------------------------------
    # For the full-res valid mask we re-check every pixel (not strided).
    # This is used by occupancy_map.py which needs per-pixel data.
    valid_mask_full = (
        (disparity_raw > 0)              &
        np.isfinite(points_3d[..., 0])   &
        np.isfinite(points_3d[..., 1])   &
        np.isfinite(points_3d[..., 2])   &
        (points_3d[..., 2] >= MIN_DEPTH_M) &
        (points_3d[..., 2] <= MAX_DEPTH_M) &
        (points_3d[..., 1] >= Y_CLIP_MIN)  &
        (points_3d[..., 1] <= Y_CLIP_MAX)
    )

    # depth_map_m is Z only, invalid pixels set to 0
    depth_map_m = np.where(valid_mask_full, points_3d[..., 2], 0.0).astype(np.float32)

    depth_valid_ratio = float(valid_mask_full.sum()) / float(eye_h * eye_w)

    # ------------------------------------------------------------------
    # Step 6 — Depth percentile statistics
    # ------------------------------------------------------------------
    # Percentiles give a better picture of the depth distribution than
    # a single mean or min.
    #
    # p10 = closest 10 % of valid points — tells you if something is
    #       very near even if most of the frame is open
    # p50 = median depth — robust "typical distance" ahead
    # p90 = far distance — tells you if the space is large / open
    depth_percentiles = {}
    if Z_v.size > 0:
        for pct in (10, 25, 50, 75, 90):
            depth_percentiles[f"p{pct}"] = float(np.percentile(Z_v, pct))
    else:
        for pct in (10, 25, 50, 75, 90):
            depth_percentiles[f"p{pct}"] = 0.0

    # ------------------------------------------------------------------
    # Step 7 — Floor plane estimation
    # ------------------------------------------------------------------
    floor_y_m = _estimate_floor_y(Y_s, Z_s, eye_h, rr, cc, keep)

    # ------------------------------------------------------------------
    # Step 8 — Per-zone depth histograms
    # ------------------------------------------------------------------
    near_depth_hist = _zone_histogram(Z_v, 0.4, 1.5)
    mid_depth_hist  = _zone_histogram(Z_v, 1.5, 3.0)
    far_depth_hist  = _zone_histogram(Z_v, 3.0, 5.0)

    # ------------------------------------------------------------------
    # Step 9 — Debug colormap
    # ------------------------------------------------------------------
    # This is ONLY for writing to a debug video or displaying when
    # the --debug flag is passed.  It is NOT used by any logic layer.
    #
    # We normalise disparity to 0–255 and apply JET colormap:
    # blue = far, red = close  (matches depth intuition)
    disp_vis  = cv2.normalize(
        disparity_raw, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    disp_vis  = cv2.medianBlur(disp_vis, 5)
    debug_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # ------------------------------------------------------------------
    # Step 10 — Assemble and return bundle
    # ------------------------------------------------------------------
    return {
        # Full-resolution outputs (used by occupancy_map.py)
        "points_3d":          points_3d,          # (H, W, 3) float32
        "valid_mask":         valid_mask_full,     # (H, W) bool
        "depth_map_m":        depth_map_m,         # (H, W) float32, 0 = invalid
        "disparity_raw":      disparity_raw,       # (H, W) float32

        # Scalar stats (used by environment_state.py and logging)
        "depth_valid_ratio":  round(depth_valid_ratio, 3),
        "depth_percentiles":  depth_percentiles,

        # Floor geometry (used by occupancy_map.py for height thresholding)
        # floor_y_m is NEGATIVE (floor is below camera, Y positive=up)
        # obstacle_height_m = floor_y_m + OBSTACLE_HEIGHT_THRESH_M
        # A point is an obstacle if its Y > obstacle_height_m
        # (i.e. it is more than 12 cm above the floor level)
        "floor_y_m":          round(floor_y_m, 3),
        "obstacle_height_m":  round(floor_y_m + OBSTACLE_HEIGHT_THRESH_M, 3),

        # Per-zone histograms (used by environment_state.py)
        "near_depth_hist":    near_depth_hist,     # (HIST_BINS,) float32
        "mid_depth_hist":     mid_depth_hist,
        "far_depth_hist":     far_depth_hist,

        # Debug only — do not use in logic layers
        "debug_colormap":     debug_colormap,      # (H, W, 3) uint8
    }


# =============================================================================
# Standalone test  — run directly: python depth_map.py
# Reads one frame from the camera, prints the bundle, saves a debug image.
# Use this to confirm the module is working before integrating into main.py
# =============================================================================

if __name__ == "__main__":
    import os

    CALIB_PATH   = "calibration/stereo_calibration.npz"
    CAMERA_INDEX = 0
    FRAME_WIDTH  = 1280
    FRAME_HEIGHT = 480
    FPS          = 30

    if not os.path.exists(CALIB_PATH):
        print(f"ERROR: calibration file not found at {CALIB_PATH}")
        raise SystemExit(1)

    # Load calibration
    cal    = np.load(CALIB_PATH)
    M1, d1 = cal["M1"], cal["d1"]
    M2, d2 = cal["M2"], cal["d2"]
    R1, R2 = cal["R1"], cal["R2"]
    P1, P2 = cal["P1"], cal["P2"]
    Q      = cal["Q"]

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Warmup
    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: could not read frame")
        raise SystemExit(1)

    h, w  = frame.shape[:2]
    eye_w = w // 2
    eye_h = h

    # Build rectification maps
    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)

    left  = frame[:, :eye_w]
    right = frame[:, eye_w:]
    rectL = cv2.remap(left,  mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, mapR1, mapR2, cv2.INTER_LINEAR)

    stereo = build_stereo_matcher()
    bundle = process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)

    # Print bundle (excluding large arrays)
    print("\n--- depth_map bundle ---")
    for k, v in bundle.items():
        if isinstance(v, np.ndarray):
            print(f"  {k:25s}: ndarray {v.shape} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:25s}: {v}")
        else:
            print(f"  {k:25s}: {v}")

    # Save debug image
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite("debug_output/depth_map_test.png", bundle["debug_colormap"])
    print("\nDebug colormap saved to debug_output/depth_map_test.png")
    print("Pull this to your PC with:  scp jetson:~/vision-navigation/debug_output/depth_map_test.png .")
