"""
yolo_detector.py
================
Layer 1.2b — Perception  (Semantic Object Detection + Depth)

What this file does
-------------------
Runs YOLOv8 nano on the left camera frame to detect and label objects,
then looks up the stereo depth at each detection to get a real distance.

This file does NOT make navigation decisions.
This file does NOT build the occupancy grid.
This file's only job is:
    left camera frame + depth bundle → labeled detections with distances

It runs as a parallel perception stream alongside occupancy_map.py.
Both feed into environment_state.py which combines them.

Why YOLO + stereo depth together?
----------------------------------
YOLO tells you WHAT something is (chair, person, door, wall).
Stereo depth tells you HOW FAR it is with metric accuracy.
Neither is enough alone:
  - YOLO alone: "there's a chair" but no distance
  - Stereo alone: "obstacle at 0.8m" but no label
  - Together: "chair at 0.8m, slightly left, blocking path"

That combined output is what environment_state.py needs to make
intelligent decisions about the scene.

Output bundle
-------------
detections : list of dicts, one per detected object, sorted by distance
  {
    "label"        : str    — YOLO class name e.g. "chair", "person"
    "confidence"   : float  — YOLO confidence 0.0–1.0
    "distance_m"   : float  — median stereo depth at detection centre
                              None if depth is unavailable at that pixel
    "position_x_m" : float  — lateral world position (negative=left)
    "bbox"         : [x1,y1,x2,y2]  — pixel coords in left frame
    "centre_px"    : [cx, cy]        — bbox centre in pixels
    "in_path"      : bool   — True if bbox overlaps the forward corridor
    "is_close"     : bool   — True if distance_m < CLOSE_THRESHOLD_M
    "is_blocking"  : bool   — in_path AND is_close
  }

detection_count   : int   — number of detections this frame
inference_time_ms : float — YOLO inference time in milliseconds
depth_hit_rate    : float — fraction of detections with valid depth

Stack position:
    depth_map.py
    occupancy_map.py    (parallel, not dependent on this file)
    >>> yolo_detector.py <<<
    environment_state.py   (consumes both occupancy + yolo bundles)
"""

from __future__ import annotations

import time
import os
from typing import Optional

import cv2
import numpy as np

# =============================================================================
# Tunable constants
# =============================================================================

# YOLO model to use.
# yolov8n.pt = nano, fastest, least accurate — good for Jetson Orin Nano
# yolov8s.pt = small, slower but better accuracy — try if nano is too noisy
YOLO_MODEL = "yolov8n.pt"

# Minimum confidence to keep a detection.
# Lower = more detections but more false positives.
# 0.35 is a good starting point for indoor navigation.
CONFIDENCE_THRESHOLD = 0.35

# Image size for YOLO inference.
# 320 = fastest on Jetson, good for navigation (don't need fine detail)
# 640 = better accuracy but triggers NMS timeout warnings in live loop
# Use 320 for live pipeline, 640 only for standalone testing
INFERENCE_SIZE = 320

# How close (metres) something must be to count as "close"
CLOSE_THRESHOLD_M = 1.5

# Forward corridor half-width in pixels — used to check if a detection
# overlaps the path directly ahead. Matched to occupancy_map.py's corridor.
# We compute this from image width at runtime so it scales with resolution.
CORRIDOR_FRACTION = 0.15   # 15% of image width on each side of centre

# Depth sampling region — we take the median depth of a patch around
# the bbox centre rather than a single pixel, which is more robust to
# stereo noise. This is the half-size of that patch in pixels.
DEPTH_PATCH_HALF = 8   # 16x16 pixel patch

# Classes from COCO that are relevant for indoor navigation.
# We keep all detections but flag these as higher priority.
NAVIGATION_RELEVANT_CLASSES = {
    "person", "bicycle", "car", "motorcycle",
    "chair", "couch", "bed", "dining table", "toilet",
    "door", "stairs", "backpack", "handbag", "suitcase",
    "bottle", "cup", "laptop", "keyboard", "mouse",
    "potted plant", "tv", "refrigerator", "sink",
    "clock", "vase", "scissors", "book",
}

# =============================================================================
# Module-level model cache
# =============================================================================
# The YOLO model is loaded once and reused every frame.
# Loading it fresh each frame would take ~2 seconds per frame.
_model = None
_model_path = None


def _get_model(model_path: str):
    """
    Load YOLO model once, cache it for subsequent calls.

    Why warmup?
    -----------
    The first YOLO inference on GPU always takes 1-6 seconds because
    PyTorch JIT-compiles CUDA kernels on first use.  We run 3 dummy
    inferences at load time so every subsequent real frame runs at full
    speed (~25ms on Orin Nano).  Without this the first real frame would
    be slow and the pipeline would appear to stall.
    """
    global _model, _model_path
    if _model is None or _model_path != model_path:
        from ultralytics import YOLO
        import torch
        _model = YOLO(model_path)
        if torch.cuda.is_available():
            _model.to("cuda")
        # Warmup — compile CUDA kernels before first real frame
        print("Warming up YOLO on GPU (compiling CUDA kernels)...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            _model(dummy, imgsz=INFERENCE_SIZE, verbose=False)
        print("YOLO ready — warmup complete.")
        _model_path = model_path
    return _model


# =============================================================================
# Depth lookup
# =============================================================================

def _sample_depth(
    depth_map_m: np.ndarray,
    valid_mask:  np.ndarray,
    cx: int,
    cy: int,
) -> Optional[float]:
    """
    Sample the median depth in a patch around pixel (cx, cy).

    Why median of a patch instead of a single pixel?
    -------------------------------------------------
    Stereo depth maps have holes (invalid pixels) and noise spikes.
    A single pixel might be invalid or wrong.
    The median of a 16x16 patch around the bbox centre is much more
    robust — it ignores holes and outliers and gives a stable reading.

    Returns None if fewer than 4 valid pixels exist in the patch.
    """
    h, w = depth_map_m.shape

    # Clamp patch to image bounds
    y1 = max(0, cy - DEPTH_PATCH_HALF)
    y2 = min(h, cy + DEPTH_PATCH_HALF)
    x1 = max(0, cx - DEPTH_PATCH_HALF)
    x2 = min(w, cx + DEPTH_PATCH_HALF)

    patch_depth = depth_map_m[y1:y2, x1:x2]
    patch_valid = valid_mask[y1:y2, x1:x2]

    valid_depths = patch_depth[patch_valid]

    if valid_depths.size < 4:
        return None

    return float(np.median(valid_depths))


# =============================================================================
# X-world position from pixel column
# =============================================================================

def _pixel_to_world_x(
    cx_px:  int,
    eye_w:  int,
    depth_m: float,
    Q:      np.ndarray,
) -> float:
    """
    Estimate the lateral world X position of a detection.

    Uses the Q matrix focal length to convert pixel offset from centre
    into metres at the given depth.

    Q[2,3] = focal_length * baseline  →  f = Q[2,3] / Q[3,2] ... actually
    the simplest approach: Q[0,3] = -cx (principal point x offset)
    and Q[2,3] = f*B.  We extract f from Q directly.

    For a rough lateral estimate: X = (cx_px - image_cx) * depth / f
    where f is the focal length in pixels from Q[2,3] / (-Q[3,2]).
    """
    # Q layout from cv2.stereoRectify:
    # Q[0,3] = -cx_left
    # Q[1,3] = -cy
    # Q[2,3] = focal_length
    # Q[3,2] = -1/baseline
    # Q[3,3] = (cx_left - cx_right) / baseline
    try:
        f  = float(Q[2, 3])          # focal_length * baseline ... actually just f
        cx = float(-Q[0, 3])         # principal point x
        # More direct: use disparity-to-depth scale
        # X_world = (px - cx) * Z / f
        x_world = (cx_px - cx) * depth_m / f if f != 0 else 0.0
    except Exception:
        # Fallback: assume centred, use image fraction
        x_world = (cx_px - eye_w / 2) / eye_w * depth_m * 0.8

    return float(x_world)


# =============================================================================
# Main entry point — called once per frame
# =============================================================================

def detect(
    frame_left:   np.ndarray,
    depth_bundle: dict,
    Q:            np.ndarray,
    model_path:   str = YOLO_MODEL,
) -> dict:
    """
    Run YOLO detection on the left frame and attach stereo depth to each box.

    Parameters
    ----------
    frame_left   : BGR uint8 left eye image (already split, not rectified
                   needed for YOLO — YOLO works on raw or rectified)
    depth_bundle : output of depth_map.process_frame()
    Q            : 4x4 disparity-to-depth matrix from stereo calibration
    model_path   : path to YOLO weights file

    Returns
    -------
    dict — see module docstring
    """
    depth_map_m = depth_bundle["depth_map_m"]   # (H, W) float32
    valid_mask  = depth_bundle["valid_mask"]    # (H, W) bool

    eye_h, eye_w = frame_left.shape[:2]

    # Forward corridor pixel bounds
    corridor_half_px = int(eye_w * CORRIDOR_FRACTION)
    corridor_x1 = eye_w // 2 - corridor_half_px
    corridor_x2 = eye_w // 2 + corridor_half_px

    # ------------------------------------------------------------------
    # Step 1 — YOLO inference
    # ------------------------------------------------------------------
    model = _get_model(model_path)

    t0 = time.perf_counter()

    # verbose=False suppresses per-frame console output
    # imgsz controls inference resolution — smaller = faster
    results = model(
        frame_left,
        imgsz     = INFERENCE_SIZE,
        conf      = CONFIDENCE_THRESHOLD,
        verbose   = False,
    )
    inference_ms = (time.perf_counter() - t0) * 1000.0

    # ------------------------------------------------------------------
    # Step 2 — Parse detections and attach depth
    # ------------------------------------------------------------------
    detections = []
    depth_hits  = 0

    result = results[0]   # single image → single result

    if result.boxes is not None and len(result.boxes) > 0:
        boxes      = result.boxes.xyxy.cpu().numpy()    # (N, 4) x1y1x2y2
        confs      = result.boxes.conf.cpu().numpy()    # (N,)
        class_ids  = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
        names      = result.names                       # {id: name}

        # YOLO runs at INFERENCE_SIZE but the frame is eye_w x eye_h.
        # Boxes are already scaled back to original frame size by ultralytics.

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf      = float(confs[i])
            label     = names[class_ids[i]]

            # Bbox centre in pixels
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Clamp to image bounds
            cx = max(0, min(eye_w - 1, cx))
            cy = max(0, min(eye_h - 1, cy))

            # Depth at detection centre
            distance_m = _sample_depth(depth_map_m, valid_mask, cx, cy)
            if distance_m is not None:
                depth_hits += 1

            # Lateral world position
            if distance_m is not None:
                position_x_m = _pixel_to_world_x(cx, eye_w, distance_m, Q)
            else:
                position_x_m = None

            # Does this detection overlap the forward corridor?
            box_cx = (x1 + x2) / 2
            in_path = (corridor_x1 <= box_cx <= corridor_x2)

            # Is it close enough to matter?
            is_close = (distance_m is not None and distance_m < CLOSE_THRESHOLD_M)

            # Is it actively blocking the path?
            is_blocking = in_path and is_close

            detections.append({
                "label":         label,
                "confidence":    round(conf, 3),
                "distance_m":    round(distance_m, 3) if distance_m is not None else None,
                "position_x_m":  round(position_x_m, 3) if position_x_m is not None else None,
                "bbox":          [int(x1), int(y1), int(x2), int(y2)],
                "centre_px":     [cx, cy],
                "in_path":       in_path,
                "is_close":      is_close,
                "is_blocking":   is_blocking,
                "is_relevant":   label in NAVIGATION_RELEVANT_CLASSES,
            })

    # Sort by distance — closest first, unknowns last
    detections.sort(
        key=lambda d: d["distance_m"] if d["distance_m"] is not None else 999.0
    )

    depth_hit_rate = depth_hits / len(detections) if detections else 0.0

    # ------------------------------------------------------------------
    # Step 3 — Debug annotated image
    # ------------------------------------------------------------------
    # Draw boxes and labels onto the left frame for debug output.
    # NOT used by any logic layer — only for saving to debug_output/.
    debug_frame = frame_left.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label    = det["label"]
        dist     = det["distance_m"]
        blocking = det["is_blocking"]

        # Red = blocking path, yellow = close but not blocking, green = far/safe
        if blocking:
            color = (0, 0, 220)      # red (BGR)
        elif det["is_close"]:
            color = (0, 200, 220)    # yellow
        else:
            color = (0, 180, 0)      # green

        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)

        dist_str = f"{dist:.2f}m" if dist is not None else "??m"
        text = f"{label} {dist_str}"
        cv2.putText(debug_frame, text, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw corridor lines
    cv2.line(debug_frame, (corridor_x1, 0), (corridor_x1, eye_h), (80, 80, 80), 1)
    cv2.line(debug_frame, (corridor_x2, 0), (corridor_x2, eye_h), (80, 80, 80), 1)

    return {
        "detections":       detections,
        "detection_count":  len(detections),
        "inference_time_ms": round(inference_ms, 1),
        "depth_hit_rate":   round(depth_hit_rate, 3),
        "debug_frame":      debug_frame,   # (H, W, 3) uint8 — debug only
    }


# =============================================================================
# Standalone test — run directly: python yolo_detector.py
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from depth_map import build_stereo_matcher, process_frame as depth_process_frame
    except ImportError:
        from modules.depth_map import build_stereo_matcher, process_frame as depth_process_frame

    CALIB_PATH   = "calibration/stereo_calibration.npz"
    CAMERA_INDEX = 0
    FRAME_WIDTH  = 1280
    FRAME_HEIGHT = 480
    FPS          = 30

    if not os.path.exists(CALIB_PATH):
        print(f"ERROR: calibration file not found at {CALIB_PATH}")
        raise SystemExit(1)

    cal    = np.load(CALIB_PATH)
    M1, d1 = cal["M1"], cal["d1"]
    M2, d2 = cal["M2"], cal["d2"]
    R1, R2 = cal["R1"], cal["R2"]
    P1, P2 = cal["P1"], cal["P2"]
    Q      = cal["Q"]

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC,       cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

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

    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)

    rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)

    print("Running depth map...")
    stereo       = build_stereo_matcher()
    depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)

    print("Running YOLO detector (first run loads model)...")
    yolo_bundle = detect(frame[:, :eye_w], depth_bundle, Q)

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print(f"\n--- yolo_detector bundle ---")
    print(f"  detection_count   : {yolo_bundle['detection_count']}")
    print(f"  inference_time_ms : {yolo_bundle['inference_time_ms']}")
    print(f"  depth_hit_rate    : {yolo_bundle['depth_hit_rate']}")
    print(f"\n  detections (sorted by distance):")

    for det in yolo_bundle["detections"]:
        dist    = f"{det['distance_m']:.2f}m" if det["distance_m"] is not None else "  --  "
        x_pos   = f"{det['position_x_m']:+.2f}m" if det["position_x_m"] is not None else "  -- "
        flags   = []
        if det["in_path"]:    flags.append("IN_PATH")
        if det["is_close"]:   flags.append("CLOSE")
        if det["is_blocking"]: flags.append("BLOCKING")
        flag_str = " ".join(flags)
        print(f"    {det['label']:20s}  conf={det['confidence']:.2f}  "
              f"dist={dist}  x={x_pos}  {flag_str}")

    # ------------------------------------------------------------------
    # Save debug image
    # ------------------------------------------------------------------
    os.makedirs("debug_output", exist_ok=True)
    cv2.imwrite("debug_output/yolo_detections.png", yolo_bundle["debug_frame"])
    print(f"\nDebug image saved to debug_output/yolo_detections.png")
    print("Pull to Mac with:")
    print("  scp 'zanewolf@192.168.50.103:~/6thSense/debug_output/yolo_detections.png' ~/Desktop/")
