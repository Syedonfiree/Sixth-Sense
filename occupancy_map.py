"""
occupancy_map.py
================
Layer 1.2a — Perception  (2D Local Occupancy Grid)

What this file does
-------------------
Takes the point cloud bundle from depth_map.py and builds a clean
per-frame 2D top-down grid of the space around the user.

Every frame is computed fresh — no decay, no persistence, no ghosts.
The grid is egocentric: the user is always at the bottom centre.
Forward = up. Left = left. Right = right.

    LEFT    CENTER    RIGHT
    |         |         |
FAR |         |         |      4.0 m ahead
    |         |         |
MID |         |         |      ~1.5 m ahead
    |         |         |
NEAR|         |         |      0.3 m ahead
    |        [U]        |      ← user position (bottom centre)

Grid coordinates
----------------
- gz (grid-z) = row index,   0 = nearest to user, increases forward
- gx (grid-x) = col index,   0 = far left, centre_gx = directly ahead

World coordinates (from depth_map.py / camera frame)
-----------------------------------------------------
- X = lateral,  negative = left,  positive = right
- Y = vertical, negative = up,    positive = down  (camera convention)
- Z = depth,    positive = forward

Obstacle detection
------------------
A point is an obstacle if its height above the floor exceeds
OBSTACLE_HEIGHT_THRESH_M (default 12 cm).
Height = -(Y - floor_y_m)  because Y is positive downward.
So a point above the floor has a negative Y relative to floor_y_m,
and negating that gives a positive height value.

Output bundle
-------------
See bottom of this docstring or the return dict in build_occupancy_map().

Stack position:
    depth_map.py
    >>> occupancy_map.py <<<
    environment_state.py       (next)
    situation_layer.py         (future)
    mode_controller.py         (future)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

# =============================================================================
# Tunable constants
# =============================================================================

# Physical extent of the grid
MAP_HALF_W_M  = 2.5    # metres left and right of user  → total width 5.0 m
MAP_FORWARD_M = 4.0    # metres ahead of user

# Cell size — 5 cm gives good resolution without being too slow
GRID_RES_M = 0.05      # metres per cell

# Depth range — must match depth_map.py
MIN_DEPTH_M = 0.55
MAX_DEPTH_M = 4.00

# How high above the floor a point must be to count as an obstacle.
# 12 cm catches: chair legs, bags, kerbs, low furniture.
# Raise to 0.20 if you get too many false positives from floor texture.
OBSTACLE_HEIGHT_THRESH_M = 0.12

# Point cloud sub-sampling stride.
# STRIDE=4 means we sample every 4th pixel in each direction.
# This is plenty for a 5 cm grid and runs fast on the Jetson.
STRIDE = 4

# Minimum number of obstacle points that must land in a cell before
# we mark it occupied.  1 = very sensitive, 3 = more robust to noise.
MIN_POINTS_TO_OCCUPY = 4

# Forward corridor — the path directly ahead of the user.
# We check this corridor for forward clearance.
# 0.35 m half-width = 0.70 m total corridor (about one shoulder width).
FORWARD_CORRIDOR_HALF_W_M = 0.35

# How far forward we scan when computing left/right clearances.
LATERAL_SCAN_DEPTH_M = 2.5

# Zone boundaries (metres from user)
ZONE_NEAR_M = (0.30, 0.90)
ZONE_MID_M  = (0.90, 2.00)
ZONE_FAR_M  = (2.00, 3.50)

# Maximum Z at which a point can vote as an obstacle.
# Points beyond this are background (far walls, ceiling fixtures) and ignored.
# This is the single most important tunable for reducing false positives in
# cluttered indoor scenes — far walls should not block the near grid.
# Separate from MAX_DEPTH_M which is the stereo sensing range limit.
OBSTACLE_MAX_Z_M = 3.0


# =============================================================================
# Grid geometry — computed once from the constants above
# =============================================================================

# These are derived once at module load time so we don't recompute every frame.
# If you change MAP_HALF_W_M, MAP_FORWARD_M, or GRID_RES_M, everything
# downstream updates automatically.

GRID_W    = int(round(2.0 * MAP_HALF_W_M / GRID_RES_M))   # columns
GRID_H    = int(round(MAP_FORWARD_M      / GRID_RES_M))   # rows
CENTRE_GX = GRID_W // 2                                   # column of user


def _world_to_grid(
    X: np.ndarray,
    Z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert world X (lateral) and Z (depth) coordinates into grid indices.

    X = 0 maps to CENTRE_GX  (directly ahead of user)
    Z = 0 maps to gz = 0     (user's position, bottom of grid)
    Z = MAP_FORWARD_M maps to gz = GRID_H - 1  (top of grid, far)

    Returns
    -------
    gx       : (N,) int   column indices
    gz       : (N,) int   row indices
    in_grid  : (N,) bool  True where the point falls inside the grid bounds
    """
    gx = ((X + MAP_HALF_W_M) / GRID_RES_M).astype(int)
    gz = (Z / GRID_RES_M).astype(int)
    in_grid = (gx >= 0) & (gx < GRID_W) & (gz >= 0) & (gz < GRID_H)
    return gx, gz, in_grid


# =============================================================================
# Core function — one call per frame
# =============================================================================

def build_occupancy_map(depth_bundle: dict) -> dict:
    """
    Build a 2D occupancy grid from the depth_map.py output bundle.

    Parameters
    ----------
    depth_bundle : dict
        Output of depth_map.process_frame().
        Required keys: points_3d, valid_mask, floor_y_m

    Returns
    -------
    dict — see module docstring for full key descriptions
    """

    points_3d  = depth_bundle["points_3d"]    # (H, W, 3) float32
    valid_mask = depth_bundle["valid_mask"]   # (H, W) bool
    floor_y_m  = depth_bundle["floor_y_m"]   # float — camera Y of floor

    eye_h, eye_w = points_3d.shape[:2]

    # ------------------------------------------------------------------
    # Step 1 — Sub-sample the point cloud
    # ------------------------------------------------------------------
    # We don't need every pixel to build a 5 cm grid.
    # Sample every STRIDE pixels in both dimensions.
    # np.meshgrid creates two 2D arrays:
    #   rr[i,j] = row index of sample (i, j)
    #   cc[i,j] = col index of sample (i, j)
    rows = np.arange(0, eye_h, STRIDE)
    cols = np.arange(0, eye_w, STRIDE)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    # Pull X, Y, Z for every sampled pixel
    X_s = points_3d[rr, cc, 0]   # lateral    (negative=left, positive=right)
    Y_s = points_3d[rr, cc, 1]   # vertical   (positive=down in camera frame)
    Z_s = points_3d[rr, cc, 2]   # depth      (positive=forward)

    # ------------------------------------------------------------------
    # Step 2 — Filter to valid, in-range points only
    # ------------------------------------------------------------------
    # We re-apply the same validity checks as depth_map.py because
    # the sub-sampling here uses a different stride pattern and we
    # want this module to be self-contained and not silently trust
    # bad points that slipped through.
    keep = (
        valid_mask[rr, cc]       &   # depth_map said this pixel is valid
        (Z_s >= MIN_DEPTH_M)     &   # not too close (stereo noise)
        (Z_s <= MAX_DEPTH_M)     &   # not too far (inaccurate)
        np.isfinite(X_s)         &   # no inf/nan from bad reprojection
        np.isfinite(Y_s)         &
        np.isfinite(Z_s)
    )

    X = X_s[keep]
    Y = Y_s[keep]
    Z = Z_s[keep]

    # ------------------------------------------------------------------
    # Step 3 — Compute height above floor for each point
    # ------------------------------------------------------------------
    # After Y-flip in depth_map.py: Y positive = up, Y negative = down.
    # floor_y_m is negative (e.g. -0.8 m means floor is 0.8 m below camera).
    #
    # Height above floor = Y - floor_y_m
    #
    # Example:
    #   floor_y_m  = -0.80  (floor is 0.8 m below camera)
    #   chair leg Y = -0.55  (chair leg is 0.55 m below camera)
    #   height = -0.55 - (-0.80) = 0.25 m above floor  → obstacle ✓
    #
    #   floor point Y = -0.78
    #   height = -0.78 - (-0.80) = 0.02 m  → not an obstacle ✓
    #
    #   ceiling point Y = +0.60
    #   height = +0.60 - (-0.80) = 1.40 m  → above threshold but we
    #   clamp using Y_CLIP_MAX in depth_map so ceiling points are already
    #   excluded from the valid_mask.
    #
    # ROBUSTNESS: use the most negative 15th percentile of all valid Y
    # as a secondary floor estimate in case the bottom-row estimate from
    # depth_map.py is unreliable (camera not seeing floor in bottom rows).
    if Y.size > 50:
        floor_y_secondary = float(np.percentile(Y, 15))
        # Use whichever is MORE negative (further down = safer threshold)
        floor_y_used = min(floor_y_m, floor_y_secondary)
    else:
        floor_y_used = floor_y_m

    height_m = Y - floor_y_used

    # A point is an obstacle if ALL of these are true:
    #   1. More than 12 cm above the floor     → excludes actual floor surface
    #   2. Less than 2.0 m above the floor     → excludes ceiling, high signs
    #   3. Within OBSTACLE_MAX_Z_M forward     → excludes far background walls
    #
    # Condition 3 is the key fix for cluttered indoor scenes.
    # A wall at 3.5 m is real but it does not block your next step.
    # Only things close enough to actually matter get projected into the grid.
    is_obstacle = (
        (height_m > OBSTACLE_HEIGHT_THRESH_M) &
        (height_m < 2.0)                      &
        (Z <= OBSTACLE_MAX_Z_M)               &
        (Y < -0.35)    # must be at least 35cm below camera — rejects ceiling artefacts
    )

    X_obs = X[is_obstacle]
    Z_obs = Z[is_obstacle]

    # ------------------------------------------------------------------
    # Step 4 — Build the grid
    # ------------------------------------------------------------------
    # We use a vote grid: each obstacle point votes for its cell.
    # Cells with enough votes (MIN_POINTS_TO_OCCUPY) are marked occupied.
    # This filters single-pixel noise — a real obstacle gets many votes,
    # a random bad pixel gets one.
    vote_grid = np.zeros((GRID_H, GRID_W), dtype=np.int16)

    if X_obs.size > 0:
        gx, gz, in_grid = _world_to_grid(X_obs, Z_obs)
        gx_v = gx[in_grid]
        gz_v = gz[in_grid]

        # np.add.at does an unbuffered in-place addition.
        # This means if two points land in the same cell, the cell
        # gets incremented twice — which is exactly what we want for voting.
        # Normal array indexing like grid[gz_v, gx_v] += 1 would only
        # count each unique index once due to numpy buffering.
        np.add.at(vote_grid, (gz_v, gx_v), 1)

    # Threshold: cells with enough votes become occupied
    # grid is uint8: 0 = free, 255 = occupied
    grid = np.where(vote_grid >= MIN_POINTS_TO_OCCUPY, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 5 — Forward clearance
    # ------------------------------------------------------------------
    # Scan the forward corridor from near to far.
    # The corridor is a vertical strip centred on CENTRE_GX.
    # The first row that contains an obstacle = forward clearance distance.
    half_corr_cells = max(1, int(FORWARD_CORRIDOR_HALF_W_M / GRID_RES_M))
    col_lo = max(0,      CENTRE_GX - half_corr_cells)
    col_hi = min(GRID_W, CENTRE_GX + half_corr_cells + 1)

    forward_clearance_m = MAP_FORWARD_M   # assume clear until obstacle found
    path_blocked = False

    for gz_row in range(GRID_H):
        # Check if any cell in the corridor at this depth is occupied
        if grid[gz_row, col_lo:col_hi].any():
            forward_clearance_m = gz_row * GRID_RES_M
            path_blocked = True
            break

    forward_clearance_m = max(forward_clearance_m, 0.0)

    # ------------------------------------------------------------------
    # Step 6 — Left and right clearance
    # ------------------------------------------------------------------
    # For each column, check if there's any obstacle within LATERAL_SCAN_DEPTH_M.
    # Then walk left and right from centre to find the first blocked column.
    scan_rows = min(GRID_H, int(LATERAL_SCAN_DEPTH_M / GRID_RES_M))
    col_has_obstacle = np.any(grid[:scan_rows, :] > 0, axis=0)

    left_free_cells = right_free_cells = 0

    for c in range(CENTRE_GX - 1, -1, -1):
        if col_has_obstacle[c]:
            break
        left_free_cells += 1

    for c in range(CENTRE_GX + 1, GRID_W):
        if col_has_obstacle[c]:
            break
        right_free_cells += 1

    left_clearance_m  = left_free_cells  * GRID_RES_M
    right_clearance_m = right_free_cells * GRID_RES_M

    # ------------------------------------------------------------------
    # Step 7 — Corridor geometry
    # ------------------------------------------------------------------
    # left_boundary_m and right_boundary_m are the distances to the
    # nearest wall or obstacle on each side (same as clearances for now,
    # but named more explicitly for environment_state.py to consume).
    left_boundary_m  = left_clearance_m
    right_boundary_m = right_clearance_m
    corridor_width_m = left_boundary_m + right_boundary_m

    # Centerline offset: how far the user is from the centre of the
    # navigable corridor.  Positive = user is to the left of centre
    # (should move right).  Negative = user is to the right.
    if corridor_width_m > 0:
        centerline_offset_m = right_boundary_m - left_boundary_m
    else:
        centerline_offset_m = 0.0

    # ------------------------------------------------------------------
    # Step 8 — Best opening angle
    # ------------------------------------------------------------------
    # Find the widest continuous free gap in the mid zone.
    # This is the direction the user should aim toward.
    look_gz_min = max(0,      int(ZONE_MID_M[0] / GRID_RES_M))
    look_gz_max = min(GRID_H, int(ZONE_MID_M[1] / GRID_RES_M))

    mid_blocked_cols = np.zeros(GRID_W, dtype=bool)
    if look_gz_max > look_gz_min:
        mid_blocked_cols = np.any(grid[look_gz_min:look_gz_max, :] > 0, axis=0)

    # Scan for the longest run of free columns
    best_start = best_end = run_start = None

    for gx_col in range(GRID_W):
        if not mid_blocked_cols[gx_col]:
            if run_start is None:
                run_start = gx_col
        else:
            if run_start is not None:
                run_end = gx_col - 1
                if best_start is None or (run_end - run_start) > (best_end - best_start):
                    best_start, best_end = run_start, run_end
                run_start = None

    # Handle a run that reaches the right edge of the grid
    if run_start is not None:
        run_end = GRID_W - 1
        if best_start is None or (run_end - run_start) > (best_end - best_start):
            best_start, best_end = run_start, run_end

    free_path_width_m   = None
    best_opening_angle  = None

    if best_start is not None:
        # Convert grid column of opening centre → world X → angle
        best_cx_cell  = 0.5 * (best_start + best_end)
        best_cx_m     = -MAP_HALF_W_M + (best_cx_cell + 0.5) * GRID_RES_M
        look_z_m      = 0.5 * (ZONE_MID_M[0] + ZONE_MID_M[1])
        free_path_width_m  = (best_end - best_start + 1) * GRID_RES_M
        best_opening_angle = float(np.degrees(np.arctan2(best_cx_m, look_z_m)))
        # Negative angle = opening is to the left
        # Positive angle = opening is to the right

    # ------------------------------------------------------------------
    # Step 9 — Nearest obstacle
    # ------------------------------------------------------------------
    # Find the closest obstacle point that lies in the forward corridor
    # or, if none, the closest obstacle anywhere.
    # We use the filtered point cloud (post-height-threshold) so we only
    # report real obstacles, not floor points.
    nearest_obstacle_m   = None
    nearest_obstacle_pos = None

    if X_obs.size > 0:
        # Map obstacle points to grid and check if they're in the grid
        gx_all, gz_all, in_g = _world_to_grid(X_obs, Z_obs)

        # Check which points land in occupied cells (double-checking vote threshold)
        in_occupied = np.zeros(X_obs.size, dtype=bool)
        in_occupied[in_g] = grid[gz_all[in_g], gx_all[in_g]] > 0

        # Prefer points in the forward corridor
        in_corridor = in_occupied & (np.abs(X_obs) <= FORWARD_CORRIDOR_HALF_W_M * 2.0)
        source = in_corridor if in_corridor.any() else in_occupied

        if source.any():
            idx = int(np.argmin(Z_obs[source]))
            nearest_obstacle_m   = float(Z_obs[source][idx])
            nearest_obstacle_pos = {
                "x_m": round(float(X_obs[source][idx]), 3),
                "z_m": round(float(Z_obs[source][idx]), 3),
            }

    # ------------------------------------------------------------------
    # Step 10 — Zone blocking flags
    # ------------------------------------------------------------------
    # Simple boolean: is there any obstacle in each zone corridor?
    def _zone_blocked(z_min_m: float, z_max_m: float) -> bool:
        gz_min = max(0,      int(z_min_m / GRID_RES_M))
        gz_max = min(GRID_H, int(z_max_m / GRID_RES_M))
        if gz_max <= gz_min:
            return False
        return bool(grid[gz_min:gz_max, col_lo:col_hi].any())

    zones = {
        "near_blocked": _zone_blocked(*ZONE_NEAR_M),
        "mid_blocked":  _zone_blocked(*ZONE_MID_M),
        "far_blocked":  _zone_blocked(*ZONE_FAR_M),
    }

    # ------------------------------------------------------------------
    # Step 11 — Grid coverage
    # ------------------------------------------------------------------
    # What fraction of the grid has at least one valid depth reading?
    # Low coverage = the depth map didn't cover much of the scene,
    # so clearance numbers may be overestimates.
    if X.size > 0:
        gx_all_v, gz_all_v, in_g_v = _world_to_grid(X, Z)
        covered_cells = set(
            zip(gz_all_v[in_g_v].tolist(), gx_all_v[in_g_v].tolist())
        )
        grid_coverage_ratio = len(covered_cells) / float(GRID_H * GRID_W)
    else:
        grid_coverage_ratio = 0.0

    # ------------------------------------------------------------------
    # Step 12 — Debug colormap
    # ------------------------------------------------------------------
    # Green = free, red = obstacle, black = no data
    # Only used for debug video output, not by any logic layer.
    debug_grid = np.zeros((GRID_H, GRID_W, 3), dtype=np.uint8)
    debug_grid[grid == 0]   = (0, 40, 0)     # dark green = free
    debug_grid[grid == 255] = (0, 0, 220)    # red = obstacle (BGR)

    # Draw user position marker
    user_row = 0
    cv2.circle(debug_grid, (CENTRE_GX, user_row), 4, (255, 255, 0), -1)

    # Draw forward corridor lines
    cv2.line(debug_grid, (col_lo, 0), (col_lo, GRID_H - 1), (80, 80, 80), 1)
    cv2.line(debug_grid, (col_hi, 0), (col_hi, GRID_H - 1), (80, 80, 80), 1)

    # Draw best opening arrow if one exists
    if best_opening_angle is not None:
        best_cx_cell = int(-MAP_HALF_W_M / GRID_RES_M +
                           ((-MAP_HALF_W_M + MAP_HALF_W_M +
                             np.tan(np.radians(best_opening_angle)) *
                             look_z_m) / GRID_RES_M) + GRID_W // 2)
        look_gz = int(look_z_m / GRID_RES_M)
        cv2.arrowedLine(
            debug_grid,
            (CENTRE_GX, 0),
            (int(CENTRE_GX + np.tan(np.radians(best_opening_angle)) * look_gz),
             look_gz),
            (0, 255, 0), 1, tipLength=0.2
        )

    # Flip vertically so far = top, near = bottom (intuitive orientation)
    debug_grid = cv2.flip(debug_grid, 0)

    # ------------------------------------------------------------------
    # Assemble and return bundle
    # ------------------------------------------------------------------
    return {
        # The grid itself
        "grid":                   grid,              # (GRID_H, GRID_W) uint8

        # Forward navigation
        "forward_clearance_m":    round(forward_clearance_m, 3),
        "path_blocked":           path_blocked,

        # Lateral navigation
        "left_clearance_m":       round(left_clearance_m,  3),
        "right_clearance_m":      round(right_clearance_m, 3),

        # Corridor geometry (for environment_state.py)
        "left_boundary_m":        round(left_boundary_m,   3),
        "right_boundary_m":       round(right_boundary_m,  3),
        "corridor_width_m":       round(corridor_width_m,  3),
        "centerline_offset_m":    round(centerline_offset_m, 3),
        # positive = user is left of centre (move right)
        # negative = user is right of centre (move left)

        # Best opening to navigate toward
        "free_path_width_m":      round(free_path_width_m,  3) if free_path_width_m  is not None else None,
        "best_opening_angle_deg": round(best_opening_angle, 2) if best_opening_angle is not None else None,

        # Nearest obstacle
        "nearest_obstacle_m":     round(nearest_obstacle_m, 3) if nearest_obstacle_m is not None else None,
        "nearest_obstacle_pos":   nearest_obstacle_pos,

        # Zone flags
        "zones":                  zones,

        # Reliability
        "obstacle_point_count":   int(X_obs.size),
        "grid_coverage_ratio":    round(grid_coverage_ratio, 3),

        # Grid geometry (for environment_state.py to use without importing constants)
        "_grid_res_m":            GRID_RES_M,
        "_map_half_w_m":          MAP_HALF_W_M,
        "_grid_w":                GRID_W,
        "_grid_h":                GRID_H,
        "_centre_gx":             CENTRE_GX,

        # Debug only — do not use in logic layers
        "debug_grid":             debug_grid,        # (GRID_H, GRID_W, 3) uint8
    }


# =============================================================================
# Standalone test — run directly: python occupancy_map.py
# Requires depth_map.py to be in the same directory (or modules/).
# Runs one frame, prints the bundle, saves a debug image.
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

    os.makedirs("debug_output", exist_ok=True)

    # ------------------------------------------------------------------
    # Save raw camera image immediately so you can compare it to the
    # occupancy map and depth output side by side on your Mac.
    # Left eye only — this is what the algorithm is actually looking at.
    # ------------------------------------------------------------------
    h, w  = frame.shape[:2]
    eye_w = w // 2
    eye_h = h
    cv2.imwrite("debug_output/camera_left.png", frame[:, :eye_w])
    print("Raw left camera image saved to debug_output/camera_left.png")

    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)

    rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)

    stereo       = build_stereo_matcher()
    depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)

    # ------------------------------------------------------------------
    # Floor diagnosis — print before running occupancy map so you can
    # see exactly what floor_y_m is being estimated as and whether it
    # matches the real Y distribution of the point cloud.
    #
    # What to look for:
    #   floor_y_m (from depth_map)  — estimated from bottom image rows
    #   p85 of all valid Y          — 85th percentile of full point cloud
    #   floor_y_used                — whichever is larger (more downward)
    #
    # If floor_y_m is much smaller than p85, depth_map's estimate was
    # too conservative and the occupancy map was falsely marking floor
    # points as obstacles.  The fix in Step 3 above handles this.
    # ------------------------------------------------------------------
    pts   = depth_bundle["points_3d"]
    valid = depth_bundle["valid_mask"]
    Y_all = pts[:, :, 1][valid]

    print("\n--- floor diagnosis ---")
    print(f"  floor_y_m from depth_map     : {depth_bundle['floor_y_m']:.3f}")
    print(f"  obstacle_height_m            : {depth_bundle['obstacle_height_m']:.3f}")
    if Y_all.size > 0:
        print(f"  all-points Y p50             : {np.percentile(Y_all, 50):.3f}")
        print(f"  all-points Y p15             : {np.percentile(Y_all, 15):.3f}  ← secondary floor estimate")
        print(f"  all-points Y p05             : {np.percentile(Y_all, 5):.3f}")
        # bottom 25% of image rows specifically
        bottom_start = int(eye_h * 0.75)
        Y_bottom = pts[bottom_start:, :, 1][valid[bottom_start:, :]]
        if Y_bottom.size > 0:
            print(f"  bottom-row Y p10             : {np.percentile(Y_bottom, 10):.3f}  ← depth_map floor estimate")
            print(f"  bottom-row Y p50             : {np.percentile(Y_bottom, 50):.3f}")
        # What floor_y_used will be after the fix
        floor_y_secondary = float(np.percentile(Y_all, 15))
        floor_y_used = min(depth_bundle['floor_y_m'], floor_y_secondary)
        print(f"  floor_y_used (after fix)     : {floor_y_used:.3f}  (more negative = safer)")
        print(f"  obstacle threshold Y         : {floor_y_used + OBSTACLE_HEIGHT_THRESH_M:.3f}")
        print(f"  points above threshold       : {(Y_all > (floor_y_used + OBSTACLE_HEIGHT_THRESH_M)).sum()}")
        print(f"  total valid points           : {Y_all.size}")
        pct = 100.0 * (Y_all > (floor_y_used + OBSTACLE_HEIGHT_THRESH_M)).sum() / Y_all.size
        print(f"  obstacle point fraction      : {pct:.1f}%  (open room should be <20%)")
        # Also show what fraction survives the Z cap
        Z_all = pts[:, :, 2][valid]
        height_all = Y_all - floor_y_used
        in_band = (
            (height_all > OBSTACLE_HEIGHT_THRESH_M) &
            (height_all < 2.0) &
            (Z_all <= OBSTACLE_MAX_Z_M)
        )
        pct_capped = 100.0 * in_band.sum() / Y_all.size
        print(f"  after Z cap ({OBSTACLE_MAX_Z_M}m)           : {pct_capped:.1f}%  (this is what the grid uses)")

    occ_bundle = build_occupancy_map(depth_bundle)

    # ------------------------------------------------------------------
    # Print occupancy bundle
    # ------------------------------------------------------------------
    print("\n--- occupancy_map bundle ---")
    for k, v in occ_bundle.items():
        if isinstance(v, np.ndarray):
            occupied = int((v == 255).sum()) if v.ndim == 2 else 0
            print(f"  {k:30s}: ndarray {v.shape}  occupied_cells={occupied}")
        elif isinstance(v, dict):
            print(f"  {k:30s}: {v}")
        else:
            print(f"  {k:30s}: {v}")

    # ------------------------------------------------------------------
    # Save debug images
    # ------------------------------------------------------------------
    # Occupancy grid — scale up so individual cells are visible
    scale = 4
    debug_large = cv2.resize(
        occ_bundle["debug_grid"],
        (GRID_W * scale, GRID_H * scale),
        interpolation=cv2.INTER_NEAREST
    )
    cv2.imwrite("debug_output/occupancy_map_test.png", debug_large)

    # Depth colormap from depth_map
    cv2.imwrite("debug_output/depth_map_test.png", depth_bundle["debug_colormap"])

    # Annotated camera image — draw a horizontal line at the bottom 25%
    # boundary so you can see exactly which rows depth_map uses for
    # floor estimation
    camera_annotated = frame[:, :eye_w].copy()
    floor_row = int(eye_h * 0.75)
    cv2.line(camera_annotated, (0, floor_row), (eye_w, floor_row), (0, 255, 255), 2)
    cv2.putText(camera_annotated, "floor estimation region below this line",
                (10, floor_row - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite("debug_output/camera_annotated.png", camera_annotated)

    print("\n--- saved to debug_output/ ---")
    print("  camera_left.png       raw left eye")
    print("  camera_annotated.png  left eye with floor region marked")
    print("  depth_map_test.png    depth colormap")
    print("  occupancy_map_test.png  2D grid")
    print("\nPull all to Mac with:")
    print("  scp zanewolf@192.168.50.103:~/6thSense/debug_output/*.png ~/Desktop/")
