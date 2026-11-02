"""
environment_state.py
====================
Layer 2.1 — Situation  (Scene Understanding)

What this file does
-------------------
Consumes the output of occupancy_map.py and yolo_detector.py and
produces a structured understanding of what the scene actually means
for navigation.

This file does NOT control outputs (haptics, audio, speech).
This file does NOT decide what mode the system is in.
This file's only job is:
    occupancy bundle + yolo bundle → what kind of space is this
                                     and what does it mean right now

The key principle: raw data → interpretation
--------------------------------------------
occupancy_map.py says:  "forward_clearance=2.1m, left=0.9m, right=1.4m"
yolo_detector.py says:  "chair at 0.8m, dining table at 2.6m"
environment_state.py says:
    "open room, moderate clutter on left, chair blocking right-centre,
     best path is slightly left, traversability ahead is 0.72"

That interpretation is what every layer above this one works from.

Output bundle
-------------
space_type          : str   — "open_room", "corridor", "doorway",
                              "cluttered_room", "dead_end", "unknown"
                              Determined from geometry + object labels combined.

clutter_density     : float — 0.0–1.0 continuous score
                              0.0 = completely clear
                              0.5 = moderate furniture/objects
                              1.0 = heavily cluttered, hard to navigate
                              NOT a boolean. Derived from obstacle grid
                              coverage + YOLO detection density.

clutter_zone        : str   — where clutter is concentrated:
                              "left", "right", "centre", "even", "none"

traversability      : dict  — per-direction walkability score 0.0–1.0
  ahead             : float — how walkable the forward path is
  left              : float — how walkable the left side is
  right             : float — how walkable the right side is

dominant_obstacle   : dict or None — the most navigation-relevant obstacle
  label             : str   — YOLO label if available, else "unknown"
  distance_m        : float
  position_x_m      : float
  blocks_path       : bool

nearest_person      : dict or None — nearest detected person specifically
  distance_m        : float
  position_x_m      : float
  in_path           : bool

objects_in_path     : list  — all YOLO detections that are blocking
doorway_candidate   : bool  — geometry suggests a doorway ahead
corridor_detected   : bool  — geometry suggests a corridor/hallway

centerline_offset_m : float — how far user is from navigable centre
                              positive = user left of centre (move right)
                              negative = user right of centre (move left)

guidance            : dict  — what the system should do right now
  state             : str   — "clear", "slight_left", "slight_right",
                              "strong_left", "strong_right", "stop",
                              "caution", "recentre"
  reason            : str   — human readable explanation
  priority          : int   — 1=info, 2=caution, 3=urgent

confidence          : float — how reliable this state estimate is 0.0–1.0
                              Low if depth_valid_ratio is low or few
                              detections matched with depth.

Stack position:
    depth_map.py
    occupancy_map.py  ─┐
    yolo_detector.py  ─┴→  >>> environment_state.py <<<
                                situation_layer.py     (future)
                                mode_controller.py     (future)



Stable version
--------------
This version keeps the SAME main output structure as the original file, so
higher-level files should not need to change.

Key improvements:
- Adds scored space classification instead of only hard thresholds.
- Adds temporal smoothing for space_type so it does not flicker frame-to-frame.
- Adds hysteresis so the system does not switch labels unless the new label is
  meaningfully stronger.
- Makes corridor and doorway detection stricter.
- Keeps urgent obstacle guidance immediate and unsmoothed for safety.
- Adds optional debug keys: raw_space_type, space_scores.

Important compatibility note
----------------------------
All original output keys are preserved:
    space_type, clutter_density, clutter_zone, traversability,
    dominant_obstacle, nearest_person, objects_in_path, wall,
    doorway_candidate, corridor_detected, centerline_offset_m,
    guidance, confidence, detection_count, debug_hud

New debug keys are added but should not break normal dictionary use:
    raw_space_type, space_scores
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional
from collections import deque, Counter

# =============================================================================
# Tunable constants
# =============================================================================

# Corridor detection — if both left and right clearances are below this
# and roughly equal, we're probably in a corridor/hallway.
CORRIDOR_MAX_SIDE_M     = 1.8
CORRIDOR_SYMMETRY_RATIO = 0.60
CORRIDOR_MIN_FORWARD_M  = 1.2
CORRIDOR_MIN_PATH_W_M   = 0.70

# Doorway detection — a narrow opening directly ahead in the mid zone.
DOORWAY_MIN_WIDTH_M  = 0.70
DOORWAY_MAX_WIDTH_M  = 1.45
DOORWAY_MAX_ANGLE    = 20.0
DOORWAY_MIN_FORWARD_M = 1.0

# Space classification scoring
SPACE_MIN_SCORE       = 0.52   # below this, raw space type becomes unknown
SPACE_SWITCH_MARGIN   = 0.15   # new label must beat current smoothed label by this much
SPACE_HISTORY_LEN     = 8      # frames used for smoothing
SPACE_HISTORY_MIN_VOTES = 3    # minimum votes needed to trust a new space type
LOW_COVERAGE_RATIO    = 0.10   # low data quality, but no instant unknown flicker

# Clutter scoring weights
CLUTTER_NEAR_WEIGHT = 1.0
CLUTTER_MID_WEIGHT  = 0.6
CLUTTER_FAR_WEIGHT  = 0.3

# Traversability — forward clearance thresholds
TRAV_CLEAR_M    = 2.0
TRAV_CAUTION_M  = 1.0
TRAV_BLOCKED_M  = 0.5

# Guidance — how much correction to recommend
SLIGHT_CORRECTION_M  = 0.4
STRONG_CORRECTION_M  = 1.0

# Priority thresholds
URGENT_DISTANCE_M  = 0.5
CAUTION_DISTANCE_M = 1.5

# Wall detection — depth histogram approach
WALL_BAND_COVERAGE   = 0.45
WALL_MAX_STD_M       = 0.35
WALL_MIN_BANDS       = 2
WALL_NUM_BANDS       = 6
WALL_CHECK_BANDS     = (1, 4)
WALL_FULL_COVERAGE   = 0.65


# =============================================================================
# Simple in-process temporal state
# =============================================================================

_SPACE_HISTORY: deque[str] = deque(maxlen=SPACE_HISTORY_LEN)
_LAST_SPACE_TYPE: str = "unknown"
_LAST_SPACE_SCORES: dict[str, float] = {}

# Additional smoothing buffers
_DOMINANT_X_HISTORY = deque(maxlen=5)
_ANGLE_HISTORY = deque(maxlen=5)
_LAST_DISTANCE = None

# Direction tuning
CENTER_ZONE_M = 0.35
DIRECTION_CONFIDENCE_MIN = 0.25
DISTANCE_HYSTERESIS = 0.3


def reset_temporal_state() -> None:
    """
    Reset smoothing history.

    Call this if the camera is restarted, the pipeline is reset, or you want
    a clean test run. Normal frame-by-frame operation does not need to call it.
    """
    global _LAST_SPACE_TYPE, _LAST_SPACE_SCORES
    _SPACE_HISTORY.clear()
    _LAST_SPACE_TYPE = "unknown"
    _LAST_SPACE_SCORES = {}


# =============================================================================
# Helpers
# =============================================================================

def _safe_float(value, default: float = 0.0) -> float:
    """Convert numeric-ish values to float without crashing on None."""
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _score_range(value: Optional[float], low: float, high: float, ideal: Optional[float] = None) -> float:
    """
    Soft score for a value being inside a useful range.

    Returns 1.0 near the ideal/middle and fades toward 0.0 outside the range.
    This is safer than hard if/else thresholds for noisy stereo values.
    """
    if value is None:
        return 0.0

    value = float(value)
    if ideal is None:
        ideal = (low + high) / 2.0

    if low <= value <= high:
        span = max(ideal - low, high - ideal, 1e-6)
        return _clamp01(1.0 - abs(value - ideal) / (span * 1.25))

    # Gentle falloff just outside the range
    falloff = max((high - low) * 0.5, 0.1)
    if value < low:
        return _clamp01(1.0 - (low - value) / falloff)
    return _clamp01(1.0 - (value - high) / falloff)


# =============================================================================
# Wall detection
# =============================================================================

def _detect_wall(occ: dict, depth_bundle: Optional[dict] = None) -> dict:
    """
    Detect walls using a depth histogram on horizontal image bands.

    Preferred method uses the depth map directly. Fallback uses occupancy grid.
    """
    if depth_bundle is not None:
        depth_map = depth_bundle["depth_map_m"]
        valid_mask = depth_bundle["valid_mask"]

        h, w = depth_map.shape
        band_h = max(1, h // WALL_NUM_BANDS)

        wall_bands = []
        wall_depths = []

        for band_idx in range(WALL_CHECK_BANDS[0], WALL_CHECK_BANDS[1]):
            row_start = band_idx * band_h
            row_end = min(h, row_start + band_h)

            band_depth = depth_map[row_start:row_end, :]
            band_valid = valid_mask[row_start:row_end, :]

            valid_depths = band_depth[band_valid]
            total_pixels = band_valid.size
            valid_count = valid_depths.size

            if total_pixels == 0 or valid_count == 0:
                continue

            coverage = valid_count / total_pixels
            if coverage < WALL_BAND_COVERAGE:
                continue

            median_d = float(np.median(valid_depths))
            std_d = float(np.std(valid_depths))

            if std_d <= WALL_MAX_STD_M and 0.4 < median_d < 4.0:
                wall_bands.append(band_idx)
                wall_depths.append(median_d)

        if len(wall_bands) >= WALL_MIN_BANDS:
            wall_dist = float(np.mean(wall_depths))

            best_band_idx = wall_bands[len(wall_bands) // 2]
            row_start = best_band_idx * band_h
            row_end = min(h, row_start + band_h)
            band_valid = valid_mask[row_start:row_end, :]
            col_has_valid = band_valid.any(axis=0)
            coverage = float(col_has_valid.mean())

            cx = w // 2
            left_cov = float(col_has_valid[:cx].mean())
            right_cov = float(col_has_valid[cx:].mean())

            if coverage >= WALL_FULL_COVERAGE:
                direction = "ahead"
            elif left_cov > right_cov * 1.5:
                direction = "left"
            elif right_cov > left_cov * 1.5:
                direction = "right"
            else:
                direction = "ahead"

            return {
                "present": True,
                "distance_m": round(wall_dist, 3),
                "coverage": round(coverage, 3),
                "is_full_wall": coverage >= WALL_FULL_COVERAGE,
                "depth_rows": len(wall_bands),
                "direction": direction,
            }

        return {
            "present": False,
            "distance_m": None,
            "coverage": 0.0,
            "is_full_wall": False,
            "depth_rows": 0,
            "direction": "none",
        }

    # Fallback — occupancy grid method
    grid = occ["grid"]
    gres = occ["_grid_res_m"]
    gh, gw = grid.shape
    cx = occ["_centre_gx"]

    row_coverage = (grid > 0).sum(axis=1) / gw
    wall_rows = np.where(row_coverage >= 0.65)[0]

    if len(wall_rows) == 0:
        return {
            "present": False,
            "distance_m": None,
            "coverage": 0.0,
            "is_full_wall": False,
            "depth_rows": 0,
            "direction": "none",
        }

    nearest_row = int(wall_rows[0])
    nearest_distance = float(nearest_row * gres)
    nearest_coverage = float(row_coverage[nearest_row])
    wall_slice = grid[nearest_row, :]
    left_occ = float((wall_slice[:cx] > 0).mean())
    right_occ = float((wall_slice[cx:] > 0).mean())

    if nearest_coverage >= WALL_FULL_COVERAGE:
        direction = "ahead"
    elif left_occ > right_occ * 1.5:
        direction = "left"
    elif right_occ > left_occ * 1.5:
        direction = "right"
    else:
        direction = "ahead"

    return {
        "present": True,
        "distance_m": round(nearest_distance, 3),
        "coverage": round(nearest_coverage, 3),
        "is_full_wall": nearest_coverage >= WALL_FULL_COVERAGE,
        "depth_rows": 1,
        "direction": direction,
    }


# =============================================================================
# Space type classification
# =============================================================================

def _edge_obstruction_score(occ: dict) -> float:
    """
    Estimate whether there are obstacles/walls on both sides of the forward view.

    Doorways should usually have solid structure on both sides of a narrower
    opening. This helps avoid mistaking a random open gap between furniture for
    a real doorway.
    """
    grid = occ["grid"]
    gres = occ["_grid_res_m"]
    gh, gw = grid.shape
    cx = occ["_centre_gx"]

    # Look roughly 0.5m to 2.0m ahead, where a doorway frame would matter.
    r0 = max(0, int(0.5 / gres))
    r1 = min(gh, int(2.0 / gres))
    if r1 <= r0:
        return 0.0

    band = grid[r0:r1, :]
    if band.size == 0:
        return 0.0

    # Side regions, avoiding the center walking path.
    left_region = band[:, :max(1, int(gw * 0.35))]
    right_region = band[:, min(gw - 1, int(gw * 0.65)):]

    left_density = float((left_region > 0).mean()) if left_region.size else 0.0
    right_density = float((right_region > 0).mean()) if right_region.size else 0.0

    # Both sides should have some structure. Use min so one-sided clutter scores low.
    both_sides = min(left_density, right_density)
    return _clamp01(both_sides / 0.08)  # 8% occupied side density is fairly strong


def _compute_space_scores(occ: dict, yolo: dict, wall: dict) -> dict[str, float]:
    """
    Compute soft scores for each possible space type.

    This avoids brittle single-frame threshold logic. The final label is chosen
    from these scores, then smoothed over time.
    """
    fc = _safe_float(occ.get("forward_clearance_m"), 0.0)
    lc = _safe_float(occ.get("left_clearance_m"), 0.0)
    rc = _safe_float(occ.get("right_clearance_m"), 0.0)
    cw = _safe_float(occ.get("corridor_width_m"), 0.0)
    fpw = occ.get("free_path_width_m")
    angle = occ.get("best_opening_angle_deg")
    cov = _safe_float(occ.get("grid_coverage_ratio"), 0.0)

    fpw_f = None if fpw is None else float(fpw)
    angle_f = None if angle is None else float(angle)

    low_data_penalty = 0.75 if cov < LOW_COVERAGE_RATIO else 1.0

    scores = {
        "corridor": 0.0,
        "doorway": 0.0,
        "open_room": 0.0,
        "cluttered_room": 0.0,
        "dead_end": 0.0,
        "unknown": 0.0,
    }

    # Dead end: full wall ahead with no meaningful opening.
    if wall.get("present") and wall.get("is_full_wall") and wall.get("direction") == "ahead":
        fpw_ok = fpw_f is not None and fpw_f > DOORWAY_MIN_WIDTH_M
        scores["dead_end"] = 0.9 if not fpw_ok else 0.45

    # Corridor: narrow-ish, left/right balanced, forward path exists.
    side_valid = lc > 0.1 and rc > 0.1
    if side_valid:
        side_ratio = min(lc, rc) / max(lc, rc) if max(lc, rc) > 0 else 0.0
    else:
        side_ratio = 0.0

    sides_narrow = _score_range(max(lc, rc), 0.2, CORRIDOR_MAX_SIDE_M, ideal=1.1)
    side_balance = _clamp01((side_ratio - CORRIDOR_SYMMETRY_RATIO) / (1.0 - CORRIDOR_SYMMETRY_RATIO))
    forward_ok = _clamp01((fc - CORRIDOR_MIN_FORWARD_M) / 1.2)
    path_ok = 1.0 if fpw_f is None else _clamp01((fpw_f - CORRIDOR_MIN_PATH_W_M) / 1.0)

    corridor_score = (
        sides_narrow * 0.35 +
        side_balance * 0.30 +
        forward_ok * 0.20 +
        path_ok * 0.15
    )

    if wall.get("present") and wall.get("direction") in ("left", "right"):
        corridor_score += 0.10

    scores["corridor"] = _clamp01(corridor_score) * low_data_penalty

    # Doorway: narrow centered opening with side structure and usable forward space.
    width_score = _score_range(fpw_f, DOORWAY_MIN_WIDTH_M, DOORWAY_MAX_WIDTH_M, ideal=0.95)
    angle_score = 0.0 if angle_f is None else _clamp01(1.0 - abs(angle_f) / DOORWAY_MAX_ANGLE)
    forward_door_score = _clamp01((fc - DOORWAY_MIN_FORWARD_M) / 1.5)
    edge_score = _edge_obstruction_score(occ)

    doorway_score = (
        width_score * 0.35 +
        angle_score * 0.25 +
        forward_door_score * 0.20 +
        edge_score * 0.20
    )

    # If there is a full wall ahead, doorway should be weaker unless a clear opening exists.
    if wall.get("present") and wall.get("is_full_wall") and wall.get("direction") == "ahead":
        doorway_score *= 0.65

    scores["doorway"] = _clamp01(doorway_score) * low_data_penalty

    # Room object hints from YOLO.
    labels = [d.get("label") for d in yolo.get("detections", [])]
    room_objects = {
        "chair", "dining table", "couch", "bed", "tv",
        "refrigerator", "sink", "toilet", "potted plant"
    }
    room_label_count = sum(1 for label in labels if label in room_objects)
    yolo_room_score = _clamp01(room_label_count / 3.0)

    # Cluttered room: multiple room objects and not very open forward.
    object_density_score = _clamp01(yolo.get("detection_count", 0) / 6.0)
    limited_forward_score = _clamp01((3.0 - fc) / 2.5)
    scores["cluttered_room"] = _clamp01(
        yolo_room_score * 0.45 + object_density_score * 0.25 + limited_forward_score * 0.30
    ) * low_data_penalty

    # Open room: good forward clearance and wide space, with low room-object clutter.
    forward_open_score = _clamp01((fc - 1.0) / 2.0)
    width_open_score = _clamp01((cw - 1.5) / 1.5)
    low_object_score = 1.0 - _clamp01(yolo.get("detection_count", 0) / 6.0)
    not_corridor_score = 1.0 - scores["corridor"] * 0.6

    scores["open_room"] = _clamp01(
        forward_open_score * 0.40 +
        width_open_score * 0.25 +
        low_object_score * 0.20 +
        not_corridor_score * 0.15
    ) * low_data_penalty

    # Unknown: grows when data quality is low or all other scores are weak.
    best_non_unknown = max(v for k, v in scores.items() if k != "unknown")
    data_unknown = _clamp01((LOW_COVERAGE_RATIO - cov) / LOW_COVERAGE_RATIO) if cov < LOW_COVERAGE_RATIO else 0.0
    weak_unknown = _clamp01((SPACE_MIN_SCORE - best_non_unknown) / SPACE_MIN_SCORE)
    scores["unknown"] = max(data_unknown * 0.75, weak_unknown * 0.65)

    return {k: round(float(_clamp01(v)), 3) for k, v in scores.items()}


def _pick_raw_space_type(space_scores: dict[str, float]) -> str:
    """Choose a raw one-frame label from scores."""
    candidates = {k: v for k, v in space_scores.items() if k != "unknown"}
    best_label, best_score = max(candidates.items(), key=lambda item: item[1])

    if best_score < SPACE_MIN_SCORE:
        return "unknown"

    # If unknown is stronger than the best label, trust unknown for the raw frame.
    if space_scores.get("unknown", 0.0) > best_score:
        return "unknown"

    return best_label


def _smooth_space_type(raw_space_type: str, space_scores: dict[str, float]) -> str:
    """
    Smooth space_type over several frames and apply hysteresis.

    This is the main anti-flicker layer. It should make a steady camera view
    stop bouncing between unknown/open_room/corridor.
    """
    global _LAST_SPACE_TYPE, _LAST_SPACE_SCORES

    _SPACE_HISTORY.append(raw_space_type)

    # Start quickly if we have no stable label yet.
    if _LAST_SPACE_TYPE == "unknown" and len(_SPACE_HISTORY) < 3:
        if raw_space_type != "unknown":
            _LAST_SPACE_TYPE = raw_space_type
            _LAST_SPACE_SCORES = space_scores
        return _LAST_SPACE_TYPE

    counts = Counter(_SPACE_HISTORY)
    common_label, common_votes = counts.most_common(1)[0]

    # Prefer not to switch into unknown unless unknown dominates the history.
    if common_label == "unknown" and common_votes < max(4, SPACE_HISTORY_LEN // 2):
        return _LAST_SPACE_TYPE

    # If history is not strong enough, keep previous stable type.
    if common_votes < SPACE_HISTORY_MIN_VOTES:
        return _LAST_SPACE_TYPE

    if common_label == _LAST_SPACE_TYPE:
        _LAST_SPACE_SCORES = space_scores
        return _LAST_SPACE_TYPE

    old_score = space_scores.get(_LAST_SPACE_TYPE, 0.0)
    new_score = space_scores.get(common_label, 0.0)

    # Hysteresis: only switch if the new label is clearly stronger.
    # Exception: if old is unknown and new is plausible, switch faster.
    if _LAST_SPACE_TYPE == "unknown":
        if new_score >= SPACE_MIN_SCORE:
            _LAST_SPACE_TYPE = common_label
            _LAST_SPACE_SCORES = space_scores
        return _LAST_SPACE_TYPE

    if new_score >= old_score + SPACE_SWITCH_MARGIN or common_votes >= SPACE_HISTORY_LEN - 1:
        _LAST_SPACE_TYPE = common_label
        _LAST_SPACE_SCORES = space_scores

    return _LAST_SPACE_TYPE


def _classify_space(occ: dict, yolo: dict, wall: dict) -> tuple[str, str, dict[str, float]]:
    """
    Determine space type using scoring + smoothing.

    Returns:
        smoothed_space_type, raw_space_type, space_scores
    """
    scores = _compute_space_scores(occ, yolo, wall)
    raw = _pick_raw_space_type(scores)
    smoothed = _smooth_space_type(raw, scores)
    return smoothed, raw, scores


# =============================================================================
# Clutter density scoring
# =============================================================================

def _compute_clutter(occ: dict, yolo: dict) -> tuple[float, str]:
    """Compute continuous clutter density score and clutter zone."""
    grid = occ["grid"]
    gres = occ["_grid_res_m"]

    near_rows = slice(0, int(0.9 / gres))
    mid_rows = slice(int(0.9 / gres), int(2.0 / gres))
    far_rows = slice(int(2.0 / gres), int(3.5 / gres))

    def _zone_density(row_slice):
        zone = grid[row_slice, :]
        if zone.size == 0:
            return 0.0
        return float((zone > 0).sum()) / zone.size

    near_density = _zone_density(near_rows)
    mid_density = _zone_density(mid_rows)
    far_density = _zone_density(far_rows)

    raw = (
        near_density * CLUTTER_NEAR_WEIGHT +
        mid_density * CLUTTER_MID_WEIGHT +
        far_density * CLUTTER_FAR_WEIGHT
    )

    clutter_density = min(1.0, raw / 1.9)

    yolo_factor = min(1.0, yolo["detection_count"] / 6.0)
    clutter_density = min(1.0, clutter_density * 0.7 + yolo_factor * 0.3)

    cx = occ["_centre_gx"]
    left_density = float((grid[:, :cx] > 0).sum()) / max(1, grid[:, :cx].size)
    right_density = float((grid[:, cx:] > 0).sum()) / max(1, grid[:, cx:].size)

    if clutter_density < 0.1:
        clutter_zone = "none"
    elif abs(left_density - right_density) < 0.05:
        clutter_zone = "even"
    elif left_density > right_density * 1.3:
        clutter_zone = "left"
    elif right_density > left_density * 1.3:
        clutter_zone = "right"
    else:
        clutter_zone = "centre"

    return round(clutter_density, 3), clutter_zone


# =============================================================================
# Traversability scoring
# =============================================================================

def _compute_traversability(occ: dict) -> dict:
    """Compute per-direction traversability scores, 0.0–1.0."""
    fc = occ["forward_clearance_m"]
    lc = occ["left_clearance_m"]
    rc = occ["right_clearance_m"]

    if fc >= TRAV_CLEAR_M:
        ahead = 1.0
    elif fc <= TRAV_BLOCKED_M:
        ahead = 0.0
    else:
        ahead = (fc - TRAV_BLOCKED_M) / (TRAV_CLEAR_M - TRAV_BLOCKED_M)

    map_half = occ["_map_half_w_m"]
    left = min(1.0, lc / map_half)
    right = min(1.0, rc / map_half)

    return {
        "ahead": round(ahead, 3),
        "left": round(left, 3),
        "right": round(right, 3),
    }


# =============================================================================
# Dominant obstacle
# =============================================================================

def _find_dominant_obstacle(occ: dict, yolo: dict, wall: dict) -> Optional[dict]:
    global _LAST_DISTANCE
    """Find the single most navigation-relevant obstacle."""
    if wall["present"] and wall["distance_m"] is not None:
        wall_label = "wall" if wall["is_full_wall"] else f"wall ({wall['direction']})"
        return {
            "label": wall_label,
            "distance_m": wall["distance_m"],
            "position_x_m": 0.0 if wall["direction"] == "ahead" else (-1.0 if wall["direction"] == "left" else 1.0),
            "blocks_path": wall["direction"] == "ahead",
            "source": "wall_detection",
        }

    blocking = [d for d in yolo["detections"] if d["is_blocking"]]
    if blocking:
        d = min(blocking, key=lambda x: x["distance_m"] or 999)
        # Smooth position_x_m
        if d.get("position_x_m") is not None:
            _DOMINANT_X_HISTORY.append(d["position_x_m"])
            smoothed_x = float(np.mean(_DOMINANT_X_HISTORY))
        else:
            smoothed_x = d.get("position_x_m")
        # Distance hysteresis
        if _LAST_DISTANCE is not None and d.get("distance_m") is not None:
            if abs(d["distance_m"] - _LAST_DISTANCE) < DISTANCE_HYSTERESIS:
                d["distance_m"] = _LAST_DISTANCE
        _LAST_DISTANCE = d.get("distance_m")
        return {
            "label": d["label"],
            "distance_m": d["distance_m"],
            "position_x_m": smoothed_x,
            "blocks_path": True,
            "source": "yolo",
        }

    close = [d for d in yolo["detections"] if d["is_close"] and d["distance_m"] is not None]
    if close:
        d = min(close, key=lambda x: x["distance_m"] or 999)
        # Smooth position_x_m
        if d.get("position_x_m") is not None:
            _DOMINANT_X_HISTORY.append(d["position_x_m"])
            smoothed_x = float(np.mean(_DOMINANT_X_HISTORY))
        else:
            smoothed_x = d.get("position_x_m")
        # Distance hysteresis
        if _LAST_DISTANCE is not None and d.get("distance_m") is not None:
            if abs(d["distance_m"] - _LAST_DISTANCE) < DISTANCE_HYSTERESIS:
                d["distance_m"] = _LAST_DISTANCE
        _LAST_DISTANCE = d.get("distance_m")
        return {
            "label": d["label"],
            "distance_m": d["distance_m"],
            "position_x_m": smoothed_x,
            "blocks_path": d["in_path"],
            "source": "yolo",
        }

    nd = occ["nearest_obstacle_m"]
    if nd is not None:
        pos = occ["nearest_obstacle_pos"]
        return {
            "label": "obstacle",
            "distance_m": nd,
            "position_x_m": pos["x_m"] if pos else None,
            "blocks_path": occ["path_blocked"],
            "source": "occupancy",
        }

    return None


# =============================================================================
# Guidance state
# =============================================================================

def _compute_guidance(
    occ: dict,
    dominant: Optional[dict],
    traversability: dict,
    centerline_offset_m: float,
    space_type: str,
) -> dict:
    """Decide what the system should tell the user right now."""
    fc = occ["forward_clearance_m"]
    lc = occ["left_clearance_m"]
    rc = occ["right_clearance_m"]
    angle = occ["best_opening_angle_deg"]

    if angle is not None:
        _ANGLE_HISTORY.append(angle)
        angle = float(np.mean(_ANGLE_HISTORY))

    # Wall ahead — special handling.
    if dominant is not None and dominant["source"] == "wall_detection":
        dist = dominant["distance_m"]
        if dominant["blocks_path"]:
            if angle is not None:
                side = "left" if angle < 0 else "right"
                return {
                    "state": f"slight_{side}",
                    "reason": f"wall at {dist:.1f}m — opening {side}",
                    "priority": 2,
                }
            return {
                "state": "caution",
                "reason": f"wall at {dist:.1f}m ahead",
                "priority": 2,
            }

    # Urgent obstacle remains immediate and unsmoothed for safety.
    if dominant is not None and dominant["distance_m"] is not None:
        if dominant["distance_m"] < URGENT_DISTANCE_M and dominant["blocks_path"]:
            x = dominant.get("position_x_m")
            if x is None or abs(x) < CENTER_ZONE_M:
                return {
                    "state": "stop",
                    "reason": f"{dominant['label']} directly ahead at {dominant['distance_m']:.1f}m",
                    "priority": 3,
                }
            elif x < 0:
                return {
                    "state": "strong_left",
                    "reason": f"{dominant['label']} at {dominant['distance_m']:.1f}m, move left",
                    "priority": 3,
                }
            else:
                return {
                    "state": "strong_right",
                    "reason": f"{dominant['label']} at {dominant['distance_m']:.1f}m, move right",
                    "priority": 3,
                }

    if dominant is not None and dominant["distance_m"] is not None:
        if dominant["distance_m"] < CAUTION_DISTANCE_M and dominant["blocks_path"]:
            return {
                "state": "caution",
                "reason": f"{dominant['label']} at {dominant['distance_m']:.1f}m ahead",
                "priority": 2,
            }

    if space_type == "corridor":
        if abs(centerline_offset_m) > STRONG_CORRECTION_M:
            state = "strong_right" if centerline_offset_m > 0 else "strong_left"
            return {
                "state": state,
                "reason": f"drifting {abs(centerline_offset_m):.1f}m from corridor centre",
                "priority": 2,
            }
        if abs(centerline_offset_m) > SLIGHT_CORRECTION_M:
            state = "slight_right" if centerline_offset_m > 0 else "slight_left"
            return {
                "state": state,
                "reason": "recentre in corridor",
                "priority": 1,
            }

    if angle is not None and occ["path_blocked"]:
        if angle < -SLIGHT_CORRECTION_M * 10:
            state = "strong_left" if angle < -20 else "slight_left"
            priority = 2 if abs(angle) > 20 else 1
            return {
                "state": state,
                "reason": f"opening at {angle:.0f}° left",
                "priority": priority,
            }
        if angle > SLIGHT_CORRECTION_M * 10:
            state = "strong_right" if angle > 20 else "slight_right"
            priority = 2 if abs(angle) > 20 else 1
            return {
                "state": state,
                "reason": f"opening at {angle:.0f}° right",
                "priority": priority,
            }

    if traversability["ahead"] > 0.8:
        return {
            "state": "clear",
            "reason": f"path clear for {fc:.1f}m",
            "priority": 1,
        }

    return {
        "state": "caution",
        "reason": "limited forward clearance",
        "priority": 2,
    }


# =============================================================================
# Confidence estimate
# =============================================================================

def _compute_confidence(occ: dict, yolo: dict, depth_bundle: dict) -> float:
    """Estimate how reliable the state estimate is."""
    depth_conf = depth_bundle["depth_valid_ratio"]
    grid_conf = occ["grid_coverage_ratio"]
    yolo_conf = yolo["depth_hit_rate"] if yolo["detection_count"] > 0 else 1.0

    confidence = depth_conf * 0.5 + grid_conf * 0.3 + yolo_conf * 0.2
    return round(float(confidence), 3)


# =============================================================================
# Main entry point — called once per frame
# =============================================================================

def analyse(
    occ_bundle: dict,
    yolo_bundle: dict,
    depth_bundle: dict,
) -> dict:
    """Produce a structured scene understanding from perception bundles."""

    # Filter low-confidence detections
    filtered_detections = [
        d for d in yolo_bundle["detections"]
        if d.get("confidence", 1.0) > 0.35
    ]
    yolo_bundle = {
        **yolo_bundle,
        "detections": filtered_detections,
        "detection_count": len(filtered_detections),
    }

    # Step 0 — Wall detection.
    wall = _detect_wall(occ_bundle, depth_bundle)

    # Step 1 — Space type with scoring + smoothing.
    space_type, raw_space_type, space_scores = _classify_space(occ_bundle, yolo_bundle, wall)

    # Step 2 — Clutter density.
    clutter_density, clutter_zone = _compute_clutter(occ_bundle, yolo_bundle)

    # Step 3 — Traversability.
    traversability = _compute_traversability(occ_bundle)

    # Step 4 — Dominant obstacle.
    dominant_obstacle = _find_dominant_obstacle(occ_bundle, yolo_bundle, wall)

    # Step 5 — Nearest person.
    persons = [
        d for d in yolo_bundle["detections"]
        if d["label"] == "person" and d["distance_m"] is not None
    ]
    nearest_person = None
    if persons:
        p = min(persons, key=lambda x: x["distance_m"])
        nearest_person = {
            "distance_m": p["distance_m"],
            "position_x_m": p["position_x_m"],
            "in_path": p["in_path"],
        }

    # Step 6 — Objects in path.
    objects_in_path = [
        {
            "label": d["label"],
            "distance_m": d["distance_m"],
            "position_x_m": d["position_x_m"],
        }
        for d in yolo_bundle["detections"]
        if d["is_blocking"]
    ]

    # Step 7 — Structural features.
    doorway_candidate = space_type == "doorway"
    corridor_detected = space_type == "corridor"
    centerline_offset = occ_bundle["centerline_offset_m"]

    # Step 8 — Guidance uses smoothed space_type, but urgent obstacle logic is immediate.
    guidance = _compute_guidance(
        occ_bundle,
        dominant_obstacle,
        traversability,
        centerline_offset,
        space_type,
    )

    # Step 9 — Confidence.
    confidence = _compute_confidence(occ_bundle, yolo_bundle, depth_bundle)

    return {
        # Scene classification
        "space_type": space_type,

        # Debug classification values
        "raw_space_type": raw_space_type,
        "space_scores": space_scores,

        # Clutter
        "clutter_density": clutter_density,
        "clutter_zone": clutter_zone,

        # Traversability
        "traversability": traversability,

        # Obstacles
        "dominant_obstacle": dominant_obstacle,
        "nearest_person": nearest_person,
        "objects_in_path": objects_in_path,

        # Wall detection
        "wall": wall,

        # Structural
        "doorway_candidate": doorway_candidate,
        "corridor_detected": corridor_detected,
        "centerline_offset_m": round(centerline_offset, 3),

        # Guidance
        "guidance": guidance,

        # Reliability
        "confidence": confidence,

        # Detection count — used by mode_controller to decide if YOLO is active
        "detection_count": yolo_bundle["detection_count"],

        # Debug HUD placeholder
        "debug_hud": None,
    }


# =============================================================================
# HUD image builder
# =============================================================================

def build_hud(
    state: dict,
    frame_left: np.ndarray,
    occ_bundle: dict,
    yolo_bundle: dict,
) -> np.ndarray:
    """Build a combined debug HUD image overlaying all state onto the camera frame."""
    import math

    hud = frame_left.copy()
    fh, fw = hud.shape[:2]

    COL_RED = (0, 0, 220)
    COL_YELLOW = (0, 200, 220)
    COL_GREEN = (0, 180, 0)
    COL_CYAN = (220, 220, 0)
    COL_WHITE = (255, 255, 255)
    COL_GREY = (120, 120, 120)
    COL_BLACK = (0, 0, 0)

    priority = state["guidance"]["priority"]
    GUIDE_COL = COL_RED if priority == 3 else (COL_YELLOW if priority == 2 else COL_GREEN)

    # YOLO detection boxes.
    for det in yolo_bundle["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        dist = det["distance_m"]
        label = det["label"]
        col = COL_RED if det["is_blocking"] else (COL_YELLOW if det["is_close"] else COL_GREEN)
        cv2.rectangle(hud, (x1, y1), (x2, y2), col, 2)
        dist_str = f"{dist:.1f}m" if dist is not None else "?"
        cv2.putText(
            hud,
            f"{label} {dist_str}",
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
            cv2.LINE_AA,
        )

    # Forward corridor lines.
    corridor_half = int(fw * 0.15)
    cx = fw // 2
    cv2.line(hud, (cx - corridor_half, 0), (cx - corridor_half, fh), COL_GREY, 1)
    cv2.line(hud, (cx + corridor_half, 0), (cx + corridor_half, fh), COL_GREY, 1)

    # Guidance arrow toward best opening.
    angle = occ_bundle["best_opening_angle_deg"]
    if angle is not None:
        arrow_len = fh // 3
        tip_x = int(cx + math.tan(math.radians(angle)) * arrow_len)
        tip_y = int(fh * 0.55)
        base_y = fh - 20
        cv2.arrowedLine(
            hud,
            (cx, base_y),
            (tip_x, tip_y),
            GUIDE_COL,
            3,
            tipLength=0.2,
            line_type=cv2.LINE_AA,
        )

    # Occupancy grid inset — bottom right.
    grid = occ_bundle["grid"]
    grid_h, grid_w = grid.shape
    inset_scale = 2
    inset_w = grid_w * inset_scale
    inset_h = grid_h * inset_scale
    inset_x = fw - inset_w - 10
    inset_y = fh - inset_h - 10

    grid_rgb = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    grid_rgb[grid == 0] = (0, 30, 0)
    grid_rgb[grid == 255] = (0, 0, 200)
    cv2.circle(grid_rgb, (grid_w // 2, 0), 3, (255, 255, 0), -1)
    grid_large = cv2.resize(grid_rgb, (inset_w, inset_h), interpolation=cv2.INTER_NEAREST)
    grid_large = cv2.flip(grid_large, 0)

    cv2.rectangle(
        hud,
        (inset_x - 2, inset_y - 2),
        (inset_x + inset_w + 2, inset_y + inset_h + 2),
        COL_GREY,
        1,
    )
    hud[inset_y:inset_y + inset_h, inset_x:inset_x + inset_w] = grid_large
    cv2.putText(
        hud,
        "occupancy",
        (inset_x, inset_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        COL_GREY,
        1,
    )

    # Text panel — top left.
    g = state["guidance"]
    trav = state["traversability"]
    dom = state["dominant_obstacle"]

    raw_space = state.get("raw_space_type", "?")
    scores = state.get("space_scores", {})
    cor_score = scores.get("corridor", 0.0)
    door_score = scores.get("doorway", 0.0)

    lines = [
        (f"SPACE   : {state['space_type']}  raw={raw_space}", COL_CYAN),
        (f"SCORES  : cor={cor_score:.2f} door={door_score:.2f}", COL_WHITE),
        (f"CLUTTER : {state['clutter_density']:.2f} ({state['clutter_zone']})", COL_WHITE),
        (f"TRAV    : fwd={trav['ahead']:.2f}  L={trav['left']:.2f}  R={trav['right']:.2f}", COL_WHITE),
        (f"CONF    : {state['confidence']:.2f}", COL_WHITE),
        ("", COL_WHITE),
        (f"GUIDANCE: {g['state'].upper()}  [P{g['priority']}]", GUIDE_COL),
        (f"REASON  : {g['reason']}", COL_WHITE),
    ]

    if dom is not None:
        dist_str = f"{dom['distance_m']:.2f}m" if dom["distance_m"] else "?"
        lines.append((f"OBSTACLE: {dom['label']} @ {dist_str}", COL_YELLOW))

    line_h = 20
    pad = 8
    box_w = 440
    box_h = len(lines) * line_h + pad * 2
    overlay = hud.copy()
    cv2.rectangle(overlay, (6, 6), (6 + box_w, 6 + box_h), COL_BLACK, -1)
    cv2.addWeighted(overlay, 0.55, hud, 0.45, 0, hud)

    for i, (text, col) in enumerate(lines):
        if not text:
            continue
        cv2.putText(
            hud,
            text,
            (pad + 6, pad + 6 + (i + 1) * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            col,
            1,
            cv2.LINE_AA,
        )

    return hud


# =============================================================================
# Standalone test — run directly: python environment_state.py
# =============================================================================

if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from occupancy_map import build_occupancy_map
        from yolo_detector import detect as yolo_detect
    except ImportError:
        from modules.depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from modules.occupancy_map import build_occupancy_map
        from modules.yolo_detector import detect as yolo_detect

    CALIB_PATH = "calibration/stereo_calibration.npz"
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 480

    if not os.path.exists(CALIB_PATH):
        print(f"ERROR: calibration file not found at {CALIB_PATH}")
        raise SystemExit(1)

    cal = np.load(CALIB_PATH)
    M1, d1 = cal["M1"], cal["d1"]
    M2, d2 = cal["M2"], cal["d2"]
    R1, R2 = cal["R1"], cal["R2"]
    P1, P2 = cal["P1"], cal["P2"]
    Q = cal["Q"]

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for _ in range(20):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: could not read frame")
        raise SystemExit(1)

    h, w = frame.shape[:2]
    eye_w = w // 2
    eye_h = h

    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)
    rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
    rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)

    print("Running perception pipeline...")
    stereo = build_stereo_matcher()
    depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)
    occ_bundle = build_occupancy_map(depth_bundle)
    yolo_bundle = yolo_detect(rectL, depth_bundle, Q)

    print("Analysing scene...")
    state = analyse(occ_bundle, yolo_bundle, depth_bundle)

    print("\n" + "=" * 50)
    print("--- environment_state bundle ---")
    print("=" * 50)
    print(f"  space_type         : {state['space_type']}  raw={state['raw_space_type']}")
    print(f"  space_scores       : {state['space_scores']}")
    print(f"  clutter_density    : {state['clutter_density']}  ({state['clutter_zone']})")
    print(
        f"  traversability     : ahead={state['traversability']['ahead']}  "
        f"left={state['traversability']['left']}  "
        f"right={state['traversability']['right']}"
    )
    print(f"  centerline_offset  : {state['centerline_offset_m']}m")
    print(f"  corridor_detected  : {state['corridor_detected']}")
    print(f"  doorway_candidate  : {state['doorway_candidate']}")
    print(f"  confidence         : {state['confidence']}")

    print("\n  dominant_obstacle  :", end=" ")
    if state["dominant_obstacle"]:
        d = state["dominant_obstacle"]
        print(
            f"{d['label']} at {d['distance_m']}m  "
            f"x={d['position_x_m']}m  blocks={d['blocks_path']}  "
            f"source={d['source']}"
        )
    else:
        print("none")

    print("\n  nearest_person     :", end=" ")
    if state["nearest_person"]:
        p = state["nearest_person"]
        print(f"{p['distance_m']}m  x={p['position_x_m']}m  in_path={p['in_path']}")
    else:
        print("none")

    print(f"\n  objects_in_path    : {len(state['objects_in_path'])} blocking")
    for obj in state["objects_in_path"]:
        print(f"    {obj['label']:20s} {obj['distance_m']}m  x={obj['position_x_m']}m")

    print("\n  guidance:")
    g = state["guidance"]
    print(f"    state    : {g['state']}")
    print(f"    reason   : {g['reason']}")
    print(f"    priority : {g['priority']}")
    print("=" * 50)

    os.makedirs("debug_output", exist_ok=True)
    hud_img = build_hud(state, rectL, occ_bundle, yolo_bundle)
    cv2.imwrite("debug_output/hud.png", hud_img)
    print("\nHUD saved to debug_output/hud.png")
    print("Pull to Mac with:")
    print("  scp 'zanewolf@192.168.50.103:~/6thSense/debug_output/hud.png' ~/Desktop/")

