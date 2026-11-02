"""
object_tracker.py
=================
Layer 1.3 — Perception  (Temporal Stabilization)

What this file does
-------------------
Takes raw per-frame YOLO detections and builds stable, confident object
tracks over time. A detection that appears in one frame and disappears
is ignored. A detection that appears consistently over multiple frames
gets confirmed and reported with a stable distance and position.

Why this exists
---------------
Raw YOLO output is noisy. The same chair might be detected in 8 out of
10 frames, at slightly different distances and positions each time.
Without stabilization, the system would say "chair at 2.1m" then
"chair at 2.3m" then "chair at 2.0m" — confusing and unreliable.

This layer builds confidence over time before committing to a report.
Only confirmed tracks feed into environment_state.py.

Velocity / motion note
----------------------
Motion tracking (is_moving, velocity) is intentionally NOT implemented.
Without an IMU, camera movement looks identical to object movement.
Every time the user turns their head, all objects would appear to move.
Motion detection will be added once an IMU is available to compensate
for camera ego-motion.

Track lifecycle
---------------
new detection → track created (confidence=0.2)
    ↓ seen again next frame
matched → confidence += CONFIDENCE_INCREMENT (0.15)
    ↓ confidence > CONFIRMATION_THRESHOLD (0.6)
confirmed → eligible to be reported to environment_state.py
    ↓ not seen for a frame
missed → confidence -= CONFIDENCE_DECAY (0.12) per missed frame
    ↓ confidence < REMOVAL_THRESHOLD (0.1)
removed → track deleted

Output
------
confirmed_tracks : list of confirmed track dicts, sorted by distance
all_tracks       : list of all tracks including unconfirmed
track_count      : int
confirmed_count  : int

Each track dict:
  id             : int   — unique track ID
  label          : str   — most frequently seen YOLO label
  distance_m     : float — smoothed distance (exponential moving average)
  position_x_m   : float — smoothed lateral position
  confidence     : float — 0.0–1.0
  age_frames     : int   — total frames this track has existed
  hits           : int   — total frames it was matched to a detection
  is_confirmed   : bool  — True when confidence > CONFIRMATION_THRESHOLD
  last_seen      : int   — frames since last matched (0 = seen this frame)
  in_path        : bool  — True if within forward corridor
  is_wall        : bool  — True if this track was identified as a wall

Stack position:
    depth_map.py
    occupancy_map.py  ─┐
    yolo_detector.py  ─┴→  >>> object_tracker.py <<<
                                environment_state.py
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Optional

import numpy as np

# =============================================================================
# Tunable constants
# =============================================================================

# Confidence gain. Strong detections build tracks. Weak detections keep
# already-known tracks alive without creating spammy new objects.
CONFIDENCE_INCREMENT = 0.22
LOW_CONFIDENCE_INCREMENT = 0.08

# Confidence loss. Confirmed tracks decay slower so one bad YOLO frame does
# not make a real object vanish.
CONFIDENCE_DECAY = 0.10
CONFIRMED_CONFIDENCE_DECAY = 0.055

# Confirmation hysteresis. A track confirms above CONFIRMATION_THRESHOLD but
# stays confirmed until it falls below UNCONFIRM_THRESHOLD.
CONFIRMATION_THRESHOLD = 0.50
UNCONFIRM_THRESHOLD = 0.32

# Track is removed below this confidence.
REMOVAL_THRESHOLD = 0.06

# Matching gates in metres.
MATCH_DISTANCE_M = 0.75
RELAXED_MATCH_DISTANCE_M = 1.05

# YOLO confidence handling. Low-confidence detections can maintain existing
# tracks, but only high-confidence detections can create new tracks.
HIGH_DETECTION_CONF = 0.45
LOW_DETECTION_CONF = 0.20

# EMA smoothing. Weak detections update more gently.
SMOOTH_ALPHA = 0.35
LOW_CONF_SMOOTH_ALPHA = 0.18

# Maximum frames a track can go unseen before being removed.
MAX_MISSED_FRAMES = 10
MAX_CONFIRMED_MISSED_FRAMES = 18

# Minimum hits before a track can be confirmed.
MIN_HITS_TO_CONFIRM = 2
MIN_STRONG_HITS_TO_CONFIRM = 1

# Labels that may flicker between each other while describing the same
# physical obstacle. This preserves object identity while label voting decides
# the most likely name.
LABEL_GROUPS = [
    {"chair", "couch", "bench", "dining table"},
    {"tv", "laptop", "monitor", "keyboard", "cell phone"},
    {"person"},
    {"door", "refrigerator", "cabinet"},
]


# =============================================================================
# ObjectTracker class
# =============================================================================

class ObjectTracker:
    """
    Maintains a set of object tracks across frames.

    Usage
    -----
    tracker = ObjectTracker()

    # In your main loop:
    track_bundle = tracker.update(yolo_bundle, wall_bundle)
    confirmed    = track_bundle["confirmed_tracks"]
    """

    def __init__(self):
        self._tracks    = {}     # id → track dict
        self._next_id   = 0
        self._frame_num = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, yolo_bundle: dict, wall: Optional[dict] = None) -> dict:
        """
        Update all tracks with new detections from this frame.

        Parameters
        ----------
        yolo_bundle : output of yolo_detector.detect()
        wall        : wall detection dict from environment_state._detect_wall()
                      Optional — if provided, wall gets its own stable track

        Returns
        -------
        dict with confirmed_tracks, all_tracks, track_count, confirmed_count
        """
        self._frame_num += 1
        detections = yolo_bundle["detections"]
        detections = [d for d in detections if self._det_confidence(d) >= LOW_DETECTION_CONF]

        # ------------------------------------------------------------------
        # Step 1 — Match detections to existing tracks
        # ------------------------------------------------------------------
        matched_track_ids  = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detections):
            if det["distance_m"] is None:
                continue   # can't match without distance

            best_track_id   = None
            best_dist       = MATCH_DISTANCE_M

            for tid, track in self._tracks.items():
                if tid in matched_track_ids:
                    continue
                if track.get("is_wall"):
                    continue
                if not self._labels_compatible(track["label"], det["label"]):
                    continue

                # Euclidean distance between detection and track in world space.
                dx = det["position_x_m"] - track["position_x_m"] if det["position_x_m"] is not None else 0
                dz = det["distance_m"]   - track["distance_m"]
                d  = float(np.sqrt(dx**2 + dz**2))

                # Exact labels get the normal gate. Related labels get a relaxed
                # gate, but weak detections cannot jump across the scene.
                gate = MATCH_DISTANCE_M if track["label"] == det["label"] else RELAXED_MATCH_DISTANCE_M
                if self._det_confidence(det) < HIGH_DETECTION_CONF:
                    gate = min(gate, MATCH_DISTANCE_M)

                if d < gate and d < best_dist:
                    best_dist     = d
                    best_track_id = tid

            if best_track_id is not None:
                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)
                self._update_track(best_track_id, det)
            # else: unmatched detection — will create new track below

        # ------------------------------------------------------------------
        # Step 2 — Create new tracks for unmatched detections
        # ------------------------------------------------------------------
        for det_idx, det in enumerate(detections):
            if det_idx in matched_det_indices:
                continue
            if det["distance_m"] is None:
                continue
            # Low-confidence detections can maintain existing tracks, but
            # should not create brand-new tracks by themselves.
            if self._det_confidence(det) < HIGH_DETECTION_CONF:
                continue
            self._create_track(det)

        # ------------------------------------------------------------------
        # Step 3 — Decay unmatched tracks
        # ------------------------------------------------------------------
        for tid in list(self._tracks.keys()):
            if tid not in matched_track_ids:
                self._miss_track(tid)

        # ------------------------------------------------------------------
        # Step 4 — Add wall as a special track if detected
        # ------------------------------------------------------------------
        if wall is not None and wall["present"] and wall["distance_m"] is not None:
            self._update_wall_track(wall)
        else:
            self._decay_wall_track()

        # ------------------------------------------------------------------
        # Step 5 — Remove dead tracks
        # ------------------------------------------------------------------
        dead = []
        for tid, t in self._tracks.items():
            max_missed = MAX_CONFIRMED_MISSED_FRAMES if t.get("is_confirmed") else MAX_MISSED_FRAMES
            if t["confidence"] < REMOVAL_THRESHOLD or t["last_seen"] > max_missed:
                dead.append(tid)
        for tid in dead:
            del self._tracks[tid]

        # ------------------------------------------------------------------
        # Step 6 — Build output
        # ------------------------------------------------------------------
        all_tracks = sorted(
            self._tracks.values(),
            key=lambda t: t["distance_m"]
        )
        confirmed_tracks = [t for t in all_tracks if t["is_confirmed"]]

        return {
            "confirmed_tracks": confirmed_tracks,
            "all_tracks":       all_tracks,
            "track_count":      len(all_tracks),
            "confirmed_count":  len(confirmed_tracks),
            "frame_num":        self._frame_num,
        }

    def reset(self) -> None:
        """Clear all tracks. Call when mode changes or scene resets."""
        self._tracks  = {}
        self._frame_num = 0

    def _det_confidence(self, det: dict) -> float:
        """Return YOLO confidence, defaulting to 1.0 for older detector bundles."""
        conf = det.get("confidence", 1.0)
        if conf is None:
            return 1.0
        try:
            return float(conf)
        except Exception:
            return 1.0

    def _labels_compatible(self, existing_label: str, new_label: str) -> bool:
        """
        Decide whether two labels can describe the same physical obstacle.

        Exact labels always match. Related labels can match because YOLO may
        flicker between them frame-to-frame. Unrelated labels do not match,
        which prevents a person from becoming a chair.
        """
        if existing_label == new_label:
            return True

        for group in LABEL_GROUPS:
            if existing_label in group and new_label in group:
                return True

        return False

    def _track_match_quality(self, track: dict, det: dict) -> str:
        """Classify a match as strong, related, or weak."""
        det_conf = self._det_confidence(det)
        if det_conf >= HIGH_DETECTION_CONF and track["label"] == det["label"]:
            return "strong"
        if det_conf >= HIGH_DETECTION_CONF:
            return "related"
        return "weak"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, det: dict) -> None:
        """Create a new track from a strong detection."""
        tid = self._next_id
        self._next_id += 1

        det_conf = self._det_confidence(det)

        self._tracks[tid] = {
            "id":           tid,
            "label":        det["label"],
            "label_counts": Counter({det["label"]: 1}),
            "distance_m":   det["distance_m"],
            "position_x_m": det["position_x_m"] if det["position_x_m"] is not None else 0.0,
            "confidence":   CONFIDENCE_INCREMENT,
            "age_frames":   1,
            "hits":         1,
            "strong_hits":  1 if det_conf >= HIGH_DETECTION_CONF else 0,
            "weak_hits":    0,
            "is_confirmed": False,
            "last_seen":    0,
            "in_path":      det["in_path"],
            "is_wall":      False,
            "bbox":         det["bbox"],
            "last_raw_confidence": det_conf,
            "source":       "yolo",
        }

    def _update_track(self, tid: int, det: dict) -> None:
        """Update an existing track with a new matching detection."""
        t = self._tracks[tid]
        match_quality = self._track_match_quality(t, det)
        det_conf = self._det_confidence(det)

        # Strong and related matches update label memory. Weak matches preserve
        # the track but do not get to rename the object.
        if match_quality in ("strong", "related"):
            vote_weight = 2 if match_quality == "strong" else 1
            t["label_counts"][det["label"]] += vote_weight
            t["label"] = t["label_counts"].most_common(1)[0][0]

        # Smooth distance and position. Weak detections move the track less so
        # low-confidence jitter does not make the object jump around.
        alpha = SMOOTH_ALPHA if match_quality != "weak" else LOW_CONF_SMOOTH_ALPHA
        t["distance_m"] = (alpha * det["distance_m"] +
                            (1 - alpha) * t["distance_m"])

        if det["position_x_m"] is not None:
            t["position_x_m"] = (alpha * det["position_x_m"] +
                                  (1 - alpha) * t["position_x_m"])

        # Weak detections keep the object alive but add less confidence.
        inc = CONFIDENCE_INCREMENT if match_quality != "weak" else LOW_CONFIDENCE_INCREMENT
        t["confidence"] = min(1.0, t["confidence"] + inc)

        t["age_frames"] += 1
        t["hits"] += 1
        if match_quality == "weak":
            t["weak_hits"] = t.get("weak_hits", 0) + 1
        else:
            t["strong_hits"] = t.get("strong_hits", 0) + 1

        t["last_seen"] = 0

        # Conservative path memory: a weak off-path frame should not instantly
        # erase an obstacle that was recently in the path.
        if det["in_path"] or match_quality != "weak":
            t["in_path"] = det["in_path"]

        t["bbox"] = det["bbox"]
        t["last_raw_confidence"] = det_conf
        t["source"] = t.get("source", "yolo")

        # Confirmation with hysteresis. The lower unconfirm threshold is handled
        # in _miss_track, so confirmed tracks do not flicker off immediately.
        if (t["confidence"] >= CONFIRMATION_THRESHOLD and
                t["hits"] >= MIN_HITS_TO_CONFIRM and
                t.get("strong_hits", 0) >= MIN_STRONG_HITS_TO_CONFIRM):
            t["is_confirmed"] = True

    def _miss_track(self, tid: int) -> None:
        """Decay a track that was not matched this frame."""
        t = self._tracks[tid]
        decay = CONFIRMED_CONFIDENCE_DECAY if t.get("is_confirmed") else CONFIDENCE_DECAY

        t["confidence"] = max(0.0, t["confidence"] - decay)
        t["last_seen"] += 1
        t["age_frames"] += 1

        # Hysteresis: once a track is confirmed, it stays confirmed until it
        # falls below the lower unconfirm threshold.
        if t["confidence"] < UNCONFIRM_THRESHOLD:
            t["is_confirmed"] = False

    def _update_wall_track(self, wall: dict) -> None:
        """
        Update or create the special wall track.

        Wall uses track ID -1 so it is always distinguishable from YOLO objects.
        It gets the same lifecycle fields as YOLO tracks so the dashboard and
        downstream logic can reason about it consistently.
        """
        wall_tid = -1

        if wall_tid not in self._tracks:
            self._tracks[wall_tid] = {
                "id":           wall_tid,
                "label":        "wall",
                "label_counts": Counter({"wall": 1}),
                "distance_m":   wall["distance_m"],
                "position_x_m": 0.0,
                "confidence":   CONFIDENCE_INCREMENT,
                "age_frames":   1,
                "hits":         1,
                "strong_hits":  1,
                "weak_hits":    0,
                "is_confirmed": False,
                "last_seen":    0,
                "in_path":      wall["direction"] == "ahead",
                "is_wall":      True,
                "coverage":     wall["coverage"],
                "direction":    wall["direction"],
                "bbox":         None,
                "last_raw_confidence": 1.0,
                "source":       "wall_detection",
            }
        else:
            t = self._tracks[wall_tid]

            # Smooth wall distance so one noisy depth frame does not make the
            # warning jump around.
            t["distance_m"] = (SMOOTH_ALPHA * wall["distance_m"] +
                                (1 - SMOOTH_ALPHA) * t["distance_m"])
            t["coverage"] = wall["coverage"]
            t["direction"] = wall["direction"]
            t["in_path"] = wall["direction"] == "ahead"
            t["confidence"] = min(1.0, t["confidence"] + CONFIDENCE_INCREMENT)
            t["age_frames"] += 1
            t["hits"] += 1
            t["strong_hits"] = t.get("strong_hits", 0) + 1
            t["weak_hits"] = t.get("weak_hits", 0)
            t["last_raw_confidence"] = 1.0
            t["source"] = "wall_detection"
            t["last_seen"] = 0

            if (t["confidence"] >= CONFIRMATION_THRESHOLD and
                    t["hits"] >= MIN_HITS_TO_CONFIRM and
                    t.get("strong_hits", 0) >= MIN_STRONG_HITS_TO_CONFIRM):
                t["is_confirmed"] = True

    def _decay_wall_track(self) -> None:
        """Decay wall track when no wall is detected this frame."""
        wall_tid = -1
        if wall_tid in self._tracks:
            self._miss_track(wall_tid)
            if self._tracks[wall_tid]["confidence"] < REMOVAL_THRESHOLD:
                del self._tracks[wall_tid]


# =============================================================================
# Standalone test — run directly: python object_tracker.py
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    import cv2

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    try:
        from depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from occupancy_map import build_occupancy_map
        from yolo_detector import detect as yolo_detect
        from environment_state import _detect_wall
    except ImportError:
        from modules.depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from modules.occupancy_map import build_occupancy_map
        from modules.yolo_detector import detect as yolo_detect
        from modules.environment_state import _detect_wall

    CALIB_PATH   = "calibration/stereo_calibration.npz"
    CAMERA_INDEX = 0
    FRAME_WIDTH  = 1280
    FRAME_HEIGHT = 480

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
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    for _ in range(20):
        cap.read()

    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1,
                   (FRAME_WIDTH//2, FRAME_HEIGHT), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2,
                   (FRAME_WIDTH//2, FRAME_HEIGHT), cv2.CV_32FC1)

    stereo  = build_stereo_matcher()
    tracker = ObjectTracker()

    print("Running tracker for 60 frames — watch confidence build up...\n")
    print(f"{'ID':>3} {'label':15} {'dist':>6} {'conf':>6} {'hits':>5} {'confirmed':>10}")
    print("-" * 55)

    for frame_idx in range(60):
        ret, frame = cap.read()
        if not ret:
            continue

        eye_w = frame.shape[1] // 2
        eye_h = frame.shape[0]

        rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
        rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)

        depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)
        occ_bundle   = build_occupancy_map(depth_bundle)
        yolo_bundle  = yolo_detect(rectL, depth_bundle, Q)
        wall         = _detect_wall(occ_bundle, depth_bundle)

        track_bundle = tracker.update(yolo_bundle, wall)

        # Print every 5 frames
        if frame_idx % 5 == 0:
            print(f"\n--- frame {frame_idx} ---")
            for t in track_bundle["all_tracks"]:
                confirmed_str = "✓ CONFIRMED" if t["is_confirmed"] else ""
                print(f"{t['id']:>3} {t['label']:15} "
                      f"{t['distance_m']:>6.2f}m "
                      f"{t['confidence']:>6.2f} "
                      f"{t['hits']:>5} "
                      f"{confirmed_str}")

    cap.release()
    print("\nFinal confirmed tracks:")
    for t in track_bundle["confirmed_tracks"]:
        print(f"  {t['label']:15} {t['distance_m']:.2f}m  "
              f"conf={t['confidence']:.2f}  hits={t['hits']}")
