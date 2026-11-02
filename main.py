"""
main.py
=======
Entry point — Full Pipeline

Wires every layer together in the correct order and runs the main loop.

Layer stack (data flows top to bottom each frame)
--------------------------------------------------

  LAYER 1.1   depth_map.py
              stereo frames → point cloud + depth stats

  LAYER 1.2a  occupancy_map.py          ← parallel branches
  LAYER 1.2b  yolo_detector.py          ← both consume depth bundle

  LAYER 1.3   object_tracker.py
              raw YOLO detections → stable confirmed tracks

  LAYER 2.1   environment_state.py
              grid + tracks → scene understanding + guidance state

  LAYER 3/4   mode_controller.py  (ModeGate subclass)
              decides WHETHER to output this frame
              (throttling, stability, mode gating)
              printing is suppressed — decision_layer owns output

  LAYER 5     query_layer.py
              background mic/text listener → on-demand answers
              active only in query mode

  LAYER 6     decision_layer.py
              decides WHICH channel + WHAT to send
              dispatches to haptic / spatial_audio / speech / tone / silent

How mode_controller and decision_layer connect
----------------------------------------------
mode_controller.process() returns:
    str  → output is warranted this frame (throttle + stability passed)
    None → too soon, or nothing changed, or query mode — stay silent

When it returns a string, main.py calls decision_layer.route() +
decision_layer.dispatch() with the full state bundle.

mode_controller's internal _print() is suppressed via ModeGate subclass
so it cannot output anything itself. It is used purely as a gate.

Controls
--------
e   — exploration mode
g   — guidance mode
q   — query mode
ESC — exit
"""

from __future__ import annotations

import os
import sys
import time
import tty
import termios
import copy

import cv2
import numpy as np

# =============================================================================
# Path setup — works whether modules are flat or in a modules/ subfolder
# =============================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from depth_map        import build_stereo_matcher, process_frame as depth_process_frame
    from occupancy_map    import build_occupancy_map
    from yolo_detector    import detect as yolo_detect
    from object_tracker   import ObjectTracker
    from environment_state import analyse, build_hud, _detect_wall
    from mode_controller  import ModeController, MODE_EXPLORATION, MODE_GUIDANCE, MODE_QUERY
    from query_mode      import QueryLayer
    from decision_layer   import OutputRouter, route, dispatch
    from temporal_state import TemporalState
except ImportError:
    from modules.depth_map         import build_stereo_matcher, process_frame as depth_process_frame
    from modules.occupancy_map     import build_occupancy_map
    from modules.yolo_detector     import detect as yolo_detect
    from modules.object_tracker    import ObjectTracker
    from modules.environment_state import analyse, build_hud, _detect_wall
    from modules.mode_controller   import ModeController, MODE_EXPLORATION, MODE_GUIDANCE, MODE_QUERY
    from modules.query_mode       import QueryLayer
    from modules.decision_layer    import OutputRouter, route, dispatch

# Optional dashboard support.
# main.py is prepared to use dashboard_server.py if it exists, but it will
# still run normally before that file is created.
try:
    from dashboard_server import DashboardServer
except ImportError:
    try:
        from modules.dashboard_server import DashboardServer
    except ImportError:
        DashboardServer = None

# =============================================================================
# Configuration
# =============================================================================

CALIB_PATH    = "calibration/stereo_calibration.npz"
CAMERA_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 480
CAMERA_FPS    = 30

# Save a debug HUD image every N frames (set 0 to disable)
SAVE_HUD_EVERY = 30
DEBUG_DIR      = "debug_output"

# Exploration-mode anti-spam settings.
# Explore mode should be mostly quiet. Urgent warnings still bypass cooldown.
EXPLORE_MIN_GAP_SEC       = 4.0
EXPLORE_REPEAT_GAP_SEC    = 10.0
EXPLORE_SUMMARY_GAP_SEC   = 18.0
EXPLORE_CLOSE_OBJECT_M    = 1.6

# Dashboard support. This stays safe even before dashboard_server.py exists.
ENABLE_DASHBOARD = True

# =============================================================================
# ANSI colours
# =============================================================================

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


# =============================================================================
# ModeGate
# =============================================================================
# Subclass of ModeController that suppresses all internal printing.
#
# Why subclass instead of modify mode_controller.py?
#   mode_controller.py is Layer 3/4 and owns throttle + stability logic.
#   We want that logic intact. We just don't want it printing — that job
#   belongs to decision_layer (Layer 6).
#
#   Subclassing lets us override only _print() so decision_layer owns
#   all output without touching the throttle or stability code.
#
# What ModeGate still does (from ModeController):
#   - check_keys()   : keyboard mode switching
#   - process()      : returns non-None when output is warranted
#   - throttle logic : P1 / P2 intervals, alert timing
#   - stability gate : message must hold for STABILITY_SEC
#   - _print_mode_banner() : prints mode switches (these ARE system events,
#                            not navigation output, so we keep them)
# =============================================================================

class ModeGate(ModeController):
    """
    ModeController with output suppressed.
    Returns the would-be output string from process() as a gate signal,
    but does not print it — decision_layer handles all output.
    """

    def _print(self, line: str) -> None:
        # Intentionally empty — decision_layer owns all navigation output
        pass


# =============================================================================
# _tracks_to_yolo_bundle
# =============================================================================
# Duplicated from mode_controller.py so main.py has no circular dependency.
# Converts confirmed tracker output into a yolo-bundle-compatible dict.

def _tracks_to_yolo_bundle(track_bundle: dict, original_yolo: dict) -> dict:
    confirmed = [
        t for t in track_bundle["confirmed_tracks"]
        if not t["is_wall"]
    ]
    detections = []
    for t in confirmed:
        detections.append({
            "label":        t["label"],
            "confidence":   t["confidence"],
            "distance_m":   round(t["distance_m"], 3),
            "position_x_m": round(t["position_x_m"], 3),
            "bbox":         t["bbox"] if t["bbox"] else [0, 0, 0, 0],
            "centre_px":    [0, 0],
            "in_path":      t["in_path"],
            "is_close":     t["distance_m"] < 1.5,
            "is_blocking":  t["in_path"] and t["distance_m"] < 1.5,
            "is_relevant":  True,
        })
    return {
        "detections":        detections,
        "detection_count":   len(detections),
        "inference_time_ms": original_yolo["inference_time_ms"],
        "depth_hit_rate":    original_yolo["depth_hit_rate"],
        "debug_frame":       original_yolo["debug_frame"],
    }


# =============================================================================
# Exploration output gate
# =============================================================================
# Final safety gate for exploration mode.
# This prevents repeated low-value speech while preserving urgent warnings.

_last_explore_output_time = 0.0
_last_explore_message = None
_last_explore_signature = None
_last_explore_summary_time = 0.0


def _object_signature(obj: dict | None) -> tuple | None:
    """Small stable signature for comparing dominant obstacles."""
    if obj is None:
        return None

    label = obj.get("label", "unknown")
    blocks = bool(obj.get("blocks_path", False))
    distance = obj.get("distance_m")
    x_pos = obj.get("position_x_m")

    # Round distance/position so tiny depth jitter does not count as a new object.
    dist_bucket = None if distance is None else round(float(distance) * 2.0) / 2.0
    x_bucket = None if x_pos is None else round(float(x_pos) * 2.0) / 2.0

    return (label, dist_bucket, x_bucket, blocks)


def _explore_message_signature(state: dict) -> tuple:
    """Signature used to avoid repeating the same explore-mode message."""
    guidance = state.get("guidance", {})
    dominant = state.get("dominant_obstacle")
    return (
        guidance.get("state"),
        guidance.get("priority"),
        guidance.get("reason"),
        state.get("space_type"),
        _object_signature(dominant),
    )


def should_dispatch_explore(state: dict, gate_text: str | None = None) -> bool:
    """
    Decide whether exploration mode should be allowed to speak/output.

    Exploration mode is for awareness, so it should not constantly talk.
    It should allow urgent hazards immediately, allow meaningful close/blocking
    obstacles with cooldown, and suppress repeated low-value updates like
    "path clear" or jittery space-type changes.
    """
    global _last_explore_output_time
    global _last_explore_message
    global _last_explore_signature
    global _last_explore_summary_time

    now = time.time()
    guidance = state.get("guidance", {})
    priority = int(guidance.get("priority", 1))
    guidance_state = guidance.get("state", "")
    reason = guidance.get("reason", "")
    dominant = state.get("dominant_obstacle")
    signature = _explore_message_signature(state)

    # Urgent warnings bypass cooldown.
    if priority >= 3:
        _last_explore_output_time = now
        _last_explore_message = reason
        _last_explore_signature = signature
        return True

    # Detect meaningful change to allow immediate output (event-driven)
    prev_sig = _last_explore_signature
    is_new_event = signature != prev_sig

    # Determine if this is a "major" change (new object, closer distance, or higher priority)
    major_change = False
    if is_new_event:
        prev_obj = prev_sig[-1] if prev_sig else None
        curr_obj = signature[-1]

        # New or different object
        if prev_obj != curr_obj:
            major_change = True

        # Priority increased
        # (compare current vs previous if available)
        try:
            prev_priority = int(prev_sig[1]) if prev_sig else None
        except Exception:
            prev_priority = None
        if prev_priority is not None and priority > prev_priority:
            major_change = True

    # Do not speak routine clear-path messages in exploration mode.
    if guidance_state == "clear":
        return False

    # Low-priority informational updates are only allowed as rare summaries.
    if priority <= 1:
        if now - _last_explore_summary_time >= EXPLORE_SUMMARY_GAP_SEC:
            _last_explore_summary_time = now
            _last_explore_output_time = now
            _last_explore_message = reason
            _last_explore_signature = signature
            return True
        return False

    # For caution messages, require a meaningful close/blocking obstacle or wall.
    meaningful_obstacle = False
    if dominant is not None:
        distance = dominant.get("distance_m")
        blocks_path = bool(dominant.get("blocks_path", False))
        source = dominant.get("source", "")
        is_wall = source == "wall_detection" or "wall" in str(dominant.get("label", ""))

        if is_wall:
            meaningful_obstacle = True
        elif blocks_path and distance is not None and float(distance) <= EXPLORE_CLOSE_OBJECT_M:
            meaningful_obstacle = True

    if not meaningful_obstacle:
        return False

    # If major change → bypass cooldown for responsiveness
    if major_change:
        _last_explore_output_time = now
        _last_explore_message = reason
        _last_explore_signature = signature
        return True

    # Otherwise apply normal cooldown
    if now - _last_explore_output_time < EXPLORE_MIN_GAP_SEC:
        return False

    # Do not repeat the same message/signature too often.
    if signature == _last_explore_signature and now - _last_explore_output_time < EXPLORE_REPEAT_GAP_SEC:
        return False

    _last_explore_output_time = now
    _last_explore_message = reason
    _last_explore_signature = signature
    return True


def _attach_debug_data(state: dict, raw_yolo: dict, confirmed_yolo: dict, track_bundle: dict) -> dict:
    """
    Add debug-only data to the state bundle.

    These extra keys are useful for the dashboard and should not break normal
    consumers because all original state keys remain unchanged.
    """
    state["raw_detections"] = copy.deepcopy(raw_yolo.get("detections", []))
    state["confirmed_detections"] = copy.deepcopy(confirmed_yolo.get("detections", []))
    state["confirmed_track_count"] = len(track_bundle.get("confirmed_tracks", []))
    state["raw_detection_count"] = raw_yolo.get("detection_count", len(state["raw_detections"]))
    return state


# =============================================================================
# Camera + calibration setup
# =============================================================================

def _setup_camera():
    """Open camera, validate, return cap + per-eye dimensions."""
    if not os.path.exists(CALIB_PATH):
        print(f"{RED}ERROR: calibration file not found at {CALIB_PATH}{RESET}")
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
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    # Flush buffer
    for _ in range(20):
        cap.read()

    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        print(f"{RED}ERROR: camera not streaming{RESET}")
        cap.release()
        raise SystemExit(1)

    h, w  = test_frame.shape[:2]
    eye_w = w // 2
    eye_h = h

    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)

    return cap, Q, mapL1, mapL2, mapR1, mapR2, eye_w, eye_h


# =============================================================================
# Main loop
# =============================================================================

def main():
    print(f"\n{BOLD}{CYAN}6thSense Navigation System{RESET}")
    print(f"{GREY}Initialising...{RESET}\n")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    cap, Q, mapL1, mapL2, mapR1, mapR2, eye_w, eye_h = _setup_camera()

    stereo  = build_stereo_matcher()
    tracker = ObjectTracker()
    temporal = TemporalState()

    # Layer 3/4 — mode gate (throttle + stability, no output)
    controller = ModeGate(initial_mode=MODE_EXPLORATION, tracker=tracker)

    # Layer 5 — query interface (background thread)
    query = QueryLayer(force_text=False)
    query.start()

    # Layer 6 — output router
    router = OutputRouter()

    # Optional live dashboard. This is inactive until dashboard_server.py exists.
    dashboard = None
    if ENABLE_DASHBOARD and DashboardServer is not None:
        try:
            dashboard = DashboardServer()
            dashboard.start()
            print(f"{CYAN}Dashboard enabled. Open http://<jetson-ip>:5050{RESET}")
        except Exception as exc:
            dashboard = None
            print(f"{YELLOW}Dashboard disabled: {exc}{RESET}")
    elif ENABLE_DASHBOARD:
        print(f"{GREY}Dashboard not found yet. Create dashboard_server.py to enable it.{RESET}")

    if SAVE_HUD_EVERY > 0:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    # Terminal raw mode so keypresses register without Enter
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    frame_count      = 0
    consecutive_fail = 0

    print(f"{GREEN}Pipeline running.{RESET}")
    print(f"{GREY}Keys: e=explore  g=guidance  q=query  esc=exit{RESET}\n")

    try:
        while True:

            # ==============================================================
            # LAYER 3/4 — keyboard / mode switching
            # ==============================================================
            should_exit = controller.check_keys()
            # Reset temporal smoothing when mode changes
            if hasattr(controller, "_last_mode"):
                if controller.mode != controller._last_mode:
                    temporal.reset()
            controller._last_mode = controller.mode
            if should_exit:
                break

            # ==============================================================
            # LAYER 1.1 — capture + rectify
            # ==============================================================
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= 10:
                    print(f"{RED}Too many camera failures — exiting{RESET}")
                    break
                continue
            consecutive_fail = 0
            frame_count += 1

            rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)

            # ==============================================================
            # LAYER 1.1 — depth map
            # ==============================================================
            depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)

            # ==============================================================
            # LAYER 1.2a — occupancy grid      ← parallel
            # LAYER 1.2b — YOLO detections     ← parallel
            # (both consume depth_bundle independently)
            # ==============================================================
            occ_bundle  = build_occupancy_map(depth_bundle)
            yolo_bundle = yolo_detect(rectL, depth_bundle, Q)

            # ==============================================================
            # LAYER 1.3 — object tracker
            # stabilizes raw YOLO detections across frames
            # ==============================================================
            wall         = _detect_wall(occ_bundle, depth_bundle)
            track_bundle = tracker.update(yolo_bundle, wall)
            confirmed_yolo = _tracks_to_yolo_bundle(track_bundle, yolo_bundle)

            # ==============================================================
            # LAYER 2.1 — scene understanding
            # ==============================================================
            state = analyse(occ_bundle, confirmed_yolo, depth_bundle)
            state = temporal.update(state)  # <-- temporal smoothing layer
            state = _attach_debug_data(state, yolo_bundle, confirmed_yolo, track_bundle)

            # Dashboard/debug metadata.
            # These fields explain which mode is active and why the output
            # layer did or did not speak/vibrate on this frame.
            state["current_mode"] = controller.mode
            state["output_decision"] = {
                "mode": controller.mode,
                "output_warranted": False,
                "allowed": False,
                "channel": None,
                "reason": "not evaluated yet",
            }

            # ==============================================================
            # LAYER 5 — query mode
            # Drain the query queue and route answers through Layer 6.
            # Only processes when in query mode, but the background listener
            # always runs so queries can be queued at any moment.
            # ==============================================================
            if controller.mode == MODE_QUERY:
                answer = query.process(state)
                state["output_decision"]["mode"] = MODE_QUERY
                state["output_decision"]["channel"] = "speech"

                if answer:
                    state["output_decision"]["output_warranted"] = True
                    state["output_decision"]["allowed"] = True
                    state["output_decision"]["reason"] = "query mode answer routed to speech"
                    # Query answer always → SPEECH via decision_layer
                    router.process(state, mode=MODE_QUERY, query_answer=answer)
                else:
                    state["output_decision"]["output_warranted"] = False
                    state["output_decision"]["allowed"] = False
                    state["output_decision"]["channel"] = None
                    state["output_decision"]["reason"] = "query mode active, no queued answer"

                # Skip the normal output pipeline in query mode
                continue

            # ==============================================================
            # LAYER 3/4 — mode gate
            # controller.process() returns non-None when the throttle +
            # stability rules say "output is warranted this frame".
            # It does NOT print anything (ModeGate suppresses _print).
            # ==============================================================
            output_warranted = controller.process(state)

            # ==============================================================
            # LAYER 6 — output routing + dispatch
            # Only fires when mode gate says this frame deserves output.
            # Decides: which channel? what message? what intensity?
            # ==============================================================
            state["output_decision"]["output_warranted"] = output_warranted is not None

            if output_warranted is not None:
                if controller.mode == MODE_EXPLORATION:
                    explore_allowed = should_dispatch_explore(state, output_warranted)
                    state["output_decision"]["allowed"] = explore_allowed
                    state["output_decision"]["channel"] = "decision_layer" if explore_allowed else None
                    state["output_decision"]["reason"] = (
                        "exploration output allowed by anti-spam gate"
                        if explore_allowed else
                        "blocked by exploration anti-spam gate"
                    )

                    if explore_allowed:
                        router.process(state, mode=controller.mode)
                else:
                    state["output_decision"]["allowed"] = True
                    state["output_decision"]["channel"] = "decision_layer"
                    state["output_decision"]["reason"] = "mode controller allowed output in non-exploration mode"
                    router.process(state, mode=controller.mode)
            else:
                state["output_decision"]["allowed"] = False
                state["output_decision"]["channel"] = None
                state["output_decision"]["reason"] = "mode controller suppressed output this frame"

            # ==============================================================
            # Debug HUD — saved to disk, not shown on screen
            # ==============================================================
            hud = None
            should_build_hud = (
                dashboard is not None or
                (SAVE_HUD_EVERY > 0 and frame_count % SAVE_HUD_EVERY == 0)
            )

            if should_build_hud:
                hud = build_hud(state, rectL, occ_bundle, confirmed_yolo)

            if SAVE_HUD_EVERY > 0 and frame_count % SAVE_HUD_EVERY == 0 and hud is not None:
                cv2.imwrite(f"{DEBUG_DIR}/hud.png", hud)

            if dashboard is not None and hud is not None:
                dashboard.update(
                    hud_image=hud,
                    state=state,
                    raw_yolo=yolo_bundle,
                    confirmed_yolo=confirmed_yolo,
                    track_bundle=track_bundle,
                    frame_count=frame_count,
                )

    except KeyboardInterrupt:
        print(f"\n{GREY}Interrupted{RESET}")

    finally:
        # Restore terminal, release camera
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        query.stop()
        if 'dashboard' in locals() and dashboard is not None:
            try:
                dashboard.stop()
            except Exception:
                pass
        cap.release()
        print(f"\n{GREY}Camera released. Done.{RESET}")
        print(f"{GREY}Total frames: {frame_count}  "
              f"Output dispatches: {router.dispatch_count}{RESET}\n")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    main()
