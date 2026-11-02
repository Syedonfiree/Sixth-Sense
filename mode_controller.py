"""
mode_controller.py
==================
Layer 3.1 — Mode Control + Output (Prototype)

What this file does
-------------------
Decides how the system behaves based on the current mode, throttles
output so the user isn't overwhelmed, and prints clean status lines
to the terminal.

In the prototype, output = print statements.
Later these print statements get replaced with haptic pulses,
spatial audio cues, and speech synthesis.

Modes
-----
EXPLORATION  (press e)
    Passive safety net. User is moving around freely.
    System stays quiet unless something is close and blocking.
    Prints at most once every 8 seconds unless urgent.

GUIDANCE     (press g)
    Active navigation assistance. User is trying to get somewhere.
    Directional help, corridor recentering, obstacle warnings.
    Prints at most once every 3 seconds unless urgent.

QUERY        (press q)
    On-demand information. Not yet implemented.
    Placeholder for future speech input layer.

ALERT        (automatic, overrides everything)
    Immediate danger. Fires every frame until resolved.
    Cannot be suppressed by throttling or mode.

Controls
--------
e — switch to exploration mode
g — switch to guidance mode
q — query mode (not yet implemented)
esc or ctrl+c — exit

Output format
-------------
[EXPLORE]  reason — detail
[GUIDANCE] reason — detail
[ALERT]    STOP — reason
[MODE]     switched to exploration

Stack position:
    depth_map.py
    occupancy_map.py  ─┐
    yolo_detector.py  ─┴→  environment_state.py
                                >>> mode_controller.py <<<
                                output_layer.py   (future)
"""

from __future__ import annotations
 
import time
import sys
import os
import select
from typing import Optional
 
# =============================================================================
# Mode definitions
# =============================================================================
 
MODE_EXPLORATION = "exploration"
MODE_GUIDANCE    = "guidance"
MODE_QUERY       = "query"
 
# =============================================================================
# Throttle intervals (seconds)
# =============================================================================

# How often each priority level is allowed to print
# Priority 3 (urgent) always fires immediately — no throttle
THROTTLE_P2_SEC = 2.0    # caution — at most once per 2 seconds
THROTTLE_P1_SEC = 4.0    # info    — at most once per 4 seconds

# In guidance mode, info fires more frequently
THROTTLE_P1_GUIDANCE_SEC = 2.5
 
# Alert must persist for this long before being considered resolved
ALERT_CLEAR_SEC = 1.5
 
# Message must be consistent for this long before speaking.
# At 5fps this is ~3 frames — quick enough to be responsive.

STABILITY_SEC = 0.5

#
# Anti-flicker / anti-spam tuning
CLEAR_STABILITY_SEC    = 0.6   # require longer stability before saying "clear"
NONCLEAR_STABILITY_SEC = 0.2   # faster for warnings/directions
MIN_REPEAT_SEC         = 1.2   # do not repeat the same message too often
DIST_BUCKET_M          = 0.75  # coarser buckets to reduce distance jitter
 
# Don't suppress unlabelled obstacles entirely — downgrade but don't hide.
# They print as info so the user still gets feedback.
SUPPRESS_UNLABELLED = False
 
# =============================================================================
# ANSI colours for terminal output
# =============================================================================
# Makes it easy to visually scan the output while walking around.
 
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GREY   = "\033[90m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
 
 
def _priority_colour(priority: int) -> str:
    if priority == 3: return RED + BOLD
    if priority == 2: return YELLOW
    return GREEN
 
 
def _tracks_to_yolo_bundle(track_bundle: dict, original_yolo: dict) -> dict:
    """
    Convert confirmed tracker output into a yolo-bundle-compatible dict
    so environment_state.py needs no interface changes.
 
    Confirmed tracks replace raw detections. Unconfirmed tracks are
    excluded — this is the whole point of the tracker.
 
    Wall tracks (id=-1) are excluded here because environment_state
    handles walls via _detect_wall() separately.
    """
    confirmed = [
        t for t in track_bundle["confirmed_tracks"]
        if not t["is_wall"]
    ]
 
    detections = []
    for t in confirmed:
        detections.append({
            "label":         t["label"],
            "confidence":    t["confidence"],
            "distance_m":    round(t["distance_m"], 3),
            "position_x_m":  round(t["position_x_m"], 3),
            "bbox":          t["bbox"] if t["bbox"] else [0, 0, 0, 0],
            "centre_px":     [0, 0],
            "in_path":       t["in_path"],
            "is_close":      t["distance_m"] < 1.5,
            "is_blocking":   t["in_path"] and t["distance_m"] < 1.5,
            "is_relevant":   True,
        })
 
    return {
        "detections":        detections,
        "detection_count":   len(detections),
        "inference_time_ms": original_yolo["inference_time_ms"],
        "depth_hit_rate":    original_yolo["depth_hit_rate"],
        "debug_frame":       original_yolo["debug_frame"],
    }
 
 
# =============================================================================
# ModeController class
# =============================================================================
 
class ModeController:
    """
    Manages mode state and throttled output for the navigation system.
 
    Usage
    -----
    controller = ModeController()
 
    # In your main loop:
    controller.check_keys()           # non-blocking keyboard check
    controller.process(state_bundle)  # pass environment_state output
    """
 
    def __init__(self, initial_mode: str = MODE_EXPLORATION, tracker=None):
        self.mode             = initial_mode
        self._tracker         = tracker
        self._last_print_time = 0.0
        self._last_alert_time = 0.0
        self._in_alert        = False
        self._last_guidance_state = None
        self._last_space_type     = None
        self._last_dom_label      = None
        self._last_dom_dist_bkt   = None
        self._frame_count         = 0

        # (Removed: Space-type smoothing and dominant obstacle smoothing fields)

        # Stability — message must hold for STABILITY_SEC before speaking
        self._current_message = None
        self._message_since   = 0.0

        # Anti-spam: track last emitted message key/time
        self._last_emit_key   = None
        self._last_emit_time  = 0.0

        self._print_mode_banner()
 
    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
 
    def check_keys(self) -> bool:
        """
        Non-blocking keyboard check.
        Returns True if the user pressed ESC (exit signal).
 
        Uses select() to check if stdin has input without blocking.
        This means the main loop keeps running at full speed and
        keyboard input is checked each frame.
        """
        # select() checks if stdin has data available to read
        # timeout=0 means return immediately whether or not there's input
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            if key == 'e':
                self._switch_mode(MODE_EXPLORATION)
            elif key == 'g':
                self._switch_mode(MODE_GUIDANCE)
            elif key == 'q':
                self._switch_mode(MODE_QUERY)
            elif key == '\x1b':   # ESC
                print(f"\n{GREY}Exiting...{RESET}")
                return True       # signal to exit
        return False
 
    def process(self, state: dict) -> Optional[str]:
        """
        Process one environment_state bundle.
        Decides whether to print based on mode and throttle rules.
        Returns the printed string if something was printed, else None.
 
        Parameters
        ----------
        state : dict — output of environment_state.analyse()
        """
        self._frame_count += 1
        now = time.perf_counter()

        guidance  = state["guidance"]
        priority  = guidance["priority"]
        g_state   = guidance["state"]
        reason    = guidance["reason"]
        dominant  = state["dominant_obstacle"]
        space = state["space_type"]
        trav      = state["traversability"]
        # (Removed: dominant obstacle smoothing logic; handled by temporal_state)
 
        # ------------------------------------------------------------------
        # Suppress raw "obstacle" label when YOLO has real detections.
        # If YOLO found something it would have labelled it. "obstacle"
        # from occupancy when YOLO is active = stereo noise, not real.
        # Only keep "obstacle" when YOLO found nothing at all (dark scene,
        # textureless surface etc) AND it's been stable.
        # ------------------------------------------------------------------
        yolo_has_detections = state.get("detection_count", 0) > 0 or len(
            [d for d in (state.get("objects_in_path") or [])]) > 0
 
        if dominant is not None:
            if (dominant["label"] == "obstacle" and
                    dominant["source"] == "occupancy" and
                    yolo_has_detections):
                dominant = None   # YOLO is active, trust it over occupancy noise
 
        # ------------------------------------------------------------------
        # Query mode — not implemented yet
        # ------------------------------------------------------------------
        if self.mode == MODE_QUERY:
            return None
 
        # ------------------------------------------------------------------
        # Priority 3 — ALERT
        # Only fire alert for wall or YOLO-labelled objects, not raw occupancy
        # ------------------------------------------------------------------
        if priority == 3 and dominant is not None:
            self._in_alert     = True
            self._last_alert_time = now
            label = dominant["label"]
            dist  = f"{dominant['distance_m']:.1f}m" if dominant["distance_m"] else ""
            line  = self._format_alert(g_state, reason, label, dist)
            self._print(line)
            self._last_print_time = now
            return line
 
        # ------------------------------------------------------------------
        # Clear alert state if danger has passed
        # ------------------------------------------------------------------
        if self._in_alert:
            if now - self._last_alert_time > ALERT_CLEAR_SEC:
                self._in_alert = False
                msg = f"{GREEN}[ALERT CLEARED]{RESET} path is clear"
                print(msg)
                self._last_print_time = now
 
        # ------------------------------------------------------------------
        # Determine throttle interval based on mode and priority
        # ------------------------------------------------------------------
        if self.mode == MODE_GUIDANCE:
            throttle = THROTTLE_P2_SEC if priority == 2 else THROTTLE_P1_GUIDANCE_SEC
        else:
            throttle = THROTTLE_P2_SEC if priority == 2 else THROTTLE_P1_SEC
 
        # ------------------------------------------------------------------
        # Build candidate message key for stability check
        # ------------------------------------------------------------------
        dom_label    = dominant["label"] if dominant else None
        dom_dist_bkt = int(dominant["distance_m"] / DIST_BUCKET_M) if dominant and dominant["distance_m"] else None

        # Message key — a string representing what we'd say right now
        msg_key = f"{g_state}|{dom_label}|{dom_dist_bkt}|{space}"

        # ------------------------------------------------------------------
        # Fix 4 — stability check
        # Message must be consistently true for STABILITY_SEC before speaking.
        # This prevents one-frame blips from triggering output.
        # ------------------------------------------------------------------
        if msg_key != self._current_message:
            # Message changed — reset stability timer
            self._current_message = msg_key
            self._message_since   = now

        # Require longer stability before declaring CLEAR to avoid flip-flop
        if g_state == "clear":
            stable_needed = CLEAR_STABILITY_SEC
        else:
            stable_needed = NONCLEAR_STABILITY_SEC

        message_stable = (now - self._message_since) >= stable_needed
 
        # ------------------------------------------------------------------
        # State change detection — bypass throttle if something meaningful changed
        # but still require stability
        # ------------------------------------------------------------------
        state_changed = (
            g_state      != self._last_guidance_state or
            dom_label    != self._last_dom_label      or
            dom_dist_bkt != self._last_dom_dist_bkt
        )
        # Detect urgency increase
        priority_increased = (
            self._last_guidance_state is not None and
            priority > (3 if self._last_guidance_state == "stop" else 1)
        )
        # space_type is intentionally excluded — it flickers too much
        # and causes false change triggers
 
        time_elapsed = (now - self._last_print_time) >= throttle

        # Do not spam identical messages
        if msg_key == self._last_emit_key and (now - self._last_emit_time) < MIN_REPEAT_SEC:
            return None

        # Allow immediate output if urgency increased
        if priority_increased:
            pass
        elif not (message_stable and (state_changed or time_elapsed)):
            return None

        # (Removed: false "clear" suppression; handled by temporal_state)
 
        # ------------------------------------------------------------------
        # Format and print based on mode
        # ------------------------------------------------------------------
        if self.mode == MODE_EXPLORATION:
            line = self._format_exploration(state, g_state, reason, priority,
                                            dominant, space, trav)
        elif self.mode == MODE_GUIDANCE:
            line = self._format_guidance(state, g_state, reason, priority,
                                         dominant, space, trav)
        else:
            return None
 
        self._print(line)
        self._last_print_time     = now
        self._last_emit_key       = msg_key
        self._last_emit_time      = now
        self._last_guidance_state = g_state
        self._last_space_type     = space
        self._last_dom_label      = dom_label
        self._last_dom_dist_bkt   = dom_dist_bkt
        return line
 
    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------
 
    def _format_alert(self, g_state: str, reason: str,
                      label: str, dist: str) -> str:
        col = RED + BOLD
        tag = f"{col}[ALERT]{RESET}"
        # Show label and distance cleanly without duplicating the reason text
        obj_str = f"{label} at {dist}" if dist else label
        return f"{tag} {col}STOP{RESET} — {obj_str} directly ahead"
 
    def _format_exploration(
        self, state, g_state, reason, priority, dominant, space, trav
    ) -> str:
        col = _priority_colour(priority)
        tag = f"{CYAN}[EXPLORE]{RESET}"
 
        if dominant and dominant["distance_m"] is not None:
            dist_str = f"{dominant['distance_m']:.1f}m"
            label    = dominant["label"]
            x_pos    = dominant.get("position_x_m")
 
            # Walls get special phrasing
            if dominant.get("source") == "wall_detection":
                ang = state.get("guidance", {}).get("reason", "")
                detail = f"wall at {dist_str} — {reason}"
            else:
                if x_pos is not None:
                    side = "left" if x_pos < -0.15 else ("right" if x_pos > 0.15 else "ahead")
                else:
                    side = "ahead"
                if dominant["blocks_path"]:
                    detail = f"{col}{label} at {dist_str} {side} — blocking{RESET}"
                else:
                    detail = f"{label} at {dist_str} {side}"
        else:
            # Only say "clear" if we truly have strong forward clearance
            if trav["ahead"] > 0.9:
                detail = f"clear ahead — {space}"
            else:
                detail = f"{space} — {reason}"
 
        return f"{tag} {detail}"
 
    def _format_guidance(
        self, state, g_state, reason, priority, dominant, space, trav
    ) -> str:
        col = _priority_colour(priority)
        tag = f"{YELLOW}[GUIDANCE]{RESET}"
 
        # Direction guidance
        direction_states = {
            "slight_left":  "slight left",
            "slight_right": "slight right",
            "strong_left":  "MOVE LEFT",
            "strong_right": "MOVE RIGHT",
            "recentre":     "recentre",
            "caution":      "caution",
            "clear":        "clear",
        }
        direction = direction_states.get(g_state, g_state)
 
        # Build detail string
        parts = [f"{col}{direction}{RESET}"]
 
        if reason:
            parts.append(reason)
 
        # Add traversability context
        trav_str = (f"fwd={trav['ahead']:.1f} "
                    f"L={trav['left']:.1f} "
                    f"R={trav['right']:.1f}")
        parts.append(f"[{trav_str}]")
 
        # Add space type if it changed
        if space != self._last_space_type:
            parts.append(f"({space})")
 
        return f"{tag} {' — '.join(parts)}"
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # (Removed: _smooth_space_type method; smoothing now handled by temporal_state)
 
    def _switch_mode(self, new_mode: str) -> None:
        if new_mode == self.mode:
            return
        self.mode = new_mode
        self._last_print_time     = 0.0
        self._last_guidance_state = None
        self._last_space_type     = None
        self._last_dom_label      = None
        self._last_dom_dist_bkt   = None
        self._in_alert            = False
        self._current_message     = None
        self._message_since       = 0.0
        # (Removed: reset smoothing on mode switch)
        if self._tracker is not None:
            self._tracker.reset()
        self._print_mode_banner()
 
    def _print_mode_banner(self) -> None:
        mode_colours = {
            MODE_EXPLORATION: CYAN,
            MODE_GUIDANCE:    YELLOW,
            MODE_QUERY:       GREY,
        }
        col  = mode_colours.get(self.mode, WHITE)
        desc = {
            MODE_EXPLORATION: "passive awareness — alerts only if something is close",
            MODE_GUIDANCE:    "active navigation — directional guidance enabled",
            MODE_QUERY:       "query mode — not yet implemented",
        }.get(self.mode, "")
        print(f"\n{col}[MODE] switched to {self.mode.upper()}{RESET}")
        print(f"{GREY}  {desc}{RESET}")
        print(f"{GREY}  keys: e=explore  g=guidance  q=query  esc=exit{RESET}\n")
 
    def _print(self, line: str) -> None:
        print(line)
 
 
# =============================================================================
# Standalone test — run directly: python mode_controller.py
# Runs the full pipeline in a live loop.
# Press e, g, q to switch modes. ESC to exit.
# =============================================================================
 
if __name__ == "__main__":
    import cv2
    import tty
    import termios
 
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
    try:
        from depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from occupancy_map import build_occupancy_map
        from yolo_detector import detect as yolo_detect
        from object_tracker import ObjectTracker
        from environment_state import analyse, build_hud, _detect_wall
    except ImportError:
        from modules.depth_map import build_stereo_matcher, process_frame as depth_process_frame
        from modules.occupancy_map import build_occupancy_map
        from modules.yolo_detector import detect as yolo_detect
        from modules.object_tracker import ObjectTracker
        from modules.environment_state import analyse, build_hud, _detect_wall
 
    CALIB_PATH   = "calibration/stereo_calibration.npz"
    CAMERA_INDEX = 0
    FRAME_WIDTH  = 1280
    FRAME_HEIGHT = 480
 
    if not os.path.exists(CALIB_PATH):
        print(f"ERROR: calibration file not found at {CALIB_PATH}")
        raise SystemExit(1)
 
    cal    = cv2.FileStorage if False else None
    cal    = __import__("numpy").load(CALIB_PATH)
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
 
    ret, test_frame = cap.read()
    if not ret:
        print("ERROR: camera not streaming")
        cap.release()
        raise SystemExit(1)
 
    h, w  = test_frame.shape[:2]
    eye_w = w // 2
    eye_h = h
 
    mapL1, mapL2 = cv2.initUndistortRectifyMap(M1, d1, R1, P1, (eye_w, eye_h), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(M2, d2, R2, P2, (eye_w, eye_h), cv2.CV_32FC1)
 
    stereo  = build_stereo_matcher()
    tracker = ObjectTracker()
 
    print(f"{CYAN}Warming up YOLO...{RESET}")
    # Pre-load YOLO by running detect once on a dummy bundle
    import numpy as _np
    _dummy_frame = _np.zeros((eye_h, eye_w, 3), dtype=_np.uint8)
 
    # Set terminal to raw mode so keypresses register immediately
    # without needing to press Enter
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
 
    controller = ModeController(initial_mode=MODE_EXPLORATION, tracker=tracker)
 
    # Debug output — save HUD every N frames
    SAVE_HUD_EVERY = 30
    os.makedirs("debug_output", exist_ok=True)
 
    frame_count      = 0
    consecutive_fail = 0
 
    print(f"{GREEN}Pipeline running. Press e/g/q to switch modes, ESC to exit.{RESET}\n")
 
    try:
        while True:
            # ----------------------------------------------------------
            # Keyboard input
            # ----------------------------------------------------------
            should_exit = controller.check_keys()
            if should_exit:
                break
 
            # ----------------------------------------------------------
            # Capture frame
            # ----------------------------------------------------------
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_fail += 1
                if consecutive_fail >= 10:
                    print("Too many camera failures, exiting")
                    break
                continue
            consecutive_fail = 0
            frame_count += 1
 
            # ----------------------------------------------------------
            # Perception pipeline
            # ----------------------------------------------------------
            rectL = cv2.remap(frame[:, :eye_w], mapL1, mapL2, cv2.INTER_LINEAR)
            rectR = cv2.remap(frame[:, eye_w:], mapR1, mapR2, cv2.INTER_LINEAR)
 
            depth_bundle = depth_process_frame(rectL, rectR, Q, stereo, eye_w, eye_h)
            occ_bundle   = build_occupancy_map(depth_bundle)
            yolo_bundle  = yolo_detect(rectL, depth_bundle, Q)
 
            # ------------------------------------------------------------------
            # Object tracker — stabilizes detections over time
            # Only confirmed tracks (seen consistently) feed environment_state
            # ------------------------------------------------------------------
            wall         = _detect_wall(occ_bundle, depth_bundle)
            track_bundle = tracker.update(yolo_bundle, wall)
 
            # Build a yolo-compatible bundle from confirmed tracks so
            # environment_state.py needs no changes to its interface
            confirmed_yolo = _tracks_to_yolo_bundle(track_bundle, yolo_bundle)
 
            # ----------------------------------------------------------
            # Scene understanding — uses confirmed tracks not raw YOLO
            # ----------------------------------------------------------
            state = analyse(occ_bundle, confirmed_yolo, depth_bundle)
 
            # ----------------------------------------------------------
            # Mode controller output
            # ----------------------------------------------------------
            controller.process(state)
 
            # ----------------------------------------------------------
            # Save HUD periodically for visual debugging
            # ----------------------------------------------------------
            if frame_count % SAVE_HUD_EVERY == 0:
                hud = build_hud(state, rectL, occ_bundle, confirmed_yolo)
                cv2.imwrite("debug_output/hud.png", hud)
 
    except KeyboardInterrupt:
        print(f"\n{GREY}Interrupted{RESET}")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        cap.release()
        print(f"{GREY}Camera released. Done.{RESET}")
 
