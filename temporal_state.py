"""
temporal_state.py
=================
Layer 2.2 — Temporal Smoothing (Short-Term Memory)

Purpose
-------
Adds short-term continuity across frames (~0.5–2 seconds).

Fixes:
- object flicker (appears/disappears)
- false "clear" states
- unstable guidance (left/right flip)
- space_type instability

IMPORTANT:
This is NOT long-term memory.
This is just smoothing perception into something human-stable.
"""

from __future__ import annotations
import time
from collections import deque, Counter
from typing import Optional


# =============================================================================
# Tunable constants
# =============================================================================

HISTORY_SIZE = 8                 # ~1–2 seconds depending on FPS
DOMINANT_HOLD_SEC = 0.6         # keep object alive briefly after loss
GUIDANCE_STABILITY_FRAMES = 3   # require consistency before change
CLEAR_HOLD_SEC = 0.8            # delay "clear" after obstacle seen


# =============================================================================
# TemporalState
# =============================================================================

class TemporalState:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_SIZE)

        # Dominant object memory
        self.last_dominant = None
        self.last_dominant_time = 0.0

        # Guidance smoothing
        self.guidance_history = deque(maxlen=GUIDANCE_STABILITY_FRAMES)
        self.last_stable_guidance = None

        # Clear suppression
        self.last_obstacle_time = 0.0

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(self, state: dict) -> dict:
        """
        Takes raw environment_state and returns smoothed version.
        """
        now = time.perf_counter()
        self.history.append(state)

        state = state.copy()

        # --------------------------------------------------------------
        # 1. Smooth space type (majority vote)
        # --------------------------------------------------------------
        space_types = [s["space_type"] for s in self.history]
        state["space_type"] = self._majority(space_types)

        # --------------------------------------------------------------
        # 2. Smooth dominant object (persistence)
        # --------------------------------------------------------------
        dominant = state.get("dominant_obstacle")

        if dominant:
            self.last_dominant = dominant
            self.last_dominant_time = now
            self.last_obstacle_time = now
        else:
            # Hold last object briefly
            if (now - self.last_dominant_time) < DOMINANT_HOLD_SEC:
                state["dominant_obstacle"] = self.last_dominant
            else:
                self.last_dominant = None

        # --------------------------------------------------------------
        # 3. Prevent false "clear"
        # --------------------------------------------------------------
        guidance = state["guidance"]
        if guidance["state"] == "clear":
            if (now - self.last_obstacle_time) < CLEAR_HOLD_SEC:
                # downgrade clear → caution instead of lying
                guidance["state"] = "caution"
                guidance["reason"] = "recent obstacle — holding caution"
                guidance["priority"] = max(guidance["priority"], 2)

        # --------------------------------------------------------------
        # 4. Smooth guidance state (debounce)
        # --------------------------------------------------------------
        g_state = guidance["state"]
        self.guidance_history.append(g_state)

        if len(self.guidance_history) == GUIDANCE_STABILITY_FRAMES:
            stable = self._majority(self.guidance_history)
            if self.last_stable_guidance is None:
                self.last_stable_guidance = stable
            else:
                # Only update if consistent
                if stable == self.last_stable_guidance:
                    guidance["state"] = stable
                else:
                    # hold previous to avoid flip-flop
                    guidance["state"] = self.last_stable_guidance

            self.last_stable_guidance = guidance["state"]

        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _majority(self, items):
        """Return most common element."""
        if not items:
            return None
        counts = Counter(items)
        return counts.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Reset (on mode switch)
    # ------------------------------------------------------------------

    def reset(self):
        self.history.clear()
        self.last_dominant = None
        self.guidance_history.clear()
        self.last_stable_guidance = None