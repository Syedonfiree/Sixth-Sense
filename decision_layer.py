"""
decision_layer.py
=================
Layer 6 — Output Router  (Channel Selection + Dispatch)

What this file does
-------------------
Sits between the mode/situation layers and the physical output hardware.
Takes a structured state bundle and decides TWO things:

    1. WHICH output channel to use   (haptic / spatial_audio / speech / tone / silent)
    2. WHAT to send down that channel (the formatted payload)

It then dispatches to the appropriate output handler.

In this prototype, every channel handler is a print statement.
Each print clearly shows what WOULD happen on real hardware.
Replacing a handler means swapping one function — nothing else changes.

Why this layer exists
---------------------
Without a router, every layer above has to know about every output type.
mode_controller.py would need to know when to use haptics vs speech.
query_layer.py would need to know about spatial audio.
That's the wrong place for those decisions.

The router centralises the rule:
    "given this situation and this mode, use THIS channel"

Output channels
---------------
HAPTIC          Vibration motor pulses. Zero cognitive load.
                Used for: urgent alerts, immediate stop signals.
                Pattern: long pulse = stop, short double = caution.
                Hardware: GPIO PWM to vibration motor (future).

SPATIAL_AUDIO   Directional audio cue panned left or right.
                Used for: directional guidance (veer left / veer right).
                The direction IS the message — no words needed.
                Hardware: stereo earpiece, left/right pan + distance pitch (future).

SPEECH          Synthesised voice output.
                Used for: query answers, space type announcements,
                          detailed obstacle descriptions.
                Hardware: TTS engine → earpiece (future).

TONE            Short non-speech audio tone.
                Used for: low-priority awareness in exploration mode.
                A soft beep is less disruptive than speech when the
                user is just moving around freely.
                Hardware: audio tone generator (future).

SILENT          No output. Situation does not warrant interrupting the user.

Routing rules (priority order)
------------------------------
Priority 3 (urgent)   → HAPTIC   always, regardless of mode
                         + SPEECH if there's a label to name

Directional states     → SPATIAL_AUDIO
(slight/strong L or R)   left pan for left, right pan for right
                          pan amount scales with correction strength

Exploration mode       → TONE for low priority (P1) awareness
                          SPEECH only if something is actually blocking

Guidance mode          → SPEECH for all P2+ guidance
                          SPATIAL_AUDIO for directional states

Query answers          → SPEECH always

Clear / no hazard      → SILENT in exploration
                          SPEECH in guidance (periodic path confirmation)

Output payload format
---------------------
Every dispatched payload is a dict:

  channel   : str   — "haptic" / "spatial_audio" / "speech" / "tone" / "silent"
  message   : str   — human-readable description of what would be output
  priority  : int   — 1 / 2 / 3
  mode      : str   — current system mode
  direction : str or None  — "left" / "right" / "ahead" / None
  pan       : float or None — -1.0 (full left) to +1.0 (full right), None if not spatial
  intensity : float — 0.0–1.0 how strong/loud/long the output should be
  reason    : str   — why this output was chosen

Prototype print format
----------------------
  [MODE]  [CHANNEL]  message
  ↳ priority=N  intensity=0.NN  reason: ...

Stack position:
    depth_map.py
    occupancy_map.py  ─┐
    yolo_detector.py  ─┴→  environment_state.py
                                mode_controller.py   (mode decision)
                                query_layer.py       (query answers)
                                >>> decision_layer.py <<<
                                output_layer.py      (future hardware drivers)
"""

from __future__ import annotations

from typing import Optional

# =============================================================================
# Channel constants
# =============================================================================

CHANNEL_HAPTIC = "haptic"
CHANNEL_SPATIAL_AUDIO = "spatial_audio"
CHANNEL_SPEECH = "speech"
CHANNEL_TONE = "tone"
CHANNEL_SILENT = "silent"

# =============================================================================
# Mode constants — mirrored here to avoid circular import
# =============================================================================

MODE_EXPLORATION = "exploration"
MODE_GUIDANCE = "guidance"
MODE_QUERY = "query"

# =============================================================================
# Routing thresholds
# =============================================================================

PAN_SLIGHT = 0.45
PAN_STRONG = 0.90

INTENSITY_P1 = 0.35
INTENSITY_P2 = 0.65
INTENSITY_P3 = 1.00

EXPLORATION_SPEECH_THRESHOLD_M = 1.5

# =============================================================================
# ANSI colours
# =============================================================================

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
BLUE = "\033[94m"
GREY = "\033[90m"
WHITE = "\033[97m"
RESET = "\033[0m"
BOLD = "\033[1m"

_CHANNEL_COLOURS = {
    CHANNEL_HAPTIC: RED + BOLD,
    CHANNEL_SPATIAL_AUDIO: BLUE + BOLD,
    CHANNEL_SPEECH: WHITE,
    CHANNEL_TONE: GREEN,
    CHANNEL_SILENT: GREY,
}

_MODE_COLOURS = {
    MODE_EXPLORATION: CYAN,
    MODE_GUIDANCE: YELLOW,
    MODE_QUERY: MAGENTA,
}


# =============================================================================
# Routing helpers
# =============================================================================

def _guidance_state_to_direction(g_state: str) -> Optional[str]:
    if "left" in g_state:
        return "left"
    if "right" in g_state:
        return "right"
    if g_state in ("caution", "stop", "recentre"):
        return "ahead"
    return None


def _guidance_state_to_pan(g_state: str) -> Optional[float]:
    if g_state == "slight_left":
        return -PAN_SLIGHT
    if g_state == "strong_left":
        return -PAN_STRONG
    if g_state == "slight_right":
        return PAN_SLIGHT
    if g_state == "strong_right":
        return PAN_STRONG
    return None


def _intensity_from_priority(priority: int) -> float:
    if priority >= 3:
        return INTENSITY_P3
    if priority == 2:
        return INTENSITY_P2
    return INTENSITY_P1


def _is_directional(g_state: str) -> bool:
    return g_state in ("slight_left", "slight_right", "strong_left", "strong_right")


# =============================================================================
# Core routing logic
# =============================================================================

def route(
    state: dict,
    mode: str,
    query_answer: Optional[str] = None,
) -> dict:
    """
    Decide which output channel to use and build the payload.
    """
    guidance = state["guidance"]
    g_state = guidance["state"]
    priority = guidance["priority"]
    reason = guidance["reason"]
    dominant = state.get("dominant_obstacle")
    space = state.get("space_type", "unknown")

    direction = _guidance_state_to_direction(g_state)
    pan = _guidance_state_to_pan(g_state)
    intensity = _intensity_from_priority(priority)

    # Query answers always go to speech
    if query_answer is not None:
        return _build(
            channel=CHANNEL_SPEECH,
            message=query_answer,
            priority=1,
            mode=mode,
            direction=None,
            pan=None,
            intensity=INTENSITY_P1,
            reason="query response always routed to speech",
        )

    # Priority 3 urgent alert
    if priority == 3:
        label = dominant["label"] if dominant else "obstacle"
        dist_str = (
            f"{dominant['distance_m']:.1f}m"
            if dominant and dominant.get("distance_m") is not None
            else ""
        )
        msg = f"STOP — {label} {dist_str} directly ahead".strip()

        if mode == MODE_GUIDANCE:
            return _build(
                channel=CHANNEL_HAPTIC,
                message=msg,
                priority=3,
                mode=mode,
                direction="ahead",
                pan=None,
                intensity=INTENSITY_P3,
                reason="priority 3 alert → haptic (+ speech in guidance mode)",
                extra={"also_speak": msg},
            )

        return _build(
            channel=CHANNEL_HAPTIC,
            message=msg,
            priority=3,
            mode=mode,
            direction="ahead",
            pan=None,
            intensity=INTENSITY_P3,
            reason="priority 3 alert → haptic pulse",
        )

    # ------------------------------------------------------------------
    # Exploration mode
    # ------------------------------------------------------------------
    if mode == MODE_EXPLORATION:
        if dominant and dominant.get("distance_m") is not None:
            if (
                dominant["distance_m"] <= EXPLORATION_SPEECH_THRESHOLD_M
                and dominant.get("blocks_path")
            ):
                label = dominant["label"]
                dist_str = f"{dominant['distance_m']:.1f}m"
                msg = f"{label} at {dist_str} — {direction or 'ahead'}"
                return _build(
                    channel=CHANNEL_SPEECH,
                    message=msg,
                    priority=priority,
                    mode=mode,
                    direction=direction,
                    pan=None,
                    intensity=intensity,
                    reason="close blocking obstacle in exploration → speech",
                )

        if dominant and dominant.get("distance_m") is not None:
            if dominant["distance_m"] <= EXPLORATION_SPEECH_THRESHOLD_M:
                return _build(
                    channel=CHANNEL_TONE,
                    message=f"soft awareness tone — {dominant['label']} nearby",
                    priority=priority,
                    mode=mode,
                    direction=direction,
                    pan=pan,
                    intensity=INTENSITY_P1,
                    reason="nearby non-blocking object in exploration → tone",
                )

        return _build(
            channel=CHANNEL_SILENT,
            message="no output — path clear in exploration mode",
            priority=1,
            mode=mode,
            direction=None,
            pan=None,
            intensity=0.0,
            reason="exploration mode + clear path → silent",
        )

    # ------------------------------------------------------------------
    # Guidance mode
    # ------------------------------------------------------------------
    if mode == MODE_GUIDANCE:
        if _is_directional(g_state):
            strength = "strong" if "strong" in g_state else "slight"
            side = "left" if "left" in g_state else "right"
            msg = f"{strength} correction — veer {side}"
            return _build(
                channel=CHANNEL_SPATIAL_AUDIO,
                message=msg,
                priority=priority,
                mode=mode,
                direction=side,
                pan=pan,
                intensity=intensity,
                reason=f"directional guidance in guidance mode → spatial audio (pan={pan:+.2f})",
            )

        if g_state == "recentre":
            return _build(
                channel=CHANNEL_SPATIAL_AUDIO,
                message="recentre — drift correction",
                priority=priority,
                mode=mode,
                direction="ahead",
                pan=0.0,
                intensity=INTENSITY_P1,
                reason="corridor recentre → spatial audio centred",
            )

        if g_state in ("caution", "stop"):
            label = dominant["label"] if dominant else "obstacle"
            dist_str = (
                f"{dominant['distance_m']:.1f}m"
                if dominant and dominant.get("distance_m") is not None
                else ""
            )
            msg = f"caution — {label} {dist_str} ahead".strip()
            return _build(
                channel=CHANNEL_SPEECH,
                message=msg,
                priority=priority,
                mode=mode,
                direction="ahead",
                pan=None,
                intensity=intensity,
                reason="caution/stop state in guidance → speech",
            )

        if g_state == "clear":
            msg = f"path clear — {space}"
            return _build(
                channel=CHANNEL_SPEECH,
                message=msg,
                priority=1,
                mode=mode,
                direction="ahead",
                pan=None,
                intensity=INTENSITY_P1,
                reason="clear path in guidance → speech confirmation",
            )

        return _build(
            channel=CHANNEL_SPEECH,
            message=reason,
            priority=priority,
            mode=mode,
            direction=direction,
            pan=None,
            intensity=intensity,
            reason=f"unhandled guidance state '{g_state}' → speech fallback",
        )

    # ------------------------------------------------------------------
    # Query mode without an answer
    # ------------------------------------------------------------------
    return _build(
        channel=CHANNEL_SILENT,
        message="query mode — waiting for user input",
        priority=1,
        mode=mode,
        direction=None,
        pan=None,
        intensity=0.0,
        reason="query mode active — no autonomous output",
    )


# =============================================================================
# Payload builder
# =============================================================================

def _build(
    channel: str,
    message: str,
    priority: int,
    mode: str,
    direction: Optional[str],
    pan: Optional[float],
    intensity: float,
    reason: str,
    extra: Optional[dict] = None,
) -> dict:
    payload = {
        "channel": channel,
        "message": message,
        "priority": priority,
        "mode": mode,
        "direction": direction,
        "pan": pan,
        "intensity": round(intensity, 2),
        "reason": reason,
    }
    if extra:
        payload.update(extra)
    return payload


# =============================================================================
# Dispatcher
# =============================================================================

def dispatch(payload: dict) -> None:
    channel = payload["channel"]

    if channel == CHANNEL_HAPTIC:
        _handle_haptic(payload)
    elif channel == CHANNEL_SPATIAL_AUDIO:
        _handle_spatial_audio(payload)
    elif channel == CHANNEL_SPEECH:
        _handle_speech(payload)
    elif channel == CHANNEL_TONE:
        _handle_tone(payload)
    elif channel == CHANNEL_SILENT:
        _handle_silent(payload)
    else:
        print(f"{GREY}[DECISION_LAYER] unknown channel: {channel}{RESET}")


# =============================================================================
# Channel handlers
# =============================================================================

def _handle_haptic(payload: dict) -> None:
    priority = payload["priority"]

    if priority >= 3:
        pattern = "■■■■■■  (1× long pulse 600ms)"
    elif priority == 2:
        pattern = "■■  ■■  (2× short pulses 200ms)"
    else:
        pattern = "■  (1× soft pulse 100ms)"

    col = _CHANNEL_COLOURS[CHANNEL_HAPTIC]
    _print_output(payload, f"HAPTIC   {pattern}", col)

    if "also_speak" in payload:
        speech_payload = _build(
            channel=CHANNEL_SPEECH,
            message=payload["also_speak"],
            priority=payload["priority"],
            mode=payload["mode"],
            direction=payload["direction"],
            pan=None,
            intensity=payload["intensity"],
            reason="haptic + speech for urgent alert in guidance mode",
        )
        _handle_speech(speech_payload)


def _handle_spatial_audio(payload: dict) -> None:
    pan = payload.get("pan", 0.0) or 0.0
    intensity = payload["intensity"]

    pan_pct = int(abs(pan) * 100)
    side = "LEFT" if pan < 0 else ("RIGHT" if pan > 0 else "CENTRE")
    pitch = "high" if intensity >= INTENSITY_P2 else "low"

    bar_left = "◄" * int((1.0 + pan) * 5)
    bar_right = "►" * int((1.0 - pan) * 5)
    pan_vis = f"{bar_left:5s}|{bar_right:5s}"

    col = _CHANNEL_COLOURS[CHANNEL_SPATIAL_AUDIO]
    _print_output(
        payload,
        f"SPATIAL  {pan_vis}  {side} {pan_pct}%  pitch={pitch}",
        col,
    )


def _handle_speech(payload: dict) -> None:
    col = _CHANNEL_COLOURS[CHANNEL_SPEECH]
    _print_output(payload, f"SPEECH   \"{payload['message']}\"", col)


def _handle_tone(payload: dict) -> None:
    intensity = payload["intensity"]
    volume = "soft" if intensity < 0.5 else "medium"
    col = _CHANNEL_COLOURS[CHANNEL_TONE]
    _print_output(payload, f"TONE     ♪ awareness beep ({volume})", col)


def _handle_silent(payload: dict) -> None:
    col = _CHANNEL_COLOURS[CHANNEL_SILENT]
    _print_output(payload, "SILENT   (no output)", col)


# =============================================================================
# Shared print formatter
# =============================================================================

def _print_output(payload: dict, channel_str: str, col: str) -> None:
    mode = payload["mode"]
    message = payload["message"]
    priority = payload["priority"]
    intensity = payload["intensity"]
    reason = payload["reason"]

    mode_col = _MODE_COLOURS.get(mode, WHITE)
    p_col = RED if priority == 3 else (YELLOW if priority == 2 else GREEN)

    print(
        f"{mode_col}[{mode.upper()[:5]}]{RESET}  "
        f"{col}[{channel_str}]{RESET}  "
        f"{WHITE}{message}{RESET}"
    )
    print(
        f"  {GREY}↳ priority={p_col}{priority}{GREY}  "
        f"intensity={intensity:.2f}  "
        f"reason: {reason}{RESET}"
    )


# =============================================================================
# OutputRouter class
# =============================================================================

class OutputRouter:
    """
    Stateful wrapper around route() + dispatch().
    """

    def __init__(self):
        self._last_payload: Optional[dict] = None
        self._dispatch_count = 0

    def process(
        self,
        state: dict,
        mode: str,
        query_answer: Optional[str] = None,
    ) -> dict:
        payload = route(state, mode, query_answer)
        dispatch(payload)
        self._last_payload = payload
        self._dispatch_count += 1
        return payload

    @property
    def last_payload(self) -> Optional[dict]:
        return self._last_payload

    @property
    def dispatch_count(self) -> int:
        return self._dispatch_count


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    def _fake_state(
        g_state,
        priority,
        dom_label=None,
        dom_dist=None,
        blocks=False,
        space="corridor",
        ahead=0.9,
    ):
        return {
            "space_type": space,
            "traversability": {"ahead": ahead, "left": 0.6, "right": 0.8},
            "dominant_obstacle": {
                "label": dom_label,
                "distance_m": dom_dist,
                "position_x_m": 0.0,
                "blocks_path": blocks,
                "source": "yolo",
            } if dom_label else None,
            "guidance": {
                "state": g_state,
                "reason": f"test reason for {g_state}",
                "priority": priority,
            },
        }

    router = OutputRouter()

    scenarios = [
        ("Clear path — exploration",
         MODE_EXPLORATION, dict(g_state="clear", priority=1, ahead=1.0), None),

        ("Nearby object — exploration (tone)",
         MODE_EXPLORATION, dict(g_state="caution", priority=2, dom_label="chair", dom_dist=1.2), None),

        ("Blocking object — exploration (speech)",
         MODE_EXPLORATION, dict(g_state="caution", priority=2, dom_label="chair", dom_dist=1.0, blocks=True), None),

        ("Urgent alert — exploration (haptic)",
         MODE_EXPLORATION, dict(g_state="strong_right", priority=3, dom_label="person", dom_dist=0.4, blocks=True), None),

        ("Slight left — guidance (spatial audio)",
         MODE_GUIDANCE, dict(g_state="slight_left", priority=2), None),

        ("Strong right — guidance (spatial audio)",
         MODE_GUIDANCE, dict(g_state="strong_right", priority=2), None),

        ("Caution ahead — guidance (speech)",
         MODE_GUIDANCE, dict(g_state="caution", priority=2, dom_label="wall", dom_dist=1.1, blocks=True), None),

        ("Urgent — guidance (haptic + speech)",
         MODE_GUIDANCE, dict(g_state="strong_left", priority=3, dom_label="door", dom_dist=0.3, blocks=True), None),

        ("Clear — guidance (speech confirmation)",
         MODE_GUIDANCE, dict(g_state="clear", priority=1, ahead=1.0), None),

        ("Query answer — any mode (speech)",
         MODE_QUERY, dict(g_state="clear", priority=1), "The path ahead is clear, corridor on your left."),
    ]

    sep = f"{GREY}{'─' * 70}{RESET}"

    print(f"\n{BOLD}{WHITE}Decision Layer — channel routing test{RESET}")
    print(f"{GREY}Each scenario shows what the output layer would do.{RESET}\n")

    for description, mode, state_args, q_answer in scenarios:
        print(sep)
        print(f"{CYAN}▶ {description}{RESET}")
        state = _fake_state(**state_args)
        router.process(state, mode=mode, query_answer=q_answer)
        print()

    print(sep)
    print(f"\n{GREEN}Total dispatches: {router.dispatch_count}{RESET}")
    print(f"{GREY}All channels exercised. Replace handler bodies for real hardware.{RESET}\n")
