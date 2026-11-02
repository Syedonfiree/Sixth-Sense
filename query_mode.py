"""
query_layer.py
==============
Layer 5.1 — Query Mode (On-Demand Speech Interface)

What this file does
-------------------
Listens for user voice queries in a background thread, queues them,
then resolves each one against the most recent environment_state bundle
when the system is in MODE_QUERY.

Queries are on-demand — the user asks something, the system answers
once and goes quiet. This is not continuous narration.

Recognised queries (keyword-based, no LLM needed)
--------------------------------------------------
  "what's in front of me"       → dominant obstacle + distance
  "is the path clear"           → traversability ahead score
  "how far is [that/it]"        → distance to dominant obstacle
  "what's on my left/right"     → left/right traversability + obstacles
  "what kind of space"          → space_type label
  "what's around me"            → full 360 summary (dominant + space + trav)
  "how close is [that/it]"      → same as "how far"
  "any obstacles"               → obstacle count + nearest
  "repeat" / "say again"        → repeats the last answer

Everything else
---------------
Falls back to a generic "I didn't catch that — try again" response.

Speech backend
--------------
Uses the `SpeechRecognition` library with Google's free web API.
Falls back gracefully to text input if the microphone isn't available
(useful for testing on a machine without a mic).

The fallback can be forced by setting FORCE_TEXT_INPUT = True.

Controls
--------
Activated when mode_controller switches to MODE_QUERY.
In text-fallback mode, the user types a query and presses Enter.
In voice mode, the user speaks into the microphone.

Output format (prototype — will be replaced by TTS + haptic)
-------------------------------------------------------------
[QUERY]  ← what the system heard
[ANSWER] ← what the system replied

Stack position:
    depth_map.py
    occupancy_map.py  ─┐
    yolo_detector.py  ─┴→  environment_state.py
                                mode_controller.py
                                >>> query_layer.py <<<
                                output_layer.py   (future)
"""

from __future__ import annotations

import re
import queue
import threading
import time
from typing import Optional

# =============================================================================
# Configuration
# =============================================================================

# Set True to skip microphone entirely and use typed input instead.
# Automatically set True if speech_recognition or pyaudio is missing.
FORCE_TEXT_INPUT = False

# How long (seconds) the mic listens for a single query before timing out
MIC_TIMEOUT_SEC      = 5.0   # wait this long for speech to start
MIC_PHRASE_LIMIT_SEC = 8.0   # max length of a single phrase

# How long to wait between polling the query queue (seconds)
QUEUE_POLL_SEC = 0.05

# =============================================================================
# ANSI colours — match mode_controller.py palette
# =============================================================================

MAGENTA = "\033[95m"
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
GREY    = "\033[90m"
WHITE   = "\033[97m"
RESET   = "\033[0m"
BOLD    = "\033[1m"

# =============================================================================
# Try to import speech_recognition — degrade gracefully if absent
# =============================================================================

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False
    print(f"{YELLOW}[QUERY] speech_recognition not installed — using text input fallback.{RESET}")
    print(f"{GREY}  Install with:  pip install SpeechRecognition pyaudio{RESET}\n")

# =============================================================================
# Keyword → intent map
# =============================================================================
# Each entry: (compiled regex, intent string)
# Patterns are tested in order; first match wins.

_INTENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"repeat|say again|what did you say", re.I),        "repeat"),
    (re.compile(r"what.s (in front|ahead|before me)",    re.I),     "ahead"),
    (re.compile(r"is (the )?path (clear|safe|open)",     re.I),     "path_clear"),
    (re.compile(r"how (far|close|near) is (it|that)",    re.I),     "distance"),
    (re.compile(r"distance",                             re.I),     "distance"),
    (re.compile(r"(on )?my left",                        re.I),     "left"),
    (re.compile(r"(on )?my right",                       re.I),     "right"),
    (re.compile(r"(what.s )?around me",                  re.I),     "surroundings"),
    (re.compile(r"what kind of space|where am i",        re.I),     "space"),
    (re.compile(r"any obstacle|anything (there|nearby|around)",re.I),"obstacles"),
    (re.compile(r"(is it|is the path|path) block",       re.I),     "blocking"),
]


def _classify(text: str) -> str:
    """Return intent string for a raw query string."""
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return "unknown"


# =============================================================================
# Answer builder — translates environment_state bundle → plain English
# =============================================================================

def _answer(intent: str, state: dict, last_answer: Optional[str]) -> str:
    """
    Build a plain-English answer from an intent + environment state bundle.

    Parameters
    ----------
    intent      : string from _classify()
    state       : latest dict from environment_state.analyse()
    last_answer : the previous answer string, for the "repeat" intent

    Returns
    -------
    A short sentence the output layer can speak or print.
    """
    dominant = state.get("dominant_obstacle")
    trav     = state.get("traversability", {})
    space    = state.get("space_type", "unknown space")
    guidance = state.get("guidance", {})
    g_state  = guidance.get("state", "clear")
    reason   = guidance.get("reason", "")

    ahead_score = trav.get("ahead", 1.0)
    left_score  = trav.get("left",  1.0)
    right_score = trav.get("right", 1.0)

    # Helper — describe the dominant obstacle concisely
    def _dom_str() -> Optional[str]:
        if not dominant:
            return None
        label = dominant.get("label", "obstacle")
        dist  = dominant.get("distance_m")
        x_pos = dominant.get("position_x_m")
        side  = ""
        if x_pos is not None:
            side = " to your left" if x_pos < -0.15 else (" to your right" if x_pos > 0.15 else " directly ahead")
        dist_str = f"{dist:.1f} metres" if dist is not None else ""
        parts = [label]
        if dist_str: parts.append(dist_str + " away")
        if side:     parts.append(side)
        return ", ".join(parts)

    # Helper — describe a traversability score in words
    def _trav_word(score: float) -> str:
        if score >= 0.85: return "clear"
        if score >= 0.60: return "mostly clear"
        if score >= 0.35: return "partially blocked"
        return "blocked"

    # ---------- intent dispatch ----------

    if intent == "repeat":
        return last_answer if last_answer else "Nothing to repeat yet."

    if intent == "ahead":
        dom = _dom_str()
        if dom:
            return f"There is a {dom}."
        if ahead_score > 0.8:
            return "The path ahead looks clear."
        return f"The path ahead is {_trav_word(ahead_score)}."

    if intent == "path_clear":
        if g_state in ("alert", "caution") or ahead_score < 0.5:
            dom = _dom_str()
            return f"No — {dom}." if dom else "No — something is blocking the path."
        if ahead_score > 0.8:
            return "Yes, the path ahead is clear."
        return f"Mostly — ahead is {_trav_word(ahead_score)}."

    if intent == "distance":
        if dominant and dominant.get("distance_m") is not None:
            label = dominant.get("label", "obstacle")
            dist  = dominant["distance_m"]
            return f"The nearest object — {label} — is {dist:.1f} metres away."
        return "No obstacle detected close enough to measure."

    if intent == "left":
        dom_left = None
        if dominant and dominant.get("position_x_m") is not None:
            if dominant["position_x_m"] < -0.15:
                dom_left = _dom_str()
        score_str = _trav_word(left_score)
        if dom_left:
            return f"To your left: {dom_left}. Traversability is {score_str}."
        return f"Left side is {score_str}."

    if intent == "right":
        dom_right = None
        if dominant and dominant.get("position_x_m") is not None:
            if dominant["position_x_m"] > 0.15:
                dom_right = _dom_str()
        score_str = _trav_word(right_score)
        if dom_right:
            return f"To your right: {dom_right}. Traversability is {score_str}."
        return f"Right side is {score_str}."

    if intent == "space":
        return f"You appear to be in a {space}."

    if intent == "surroundings":
        parts = [f"You are in a {space}."]
        dom = _dom_str()
        if dom:
            parts.append(f"Nearest object: {dom}.")
        parts.append(
            f"Ahead is {_trav_word(ahead_score)}, "
            f"left is {_trav_word(left_score)}, "
            f"right is {_trav_word(right_score)}."
        )
        return " ".join(parts)

    if intent == "obstacles":
        detections = state.get("dominant_obstacle")
        dom = _dom_str()
        if dom:
            return f"Yes — nearest is a {dom}."
        if ahead_score > 0.85 and left_score > 0.85 and right_score > 0.85:
            return "No obstacles detected nearby."
        return f"Path is {_trav_word(ahead_score)} — no specific object identified."

    if intent == "blocking":
        if dominant and dominant.get("blocks_path"):
            dom = _dom_str()
            return f"Yes — {dom} is blocking your path."
        return "No, the path is not directly blocked."

    # fallback
    return "Sorry, I didn't understand that. Try asking what's ahead, or is the path clear."


# =============================================================================
# QueryLayer class
# =============================================================================

class QueryLayer:
    """
    Layer 5 — On-demand query interface.

    Runs a background listener thread that pushes recognised queries
    into an internal queue. Call process() each frame to drain the queue
    and resolve answers against the current environment state.

    Usage
    -----
    query_layer = QueryLayer()
    query_layer.start()        # begins background listener

    # In main loop (only when mode == MODE_QUERY):
    answer = query_layer.process(state_bundle)
    if answer:
        print(answer)          # or pass to TTS / haptic layer

    query_layer.stop()         # clean shutdown
    """

    def __init__(self, force_text: bool = False):
        self._use_text   = force_text or FORCE_TEXT_INPUT or not _SR_AVAILABLE
        self._q: queue.Queue[str] = queue.Queue()
        self._last_answer: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        if not self._use_text:
            self._recogniser = sr.Recognizer()
            self._recogniser.dynamic_energy_threshold = True
            self._recogniser.pause_threshold = 0.8  # seconds of silence = end of phrase
            try:
                self._mic = sr.Microphone()
            except Exception:
                print(f"{YELLOW}[QUERY] Microphone unavailable — falling back to text input.{RESET}")
                self._use_text = True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background listener thread."""
        if self._running:
            return
        self._running = True
        target = self._text_listener if self._use_text else self._mic_listener
        self._thread = threading.Thread(target=target, daemon=True, name="query-listener")
        self._thread.start()
        mode = "text" if self._use_text else "voice"
        print(f"\n{MAGENTA}[QUERY] Layer 5 started ({mode} input).{RESET}")
        if self._use_text:
            print(f"{GREY}  Type a query and press Enter. Examples:{RESET}")
        else:
            print(f"{GREY}  Speak into the microphone. Examples:{RESET}")
        print(f"{GREY}  'What's in front of me'  'Is the path clear'  'What's around me'{RESET}\n")

    def stop(self) -> None:
        """Signal the listener thread to stop. Non-blocking."""
        self._running = False

    # ------------------------------------------------------------------
    # Main interface — call each frame when in query mode
    # ------------------------------------------------------------------

    def process(self, state: dict) -> Optional[str]:
        """
        Drain pending queries and resolve the first one against state.

        Returns the answer string if a query was resolved this frame,
        otherwise returns None.

        Only processes one query per call to avoid overwhelming the user.
        Remaining queries stay in the queue for the next frame.
        """
        try:
            raw_text = self._q.get_nowait()
        except queue.Empty:
            return None

        self._print_query(raw_text)

        intent = _classify(raw_text)
        answer = _answer(intent, state, self._last_answer)
        self._last_answer = answer

        self._print_answer(answer)
        return answer

    def enqueue(self, text: str) -> None:
        """
        Manually push a query string (useful for testing or remote input).
        Thread-safe.
        """
        self._q.put(text.strip())

    def pending_count(self) -> int:
        """Number of unprocessed queries in the queue."""
        return self._q.qsize()

    # ------------------------------------------------------------------
    # Background listener threads
    # ------------------------------------------------------------------

    def _text_listener(self) -> None:
        """
        Text-input fallback listener.
        Runs in a daemon thread — blocks on input() which is fine here
        because it's not the main thread.
        """
        import sys
        while self._running:
            try:
                # input() blocks, but this is a daemon thread so it won't
                # prevent the process from exiting.
                text = input(f"{MAGENTA}[QUERY ›]{RESET} ")
                text = text.strip()
                if text:
                    self._q.put(text)
            except EOFError:
                break
            except KeyboardInterrupt:
                break

    def _mic_listener(self) -> None:
        """
        Microphone listener using SpeechRecognition.
        Runs in a daemon thread, continuously listens while active.
        """
        with self._mic as source:
            # Adjust for ambient noise once on startup
            print(f"{GREY}[QUERY] Calibrating microphone... (1 second){RESET}")
            self._recogniser.adjust_for_ambient_noise(source, duration=1.0)
            print(f"{GREEN}[QUERY] Microphone ready. Speak your query.{RESET}\n")

            while self._running:
                try:
                    audio = self._recogniser.listen(
                        source,
                        timeout=MIC_TIMEOUT_SEC,
                        phrase_time_limit=MIC_PHRASE_LIMIT_SEC,
                    )
                    text = self._recogniser.recognize_google(audio)
                    self._q.put(text)

                except sr.WaitTimeoutError:
                    # No speech detected — that's fine, keep waiting
                    pass
                except sr.UnknownValueError:
                    # Speech detected but couldn't be understood
                    self._q.put("__unintelligible__")
                except sr.RequestError as e:
                    print(f"{RED}[QUERY] Speech API error: {e}{RESET}")
                    time.sleep(2.0)
                except Exception as e:
                    print(f"{RED}[QUERY] Listener error: {e}{RESET}")
                    time.sleep(1.0)

    # ------------------------------------------------------------------
    # Output formatting (prototype — replace with TTS / haptic later)
    # ------------------------------------------------------------------

    def _print_query(self, text: str) -> None:
        if text == "__unintelligible__":
            print(f"{MAGENTA}[QUERY]{RESET}  {GREY}(unintelligible){RESET}")
        else:
            print(f"{MAGENTA}[QUERY]{RESET}  {WHITE}{text}{RESET}")

    def _print_answer(self, answer: str) -> None:
        print(f"{CYAN}[ANSWER]{RESET} {answer}\n")


# =============================================================================
# Integration helper — drop into mode_controller.py
# =============================================================================

def make_query_layer(force_text: bool = False) -> QueryLayer:
    """
    Convenience factory. Returns a QueryLayer ready to start().

    Example usage in mode_controller.py
    ------------------------------------
    from query_layer import make_query_layer

    query = make_query_layer()
    query.start()

    # In ModeController.process(), add:
    if self.mode == MODE_QUERY:
        answer = query.process(state)
        return answer
    """
    return QueryLayer(force_text=force_text)


# =============================================================================
# Standalone test — python query_layer.py
# Runs without a camera. Uses a static fake state bundle.
# =============================================================================

if __name__ == "__main__":
    import json

    FAKE_STATE = {
        "space_type": "corridor",
        "dominant_obstacle": {
            "label":        "chair",
            "distance_m":   1.2,
            "position_x_m": -0.3,
            "blocks_path":  False,
            "source":       "yolo",
        },
        "traversability": {
            "ahead": 0.6,
            "left":  0.3,
            "right": 0.9,
        },
        "guidance": {
            "state":    "slight_left",
            "reason":   "chair on left, veer right",
            "priority": 2,
        },
    }

    print(f"\n{BOLD}{CYAN}QueryLayer — standalone test{RESET}")
    print(f"{GREY}Using fake environment state:{RESET}")
    print(json.dumps(FAKE_STATE, indent=2))
    print()

    layer = QueryLayer(force_text=True)
    layer.start()

    print(f"{GREY}Processing queries. Ctrl+C to exit.{RESET}\n")

    try:
        while True:
            layer.process(FAKE_STATE)
            time.sleep(QUEUE_POLL_SEC)
    except KeyboardInterrupt:
        print(f"\n{GREY}Exiting query layer test.{RESET}")
    finally:
        layer.stop()
