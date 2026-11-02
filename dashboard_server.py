"""
dashboard_server.py
===================
Live web dashboard for 6thSense.

main.py calls:

    dashboard = DashboardServer()
    dashboard.start()
    dashboard.update(...)

Then open:

    http://<jetson-ip>:5050

Routes:
    /             Web dashboard
    /video_feed   Live MJPEG HUD stream
    /state        Latest JSON state/debug data
"""

from __future__ import annotations

import json
import time
import logging
import threading
from typing import Any, Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request
from collections import deque


class DashboardServer:
    """
    Lightweight Flask dashboard server.

    It does not run perception.
    It only displays the latest HUD/state pushed from main.py.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5050):
        self.host = host
        self.port = port

        self.app = Flask(__name__)

        # Keep the terminal readable. Without this, every dashboard poll prints
        # a GET /state line and clogs the main navigation output.
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self.app.logger.disabled = True

        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._latest_jpeg: Optional[bytes] = None
        self._latest_state: dict[str, Any] = {}
        self._latest_raw_yolo: dict[str, Any] = {}
        self._latest_confirmed_yolo: dict[str, Any] = {}
        self._latest_track_bundle: dict[str, Any] = {}
        self._latest_frame_count: int = 0
        self._latest_update_time: float = 0.0
        self._latest_raw_log: str = ""

        self._fps_times: list[float] = []

        # Capture system (for 5–10s recording + review)
        self._history = deque(maxlen=300)  # ~10s at ~30fps
        self._capture_active = False
        self._capture_buffer = []
        self._capture_end_time = 0.0

        self._setup_routes()

    # -------------------------------------------------------------------------
    # Public API used by main.py
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start Flask in a background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """
        Mark server stopped.

        Flask's development server does not shut down cleanly from another
        thread, but daemon=True means it exits when main.py exits.
        """
        self._running = False

    def update(
        self,
        hud_image,
        state: dict,
        raw_yolo: Optional[dict] = None,
        confirmed_yolo: Optional[dict] = None,
        track_bundle: Optional[dict] = None,
        frame_count: int = 0,
        raw_log: Optional[str] = None,
    ) -> None:
        """
        Push latest dashboard data.

        hud_image should be a BGR OpenCV image from build_hud(...).

        Performance note:
        The dashboard must never try to JSON-serialize full debug frames or
        large numpy arrays. That will slow the perception loop badly. This
        method stores a compressed, resized JPEG plus lightweight state only.
        """

        if hud_image is None:
            return

        # Resize the HUD stream for speed. The original 1280-wide image is
        # useful for saving to disk, but too heavy for live browser streaming.
        h, w = hud_image.shape[:2]
        max_w = 960
        if w > max_w:
            scale = max_w / float(w)
            hud_image = cv2.resize(
                hud_image,
                (max_w, int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        ok, encoded = cv2.imencode(
            ".jpg",
            hud_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 65],
        )
        if not ok:
            return

        now = time.time()

        with self._lock:
            self._latest_jpeg = encoded.tobytes()
            self._latest_state = self._light_state(state)
            self._latest_raw_yolo = self._light_yolo(raw_yolo or {})
            self._latest_confirmed_yolo = self._light_yolo(confirmed_yolo or {})
            self._latest_track_bundle = self._light_tracks(track_bundle or {})
            self._latest_frame_count = int(frame_count)
            self._latest_update_time = now

            # Store rolling raw log (append, keep last 5000 chars)
            if raw_log is not None:
                self._latest_raw_log = (self._latest_raw_log + "\n" + str(raw_log))[-5000:]

            self._fps_times.append(now)
            self._fps_times = [t for t in self._fps_times if now - t <= 2.0]

            # Store history snapshot (lightweight but useful for review)
            light_state = self._light_state(state)
            snapshot = {
                "t": now,
                "frame_count": int(frame_count),
                "mode": state.get("current_mode") or light_state.get("current_mode") or "unknown",
                "space_type": light_state.get("space_type"),
                "raw_space_type": light_state.get("raw_space_type"),
                "guidance": light_state.get("guidance"),
                "dominant": light_state.get("dominant_obstacle"),
                "decision": state.get("output_decision") or light_state.get("output_decision"),
                "confidence": light_state.get("confidence"),
                "raw_detections": light_state.get("raw_detections", []),
                "confirmed_detections": light_state.get("confirmed_detections", []),
                "objects_in_path": light_state.get("objects_in_path", []),
                "state": light_state,
            }
            self._history.append(snapshot)

            # Capture logic
            if self._capture_active:
                self._capture_buffer.append(snapshot)
                if now >= self._capture_end_time:
                    self._capture_active = False

    # -------------------------------------------------------------------------
    # Flask internals
    # -------------------------------------------------------------------------

    def _run(self) -> None:
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )

    def _setup_routes(self) -> None:
        @self.app.route("/")
        def index():
            return render_template_string(DASHBOARD_HTML)

        @self.app.route("/video_feed")
        def video_feed():
            response = Response(
                self._mjpeg_generator(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            response.headers["X-Accel-Buffering"] = "no"
            return response

        @self.app.route("/state")
        def state():
            with self._lock:
                now = time.time()
                age = now - self._latest_update_time if self._latest_update_time else None
                fps = len(self._fps_times) / 2.0 if self._fps_times else 0.0

                payload = {
                    "frame_count": self._latest_frame_count,
                    "updated_age_sec": round(age, 3) if age is not None else None,
                    "dashboard_fps_est": round(fps, 2),
                    "state": self._latest_state,
                    "raw_yolo": self._latest_raw_yolo,
                    "confirmed_yolo": self._latest_confirmed_yolo,
                    "track_bundle": self._latest_track_bundle,
                    "raw_log": self._latest_raw_log,
                }

            return jsonify(payload)

        @self.app.route("/health")
        def health():
            return jsonify({"ok": True, "running": self._running})

        @self.app.route("/capture/start")
        def capture_start():
            seconds = float(request.args.get("seconds", 5))
            now = time.time()
            with self._lock:
                self._capture_active = True
                self._capture_buffer = []
                self._capture_end_time = now + seconds
            return jsonify({"status": "recording", "seconds": seconds})

        @self.app.route("/capture/status")
        def capture_status():
            with self._lock:
                return jsonify({
                    "active": self._capture_active,
                    "frames": len(self._capture_buffer),
                })

        @self.app.route("/capture/data")
        def capture_data():
            with self._lock:
                return jsonify(self._capture_buffer)

        @self.app.route("/capture/clear")
        def capture_clear():
            with self._lock:
                self._capture_buffer = []
                self._capture_active = False
            return jsonify({"status": "cleared"})

    def _mjpeg_generator(self):
        last_frame = None

        while True:
            with self._lock:
                frame = self._latest_jpeg

            if frame is None:
                time.sleep(0.05)
                continue

            # Only send when a new frame arrives. This prevents the browser
            # from building up a backlog of repeated old frames.
            if frame is last_frame:
                time.sleep(0.01)
                continue

            last_frame = frame

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Cache-Control: no-store\r\n\r\n" +
                frame +
                b"\r\n"
            )
    # -------------------------------------------------------------------------
    # Lightweight payload builders
    # -------------------------------------------------------------------------

    def _light_detection(self, detection: dict) -> dict:
        """Return only dashboard-useful detection fields."""
        return {
            "label": detection.get("label"),
            "confidence": self._safe_json_copy(detection.get("confidence")),
            "distance_m": self._safe_json_copy(detection.get("distance_m")),
            "position_x_m": self._safe_json_copy(detection.get("position_x_m")),
            "in_path": self._safe_json_copy(detection.get("in_path")),
            "is_close": self._safe_json_copy(detection.get("is_close")),
            "is_blocking": self._safe_json_copy(detection.get("is_blocking")),
            "bbox": self._safe_json_copy(detection.get("bbox")),
        }

    def _light_yolo(self, yolo: dict) -> dict:
        """Strip heavy YOLO debug frames before sending JSON."""
        detections = yolo.get("detections", []) or []
        return {
            "detection_count": self._safe_json_copy(yolo.get("detection_count", len(detections))),
            "inference_time_ms": self._safe_json_copy(yolo.get("inference_time_ms")),
            "depth_hit_rate": self._safe_json_copy(yolo.get("depth_hit_rate")),
            "detections": [self._light_detection(d) for d in detections[:20]],
        }

    def _light_tracks(self, track_bundle: dict) -> dict:
        """Strip tracker data down to fields useful for the dashboard."""
        tracks = track_bundle.get("confirmed_tracks", []) or []
        return {
            "confirmed_track_count": len(tracks),
            "confirmed_tracks": [
                {
                    "label": track.get("label"),
                    "confidence": self._safe_json_copy(track.get("confidence")),
                    "distance_m": self._safe_json_copy(track.get("distance_m")),
                    "position_x_m": self._safe_json_copy(track.get("position_x_m")),
                    "in_path": self._safe_json_copy(track.get("in_path")),
                    "is_wall": self._safe_json_copy(track.get("is_wall")),
                    "bbox": self._safe_json_copy(track.get("bbox")),
                }
                for track in tracks[:20]
            ],
        }

    def _light_state(self, state: dict) -> dict:
        """
        Keep only the compact state fields that are useful for live debugging.

        This avoids accidentally serializing raw frames, numpy arrays, or other
        heavy data attached to the state bundle.
        """
        keep = [
            "space_type",
            "raw_space_type",
            "space_scores",
            "clutter_density",
            "clutter_zone",
            "traversability",
            "dominant_obstacle",
            "nearest_person",
            "objects_in_path",
            "wall",
            "doorway_candidate",
            "corridor_detected",
            "centerline_offset_m",
            "guidance",
            "confidence",
            "detection_count",
            "raw_detection_count",
            "confirmed_track_count",
            "current_mode",
            "output_decision",
            "raw_detections",
            "confirmed_detections",
        ]

        light = {}
        for key in keep:
            if key not in state:
                continue

            if key in ("raw_detections", "confirmed_detections"):
                detections = state.get(key, []) or []
                light[key] = [self._light_detection(d) for d in detections[:20]]
            else:
                light[key] = self._safe_json_copy(state.get(key))

        return light


    # -------------------------------------------------------------------------
    # JSON safety helpers
    # -------------------------------------------------------------------------

    def _safe_json_copy(self, value: Any) -> Any:
        """
        Convert numpy values and other non-JSON-safe objects into safe values.
        """
        try:
            return json.loads(json.dumps(value, default=self._json_default))
        except Exception:
            return {"error": "could not serialize dashboard object"}

    def _json_default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()

            if isinstance(obj, (np.integer,)):
                return int(obj)

            if isinstance(obj, (np.floating,)):
                return float(obj)

            if isinstance(obj, (np.bool_,)):
                return bool(obj)

        except Exception:
            pass

        try:
            return str(obj)
        except Exception:
            return None


DASHBOARD_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>6thSense Live Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <style>
    :root {
      --bg: #071014;
      --panel: #0d1b22;
      --panel2: #102732;
      --text: #eaf7ff;
      --muted: #8fa7b3;
      --green: #27e29a;
      --yellow: #ffd166;
      --red: #ff5c5c;
      --cyan: #39c8ff;
      --border: rgba(255,255,255,0.09);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(57,200,255,0.12), transparent 32rem),
        radial-gradient(circle at bottom right, rgba(39,226,154,0.08), transparent 28rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(7,16,20,0.86);
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(14px);
    }

    h1 {
      margin: 0;
      font-size: 22px;
      letter-spacing: 0.3px;
    }

    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 3px;
    }

    .status-pill {
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.05);
      font-size: 13px;
      color: var(--muted);
    }

    .status-pill.live {
      color: var(--green);
      border-color: rgba(39,226,154,0.35);
      box-shadow: 0 0 24px rgba(39,226,154,0.1);
    }

    .toolbar {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }

    button {
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.075);
      color: var(--text);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      cursor: pointer;
      transition: transform 0.12s ease, background 0.12s ease, border-color 0.12s ease;
    }

    button:hover {
      background: rgba(57,200,255,0.14);
      border-color: rgba(57,200,255,0.35);
      transform: translateY(-1px);
    }

    button.danger:hover {
      background: rgba(255,92,92,0.14);
      border-color: rgba(255,92,92,0.35);
    }

    .capture-status {
      color: var(--muted);
      font-size: 12px;
      min-width: 120px;
    }

    .capture-status.recording {
      color: var(--yellow);
    }

    .timeline {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
      padding-right: 4px;
    }

    .timeline-row {
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.045);
      border-radius: 12px;
      padding: 10px;
      cursor: pointer;
      display: grid;
      grid-template-columns: 56px 1fr auto;
      gap: 10px;
      align-items: center;
    }

    .timeline-row:hover,
    .timeline-row.selected {
      background: rgba(57,200,255,0.12);
      border-color: rgba(57,200,255,0.35);
    }

    .timeline-time {
      color: var(--cyan);
      font-size: 12px;
      font-weight: 700;
    }

    .timeline-main {
      font-size: 12px;
      line-height: 1.35;
    }

    .timeline-meta {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 11px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.06);
      color: var(--muted);
      white-space: nowrap;
    }

    .badge.good {
      color: var(--green);
      border-color: rgba(39,226,154,0.35);
    }

    .badge.warn {
      color: var(--yellow);
      border-color: rgba(255,209,102,0.35);
    }

    .badge.bad {
      color: var(--red);
      border-color: rgba(255,92,92,0.35);
    }

    .friendly-box {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(255,255,255,0.045);
      padding: 12px;
      font-size: 13px;
      line-height: 1.45;
    }

    .friendly-box strong {
      color: var(--cyan);
    }

    main {
      padding: 20px;
      display: grid;
      grid-template-columns: minmax(540px, 1.45fr) minmax(380px, 0.95fr);
      gap: 18px;
    }

    .panel {
      background: linear-gradient(180deg, rgba(16,39,50,0.88), rgba(13,27,34,0.92));
      border: 1px solid var(--border);
      border-radius: 18px;
      box-shadow: 0 20px 70px rgba(0,0,0,0.28);
      overflow: hidden;
    }

    .panel-title {
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .video-wrap {
      padding: 14px;
    }

    #video {
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
      border: 1px solid var(--border);
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      padding: 14px;
    }

    .card {
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.045);
      border-radius: 14px;
      padding: 13px;
      min-height: 84px;
    }

    .card .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      margin-bottom: 8px;
    }

    .card .value {
      font-size: 22px;
      font-weight: 750;
      line-height: 1.1;
    }

    .card .small {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .priority-1 { color: var(--green); }
    .priority-2 { color: var(--yellow); }
    .priority-3 { color: var(--red); }

    .section {
      padding: 14px 16px;
      border-top: 1px solid var(--border);
    }

    .section:first-child {
      border-top: none;
    }

    .section h2 {
      margin: 0 0 10px 0;
      font-size: 15px;
    }

    .kv {
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 7px 12px;
      font-size: 13px;
    }

    .k {
      color: var(--muted);
    }

    .v {
      color: var(--text);
      overflow-wrap: anywhere;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      overflow: hidden;
      border-radius: 12px;
    }

    th, td {
      border-bottom: 1px solid var(--border);
      text-align: left;
      padding: 8px 7px;
      vertical-align: top;
    }

    th {
      color: var(--muted);
      font-weight: 600;
      background: rgba(255,255,255,0.04);
    }

    tr:last-child td {
      border-bottom: none;
    }

    .json-box {
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #c7dce6;
      background: rgba(0,0,0,0.24);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      max-height: 280px;
      overflow: auto;
    }

    .bar {
      height: 8px;
      background: rgba(255,255,255,0.08);
      border-radius: 999px;
      overflow: hidden;
      margin-top: 8px;
    }

    .bar > div {
      height: 100%;
      background: linear-gradient(90deg, var(--green), var(--cyan));
      width: 0%;
    }

    @media (max-width: 1050px) {
      main {
        grid-template-columns: 1fr;
      }

      .grid {
        grid-template-columns: 1fr;
      }
    }

    /* Collapsible sections */
    .collapsible {
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .collapsible::after {
      content: "▾";
      font-size: 12px;
      color: var(--muted);
    }
    .collapsible.closed::after {
      content: "▸";
    }
    .collapsible-content { display: block; }
    .collapsible-content.closed { display: none; }
  </style>
</head>

<body>
  <header>
    <div>
      <h1>6thSense Live Dashboard</h1>
      <div class="sub">Scene understanding, YOLO/tracker debugging, guidance, and HUD stream</div>
    </div>
    <div class="toolbar">
      <div id="livePill" class="status-pill">Waiting for frames...</div>
      <button onclick="startCapture(5)">Record 5s</button>
      <button onclick="startCapture(10)">Record 10s</button>
      <button class="danger" onclick="clearCapture()">Clear</button>
      <div id="captureStatus" class="capture-status">Capture idle</div>
    </div>
  </header>

  <main>
    <div>
      <div class="panel">
        <div class="panel-title">
          <span>Live HUD</span>
          <span id="frameInfo">Frame —</span>
        </div>
        <div class="video-wrap">
          <img id="video" src="/video_feed" />
        </div>
      </div>

      <div class="panel" style="margin-top:18px;">
        <div class="panel-title">
          <span>Core State</span>
          <span id="fpsInfo">FPS —</span>
        </div>
        <div class="grid">
          <div class="card">
            <div class="label">Space Type</div>
            <div id="spaceType" class="value">—</div>
            <div id="rawSpaceType" class="small">raw: —</div>
          </div>

          <div class="card">
            <div class="label">Guidance</div>
            <div id="guidanceState" class="value">—</div>
            <div id="guidanceReason" class="small">—</div>
          </div>

          <div class="card">
            <div class="label">Confidence</div>
            <div id="confidence" class="value">—</div>
            <div class="bar"><div id="confidenceBar"></div></div>
          </div>
        </div>

        <div class="section">
          <h2>Traversability</h2>
          <div class="kv">
            <div class="k">Ahead</div><div id="travAhead" class="v">—</div>
            <div class="k">Left</div><div id="travLeft" class="v">—</div>
            <div class="k">Right</div><div id="travRight" class="v">—</div>
            <div class="k">Clutter</div><div id="clutter" class="v">—</div>
            <div class="k">Centerline Offset</div><div id="offset" class="v">—</div>
          </div>
        </div>

        <div class="section">
          <h2>Space Scores</h2>
          <div id="spaceScores" class="json-box">—</div>
        </div>
      </div>
    </div>

    <div>
      <div class="panel">
        <div class="panel-title">
          <span>Navigation Debug</span>
          <span id="ageInfo">—</span>
        </div>

        <div class="section">
          <h2>Dominant Obstacle</h2>
          <div id="dominantObstacle" class="json-box">—</div>
        </div>

        <div class="section">
          <h2>Wall</h2>
          <div id="wall" class="json-box">—</div>
        </div>

        <div class="section">
          <h2>Raw YOLO Detections</h2>
          <div id="rawTable">—</div>
        </div>

        <div class="section">
          <h2>Confirmed Tracker Detections</h2>
          <div id="confirmedTable">—</div>
        </div>

        <div class="section">
          <h2>Objects In Path</h2>
          <div id="objectsInPath" class="json-box">—</div>
        </div>

        <div class="section">
          <h2>Mode-Aware Output Decision</h2>
          <div id="outputDecisionFriendly" class="friendly-box">Waiting for output decision data...</div>
        </div>

        <div class="section">
          <h2>Live Output Channels</h2>
          <div class="kv">
            <div class="k">Speech</div><div id="liveSpeech" class="v">—</div>
            <div class="k">Haptics</div><div id="liveHaptics" class="v">—</div>
            <div class="k">Audio</div><div id="liveAudio" class="v">—</div>
            <div class="k">Decision</div><div id="liveDecision" class="v">—</div>
          </div>
        </div>

        <div class="section">
          <h2>Capture Review</h2>
          <div id="captureHint" class="friendly-box" style="margin-bottom:10px;">
            Press <strong>Record 5s</strong> or <strong>Record 10s</strong>, then review the timeline here.
          </div>
          <div id="captureTimeline" class="timeline">No capture yet.</div>
        </div>

        <div class="section">
          <h2>Selected Capture Frame</h2>
          <div id="selectedCapture" class="json-box">Click a timeline row after recording.</div>
        </div>

        <div class="section">
          <h2 class="collapsible" onclick="toggleSection(this)">Raw System Output (Unfiltered)</h2>
          <div class="collapsible-content">
            <div id="rawLog" class="json-box" style="max-height:200px;overflow:auto;">—</div>
          </div>
        </div>
        <div class="section">
          <h2 class="collapsible" onclick="toggleSection(this)">Full State JSON</h2>
          <div class="collapsible-content">
            <div id="fullJson" class="json-box">—</div>
          </div>
        </div>
      </div>
    </div>
  </main>

<script>
  function toggleSection(el) {
    el.classList.toggle("closed");
    const content = el.nextElementSibling;
    if (content) content.classList.toggle("closed");
  }

  window.addEventListener("load", () => {
    document.querySelectorAll(".collapsible").forEach(el => {
      el.classList.add("closed");
      const content = el.nextElementSibling;
      if (content) content.classList.add("closed");
    });
  });

  const $ = (id) => document.getElementById(id);

  let captureData = [];
  let selectedCaptureIndex = -1;

  function fmt(value, fallback = "—") {
    if (value === null || value === undefined) return fallback;
    if (typeof value === "number") return Number(value).toFixed(2);
    return String(value);
  }

  function pretty(obj) {
    if (obj === null || obj === undefined) return "—";
    return JSON.stringify(obj, null, 2);
  }

  function setPriorityClass(el, priority) {
    el.classList.remove("priority-1", "priority-2", "priority-3");
    el.classList.add("priority-" + String(priority || 1));
  }

  function detectionTable(dets) {
    if (!dets || dets.length === 0) {
      return "<div style='color:var(--muted);font-size:13px;'>No detections</div>";
    }

    const rows = dets.slice(0, 12).map(d => `
      <tr>
        <td>${fmt(d.label)}</td>
        <td>${fmt(d.confidence)}</td>
        <td>${fmt(d.distance_m)}</td>
        <td>${fmt(d.position_x_m)}</td>
        <td>${d.in_path ? "yes" : "no"}</td>
        <td>${d.is_blocking ? "yes" : "no"}</td>
      </tr>
    `).join("");

    return `
      <table>
        <thead>
          <tr>
            <th>Label</th>
            <th>Conf</th>
            <th>Dist</th>
            <th>X</th>
            <th>Path</th>
            <th>Block</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  }

  async function startCapture(sec) {
    selectedCaptureIndex = -1;
    $("selectedCapture").textContent = "Recording... select a row when capture completes.";
    await fetch(`/capture/start?seconds=${sec}`);
    await updateCaptureStatus();
  }

  async function clearCapture() {
    await fetch(`/capture/clear`);
    captureData = [];
    selectedCaptureIndex = -1;
    $("captureTimeline").innerHTML = "No capture yet.";
    $("selectedCapture").textContent = "Click a timeline row after recording.";
    await updateCaptureStatus();
  }

  async function updateCaptureStatus() {
    try {
      const res = await fetch("/capture/status", { cache: "no-store" });
      const status = await res.json();
      const el = $("captureStatus");
      if (status.active) {
        el.textContent = `Recording… ${status.frames} frames`;
        el.className = "capture-status recording";
      } else {
        el.textContent = status.frames > 0 ? `Capture ready: ${status.frames} frames` : "Capture idle";
        el.className = "capture-status";
      }
      return status;
    } catch (err) {
      $("captureStatus").textContent = "Capture status error";
      return { active: false, frames: 0 };
    }
  }

  async function loadCapture() {
    const res = await fetch("/capture/data", { cache: "no-store" });
    captureData = await res.json();
    renderCaptureTimeline();
  }

  function modeLabel(mode) {
    if (!mode) return "unknown";
    return String(mode).replace("mode_", "").replace("MODE_", "").toLowerCase();
  }

  function decisionBadge(decision) {
    if (!decision) return `<span class="badge">no decision</span>`;
    if (decision.allowed === true) return `<span class="badge good">output allowed</span>`;
    if (decision.output_warranted === true) return `<span class="badge warn">blocked</span>`;
    return `<span class="badge">quiet</span>`;
  }

  function captureRowSummary(snap, idx, startTime) {
    const guidance = snap.guidance || {};
    const dominant = snap.dominant || {};
    const decision = snap.decision || {};
    const t = Math.max(0, snap.t - startTime);
    const obstacle = dominant.label
      ? `${dominant.label} ${dominant.distance_m !== null && dominant.distance_m !== undefined ? fmt(dominant.distance_m) + "m" : ""}`
      : "no dominant obstacle";

    return `
      <div class="timeline-row ${idx === selectedCaptureIndex ? "selected" : ""}" onclick="selectCapture(${idx})">
        <div class="timeline-time">+${t.toFixed(1)}s</div>
        <div class="timeline-main">
          <div><strong>${modeLabel(snap.mode)}</strong> · ${fmt(guidance.state)} · ${obstacle}</div>
          <div class="timeline-meta">${fmt(decision.reason, "no decision reason")} · space=${fmt(snap.space_type)}</div>
        </div>
        <div>${decisionBadge(decision)}</div>
      </div>
    `;
  }

  function renderCaptureTimeline() {
    const el = $("captureTimeline");
    if (!captureData || captureData.length === 0) {
      el.innerHTML = "No capture yet.";
      return;
    }

    const startTime = captureData[0].t;
    el.innerHTML = captureData
      .map((snap, idx) => captureRowSummary(snap, idx, startTime))
      .join("");
  }

  function selectCapture(idx) {
    selectedCaptureIndex = idx;
    renderCaptureTimeline();
    const snap = captureData[idx];
    if (!snap) return;

    const friendly = explainSnapshot(snap);
    $("selectedCapture").textContent = friendly + "\n\nRAW SNAPSHOT\n" + pretty(snap);
  }

  function explainSnapshot(snap) {
    const mode = modeLabel(snap.mode);
    const guidance = snap.guidance || {};
    const decision = snap.decision || {};
    const dominant = snap.dominant || {};

    const obstacleText = dominant.label
      ? `${dominant.label} at ${fmt(dominant.distance_m)}m, x=${fmt(dominant.position_x_m)}, blocking=${dominant.blocks_path ? "yes" : "no"}`
      : "No dominant obstacle.";

    const decisionText = decision.allowed === true
      ? "Output was allowed."
      : decision.output_warranted === true
        ? "Output was considered but blocked."
        : "System stayed quiet.";

    return [
      `MODE: ${mode}`,
      `SPACE: ${fmt(snap.space_type)} (raw=${fmt(snap.raw_space_type)})`,
      `GUIDANCE: ${fmt(guidance.state)} / priority ${fmt(guidance.priority)} / ${fmt(guidance.reason)}`,
      `DOMINANT: ${obstacleText}`,
      `DECISION: ${decisionText}`,
      `WHY: ${fmt(decision.reason, "No output-decision reason available yet.")}`,
    ].join("\n");
  }

  function explainCurrentOutputDecision(state) {
    const mode = modeLabel(state.current_mode);
    const guidance = state.guidance || {};
    const decision = state.output_decision || {};
    const dominant = state.dominant_obstacle || {};

    let ruleText = "";
    if (mode.includes("exploration") || mode === "exploration") {
      ruleText = "Exploration mode should be mostly quiet. Speech should happen only for meaningful close/blocking obstacles, walls, or urgent changes.";
    } else if (mode.includes("guidance") || mode === "guidance") {
      ruleText = "Guidance mode can be more active. Directional haptics should carry most frequent updates, while speech should stay reserved for important context.";
    } else if (mode.includes("query") || mode === "query") {
      ruleText = "Query mode is speech-forward. Normal navigation output should be reduced while the system answers the user.";
    } else {
      ruleText = "Mode not available yet. Add current_mode in main.py for better rule explanations.";
    }

    const allowed = decision.allowed === true
      ? "Allowed"
      : decision.output_warranted === true
        ? "Blocked"
        : "Quiet";

    const obstacle = dominant.label
      ? `${dominant.label} at ${fmt(dominant.distance_m)}m`
      : "no dominant obstacle";

    return `
      <div><strong>Mode:</strong> ${mode}</div>
      <div><strong>Decision:</strong> ${allowed}</div>
      <div><strong>Reason:</strong> ${fmt(decision.reason, "No output-decision debug field yet")}</div>
      <div><strong>Guidance:</strong> ${fmt(guidance.state)} / priority ${fmt(guidance.priority)}</div>
      <div><strong>Dominant:</strong> ${obstacle}</div>
      <br>
      <div>${ruleText}</div>
    `;
  }

  async function refresh() {
    try {
      const res = await fetch("/state", { cache: "no-store" });
      const data = await res.json();

      const state = data.state || {};
      const guidance = state.guidance || {};
      const trav = state.traversability || {};

      const age = data.updated_age_sec;
      const live = age !== null && age < 2.0;

      $("livePill").textContent = live ? "LIVE" : "STALE / WAITING";
      $("livePill").className = live ? "status-pill live" : "status-pill";

      $("frameInfo").textContent = "Frame " + fmt(data.frame_count, "—");
      $("fpsInfo").textContent = "Dashboard FPS " + fmt(data.dashboard_fps_est);
      $("ageInfo").textContent = age === null ? "no data" : fmt(age) + "s ago";

      $("spaceType").textContent = fmt(state.space_type);
      $("rawSpaceType").textContent = "raw: " + fmt(state.raw_space_type);

      $("guidanceState").textContent = fmt(guidance.state).toUpperCase();
      $("guidanceReason").textContent = fmt(guidance.reason);
      setPriorityClass($("guidanceState"), guidance.priority);

      const conf = Number(state.confidence || 0);
      $("confidence").textContent = fmt(conf);
      $("confidenceBar").style.width = Math.max(0, Math.min(100, conf * 100)) + "%";

      $("travAhead").textContent = fmt(trav.ahead);
      $("travLeft").textContent = fmt(trav.left);
      $("travRight").textContent = fmt(trav.right);
      $("clutter").textContent = `${fmt(state.clutter_density)} (${fmt(state.clutter_zone)})`;
      $("offset").textContent = fmt(state.centerline_offset_m) + " m";

      $("spaceScores").textContent = pretty(state.space_scores);
      $("dominantObstacle").textContent = pretty(state.dominant_obstacle);
      $("wall").textContent = pretty(state.wall);
      $("objectsInPath").textContent = pretty(state.objects_in_path);
      $("outputDecisionFriendly").innerHTML = explainCurrentOutputDecision(state);

      // Populate live output channels panel
      const decision = state.output_decision || {};
      $("liveSpeech").textContent = decision.speech || "—";
      $("liveHaptics").textContent = decision.haptic || "—";
      $("liveAudio").textContent = decision.audio || "—";
      $("liveDecision").textContent = decision.reason || "—";

      const rawDets =
        state.raw_detections ||
        (data.raw_yolo && data.raw_yolo.detections) ||
        [];

      const confirmedDets =
        state.confirmed_detections ||
        (data.confirmed_yolo && data.confirmed_yolo.detections) ||
        [];

      $("rawTable").innerHTML = detectionTable(rawDets);
      $("confirmedTable").innerHTML = detectionTable(confirmedDets);

      $("fullJson").textContent = pretty(state);
      $("rawLog").textContent = data.raw_log || "—";

    } catch (err) {
      $("livePill").textContent = "DASHBOARD ERROR";
      $("livePill").className = "status-pill";
      console.error(err);
    }
    const captureStatus = await updateCaptureStatus();
    if (!captureStatus.active && captureStatus.frames > 0) {
      await loadCapture();
    }
  }

  refresh();
  setInterval(refresh, 1000);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting standalone 6thSense dashboard server...")
    print("This mode only serves the page. Run main.py for live data.")
    server = DashboardServer()
    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
        print("Dashboard stopped.")