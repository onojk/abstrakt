#!/usr/bin/env python3
"""Abstrakt browser UI — single-page Flask app for audio-reactive video generation."""

import os
import shutil
import sys
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import psutil
from flask import Flask, Response, jsonify, render_template, request
import subprocess

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRAKT_SH = os.path.join(REPO_ROOT, "abstrakt.sh")
JOBS_DIR    = os.path.join(REPO_ROOT, "jobs")

# ── Config ────────────────────────────────────────────────────────────────────
RESOLUTION_PRESETS = {
    "480p":  {"width": 852,  "height": 480,  "crf": 32, "preset": "ultrafast", "fps": 24},
    "720p":  {"width": 1280, "height": 720,  "crf": 22, "preset": "fast",      "fps": 30},
    "1080p": {"width": 1920, "height": 1080, "crf": 18, "preset": "medium",    "fps": 30},
    "4K":    {"width": 3840, "height": 2160, "crf": 20, "preset": "medium",    "fps": 30},
}

KALEIDO_DEFAULTS = {
    "apply_kden":    True,
    "fill_mandala":  False,
    "skip_mirror":   False,
    "kaleido_sides": 12,
    "seed_quad":     "br",
}


VISUALIZERS = ["warpfield", "02_kaleidoscope_spokes", "09_beat_reactive", "aurora", "coalescing_grid", "image_warp", "kaleido_stack", "kaleido_stack_inked", "kaleido_qbist", "kaleido_qbist_strings", "plasma", "plasma_warm", "camo_plasma", "glitch_scroll", "chaos_explosions", "speaker_grid", "string_resonance", "stick_figure_dance", "clusters_with_strings", "worm_swarm"]

MAX_UPLOAD_MB = 100

_STEP_MARKERS = [
    ("[abstrakt] 1/4",   15,  "Converting audio..."),
    ("[abstrakt] 2/4",   40,  "Rendering visualizer (this is the slow step)..."),
    ("[abstrakt] 3/4",   70,  "Applying kaleido effects..."),
    ("[abstrakt] 4/4",   90,  "Muxing audio..."),
    ("[abstrakt] Done:", 100, "Complete"),
]

# ── App ───────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

jobs: dict[str, "RenderJob"] = {}
_jobs_lock = threading.Lock()
_active_job: Optional[str] = None
_active_lock = threading.Lock()


# ── Job model ─────────────────────────────────────────────────────────────────
@dataclass
class RenderJob:
    job_id:      str
    job_dir:     str
    audio_path:  str
    visualizer:  str
    resolution:  str
    duration:    int
    log_path:    str  = field(init=False)
    output_path: str  = field(init=False)
    status:      str  = field(default="queued")
    progress:    int  = field(default=0)
    step_text:   str  = field(default="Queued")
    error:       str  = field(default="")
    _proc:       Optional[subprocess.Popen] = field(default=None, repr=False)
    _cancelled:  bool = field(default=False, repr=False)

    def __post_init__(self):
        self.log_path    = os.path.join(self.job_dir, "log.txt")
        self.output_path = os.path.join(self.job_dir, "output.mp4")

    def start(self):
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _build_cmd(self) -> list[str]:
        res = RESOLUTION_PRESETS[self.resolution]
        cmd = [
            ABSTRAKT_SH,
            self.audio_path,
            "-o", self.output_path,
            "-r", f"{res['width']}x{res['height']}",
            "--fps",    str(res["fps"]),
            "--crf",    str(res["crf"]),
            "--preset", res["preset"],
            "--visualizer", self.visualizer,
            "--kaleido-sides", str(KALEIDO_DEFAULTS["kaleido_sides"]),
            "--seed-quad", KALEIDO_DEFAULTS["seed_quad"],
        ]
        if self.duration > 0:
            cmd += ["--duration", str(self.duration)]
        if KALEIDO_DEFAULTS["apply_kden"]:
            cmd.append("--apply-kden")
        if KALEIDO_DEFAULTS["fill_mandala"]:
            cmd.append("--fill-mandala")
        if KALEIDO_DEFAULTS["skip_mirror"]:
            cmd.append("--skip-mirror")
        return cmd

    def _run(self):
        self.status    = "running"
        self.progress  = 5
        self.step_text = "Starting..."

        cmd = self._build_cmd()
        try:
            with open(self.log_path, "w") as log_fh:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    cwd=self.job_dir,
                )
            self._proc.wait()
            rc = self._proc.returncode
        except Exception as exc:
            self.error  = str(exc)
            self.status = "failed"
            _clear_active(self.job_id)
            return

        if self._cancelled:
            self.status = "cancelled"
            _clear_active(self.job_id)
            return

        if rc != 0:
            self.error  = f"abstrakt.sh exited with code {rc}"
            self.status = "failed"
        elif not os.path.isfile(self.output_path):
            self.error  = "Pipeline completed but output file not found"
            self.status = "failed"
        else:
            self.progress  = 100
            self.step_text = "Complete"
            self.status    = "done"

        _clear_active(self.job_id)

    def cancel(self):
        self._cancelled = True
        proc = self._proc
        if proc is None:
            return
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        except psutil.NoSuchProcess:
            pass


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clear_active(job_id: str):
    global _active_job
    with _active_lock:
        if _active_job == job_id:
            _active_job = None


def _parse_progress(log_path: str) -> tuple[int, str]:
    """Scan log for [abstrakt] markers; return (pct, step_text)."""
    try:
        with open(log_path, "r", errors="replace") as f:
            content = f.read()
    except OSError:
        return 5, "Starting..."

    best_pct, best_text = 5, "Starting..."
    for marker, pct, text in _STEP_MARKERS:
        if marker in content:
            best_pct, best_text = pct, text
    return best_pct, best_text


def _job_dict(job: RenderJob) -> dict:
    if job.status == "running":
        pct, txt = _parse_progress(job.log_path)
        job.progress  = pct
        job.step_text = txt
    return {
        "job_id":    job.job_id,
        "status":    job.status,
        "progress":  job.progress,
        "step_text": job.step_text,
        "error":     job.error,
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           visualizers=VISUALIZERS,
                           resolutions=list(RESOLUTION_PRESETS.keys()))


@app.route("/start_render", methods=["POST"])
def start_render():
    global _active_job

    # One-at-a-time enforcement
    with _active_lock:
        if _active_job is not None:
            existing = jobs.get(_active_job)
            if existing and existing.status in ("queued", "running"):
                return jsonify({"error": "A render is already in progress"}), 409
            _active_job = None

    audio_file = request.files.get("audio")
    if not audio_file or not audio_file.filename:
        return jsonify({"error": "No audio file provided"}), 400

    visualizer = request.form.get("visualizer", "warpfield")
    if visualizer not in VISUALIZERS:
        return jsonify({"error": f"Unknown visualizer: {visualizer}"}), 400

    resolution = request.form.get("resolution", "1080p")
    if resolution not in RESOLUTION_PRESETS:
        return jsonify({"error": f"Unknown resolution: {resolution}"}), 400

    try:
        duration = int(request.form.get("duration", "0"))
        if duration < 0:
            raise ValueError
    except ValueError:
        return jsonify({"error": "duration must be a non-negative integer"}), 400

    job_id  = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    ext        = os.path.splitext(audio_file.filename)[1] or ".mp3"
    audio_path = os.path.join(job_dir, f"audio{ext}")
    audio_file.save(audio_path)

    job = RenderJob(
        job_id=job_id,
        job_dir=job_dir,
        audio_path=audio_path,
        visualizer=visualizer,
        resolution=resolution,
        duration=duration,
    )

    with _jobs_lock:
        jobs[job_id] = job
    with _active_lock:
        _active_job = job_id

    job.start()
    return jsonify({"job_id": job_id}), 202


@app.route("/job_status/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(_job_dict(job))


@app.route("/job_log/<job_id>")
def job_log(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    try:
        with open(job.log_path, "r", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    return app.response_class(text, mimetype="text/plain")


@app.route("/cancel/<job_id>", methods=["POST"])
def cancel_job(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    if job.status not in ("queued", "running"):
        return jsonify({"error": "Job is not running"}), 409
    job.status = "cancelled"
    job.cancel()
    return jsonify({"status": "cancelled"})


@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"error": "Unknown job"}), 404
    if job.status != "done":
        return jsonify({"error": "Job not complete"}), 409
    if not os.path.isfile(job.output_path):
        return jsonify({"error": "Output file missing"}), 500

    out_path = job.output_path
    job_dir  = job.job_dir
    fname    = f"abstrakt_{job_id[:8]}.mp4"
    filesize = os.path.getsize(out_path)

    def _stream():
        try:
            with open(out_path, "rb") as fh:
                while True:
                    chunk = fh.read(65536)
                    if not chunk:
                        break
                    yield chunk
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)
            with _jobs_lock:
                jobs.pop(job_id, None)

    headers = {
        "Content-Disposition": f'attachment; filename="{fname}"',
        "Content-Length": str(filesize),
        "Content-Type": "video/mp4",
    }
    return Response(_stream(), status=200, headers=headers)


@app.route("/server_load")
def server_load():
    return jsonify({
        "cpu_percent": psutil.cpu_percent(interval=None),
        "mem_percent": psutil.virtual_memory().percent,
    })


@app.errorhandler(413)
def request_too_large(_e):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)"}), 413


# ── Startup validation ────────────────────────────────────────────────────────
def _startup_check():
    errors = []
    if not os.path.isfile(ABSTRAKT_SH):
        errors.append(f"abstrakt.sh not found: {ABSTRAKT_SH}")
    elif not os.access(ABSTRAKT_SH, os.X_OK):
        errors.append(f"abstrakt.sh not executable (run: chmod +x {ABSTRAKT_SH})")

    pygame_project  = os.environ.get(
        "PYGAME_PROJECT", os.path.join(os.environ.get("HOME", ""), "pygame-eq-visualizer"))
    kaleido_project = os.environ.get(
        "KALEIDO_PROJECT", os.path.join(os.environ.get("HOME", ""), "kaleido-video-generator"))

    if not os.path.isdir(pygame_project):
        errors.append(f"PYGAME_PROJECT not found: {pygame_project}")
    if not os.path.isdir(kaleido_project):
        errors.append(f"KALEIDO_PROJECT not found: {kaleido_project}")

    if errors:
        print("[abstrakt-webui] Startup validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    os.makedirs(JOBS_DIR, exist_ok=True)
    _startup_check()
    app.run(host="127.0.0.1", port=5000, threaded=True)
