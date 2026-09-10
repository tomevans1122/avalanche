"""Local dashboard for starting and monitoring research training runs.

This intentionally does not expose a public network service or a terrain-safety
decision. Bind it to localhost and use it to inspect research experiments.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024
_lock = threading.Lock()
_run: dict[str, object] = {"process": None, "started": None, "output": None, "error": None}


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _run_status() -> dict[str, object]:
    with _lock:
        process = _run["process"]
        running = isinstance(process, subprocess.Popen) and process.poll() is None
        output = Path(_run["output"]) if _run["output"] else None
        log = output / "train.log" if output else None
        log_text = log.read_text(errors="replace")[-6000:] if log and log.exists() else ""
        return {
            "running": running,
            "return_code": process.poll() if isinstance(process, subprocess.Popen) else None,
            "started": _run["started"],
            "output": str(output.relative_to(PROJECT_ROOT)) if output else None,
            "error": _run["error"],
            "log": log_text,
        }


@app.get("/")
def index():
    return render_template("index.html", status=_run_status())


@app.get("/api/status")
def status():
    return jsonify(_run_status())


@app.post("/api/train")
def start_training():
    manifest_value = request.form.get("manifest", "data/manifest.csv").strip()
    try:
        horizon = float(request.form.get("horizon_hours", "6"))
        if not 0 < horizon <= 72:
            raise ValueError("Forecast horizon must be between 0 and 72 hours.")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    manifest = (PROJECT_ROOT / manifest_value).resolve()
    if not _inside_project(manifest) or not manifest.is_file():
        return jsonify(error="Manifest must be an existing file inside this project."), 400

    with _lock:
        process = _run["process"]
        if isinstance(process, subprocess.Popen) and process.poll() is None:
            return jsonify(error="A training run is already active."), 409
        RUNS_ROOT.mkdir(exist_ok=True)
        output = RUNS_ROOT / f"web-{time.strftime('%Y%m%d-%H%M%S')}"
        output.mkdir()
        log_path = output / "train.log"
        command = [
            sys.executable, "-m", "avalanche_forecast.train",
            "--manifest", str(manifest), "--output", str(output),
            "--horizon-hours", str(horizon),
        ]
        environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        log_file = log_path.open("w")
        _run.update(
            process=subprocess.Popen(
                command, cwd=PROJECT_ROOT, env=environment,
                stdout=log_file, stderr=subprocess.STDOUT, text=True,
            ),
            started=time.strftime("%Y-%m-%d %H:%M:%S"), output=output, error=None,
        )
    return jsonify(_run_status()), 202


def main() -> None:
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
