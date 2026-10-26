"""Local dashboard for starting and monitoring research training runs.

This intentionally does not expose a public network service or a terrain-safety
decision. Bind it to localhost and use it to inspect research experiments.
"""

import os
import subprocess
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from .official_report import fetch_official_report, pilot_region


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = PROJECT_ROOT / "runs"
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024
_lock = threading.Lock()
_run: dict[str, object] = {"process": None, "started": None, "output": None, "error": None}
_results_cache: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}

DEFAULT_PU_MANIFEST = "data/lyngen-frost-sentinel-pu-manifest.csv"
DEFAULT_PU_CHECKPOINT = "runs/lyngen-frost-pu/multimodal-pu-model.pt"


def _inside_project(path: Path) -> bool:
    try:
        path.resolve().relative_to(PROJECT_ROOT)
        return True
    except ValueError:
        return False


def _project_file(value: str) -> Path:
    """Resolve a dashboard input while preventing paths outside this checkout."""
    path = (PROJECT_ROOT / value).resolve()
    if not _inside_project(path) or not path.is_file():
        raise ValueError("File must be an existing file inside this project.")
    return path


def _tile_lookup() -> tuple[dict[str, tuple[float, float, float, float]], list[float]]:
    """Return fixed Sentinel tile footprints for the configured pilot envelope."""
    from .data_pipeline import generate_tiles, load_config

    config = load_config()
    tiles = {tile.tile_id: tile.bbox_wgs84 for tile in generate_tiles(config)}
    return tiles, list(config["bbox_wgs84"])


def _score_pu_sample(checkpoint: Path, sample_path: Path) -> tuple[float, dict[str, Any]]:
    """Run one PU checkpoint on one matched imagery/weather sequence on CPU.

    The sigmoid is exposed only as a model ranking score. PU training has no
    confirmed-negative calibration set, so it must never be presented as an
    avalanche probability.
    """
    import torch

    from .model import TemporalResNet50

    with np.load(sample_path) as sample:
        imagery = sample["imagery"].astype(np.float32, copy=False)
        meteo = sample["meteo"].astype(np.float32, copy=False)
    if imagery.ndim != 4 or meteo.ndim != 2 or imagery.shape[0] != meteo.shape[0]:
        raise ValueError(f"Invalid imagery/weather sequence in {sample_path.name}.")
    if not np.isfinite(imagery).all() or not np.isfinite(meteo).all():
        raise ValueError(f"Non-finite imagery/weather value in {sample_path.name}.")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TemporalResNet50(imagery.shape[1], meteo.shape[1], pretrained=False)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        logit = model(torch.from_numpy(imagery).unsqueeze(0), torch.from_numpy(meteo).unsqueeze(0))
        score = float(torch.sigmoid(logit).item())
    return score, {
        "sequence_length": int(imagery.shape[0]),
        "imagery_shape": list(imagery.shape),
        "meteo_last_step": [round(float(value), 4) for value in meteo[-1]],
    }


def _sar_preview_png(sample_path: Path) -> BytesIO:
    """Render the last preprocessed Sentinel-1 frame as a labelled SAR reference.

    The model receives VV dB, VH dB and a validity mask. This deliberately
    renders the VV backscatter band only, retaining invalid pixels in black,
    rather than inventing a natural-colour satellite image from radar data.
    """
    from PIL import Image

    with np.load(sample_path) as sample:
        imagery = sample["imagery"].astype(np.float32, copy=False)
    if imagery.ndim != 4 or imagery.shape[1] < 1:
        raise ValueError(f"Invalid imagery sequence in {sample_path.name}.")
    frame = imagery[-1, 0]
    valid = imagery[-1, 2] > .5 if imagery.shape[1] >= 3 else np.isfinite(frame)
    values = frame[valid & np.isfinite(frame)]
    if not len(values):
        raise ValueError(f"No valid SAR pixels in {sample_path.name}.")
    low, high = np.percentile(values, (2, 98))
    if high <= low:
        high = low + 1
    scaled = np.clip((frame - low) / (high - low) * 255, 0, 255).astype(np.uint8)
    scaled[~valid] = 0
    image = Image.fromarray(scaled, mode="L").resize((512, 512), Image.Resampling.NEAREST)
    payload = BytesIO()
    image.save(payload, format="PNG", optimize=True)
    payload.seek(0)
    return payload


def _map_results(manifest: Path, checkpoint: Path, limit: int) -> dict[str, Any]:
    """Score the newest distinct fixed tiles and retain their true footprints."""
    cache_key = (str(manifest), manifest.stat().st_mtime_ns, str(checkpoint), checkpoint.stat().st_mtime_ns, limit)
    if cached := _results_cache.get(cache_key):
        return cached

    rows = []
    with manifest.open(newline="") as file:
        import csv
        reader = csv.DictReader(file)
        required = {"path", "target", "tile_id", "analysis_time"}
        if reader.fieldnames is None or required.difference(reader.fieldnames):
            raise ValueError(f"Manifest must contain {sorted(required)}.")
        rows = list(reader)
    tiles, region_bbox = _tile_lookup()
    # One current result per footprint keeps overlapping history legible. ISO UTC
    # timestamps sort lexicographically, and newer rows win when timestamps tie.
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item["analysis_time"], reverse=True):
        if row["tile_id"] in tiles and row["tile_id"] not in seen:
            selected.append(row)
            seen.add(row["tile_id"])
        if len(selected) >= limit:
            break

    features = []
    for row in selected:
        sample_path = _project_file(row["path"])
        score, input_summary = _score_pu_sample(checkpoint, sample_path)
        features.append({
            "tile_id": row["tile_id"], "bbox_wgs84": list(tiles[row["tile_id"]]),
            "analysis_time": row["analysis_time"], "target": row["target"],
            "sample_path": str(sample_path.relative_to(PROJECT_ROOT)),
            "score": round(score, 4), "input": input_summary,
        })
    response = {
        "region": pilot_region(), "region_bbox_wgs84": region_bbox, "features": features,
        "grid_bboxes_wgs84": [list(tile_bbox) for tile_bbox in tiles.values()],
        "available_manifest_rows": len(rows), "displayed_tiles": len(features),
        "warning": "Uncalibrated PU report-likelihood research score; not an avalanche probability or safety advice.",
        "coverage_note": "Each imagery patch is the final Sentinel-1 frame for that tile's own sample time. The reference grid is not a single-date mosaic; blank cells have no model result.",
    }
    _results_cache.clear()
    _results_cache[cache_key] = response
    return response


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
            "model_region": pilot_region(),
            "model_result": (
                "Selected Sentinel tiles can be scored as uncalibrated PU research results in the map below."
                if (PROJECT_ROOT / DEFAULT_PU_CHECKPOINT).is_file() else
                ("A trained checkpoint exists, but no calibrated inference input is available."
                 if output else "No trained model result is available.")
            ),
        }


@app.get("/")
def index():
    return render_template("index.html", status=_run_status())


@app.get("/api/status")
def status():
    return jsonify({**_run_status(), "official_report": fetch_official_report()})


@app.get("/api/results")
def results():
    """Return geographic, selected-tile PU model results for the local map."""
    try:
        manifest = _project_file(request.args.get("manifest", DEFAULT_PU_MANIFEST))
        checkpoint = _project_file(request.args.get("checkpoint", DEFAULT_PU_CHECKPOINT))
        limit = int(request.args.get("limit", "12"))
        if not 1 <= limit <= 24:
            raise ValueError("Result tile limit must be between 1 and 24.")
        return jsonify(_map_results(manifest, checkpoint, limit))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/sar-preview")
def sar_preview():
    """Serve the final processed Sentinel-1 frame for a mapped model input."""
    try:
        sample = _project_file(request.args.get("sample", ""))
        if sample.suffix != ".npz":
            raise ValueError("Satellite preview must be generated from an NPZ input sample.")
        return send_file(_sar_preview_png(sample), mimetype="image/png", max_age=3600)
    except (OSError, ValueError, KeyError) as exc:
        return jsonify(error=str(exc)), 400


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
