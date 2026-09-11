"""Auditable preparation of a single-region Sentinel-1 avalanche dataset.

Raw event labels are intentionally not inferred from a missing report. A sample
without an explicit negative observation is written to the unknown audit, not
to the training manifest.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pyproj import Transformer
from shapely.geometry import box, shape


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
WEATHER_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
REGOBS_MAP_URL = "https://gis4.nve.no/map/rest/services/Mapservices/regobs2/MapServer"
_CDSE_TOKEN: str | None = None
_CDSE_TOKEN_EXPIRES_AT: datetime | None = None


@dataclass(frozen=True)
class Tile:
    tile_id: str
    bbox_wgs84: tuple[float, float, float, float]
    bbox_projected: tuple[float, float, float, float]


def load_config(path: str | Path = ROOT / "config" / "pilot-study.json") -> dict[str, Any]:
    with Path(path).open() as file:
        return json.load(file)


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def generate_tiles(config: dict[str, Any]) -> list[Tile]:
    """Create fixed-size UTM terrain tiles covering the configured WGS84 bbox."""
    west, south, east, north = config["bbox_wgs84"]
    crs = config["terrain_crs"]
    to_utm = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    min_x, min_y = to_utm.transform(west, south)
    max_x, max_y = to_utm.transform(east, north)
    side = config["tile_resolution_metres"] * config["tile_size_pixels"]
    tiles = []
    y, row = min_y, 0
    while y < max_y:
        x, col = min_x, 0
        while x < max_x:
            x2, y2 = min(x + side, max_x), min(y + side, max_y)
            lon1, lat1 = to_wgs.transform(x, y)
            lon2, lat2 = to_wgs.transform(x2, y2)
            tiles.append(Tile(f"r{row:03d}c{col:03d}", (lon1, lat1, lon2, lat2), (x, y, x2, y2)))
            x, col = x + side, col + 1
        y, row = y + side, row + 1
    return tiles


def search_sentinel1(config: dict[str, Any], limit: int = 500, max_pages: int = 10) -> dict[str, Any]:
    """Query the public CDSE STAC catalogue for VV/VH Sentinel-1 GRD scenes."""
    payload = {
        "collections": [config["primary_collection"]], "bbox": config["bbox_wgs84"],
        "datetime": config["date_range"], "limit": limit,
        "query": {"sar:polarizations": {"eq": ["VV", "VH"]}, "sar:instrument_mode": {"eq": "IW"}},
        "sortby": [{"field": "datetime", "direction": "asc"}],
    }
    features, request_body = [], payload
    for _ in range(max_pages):
        response = requests.post(STAC_URL, json=request_body, timeout=60)
        response.raise_for_status()
        page = response.json()
        features.extend(page.get("features", []))
        next_links = [link for link in page.get("links", []) if link.get("rel") == "next" and link.get("method", "GET") == "POST"]
        if not next_links:
            break
        request_body = next_links[0].get("body")
        if not request_body:
            break
    return {"type": "FeatureCollection", "features": features, "pages_requested": max_pages, "query": payload}


def cdse_token() -> str:
    global _CDSE_TOKEN, _CDSE_TOKEN_EXPIRES_AT
    if _CDSE_TOKEN and _CDSE_TOKEN_EXPIRES_AT and datetime.now(timezone.utc) < _CDSE_TOKEN_EXPIRES_AT:
        return _CDSE_TOKEN
    client_id, secret = os.getenv("CDSE_CLIENT_ID"), os.getenv("CDSE_CLIENT_SECRET")
    if not client_id or not secret:
        raise RuntimeError("Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET; do not store them in the manifest.")
    response = requests.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": client_id, "client_secret": secret}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    _CDSE_TOKEN = payload["access_token"]
    _CDSE_TOKEN_EXPIRES_AT = datetime.now(timezone.utc) + timedelta(seconds=max(60, int(payload.get("expires_in", 300)) - 60))
    return _CDSE_TOKEN


def download_processed_tile(tile: Tile, acquisition_time: str, destination: Path, resolution: int) -> None:
    """Download terrain-corrected Gamma0 VV/VH GeoTIFF via CDSE Process API.

    This requests orthorectification and terrain backscatter before local speckle
    filtering. Keep the response provenance beside the output; do not substitute
    this for quality control of layover/shadow terrain.
    """
    moment = utc(acquisition_time)
    start, end = moment - timedelta(minutes=1), moment + timedelta(minutes=1)
    evalscript = """//VERSION=3
function setup() { return {input: [\"VV\", \"VH\", \"dataMask\", \"shadowMask\"], output: {bands: 4, sampleType: \"FLOAT32\"}}; }
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask, s.shadowMask]; }"""
    payload = {
        "input": {"bounds": {"bbox": list(tile.bbox_wgs84), "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
                  "data": [{"type": "sentinel-1-grd", "dataFilter": {"timeRange": {"from": start.isoformat().replace("+00:00", "Z"), "to": end.isoformat().replace("+00:00", "Z")}, "acquisitionMode": "IW", "polarization": "DV"}, "processing": {"orthorectify": True, "demInstance": "COPERNICUS", "backCoeff": "GAMMA0_TERRAIN"}}]},
        "output": {"width": 128, "height": 128, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]}, "evalscript": evalscript,
    }
    response = requests.post(PROCESS_URL, json=payload, headers={"Authorization": f"Bearer {cdse_token()}"}, timeout=180)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    destination.with_suffix(".json").write_text(json.dumps({"tile": asdict(tile), "acquisition_time": acquisition_time, "request": payload, "sha256": sha256(destination)}, indent=2))


def boxcar(array: np.ndarray) -> np.ndarray:
    """NaN-aware 3x3 filter; SAR-specific calibration remains upstream."""
    padded = np.pad(array, ((1, 1), (1, 1)), mode="constant", constant_values=np.nan)
    windows = np.stack([padded[i:i + array.shape[0], j:j + array.shape[1]] for i in range(3) for j in range(3)])
    return np.nanmean(windows, axis=0)


def sentinel_quality(path: Path) -> dict[str, float]:
    """Calculate explicit source-data and terrain-shadow rejection fractions."""
    try:
        import rasterio
    except ImportError as exc:
        raise RuntimeError("Install rasterio to preprocess GeoTIFFs.") from exc
    with rasterio.open(path) as source:
        if source.count != 4:
            raise ValueError(f"{path} must contain VV, VH, dataMask and shadowMask bands")
        values = source.read().astype(np.float32)
    valid_source = values[2] >= .5
    # CDSE Sentinel-1 defines 0 as likely not radar shadow and 1 as shadow.
    valid_terrain = values[3] < .5
    valid_backscatter = np.isfinite(values[:2]).all(axis=0) & (values[:2] > 0).all(axis=0)
    valid = valid_source & valid_terrain & valid_backscatter
    return {"valid_fraction": float(valid.mean()), "no_data_fraction": float((~valid_source).mean()),
            "shadow_fraction": float((~valid_terrain).mean()), "invalid_backscatter_fraction": float((~valid_backscatter).mean())}


def read_preprocessed_tiff(path: Path, minimum_valid_fraction: float = .99) -> np.ndarray:
    """Return VV/VH dB plus an explicit validity channel after quality gating."""
    quality = sentinel_quality(path)
    if quality["valid_fraction"] < minimum_valid_fraction:
        raise ValueError(f"{path} valid fraction {quality['valid_fraction']:.3f} is below {minimum_valid_fraction:.3f}; {quality}")
    import rasterio
    with rasterio.open(path) as source:
        raw = source.read().astype(np.float32)
    valid = ((raw[2] >= .5) & (raw[3] < .5) & np.isfinite(raw[:2]).all(axis=0) & (raw[:2] > 0).all(axis=0))
    values = raw[:2].copy()
    values[:, ~valid] = np.nan
    values = 10 * np.log10(values)
    values = np.stack([boxcar(band) for band in values])
    if not np.isfinite(values[:, valid]).all():
        raise ValueError(f"{path} contains invalid backscatter after filtering")
    # Invalid values are zeroed only alongside this explicit mask channel; they
    # are never indistinguishable from valid backscatter to the CNN.
    values[:, ~valid] = 0
    return np.concatenate([values, valid[np.newaxis].astype(np.float32)])


def fetch_hourly_weather(tile: Tile, start: datetime, end: datetime, destination: Path) -> None:
    """Acquire hourly historical weather at a tile centre, with source metadata."""
    west, south, east, north = tile.bbox_wgs84
    params = {"latitude": (south + north) / 2, "longitude": (west + east) / 2,
              "start_date": start.date().isoformat(), "end_date": end.date().isoformat(), "timezone": "UTC",
              "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m,wind_direction_10m"}
    response = requests.get(WEATHER_ARCHIVE_URL, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload["hourly"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    destination.with_suffix(".json").write_text(json.dumps({"provider": "Open-Meteo archive", "request": params, "response_metadata": {k: v for k, v in payload.items() if k != "hourly"}}, indent=2))


def _regobs_time(properties: dict[str, Any]) -> str | None:
    """Resolve the published observation timestamp across MapServer field variants."""
    by_lower = {str(key).lower(): value for key, value in properties.items()}
    for key in ("dtobstime", "obstime", "regdate", "registrationtime", "date"):
        value = by_lower.get(key)
        if value:
            if isinstance(value, (int, float)):
                return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
            return str(value)
    return None


def regobs_where(config: dict[str, Any]) -> str:
    """Build a bounded, server-side date and warning-region filter."""
    start, end = (utc(value).strftime("%Y-%m-%d %H:%M:%S") for value in config["date_range"].split("/"))
    region = config["regobs_region_name"].replace("'", "''")
    return f"ForecastRegionName = '{region}' AND DtObsTime >= DATE '{start}' AND DtObsTime <= DATE '{end}'"


def fetch_regobs_raw(config: dict[str, Any], output: Path) -> None:
    """Download raw RegObs avalanche reports as review-required GeoJSON.

    Layers 4 (individual avalanche observations) and 16 (avalanche activity)
    are queried by region bbox. Their data are public, but reports are not
    elevated to confirmed events automatically: manual source review is needed
    before they may become positive model labels.
    """
    bbox, where = ",".join(str(value) for value in config["bbox_wgs84"]), regobs_where(config)
    collected = []
    for layer in (4, 16):
        offset = 0
        while True:
            params = {"where": where, "geometry": bbox, "geometryType": "esriGeometryEnvelope", "inSR": 4326, "outSR": 4326, "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "returnGeometry": "true", "f": "geojson", "resultOffset": offset, "resultRecordCount": 1000, "orderByFields": "DtObsTime ASC"}
            response = requests.get(f"{REGOBS_MAP_URL}/{layer}/query", params=params, timeout=60)
            response.raise_for_status()
            page = response.json()
            features = page.get("features", [])
            for feature in features:
                source = feature.get("properties", {})
                when = _regobs_time(source)
                if not when or not feature.get("geometry"):
                    continue
                collected.append({"type": "Feature", "geometry": feature["geometry"], "properties": {"kind": "event", "status": "reported_unreviewed", "event_time": when, "source": "NVE RegObs", "source_layer": layer, "source_properties": source, "attribution": "RegObs / original observer; review required before training"}})
            if len(features) < 1000:
                break
            offset += len(features)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"type": "FeatureCollection", "name": "NVE RegObs raw avalanche reports", "licence": "NLOD / CC BY 3.0 Norway compatible; retain RegObs and observer attribution", "features": collected}, indent=2))


def screen_regobs_proxy(raw_inventory: dict[str, Any], config: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Produce proxy positives using published report completeness, not truth claims.

    The rules are a transparent research convenience. They never convert a
    report to `confirmed`; the target remains a high-confidence *report*.
    """
    rules = config["regobs_proxy_rules"]
    allowed = set(rules["allowed_competence_levels"])
    required = rules["require_avalanche_attributes"]
    accepted: dict[str, dict[str, Any]] = {}
    rejected = []
    for feature in raw_inventory.get("features", []):
        props = feature.get("properties", {})
        source = props.get("source_properties", {})
        reg_id = str(source.get(rules["deduplicate_by"], ""))
        reasons = []
        if not feature.get("geometry") or not props.get("event_time"):
            reasons.append("missing_geometry_or_time")
        if source.get("CompetenceLevelName") not in allowed:
            reasons.append("competence_not_in_configured_proxy_levels")
        if any(not source.get(field) for field in required):
            reasons.append("missing_required_avalanche_attribute")
        if not reg_id:
            reasons.append("missing_registration_id")
        if reasons:
            rejected.append({"regobs_id": reg_id or None, "reason": ";".join(reasons)})
            continue
        candidate = {**feature, "properties": {**props, "status": "reported_high_confidence", "label_semantics": "proxy_positive: high-quality RegObs report; not independently confirmed avalanche", "proxy_rules": rules}}
        # One registration may have several activity rows. Keep one point and
        # retain all row numbers for provenance rather than multiplying labels.
        if reg_id in accepted:
            accepted[reg_id]["properties"].setdefault("source_rows", []).append(source.get("Radnummer"))
        else:
            candidate["properties"]["source_rows"] = [source.get("Radnummer")]
            accepted[reg_id] = candidate
    return {"type": "FeatureCollection", "name": "NVE RegObs auto-screened proxy observations", "label_target": config["label_target"], "features": list(accepted.values())}, rejected


def align_weather(weather: pd.DataFrame, timestamps: list[datetime]) -> np.ndarray:
    weather = weather.copy()
    weather["time"] = pd.to_datetime(weather["time"], utc=True)
    columns = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "wind_direction_10m"]
    rows = []
    for timestamp in timestamps:
        preceding = weather.loc[weather["time"] <= pd.Timestamp(timestamp)]
        if preceding.empty:
            raise ValueError(f"weather has no observation at or before {timestamp.isoformat()}")
        values = preceding.iloc[-1][columns].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("weather source contains missing values")
        rows.append(values)
    return np.stack(rows)


def label_tile(inventory: dict[str, Any], tile: Tile, analysis_time: datetime, horizon_hours: float) -> tuple[int | None, str]:
    """Return positive, explicit negative, or unknown for a tile/time horizon."""
    tile_geometry = box(*tile.bbox_wgs84)
    end = analysis_time + timedelta(hours=horizon_hours)
    negative_coverage = False
    for feature in inventory.get("features", []):
        properties, geometry = feature.get("properties", {}), feature.get("geometry")
        if not geometry or not tile_geometry.intersects(shape(geometry)):
            continue
        kind = properties.get("kind")
        if kind == "event" and properties.get("status") in {"confirmed", "reported_high_confidence"}:
            when = utc(properties["event_time"])
            if analysis_time < when <= end:
                reason = "confirmed_event" if properties.get("status") == "confirmed" else "proxy_reported_high_confidence"
                return 1, reason
        if kind == "coverage" and properties.get("negative_observation") is True:
            observed_from, observed_to = utc(properties["observed_from"]), utc(properties["observed_to"])
            negative_coverage |= observed_from <= analysis_time and observed_to >= end
    return (0, "explicit_negative_coverage") if negative_coverage else (None, "unknown_no_negative_coverage")


def split_groups(rows: list[dict[str, Any]], train_fraction: float = .70, validation_fraction: float = .15) -> None:
    """Chronologically hold out whole storm groups; never random-split tiles."""
    groups = sorted({row["storm_group"] for row in rows})
    train_end, validation_end = int(len(groups) * train_fraction), int(len(groups) * (train_fraction + validation_fraction))
    assignment = {group: "train" if i < train_end else "validation" if i < validation_end else "test" for i, group in enumerate(groups)}
    for row in rows:
        row["split"] = assignment[row["storm_group"]]


def build_manifest(config: dict[str, Any], inventory_path: Path, imagery_root: Path, weather_root: Path, output: Path) -> None:
    """Create NPZ samples, manifest, unknown audit, and immutable provenance."""
    inventory = json.loads(inventory_path.read_text())
    tiles = {tile.tile_id: tile for tile in generate_tiles(config)}
    rows, unknown = [], []
    for tile_id, tile in tiles.items():
        frames = sorted((imagery_root / tile_id).glob("*.tif"))
        if len(frames) < config["sequence_length"]:
            continue
        weather_path = weather_root / f"{tile_id}.csv"
        if not weather_path.exists():
            continue
        weather = pd.read_csv(weather_path)
        for index in range(config["sequence_length"] - 1, len(frames)):
            selected = frames[index + 1 - config["sequence_length"]:index + 1]
            timestamps = [utc(path.stem) for path in selected]
            label, reason = label_tile(inventory, tile, timestamps[-1], config["analysis_horizon_hours"])
            if label is None:
                unknown.append({"tile_id": tile_id, "analysis_time": timestamps[-1].isoformat(), "reason": reason})
                continue
            imagery = np.stack([read_preprocessed_tiff(path) for path in selected])
            meteo = align_weather(weather, timestamps)
            sample_path = output.parent / "samples" / f"{tile_id}_{timestamps[-1].strftime('%Y%m%dT%H%M%S')}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sample_path, imagery=imagery, meteo=meteo, label=np.int64(label))
            storm = timestamps[-1].date() - timedelta(days=timestamps[-1].weekday() % config["storm_group_days"])
            rows.append({"path": str(sample_path.relative_to(ROOT)), "tile_id": tile_id, "analysis_time": timestamps[-1].isoformat(), "label": label, "storm_group": storm.isoformat()})
    split_groups(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["path", "split", "tile_id", "analysis_time", "label", "storm_group"]).to_csv(output, index=False)
    pd.DataFrame(unknown, columns=["tile_id", "analysis_time", "reason"]).to_csv(output.with_name("unknown-samples.csv"), index=False)
    output.with_name("provenance.json").write_text(json.dumps({"config": config, "config_sha256": sha256(ROOT / "config" / "pilot-study.json"), "event_inventory": str(inventory_path), "event_inventory_sha256": sha256(inventory_path), "code_revision": git_revision(), "created_at": datetime.now(timezone.utc).isoformat(), "label_target": config.get("label_target", "confirmed_avalanche_event_within_horizon"), "label_policy": "positive only for confirmed events or explicitly documented RegObs proxy positives; negative only for explicit negative coverage; all other samples audited as unknown"}, indent=2))


def _weather_window(weather: pd.DataFrame, analysis_time: datetime, hours: int) -> np.ndarray | None:
    frame = weather if isinstance(weather.index, pd.DatetimeIndex) else weather.assign(time=pd.to_datetime(weather["time"], utc=True)).set_index("time")
    wanted = pd.date_range(pd.Timestamp(analysis_time) - pd.Timedelta(hours=hours - 1), pd.Timestamp(analysis_time), freq="h")
    selected = frame.reindex(wanted)
    columns = ["temperature_2m", "precipitation", "snowfall", "wind_speed_10m", "wind_direction_10m"]
    if selected[columns].isna().any().any():
        return None
    return selected[columns].to_numpy(dtype=np.float32)


def build_weather_proxy_manifest(config: dict[str, Any], inventory_path: Path, weather_root: Path, output: Path) -> None:
    """Build positive/unlabeled weather samples without fabricating negatives."""
    inventory, tiles = json.loads(inventory_path.read_text()), generate_tiles(config)
    cache = {tile.tile_id: pd.read_csv(weather_root / f"{tile.tile_id}.csv", parse_dates=["time"]).assign(time=lambda frame: pd.to_datetime(frame["time"], utc=True)).set_index("time") for tile in tiles if (weather_root / f"{tile.tile_id}.csv").exists()}
    positives, reported_times = [], []
    for feature in inventory.get("features", []):
        if feature.get("properties", {}).get("status") != "reported_high_confidence" or not feature.get("geometry"):
            continue
        point = shape(feature["geometry"])
        tile = next((candidate for candidate in tiles if box(*candidate.bbox_wgs84).intersects(point)), None)
        if tile is None or tile.tile_id not in cache:
            continue
        reported_at = utc(feature["properties"]["event_time"])
        # Hourly reanalysis cannot support a sub-hour cutoff. Floor rather than
        # round up so no weather later than the model analysis time can leak in.
        analysis_time = (reported_at - timedelta(hours=config["analysis_horizon_hours"])).replace(minute=0, second=0, microsecond=0)
        sequence = _weather_window(cache[tile.tile_id], analysis_time, config["weather_history_hours"])
        if sequence is not None:
            positives.append((tile, analysis_time, sequence, str(feature["properties"].get("source_properties", {}).get("RegID", ""))))
            reported_times.append(reported_at)
    if not positives:
        raise RuntimeError("No proxy positives align with hourly weather. Fetch weather before building this manifest.")
    rng, rows = np.random.default_rng(42), []
    sample_root = output.parent / "weather-proxy-samples"; sample_root.mkdir(parents=True, exist_ok=True)
    def write(tile: Tile, analysis_time: datetime, sequence: np.ndarray, target: str, source_id: str = ""):
        path = sample_root / f"{target}_{tile.tile_id}_{analysis_time.strftime('%Y%m%dT%H%M%S')}.npz"
        np.savez_compressed(path, meteo=sequence, label=np.int64(target == "positive"))
        storm = analysis_time.date() - timedelta(days=analysis_time.weekday() % config["storm_group_days"])
        rows.append({"path": str(path.relative_to(ROOT)), "target": target, "tile_id": tile.tile_id, "analysis_time": analysis_time.isoformat(), "storm_group": storm.isoformat(), "source_regobs_id": source_id})
    for tile, time, sequence, source_id in positives:
        write(tile, time, sequence, "positive", source_id)
    exclusion = [event - timedelta(hours=config["analysis_horizon_hours"]) for event in reported_times]
    desired, candidates, attempts = len(positives) * config["unlabeled_to_positive_ratio"], [], 0
    cache_tiles = [next(tile for tile in tiles if tile.tile_id == tile_id) for tile_id in cache]
    # Rejection sample directly from hourly indices. This avoids materialising
    # every possible 72-hour window across multi-year histories.
    while len(candidates) < desired and attempts < desired * 100:
        tile = cache_tiles[int(rng.integers(len(cache_tiles)))]; weather = cache[tile.tile_id]
        raw_time = weather.index[int(rng.integers(config["weather_history_hours"] - 1, len(weather.index)))]
        moment = raw_time.to_pydatetime(); attempts += 1
        if all(abs((moment - event).total_seconds()) > 48 * 3600 for event in exclusion):
            if (sequence := _weather_window(weather, moment, config["weather_history_hours"])) is not None:
                candidates.append((tile, moment, sequence))
    if len(candidates) < desired:
        raise RuntimeError(f"Only {len(candidates)} unlabeled weather windows available after {attempts} draws; need {desired}.")
    for tile, time, sequence in candidates:
        write(tile, time, sequence, "unlabeled")
    split_groups(rows)
    output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(output, index=False)
    output.with_name("weather-proxy-provenance.json").write_text(json.dumps({"target": config["label_target"], "training_method": "positive-unlabeled", "unlabeled_policy": "windows at least 48 hours from a proxy report; not negative labels", "class_prior_assumption": config["pu_assumed_class_prior"], "proxy_inventory_sha256": sha256(inventory_path), "config_sha256": sha256(ROOT / "config" / "pilot-study.json"), "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))


def proxy_event_tiles(config: dict[str, Any], inventory_path: Path) -> list[Tile]:
    """Return only terrain tiles containing proxy-positive report points."""
    tiles, selected = generate_tiles(config), set()
    for feature in json.loads(inventory_path.read_text()).get("features", []):
        if feature.get("properties", {}).get("status") != "reported_high_confidence" or not feature.get("geometry"):
            continue
        point = shape(feature["geometry"])
        selected.update(tile.tile_id for tile in tiles if box(*tile.bbox_wgs84).intersects(point))
    return [tile for tile in tiles if tile.tile_id in selected]


def sentinel_proxy_plan(config: dict[str, Any], inventory_path: Path, catalogue_path: Path,
                        max_examples: int | None = None, history_days: int = 90) -> dict[str, Any]:
    """Select pre-analysis Sentinel-1 sequences for report-proxy examples.

    This is intentionally a plan, rather than a label manifest: satellite
    acquisitions are sparse, so their timestamp must remain distinct from the
    weather/model analysis time.  No acquisition later than that analysis time
    is selected.
    """
    tiles = generate_tiles(config)
    features = json.loads(catalogue_path.read_text()).get("features", [])
    examples, skipped = [], []
    for feature in sorted(json.loads(inventory_path.read_text()).get("features", []),
                          key=lambda row: row.get("properties", {}).get("event_time", "")):
        properties = feature.get("properties", {})
        if properties.get("status") != "reported_high_confidence" or not feature.get("geometry"):
            continue
        point = shape(feature["geometry"])
        tile = next((candidate for candidate in tiles if box(*candidate.bbox_wgs84).intersects(point)), None)
        if tile is None:
            skipped.append({"reason": "outside_tile_grid", "event_time": properties.get("event_time")})
            continue
        analysis_time = utc(properties["event_time"]) - timedelta(hours=config["analysis_horizon_hours"])
        earliest = analysis_time - timedelta(days=history_days)
        tile_geometry = box(*tile.bbox_wgs84)
        preceding = sorted({utc(scene["properties"]["datetime"]) for scene in features
                            if scene.get("properties", {}).get("datetime") and scene.get("geometry")
                            and tile_geometry.intersects(shape(scene["geometry"]))
                            and earliest <= utc(scene["properties"]["datetime"]) <= analysis_time})
        if len(preceding) < config["sequence_length"]:
            skipped.append({"reason": "insufficient_pre_analysis_scenes", "event_time": properties["event_time"], "tile_id": tile.tile_id})
            continue
        examples.append({
            "source_regobs_id": str(properties.get("source_properties", {}).get("RegID", "")),
            "tile_id": tile.tile_id,
            "event_time": properties["event_time"],
            "analysis_time": analysis_time.isoformat().replace("+00:00", "Z"),
            "acquisitions": [moment.isoformat().replace("+00:00", "Z") for moment in preceding[-config["sequence_length"]:]],
            "candidate_acquisitions": [moment.isoformat().replace("+00:00", "Z") for moment in preceding],
            "label_semantics": properties.get("label_semantics"),
        })
        if max_examples is not None and len(examples) >= max_examples:
            break
    return {"purpose": "pre-analysis Sentinel-1 sequence plan; not a training manifest", "sequence_length": config["sequence_length"], "history_days": history_days, "examples": examples, "skipped": skipped}


def select_quality_approved_sentinel_sequences(config: dict[str, Any], plan_path: Path,
                                                imagery_root: Path, max_attempts: int = 24,
                                                start: int = 0, max_examples: int | None = None,
                                                checkpoint_path: Path | None = None) -> dict[str, Any]:
    """Download only enough candidate frames to form quality-approved sequences.

    Each rejection is retained in the output audit.  This avoids treating a
    catalogue hit as usable imagery and never substitutes a post-analysis scene.
    """
    plan = json.loads(plan_path.read_text())
    tiles = {tile.tile_id: tile for tile in generate_tiles(config)}
    selected, rejected = [], []
    candidates = plan.get("examples", [])[start:None if max_examples is None else start + max_examples]
    def result() -> dict[str, Any]:
        return {"purpose": "quality-approved, pre-analysis Sentinel-1 sequences; not a training manifest",
                "minimum_valid_fraction": .99, "sequence_length": config["sequence_length"],
                "source_plan": str(plan_path), "start": start, "requested_examples": len(candidates),
                "examples": selected, "rejected": rejected}
    for example in candidates:
        tile = tiles.get(example["tile_id"])
        if tile is None:
            rejected.append({"source_regobs_id": example.get("source_regobs_id"), "reason": "unknown_tile"})
            continue
        valid_frames, attempts = [], 0
        # Start with the most recent eligible scene; results are re-sorted for
        # temporal model input after enough quality-approved frames are found.
        for acquisition in reversed(example.get("candidate_acquisitions", example.get("acquisitions", []))):
            if attempts >= max_attempts or len(valid_frames) >= config["sequence_length"]:
                break
            attempts += 1
            filename = utc(acquisition).strftime("%Y-%m-%dT%H:%M:%SZ.tif")
            path = imagery_root / tile.tile_id / filename
            try:
                if not path.exists():
                    download_processed_tile(tile, acquisition, path, config["tile_resolution_metres"])
                quality = sentinel_quality(path)
                if quality["valid_fraction"] < .99:
                    rejected.append({"source_regobs_id": example.get("source_regobs_id"), "tile_id": tile.tile_id,
                                     "acquisition": acquisition, "reason": "below_valid_coverage_threshold", "quality": quality})
                    continue
                # Verify the complete three-channel tensor contract now, not at
                # a later training stage.
                read_preprocessed_tiff(path)
                valid_frames.append(acquisition)
            except (OSError, ValueError, requests.RequestException) as exc:
                rejected.append({"source_regobs_id": example.get("source_regobs_id"), "tile_id": tile.tile_id,
                                 "acquisition": acquisition, "reason": "download_or_preprocess_failure", "detail": str(exc)})
        if len(valid_frames) == config["sequence_length"]:
            selected.append({**example, "acquisitions": sorted(valid_frames), "attempts": attempts,
                             "selection_status": "quality_approved"})
        else:
            rejected.append({"source_regobs_id": example.get("source_regobs_id"), "tile_id": tile.tile_id,
                             "reason": "insufficient_quality_approved_frames", "approved_frames": len(valid_frames), "attempts": attempts})
        if checkpoint_path:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(result(), indent=2))
    return result()


def sentinel_unlabeled_plan(config: dict[str, Any], weather_manifest: Path, catalogue_path: Path,
                            history_days: int = 90) -> dict[str, Any]:
    """Plan pre-analysis Sentinel-1 sequences for existing PU unlabeled windows."""
    tiles = {tile.tile_id: tile for tile in generate_tiles(config)}
    scenes = json.loads(catalogue_path.read_text()).get("features", [])
    examples, skipped = [], []
    rows = pd.read_csv(weather_manifest)
    for row in rows.loc[rows.target == "unlabeled"].to_dict("records"):
        tile = tiles.get(row["tile_id"])
        if tile is None:
            skipped.append({"path": row["path"], "reason": "unknown_tile"}); continue
        analysis_time = utc(row["analysis_time"]); earliest = analysis_time - timedelta(days=history_days)
        footprint = box(*tile.bbox_wgs84)
        candidates = sorted({utc(scene["properties"]["datetime"]) for scene in scenes
                             if scene.get("geometry") and scene.get("properties", {}).get("datetime")
                             and footprint.intersects(shape(scene["geometry"]))
                             and earliest <= utc(scene["properties"]["datetime"]) <= analysis_time})
        if len(candidates) < config["sequence_length"]:
            skipped.append({"path": row["path"], "tile_id": tile.tile_id, "reason": "insufficient_pre_analysis_scenes"}); continue
        examples.append({"source_weather_path": row["path"], "target": "unlabeled", "split": row["split"],
                         "tile_id": tile.tile_id, "analysis_time": row["analysis_time"],
                         "candidate_acquisitions": [moment.isoformat().replace("+00:00", "Z") for moment in candidates],
                         "acquisitions": [moment.isoformat().replace("+00:00", "Z") for moment in candidates[-config["sequence_length"]:]]})
    return {"purpose": "pre-analysis Sentinel-1 plan for PU unlabeled examples", "examples": examples,
            "skipped": skipped, "sequence_length": config["sequence_length"], "history_days": history_days}


def build_multimodal_proxy_manifest(config: dict[str, Any], weather_manifest: Path,
                                    positive_selection: Path, unlabeled_selection: Path,
                                    imagery_root: Path, weather_root: Path, output: Path) -> None:
    """Join quality-approved imagery and scene-time weather into PU `.npz` samples."""
    weather_rows = pd.read_csv(weather_manifest)
    def regobs_id(value: Any) -> str:
        if pd.isna(value):
            return ""
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)
    positives_by_id = {regobs_id(row.source_regobs_id): row for row in weather_rows.loc[weather_rows.target == "positive"].itertuples(index=False)}
    rows, omitted = [], []
    sources = [(json.loads(positive_selection.read_text()).get("examples", []), "positive"),
               (json.loads(unlabeled_selection.read_text()).get("examples", []), "unlabeled")]
    for examples, target in sources:
        for example in examples:
            if target == "positive":
                source = positives_by_id.get(regobs_id(example.get("source_regobs_id")))
                if source is None:
                    omitted.append({"reason": "positive_not_in_weather_manifest", "source_regobs_id": example.get("source_regobs_id")}); continue
                split, source_id = source.split, str(source.source_regobs_id)
            else:
                split, source_id = example["split"], ""
            tile_id = example["tile_id"]
            paths = [imagery_root / tile_id / utc(value).strftime("%Y-%m-%dT%H:%M:%SZ.tif") for value in example["acquisitions"]]
            weather_path = weather_root / f"{tile_id}.csv"
            if not weather_path.exists() or not all(path.exists() for path in paths):
                omitted.append({"reason": "missing_imagery_or_weather", "tile_id": tile_id}); continue
            try:
                imagery = np.stack([read_preprocessed_tiff(path) for path in paths])
                timestamps = [utc(value) for value in example["acquisitions"]]
                meteo = align_weather(pd.read_csv(weather_path), timestamps)
            except ValueError as exc:
                omitted.append({"reason": "preprocess_or_alignment_failure", "tile_id": tile_id, "detail": str(exc)}); continue
            sample_path = output.parent / "multimodal-proxy-samples" / f"{target}_{tile_id}_{utc(example['analysis_time']).strftime('%Y%m%dT%H%M%S')}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(sample_path, imagery=imagery, meteo=meteo, label=np.int64(target == "positive"))
            storm = utc(example["analysis_time"]).date() - timedelta(days=utc(example["analysis_time"]).weekday() % config["storm_group_days"])
            rows.append({"path": str(sample_path.relative_to(ROOT)), "split": split, "target": target, "tile_id": tile_id,
                         "analysis_time": example["analysis_time"], "storm_group": storm.isoformat(), "source_regobs_id": source_id})
    pd.DataFrame(rows).to_csv(output, index=False)
    output.with_name("multimodal-proxy-omitted.csv").write_text(pd.DataFrame(omitted).to_csv(index=False))
    output.with_name("multimodal-proxy-provenance.json").write_text(json.dumps({"training_method": "positive-unlabeled",
        "warning": "Unlabeled rows are not negatives; do not use train.py BCE training.", "positive_selection": str(positive_selection),
        "unlabeled_selection": str(unlabeled_selection), "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    catalogue = sub.add_parser("catalogue"); catalogue.add_argument("--output", default="data/catalogue/sentinel-1-items.json"); catalogue.add_argument("--limit", type=int, default=500); catalogue.add_argument("--pages", type=int, default=10)
    tiles = sub.add_parser("tiles"); tiles.add_argument("--output", default="data/tiles.json")
    weather = sub.add_parser("weather"); weather.add_argument("--output", default="data/weather"); weather.add_argument("--proxy-inventory", help="fetch only tiles containing proxy-positive reports")
    download = sub.add_parser("download"); download.add_argument("--catalogue", default="data/catalogue/sentinel-1-items.json"); download.add_argument("--output", default="data/processed/sentinel-1"); download.add_argument("--max-scenes", required=True, type=int)
    build = sub.add_parser("build-manifest"); build.add_argument("--inventory", required=True); build.add_argument("--imagery-root", default="data/processed/sentinel-1"); build.add_argument("--weather-root", default="data/weather"); build.add_argument("--output", default="data/manifest.csv")
    regobs = sub.add_parser("regobs"); regobs.add_argument("--output", default="data/inventory/regobs-raw.geojson")
    screen = sub.add_parser("regobs-screen"); screen.add_argument("--input", default="data/inventory/regobs-raw.geojson"); screen.add_argument("--output", default="data/inventory/lyngen-regobs-proxy.geojson")
    weather_proxy = sub.add_parser("weather-proxy"); weather_proxy.add_argument("--inventory", default="data/inventory/lyngen-regobs-proxy.geojson"); weather_proxy.add_argument("--weather-root", default="data/weather"); weather_proxy.add_argument("--output", default="data/weather-proxy-manifest.csv")
    plan = sub.add_parser("sentinel-proxy-plan"); plan.add_argument("--inventory", default="data/inventory/lyngen-regobs-proxy.geojson"); plan.add_argument("--catalogue", default="data/catalogue/sentinel-1-items.json"); plan.add_argument("--output", default="data/sentinel-1-proxy-plan.json"); plan.add_argument("--max-examples", type=int); plan.add_argument("--history-days", type=int, default=90)
    selection = sub.add_parser("sentinel-select"); selection.add_argument("--plan", default="data/sentinel-1-proxy-plan.json"); selection.add_argument("--imagery-root", default="data/processed/sentinel-1"); selection.add_argument("--output", default="data/sentinel-1-proxy-selection.json"); selection.add_argument("--max-attempts", type=int, default=24); selection.add_argument("--start", type=int, default=0); selection.add_argument("--max-examples", type=int)
    unlabeled_plan = sub.add_parser("sentinel-unlabeled-plan"); unlabeled_plan.add_argument("--weather-manifest", default="data/weather-proxy-manifest.csv"); unlabeled_plan.add_argument("--catalogue", default="data/catalogue/sentinel-1-items.json"); unlabeled_plan.add_argument("--output", default="data/sentinel-1-unlabeled-plan.json"); unlabeled_plan.add_argument("--history-days", type=int, default=90)
    multimodal = sub.add_parser("build-multimodal-proxy"); multimodal.add_argument("--weather-manifest", default="data/weather-proxy-manifest.csv"); multimodal.add_argument("--positive-selection", default="data/sentinel-1-proxy-selection.json"); multimodal.add_argument("--unlabeled-selection", default="data/sentinel-1-unlabeled-selection.json"); multimodal.add_argument("--imagery-root", default="data/processed/sentinel-1"); multimodal.add_argument("--weather-root", default="data/weather"); multimodal.add_argument("--output", default="data/multimodal-proxy-manifest.csv")
    args = parser.parse_args(); config = load_config()
    if args.command == "catalogue":
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(search_sentinel1(config, args.limit, args.pages), indent=2))
    elif args.command == "tiles":
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps([asdict(tile) for tile in generate_tiles(config)], indent=2))
    elif args.command == "weather":
        start, end = (utc(value) for value in config["date_range"].split("/"))
        selected = proxy_event_tiles(config, ROOT / args.proxy_inventory) if args.proxy_inventory else generate_tiles(config)
        for tile in selected:
            path = ROOT / args.output / f"{tile.tile_id}.csv"
            if not path.exists():
                fetch_hourly_weather(tile, start, end, path)
    elif args.command == "download":
        features = json.loads((ROOT / args.catalogue).read_text()).get("features", [])[:args.max_scenes]
        if not features:
            raise RuntimeError("Catalogue contains no Sentinel-1 scenes.")
        for feature in features:
            timestamp = feature["properties"].get("datetime")
            if not timestamp:
                raise ValueError(f"scene {feature.get('id')} lacks datetime")
            filename = utc(timestamp).strftime("%Y-%m-%dT%H:%M:%SZ.tif")
            for tile in generate_tiles(config):
                download_processed_tile(tile, timestamp, ROOT / args.output / tile.tile_id / filename, config["tile_resolution_metres"])
    elif args.command == "regobs":
        fetch_regobs_raw(config, ROOT / args.output)
    elif args.command == "regobs-screen":
        screened, rejected = screen_regobs_proxy(json.loads((ROOT / args.input).read_text()), config)
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(screened, indent=2))
        pd.DataFrame(rejected, columns=["regobs_id", "reason"]).to_csv(output.with_name("regobs-screen-excluded.csv"), index=False)
    elif args.command == "weather-proxy":
        build_weather_proxy_manifest(config, ROOT / args.inventory, ROOT / args.weather_root, ROOT / args.output)
    elif args.command == "sentinel-proxy-plan":
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(sentinel_proxy_plan(config, ROOT / args.inventory, ROOT / args.catalogue, args.max_examples, args.history_days), indent=2))
    elif args.command == "sentinel-select":
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(select_quality_approved_sentinel_sequences(config, ROOT / args.plan, ROOT / args.imagery_root, args.max_attempts, args.start, args.max_examples, output), indent=2))
    elif args.command == "sentinel-unlabeled-plan":
        output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(sentinel_unlabeled_plan(config, ROOT / args.weather_manifest, ROOT / args.catalogue, args.history_days), indent=2))
    elif args.command == "build-multimodal-proxy":
        build_multimodal_proxy_manifest(config, ROOT / args.weather_manifest, ROOT / args.positive_selection,
                                       ROOT / args.unlabeled_selection, ROOT / args.imagery_root, ROOT / args.weather_root, ROOT / args.output)
    else:
        build_manifest(config, ROOT / args.inventory, ROOT / args.imagery_root, ROOT / args.weather_root, ROOT / args.output)


if __name__ == "__main__":
    main()
