# Lyngen / Norway v1 data pipeline

The pilot is confined to Norway's Lyngen avalanche-warning region. It is
intentionally not a national or pan-Alpine dataset.

## Inputs and credentials

Set `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET` in `.env` to use the CDSE Process
API. The catalogue search is public; processed pixel downloads are authenticated.
The imagery request asks CDSE for Sentinel-1 IW, dual-polarisation GRD, terrain
orthorectification and Gamma0-terrain backscatter. Each response also preserves
the CDSE `dataMask` and Sentinel-1 terrain `shadowMask`. A tile is rejected
when fewer than 99% of pixels have valid, non-shadowed VV/VH values. Remaining
invalid pixels are set to zero only beside an explicit third `validity_mask`
channel, so they are never indistinguishable from valid backscatter. A local
NaN-aware 3x3 speckle filter is applied only after this quality gate.

RegObs source observations are acquired with:

```bash
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline regobs
```

This produces `data/inventory/regobs-raw.geojson`, retaining the source layer,
properties and attribution. It is **not trainable directly**: RegObs reports
are user-generated. A qualified reviewer must copy verified observations into a
separate reviewed inventory and set `status` to `confirmed`.

When no independent reviewer is available, create the explicitly weaker proxy
target instead:

```bash
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline regobs-screen
```

This retains reports with a three-star-or-higher published competence level,
usable time/location, avalanche size and extent fields, then de-duplicates by
RegObs registration ID. It writes `lyngen-regobs-proxy.geojson` and an exclusion
audit. The target is *a high-confidence reported avalanche observation within
six hours*, not an independently confirmed avalanche. Pass this proxy file to
`build-manifest`; the provenance file records the target semantics.

The reviewed event inventory must be a GeoJSON FeatureCollection with both:

```json
{"type":"Feature","properties":{"kind":"event","status":"confirmed","event_time":"2024-01-15T15:00:00Z"},"geometry":{}}
{"type":"Feature","properties":{"kind":"coverage","negative_observation":true,"observed_from":"2024-01-15T00:00:00Z","observed_to":"2024-01-16T00:00:00Z"},"geometry":{}}
```

The coverage geometry has to document a real observation with an explicit
negative finding. Absence of an event record is retained as `unknown`, written
to `data/unknown-samples.csv`, and never used as label `0`.

## Commands

```bash
# Inspect the reproducible spatial grid and catalogue metadata.
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline tiles
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline catalogue --limit 500

# Download a deliberately bounded number of terrain-corrected scene/tile rasters.
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline download --max-scenes 4

# Fetch hourly weather at each tile centre, then build labelled sequences.
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline weather
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline build-manifest \
  --inventory data/inventory/lyngen-regobs-proxy.geojson
```

`build-manifest` creates `data/manifest.csv`, `data/samples/*.npz`,
`data/unknown-samples.csv`, and `data/provenance.json`. Provenance captures
the configuration and inventory hashes plus the Git revision.

Split assignment is chronological by whole seven-day storm groups: early groups
are training, later groups validation, newest groups test. No random tile split
is used. Before a research run, inspect class balance and confirm groups do not
overlap:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m avalanche_forecast.train \
  --manifest data/manifest.csv --output runs/vanoise-v1 --horizon-hours 6
```

Varsom publishes Lyngen regional bulletins and RegObs observations can be used
as contextual information in the dashboard. Neither replaces the reviewed,
explicit-negative coverage needed for supervised model labels.

## Weather-primary proxy training

For the six-hour product, Sentinel-1 has insufficient temporal cadence. Build
hourly weather sequences around auto-screened RegObs reports instead:

```bash
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline weather
PYTHONPATH=src python3 -m avalanche_forecast.data_pipeline weather-proxy
PYTHONPATH=src python3 -m avalanche_forecast.weather_proxy \
  --manifest data/weather-proxy-manifest.csv --output runs/lyngen-weather-pu \
  --class-prior 0.05
```

This is positive-unlabeled training. Unlabeled windows are at least 48 hours
from a proxy report and are never interpreted as negative labels. The output is
an uncalibrated report-likelihood ranking score, not an avalanche probability
or terrain-safety decision.
