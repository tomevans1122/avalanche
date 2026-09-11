# Alpine avalanche likelihood baseline

This is a research baseline for estimating the probability that an avalanche
release will be recorded in a terrain cell during a short forecast horizon. It
is **not** an avalanche-safety product and must not be used to decide whether
terrain is safe. Operational use needs local avalanche-service review,
calibration, monitoring, and a conservative human decision process.

## Why this design

The visual backbone is the established ImageNet-pretrained **ResNet-50**. It
encodes each image in a historical sequence; a GRU then combines the image
embeddings and time-aligned weather/snowpack measurements to predict an event
in the next 6 hours. The output is a calibrated probability, not a binary
"avalanche/no avalanche" verdict.

Use Sentinel-1 GRD (VV/VH) as the primary image source: SAR works through
clouds and without daylight. Use cloud-masked Sentinel-2 Level-2A (B03, B04,
B08, B11, B12) as an optional additional branch or a separately trained
model. Sentinel-2’s five-day nominal revisit and cloud cover make it unsuitable
as the only source for a few-hour target. Copernicus imagery should be found
through the [CDSE STAC catalogue](https://documentation.dataspace.copernicus.eu/APIs/STAC.html).
ESA documents Sentinel-2’s 13 bands and 10–60 m resolutions
[here](https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Facts_and_figures).

For actual next-hours skill, include frequent, time-aligned predictors such as
new snow, wind speed/direction, air temperature, precipitation, snow depth,
and—where available—snowpack/stability observations. Image history detects
spatial context and surface change; it cannot observe a rapidly changing weak
layer reliably enough on its own.

## Data contract

Prepare one compressed `.npz` per sample and a CSV manifest. A sample represents
one fixed projected terrain tile ending at `analysis_time`:

```text
sample.npz
  imagery: float32 [T, C, H, W]  # Sentinel-1: C=3, ordered VV,VH,validity_mask
  meteo:   float32 [T, M]        # standardized, no values after analysis_time
  label:   int64 scalar          # 1 if a validated release occurs within horizon
```

The manifest must contain `path,split,tile_id,analysis_time`. Split by spatial
region and storm/date (not random tiles), so near-duplicate scenes and a single
storm cannot leak into validation. Labels need trusted timestamped avalanche
inventories, snapped to terrain cells, and a documented treatment of unobserved
events. Do not label an unknown cell as a confident negative.

The ready-to-fill default pilot definition is in
[`config/pilot-study.json`](config/pilot-study.json); the credential and
inventory handoff is in [`docs/data-readiness.md`](docs/data-readiness.md).
The full reproducible Lyngen/Norway preparation workflow is documented in
[`docs/pipeline.md`](docs/pipeline.md).

All frames in one sample must use the same CRS, extent, resolution and terrain
mask. Apply radiometric terrain correction, incidence-angle normalization,
speckle handling, layover/shadow masks and per-channel normalization to
Sentinel-1 before writing the arrays. Resample weather to each image timestamp.

## Install and train

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-cpu.txt
PYTHONPATH=src python3 -m avalanche_forecast.train \
  --manifest data/manifest.csv --output runs/resnet50-s1
```

Use `--no-pretrained` only for an offline smoke run. The default downloads
ImageNet ResNet-50 weights on first use. Set `--horizon-hours` to the exact
label horizon used when creating data; it is recorded in the run metadata.

## Local dashboard

After installation, launch a local-only dashboard with:

```bash
PYTHONPATH=src python3 -m avalanche_forecast.webapp
```

Open http://127.0.0.1:5000. Enter a manifest path within this project and the
same forecast horizon used to generate its labels. The dashboard starts one
training process at a time and displays `runs/web-*/train.log`; it does not
make or display a terrain-safety prediction.

For an end-to-end interface check only, create eight **synthetic** samples:

```bash
PYTHONPATH=src python3 -m avalanche_forecast.demo_data
```

Then enter `data/demo-manifest.csv` in the dashboard. This data is generated
noise with artificial labels and must never be used for model evaluation or
forecasting.

### Official report panel

The dashboard can show a JSON bulletin from an official regional avalanche
service beside the model state. Set `OFFICIAL_REPORT_URL`, any required
`OFFICIAL_REPORT_API_KEY`, and `OFFICIAL_REPORT_REGION` in a local `.env` or
shell environment. The panel explicitly flags a region mismatch and does not
turn a bulletin into a model label or a safety recommendation. Services such as
[Avalanche.report](https://lawinen-warnung.eu/more/open-data) publish CAAML
bulletins for their covered regions; an API source must be selected that covers
the configured pilot area.

## Next steps for the Lyngen pilot

The current artefact is a weather-only, positive-unlabelled (PU) research
baseline. It was trained on automated high-confidence RegObs report proxies,
not on an exhaustive, independently verified avalanche inventory. Its score is
therefore an **uncalibrated report-likelihood ranking**, not an avalanche
probability or safety recommendation.

1. Evaluate the weather baseline once on the untouched chronological test
   split. Report ranking performance, precision at practical alert budgets, and
   variation by week and storm; do not tune using that split.
2. Calibrate a probability only if a separate validation analysis establishes
   sufficient, stable skill. Otherwise retain the output as a research score.
3. Download and preprocess Sentinel-1 VV/VH tiles aligned to each analysis
   time, applying the quality controls described above.
4. Build multimodal `.npz` samples containing the Sentinel-1 history,
   72-hour aligned weather history, terrain features, provenance, and explicit
   positive/unlabelled/unknown status.
5. Train the ResNet-50 plus temporal-GRU model and compare it fairly with the
   weather-only baseline on the locked test split. Retain imagery only if it
   provides a clear, repeatable held-out improvement.
6. In the dashboard, present the selected research score alongside—not in
   place of—the official Varsom bulletin and RegObs observations for Lyngen.
7. Consider geographic expansion or revised labelling only after the
   single-region pilot has passed the above evaluation and provenance review.

## Validation before deployment

Evaluate by season, region, elevation/aspect band, cloud/visibility class and
event size. Report PR-AUC, recall at a fixed alert budget, Brier score,
reliability curves, false negatives, and lead time. Fit calibration and choose
any alert threshold only on a validation set; lock it before an independent
seasonal test. A model that cannot beat weather/snowpack-only and persistence
baselines should not be shipped.

This repository intentionally contains no alerting endpoint. A production
system should fail closed on stale imagery/weather, missing quality masks or an
out-of-domain tile, then surface the inputs, uncertainty and provenance to a
qualified forecaster.
