# Data readiness

`config/pilot-study.json` defines a deliberately small Mont Blanc pilot area
(2022–23 and 2023–24 winters) for testing the pipeline. It is not an
operational coverage definition.

## Copernicus access

Create a CDSE account, then create an OAuth client in the Sentinel Hub services
dashboard. Copy `.env.example` to `.env` and fill `CDSE_CLIENT_ID` and
`CDSE_CLIENT_SECRET` locally. Do not add `.env` to Git.

CDSE documents [account registration](https://documentation.dataspace.copernicus.eu/Registration.html)
and [OAuth client authentication](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html).

## Event inventory requirement

Before imagery is downloaded into training samples, obtain a licensed,
timestamped avalanche-event inventory for the pilot area. Each record must at
least identify an occurrence time and location/outline, observation confidence,
and source. Confirm its licence permits machine-learning use.

The preparation process must join an event only to samples whose analysis time
precedes the event by the configured six-hour horizon. It must retain unknown
terrain as unknown—not silently make it a negative label.

## Files that will make a run possible

```text
.env                          # local CDSE credentials; never committed
data/inventory/events.geojson # approved event inventory
data/manifest.csv             # generated sample index
data/samples/*.npz            # preprocessed image/weather sequences
```

Do not generate `manifest.csv` until both credentials and the approved event
inventory are available. A manifest made with invented labels would be
misleading and unusable for evaluation.
