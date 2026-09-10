"""Create a tiny synthetic dataset solely to exercise the local dashboard."""

import csv
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sample_dir = root / "data" / "demo-samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    rows = []
    # The spatial patterns merely make labels learnable in a smoke test; they
    # are not satellite measurements or avalanche observations.
    for index in range(8):
        label = index % 2
        imagery = rng.normal(0, 1, size=(2, 2, 64, 64)).astype(np.float32)
        imagery[:, :, 24:40, 24:40] += label * 0.5
        meteo = rng.normal(0, 1, size=(2, 3)).astype(np.float32)
        meteo[:, 0] += label * 0.5
        path = sample_dir / f"sample-{index:02d}.npz"
        np.savez_compressed(path, imagery=imagery, meteo=meteo, label=np.int64(label))
        rows.append({
            "path": str(path.relative_to(root)), "split": "train",
            "tile_id": f"synthetic-{index}", "analysis_time": "2024-01-15T12:00:00Z",
        })
    manifest = root / "data" / "demo-manifest.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created synthetic demo manifest: {manifest}")


if __name__ == "__main__":
    main()
