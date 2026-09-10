"""Manifest-backed dataset with strict no-future-data checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AvalancheSequenceDataset(Dataset):
    """Loads preprocessed fixed-grid image and meteorology sequences from NPZ."""

    def __init__(self, manifest: str | Path, split: str):
        rows = pd.read_csv(manifest)
        required = {"path", "split", "tile_id", "analysis_time"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"manifest lacks columns: {sorted(missing)}")
        self.rows = rows.loc[rows.split == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"no samples for split={split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        with np.load(row.path) as sample:
            missing = {"imagery", "meteo", "label"}.difference(sample.files)
            if missing:
                raise ValueError(f"{row.path} lacks arrays: {sorted(missing)}")
            imagery = sample["imagery"].astype(np.float32, copy=False)
            meteo = sample["meteo"].astype(np.float32, copy=False)
            label = sample["label"]
        if imagery.ndim != 4 or meteo.ndim != 2 or imagery.shape[0] != meteo.shape[0]:
            raise ValueError(f"invalid sequence dimensions in {row.path}")
        if not np.isfinite(imagery).all() or not np.isfinite(meteo).all():
            raise ValueError(f"non-finite values in {row.path}; impute upstream")
        return {
            "imagery": torch.from_numpy(imagery),
            "meteo": torch.from_numpy(meteo),
            "label": torch.tensor(float(label), dtype=torch.float32),
        }
