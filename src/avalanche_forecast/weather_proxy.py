"""Weather-primary positive-unlabeled research model for RegObs report timing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class WeatherProxyDataset(Dataset):
    def __init__(self, manifest: str | Path, split: str = "train"):
        self.rows = pd.read_csv(manifest)
        needed = {"path", "target"}
        if missing := needed.difference(self.rows):
            raise ValueError(f"manifest lacks {sorted(missing)}")
        if "split" in self.rows:
            self.rows = self.rows.loc[self.rows.split == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"no samples in split {split!r}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with np.load(row.path) as sample:
            meteo = sample["meteo"].astype(np.float32)
        return {"meteo": torch.from_numpy(meteo), "positive": torch.tensor(float(row.target == "positive"))}


class WeatherGRU(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.encoder = nn.GRU(features, 64, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(64), nn.Dropout(.2), nn.Linear(64, 1))

    def forward(self, meteo):
        values, _ = self.encoder(meteo)
        return self.head(values[:, -1]).squeeze(-1)


def non_negative_pu_loss(logits: torch.Tensor, positive: torch.Tensor, class_prior: float, beta: float = 0.0) -> torch.Tensor:
    """nnPU risk; output is a ranking score, not a calibrated probability."""
    positives, unlabeled = logits[positive.bool()], logits[~positive.bool()]
    if len(positives) == 0 or len(unlabeled) == 0:
        raise ValueError("each PU batch needs positive and unlabeled samples")
    positive_risk = class_prior * nn.functional.binary_cross_entropy_with_logits(positives, torch.ones_like(positives))
    negative_risk = nn.functional.binary_cross_entropy_with_logits(unlabeled, torch.zeros_like(unlabeled)) - class_prior * nn.functional.binary_cross_entropy_with_logits(positives, torch.zeros_like(positives))
    return positive_risk + torch.maximum(negative_risk, torch.tensor(-beta, device=logits.device))


def main():
    parser = argparse.ArgumentParser(description="Train an uncalibrated RegObs-report PU score from hourly weather.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-prior", type=float, required=True, help="Assumed report prevalence; score is not a probability.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not 0 < args.class_prior < 1:
        raise ValueError("class prior must be between 0 and 1")
    dataset = WeatherProxyDataset(args.manifest, args.split)
    first = dataset[0]
    weights = [1.0 if row.target == "positive" else max((dataset.rows.target == "positive").sum() / max((dataset.rows.target == "unlabeled").sum(), 1), .05) for _, row in dataset.rows.iterrows()]
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=WeightedRandomSampler(weights, num_samples=max(len(dataset), args.batch_size * 4), replacement=True))
    model, device = WeatherGRU(first["meteo"].shape[1]), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device); optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            x, y = batch["meteo"].to(device), batch["positive"].to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = non_negative_pu_loss(model(x), y, args.class_prior)
            loss.backward(); optimiser.step(); losses.append(loss.item())
        print(f"epoch={epoch + 1} pu_risk={np.mean(losses):.4f}")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "weather-pu-model.pt")
    (output / "run.json").write_text(json.dumps({"target": "high-confidence RegObs report within configured horizon", "class_prior": args.class_prior, "warning": "Uncalibrated PU score, not avalanche probability or safety advice."}, indent=2))


if __name__ == "__main__":
    main()
