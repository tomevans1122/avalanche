"""PU-safe ResNet-50 + GRU training for multimodal RegObs report proxies."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .model import TemporalResNet50
from .weather_proxy import non_negative_pu_loss


class MultimodalProxyDataset(Dataset):
    """Load imagery/weather samples while preserving positive vs unlabelled status."""

    def __init__(self, manifest: str | Path, split: str = "train"):
        rows = pd.read_csv(manifest)
        required = {"path", "target", "split"}
        if missing := required.difference(rows):
            raise ValueError(f"manifest lacks {sorted(missing)}")
        self.rows = rows.loc[rows.split == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"no samples in split {split!r}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows.iloc[index]
        with np.load(row.path) as sample:
            imagery, meteo = sample["imagery"].astype(np.float32), sample["meteo"].astype(np.float32)
        if imagery.ndim != 4 or meteo.ndim != 2 or imagery.shape[0] != meteo.shape[0]:
            raise ValueError(f"invalid sequence dimensions in {row.path}")
        if not np.isfinite(imagery).all() or not np.isfinite(meteo).all():
            raise ValueError(f"non-finite values in {row.path}")
        return {"imagery": torch.from_numpy(imagery), "meteo": torch.from_numpy(meteo),
                "positive": torch.tensor(float(row.target == "positive"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-prior", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if not 0 < args.class_prior < 1:
        raise ValueError("class prior must be between 0 and 1")

    dataset = MultimodalProxyDataset(args.manifest)
    first = dataset[0]
    positives = (dataset.rows.target == "positive").sum()
    weights = [1.0 if row.target == "positive" else max(positives / max(len(dataset) - positives, 1), .05)
               for _, row in dataset.rows.iterrows()]
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=WeightedRandomSampler(weights, num_samples=max(len(dataset), args.batch_size * 4), replacement=True),
                        drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalResNet50(first["imagery"].shape[1], first["meteo"].shape[1], pretrained=not args.no_pretrained).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in loader:
            imagery, meteo, positive = (batch[key].to(device) for key in ("imagery", "meteo", "positive"))
            if positive.sum() == 0 or positive.sum() == len(positive):
                continue
            optimiser.zero_grad(set_to_none=True)
            loss = non_negative_pu_loss(model(imagery, meteo), positive, args.class_prior)
            loss.backward(); optimiser.step(); losses.append(loss.item())
        if not losses:
            raise RuntimeError("no mixed positive/unlabelled batches; increase batch size")
        print(f"epoch={epoch + 1} pu_risk={np.mean(losses):.4f}", flush=True)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "multimodal-pu-model.pt")
    (output / "run.json").write_text(json.dumps({"manifest": args.manifest, "class_prior": args.class_prior,
        "image_channels": first["imagery"].shape[1], "meteo_features": first["meteo"].shape[1],
        "warning": "Uncalibrated PU report-likelihood score, not an avalanche probability or safety advice."}, indent=2))


if __name__ == "__main__":
    main()
