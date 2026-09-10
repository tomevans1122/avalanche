"""Minimal, reproducible training entry point for the research baseline."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import AvalancheSequenceDataset
from .model import TemporalResNet50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--horizon-hours", type=float, default=6)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_set = AvalancheSequenceDataset(args.manifest, "train")
    first = train_set[0]
    model = TemporalResNet50(
        image_channels=first["imagery"].shape[1],
        meteo_features=first["meteo"].shape[1],
        pretrained=not args.no_pretrained,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    # Replace pos_weight after estimating it from the *training split only*.
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(args.epochs):
        running = 0.0
        for batch in loader:
            imagery, meteo, label = (batch[k].to(device) for k in ("imagery", "meteo", "label"))
            optimiser.zero_grad(set_to_none=True)
            loss = loss_fn(model(imagery, meteo), label)
            loss.backward()
            optimiser.step()
            running += loss.item() * label.numel()
        print(f"epoch={epoch + 1} loss={running / len(train_set):.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "run.json").write_text(json.dumps({
        "horizon_hours": args.horizon_hours,
        "image_channels": first["imagery"].shape[1],
        "meteo_features": first["meteo"].shape[1],
        "warning": "Research output only; do not use for terrain-safety decisions.",
    }, indent=2))


if __name__ == "__main__":
    main()
