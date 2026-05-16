from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .device import cpu_state_dict, resolve_device
from .models import MLPValue


def train_value(
    data_path: Path,
    kind: str,
    output: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_name: str = "huber",
    device_name: str = "auto",
) -> None:
    if kind != "mlp":
        raise ValueError("only kind='mlp' is supported for value training")
    device = resolve_device(device_name)
    print(f"device={device}")
    data = np.load(data_path)
    x = torch.from_numpy(data["x_mlp"]).float()
    y = torch.from_numpy(data["y_value"]).float()
    dataset = TensorDataset(x, y)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(20260516))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin_memory)

    model = MLPValue(x.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn: nn.Module = nn.SmoothL1Loss() if loss_name == "huber" else nn.MSELoss()
    best_val = float("inf")
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_loss += loss_fn(model(xb), yb).item() * len(xb)
        train_loss = total_loss / max(train_size, 1)
        val_loss /= max(val_size, 1)
        print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model": cpu_state_dict(model),
                    "kind": "value_mlp",
                    "input_dim": x.shape[1],
                    "val_loss": val_loss,
                    "loss": loss_name,
                },
                output,
            )
            print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an MLP value network.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kind", choices=["mlp"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train_value(args.data, args.kind, args.output, args.epochs, args.batch_size, args.lr, args.loss, args.device)


if __name__ == "__main__":
    main()
