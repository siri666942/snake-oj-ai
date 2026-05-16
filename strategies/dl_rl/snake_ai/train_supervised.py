from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .device import cpu_state_dict, resolve_device
from .models import CNNPolicy, MLPPolicy


def train_supervised(data_path: Path, kind: str, output: Path, epochs: int, batch_size: int, lr: float, device_name: str = "auto") -> None:
    device = resolve_device(device_name)
    print(f"device={device}")
    data = np.load(data_path)
    x_np = data["x_mlp"] if kind == "mlp" else data["x_cnn"]
    y_np = data["y"]
    x = torch.from_numpy(x_np).float()
    y = torch.from_numpy(y_np).long()
    dataset = TensorDataset(x, y)
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(20260516))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=pin_memory)

    model = MLPPolicy(x.shape[1]) if kind == "mlp" else CNNPolicy()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
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
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb).argmax(dim=1)
                correct += int((pred == yb).sum())
                total += len(yb)
        acc = correct / max(total, 1)
        print(f"epoch={epoch} train_loss={total_loss / train_size:.4f} val_acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model": cpu_state_dict(model), "kind": kind, "input_dim": x.shape[1], "val_acc": acc}, output)
            print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN/MLP supervised imitation models.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--kind", choices=["mlp", "cnn"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    train_supervised(args.data, args.kind, args.output, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
