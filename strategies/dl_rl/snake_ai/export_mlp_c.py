from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .models import MLPPolicy


def fmt(values) -> str:
    return ", ".join(f"{float(v):.8g}f" for v in values)


def export(checkpoint: Path, output: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model = MLPPolicy(int(ckpt["input_dim"]))
    model.load_state_dict(ckpt["model"])
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "/* Generated MLP weights for snake OJ inference. */",
        f"#define MLP_INPUT_DIM {ckpt['input_dim']}",
    ]
    for name, tensor in model.state_dict().items():
        ident = name.replace(".", "_")
        flat = tensor.detach().cpu().reshape(-1).numpy()
        lines.append(f"static const int {ident}_len = {len(flat)};")
        lines.append(f"static const float {ident}[] = {{{fmt(flat)}}};")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"exported {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained MLP weights as C arrays.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("strategies/dl_rl/export/mlp_weights.h"))
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()

