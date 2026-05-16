from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求使用 cuda，但当前 Python 的 PyTorch 不支持 CUDA。"
            "请安装 CUDA 版 PyTorch 后重试。"
        )
    if requested not in {"cpu", "cuda"}:
        raise ValueError(f"unknown device: {requested}")
    return torch.device(requested)


def cpu_state_dict(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}
