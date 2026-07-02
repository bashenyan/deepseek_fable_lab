"""运行时设备与推理配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
DTypeName = Literal["auto", "bf16", "fp16", "fp32"]


@dataclass(frozen=True)
class RuntimeConfig:
    """统一的设备与精度配置。"""

    device: DeviceName = "auto"
    dtype: DTypeName = "auto"


def resolve_device(device: DeviceName) -> str:
    """根据运行环境解析目标设备。"""

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("当前环境没有可用 MPS")
    return device


def resolve_dtype(dtype: DTypeName, device: str):
    """解析 torch dtype。"""

    if dtype == "auto":
        if device == "cuda":
            return torch.float16
        if device == "mps":
            return torch.float16
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def pick_input_device(model) -> torch.device:
    """推理时选择输入张量所在设备。"""

    if hasattr(model, "device"):
        try:
            return model.device
        except Exception:
            pass
    return next(model.parameters()).device
