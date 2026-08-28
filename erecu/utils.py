from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(value: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(root: str | Path, recursive: bool = True) -> list[Path]:
    root = Path(root)
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted((path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS), key=lambda x: x.name.lower())


def dice_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.flatten(1)
    target = target.flatten(1)
    if weight is not None:
        weight = weight.flatten(1)
        intersection = (pred * target * weight).sum(dim=1)
        denominator = ((pred + target) * weight).sum(dim=1)
    else:
        intersection = (pred * target).sum(dim=1)
        denominator = pred.sum(dim=1) + target.sum(dim=1)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def masked_bce(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:

    with torch.autocast(device_type=pred.device.type, enabled=False):
        value = torch.nn.functional.binary_cross_entropy(pred.float(), target.float(), reduction="none")
    if weight is None:
        return value.mean()
    weight = weight.float()
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def safe_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + eps)


def safe_logit(probability: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Numerically safe probability-to-logit conversion under CUDA fp16 AMP.

    In float16, ``1 - 1e-4`` rounds to exactly ``1``. Calling ``torch.logit``
    on a saturated probability then yields ``inf`` and poisons the backward
    pass. Do the clamp and logit in float32 and deliberately return float32.
    """
    with torch.autocast(device_type=probability.device.type, enabled=False):
        return torch.logit(probability.float().clamp(eps, 1.0 - eps))


def gradient_magnitude(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Differentiable first-order boundary magnitude on a probability map."""
    dx = torch.nn.functional.pad(x[..., 1:] - x[..., :-1], (0, 1, 0, 0))
    dy = torch.nn.functional.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + eps).clamp(0.0, 1.0)
