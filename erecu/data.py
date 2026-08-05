from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF

from .utils import discover_images


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _resize_rgb(image: Image.Image, size: int) -> Image.Image:
    return TF.resize(image, [size, size], interpolation=TF.InterpolationMode.BICUBIC, antialias=True)


class ImageOnlyDataset(Dataset):
    """Training dataset deliberately exposes no GT/edge/instance path."""

    def __init__(self, image_root: str | list[str], image_size: int, limit: int | None = None, hflip: float = 0.5) -> None:
        roots = image_root if isinstance(image_root, list) else [image_root]
        self.image_paths = []
        for item in roots:
            root = Path(item)
            self.image_paths.extend([Path(line.strip()) for line in root.read_text(encoding="utf-8").splitlines() if line.strip()] if root.is_file() else discover_images(root))
        self.image_paths = sorted(set(self.image_paths), key=lambda path: str(path).lower())
        if limit is not None:
            self.image_paths = self.image_paths[:limit]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {image_root}")
        self.image_size = image_size
        self.hflip = hflip
        self.student_jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.18, hue=0.04)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        image = _resize_rgb(image, self.image_size)
        do_flip = bool(torch.rand(()) < self.hflip)
        if do_flip:
            image = TF.hflip(image)
        native = TF.to_tensor(image)
        teacher = TF.normalize(native, IMAGENET_MEAN, IMAGENET_STD)
        student_image = self.student_jitter(image)
        if torch.rand(()) < 0.15:
            student_image = TF.rgb_to_grayscale(student_image, num_output_channels=3)
        student = TF.normalize(TF.to_tensor(student_image), IMAGENET_MEAN, IMAGENET_STD)
        return {"id": path.stem, "native": native, "teacher": teacher, "student": student}


class EvalImageDataset(Dataset):
    def __init__(self, image_root: str, image_size: int) -> None:
        root = Path(image_root)
        self.image_paths = [Path(line.strip()) for line in root.read_text(encoding="utf-8").splitlines() if line.strip()] if root.is_file() else discover_images(root)
        if not self.image_paths:
            raise FileNotFoundError(f"No images found under {image_root}")
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        original_size = (image.height, image.width)
        resized = _resize_rgb(image, self.image_size)
        native = TF.to_tensor(resized)
        return {
            "id": path.stem,
            "native": native,
            "image": TF.normalize(native, IMAGENET_MEAN, IMAGENET_STD),
            "original_size": original_size,
        }


def train_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "native": torch.stack([item["native"] for item in batch]),
        "teacher": torch.stack([item["teacher"] for item in batch]),
        "student": torch.stack([item["student"] for item in batch]),
    }


def eval_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": [item["id"] for item in batch],
        "native": torch.stack([item["native"] for item in batch]),
        "image": torch.stack([item["image"] for item in batch]),
        "original_size": [item["original_size"] for item in batch],
    }
