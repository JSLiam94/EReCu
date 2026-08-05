from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from erecu.data import EvalImageDataset, eval_collate
from erecu.model import EReCuModel
from erecu.utils import load_yaml


def load_model(
    config: dict,
    checkpoint: str,
    device: torch.device,
) -> EReCuModel:
    model = EReCuModel(
        image_size=config["model"]["image_size"],
        resnet_pretrained=False,
        mnp_sigma1=config["mnp"]["sigma1"],
        mnp_sigma2=config["mnp"]["sigma2"],
        mnp_threshold=config["mnp"]["threshold"],
        mnp_samples=config["mnp"].get("samples", 5),
        mnp_patch_size=config["mnp"].get("patch_size", 15),
        tas_seed_blend=config["pseudo"].get("tas_seed_blend", 0.65),
        tas_temperature=config["pseudo"].get("tas_temperature", 0.15),
        layers=tuple(config["model"]["teacher_layers"]),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "teacher" in state:
        state = state["teacher"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def _round_to_patch(value: float, patch_size: int = 8) -> int:
    return max(patch_size, int(round(value / patch_size)) * patch_size)


def otsu_threshold(image: np.ndarray) -> int:
    histogram = np.bincount(image.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram / max(float(histogram.sum()), 1.0)
    omega = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256, dtype=np.float64))
    total_mean = means[-1]
    between = (total_mean * omega - means) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.argmax(between))


@torch.no_grad()
def predict(
    model: EReCuModel,
    image: torch.Tensor,
    native: torch.Tensor,
) -> torch.Tensor:

    base_size = image.shape[-2:]
    logit_views = []
    for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
        size = tuple(_round_to_patch(dim * scale) for dim in base_size)
        scaled_image = image if size == base_size else F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        scaled_native = native if size == base_size else F.interpolate(native, size=size, mode="bilinear", align_corners=False)
        for flipped in (False, True):
            view_image = torch.flip(scaled_image, dims=[-1]) if flipped else scaled_image
            view_native = torch.flip(scaled_native, dims=[-1]) if flipped else scaled_native
            output = model(
                view_image,
                view_native,
                compute_mnp=False,
                quality_seed=False,
            )
            prediction = output.probability_grid
            prediction = F.interpolate(prediction, size=base_size, mode="bilinear", align_corners=False)
            if flipped:
                prediction = torch.flip(prediction, dims=[-1])
            logit_views.append(torch.logit(prediction.clamp(1e-4, 1 - 1e-4)))
    return torch.sigmoid(torch.stack(logit_views, dim=0).mean(dim=0))


def main() -> None:
    parser = argparse.ArgumentParser("EReCu inference: image inputs only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--images", required=True, help="directory or one-path-per-line manifest")
    parser.add_argument("--expected-count", type=int, default=None, help="fail before model loading if image count differs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4, help="inference batch size; lower this if GPU memory is limited")
    args = parser.parse_args()
    config = load_yaml(args.config)
    dataset = EvalImageDataset(args.images, config["model"]["image_size"])
    if args.expected_count is not None and len(dataset) != args.expected_count:
        raise RuntimeError(f"Inference image count mismatch: expected={args.expected_count}, found={len(dataset)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(config, args.checkpoint, device)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", collate_fn=eval_collate)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for batch in tqdm(loader, desc="Infer"):
        image = batch["image"].to(device)
        native = batch["native"].to(device)
        predictions = predict(model, image, native)
        for index, (stem, (height, width)) in enumerate(zip(batch["id"], batch["original_size"])):
            pred = torch.nn.functional.interpolate(predictions[index : index + 1], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
            probability = pred.detach().clamp(0, 1).cpu().numpy()
            prediction_uint8 = np.rint(probability * 255.0).astype(np.uint8)
            threshold = otsu_threshold(prediction_uint8)
            array = (prediction_uint8 >= threshold).astype(np.uint8) * 255
            Image.fromarray(array).save(output / f"{stem}.png")


if __name__ == "__main__":
    main()
