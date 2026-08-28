from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights

from erecu.data import ImageOnlyDataset, train_collate
from erecu.dinov2_model import DinoV2EReCuModel
from erecu.dinov2_trainer import DinoV2Trainer
from erecu.utils import load_yaml, save_json, seed_everything


def preflight_pretrained_assets(config: dict) -> None:
    """Fail before dataset/model construction when required offline weights are absent."""
    errors: list[str] = []
    model_cfg = config["model"]
    backbone_name = str(model_cfg["backbone"])
    local_only = bool(model_cfg.get("backbone_local_files_only", False))
    backbone_path = Path(backbone_name).expanduser()

    if backbone_path.is_file():
        if backbone_path.suffix.lower() != ".pth":
            errors.append(
                f"Unsupported DINOv2 checkpoint file: {backbone_path}. "
                "Use the official dinov2_vitb14_pretrain.pth checkpoint."
            )
        elif backbone_path.stat().st_size == 0:
            errors.append(f"DINOv2 checkpoint is empty: {backbone_path}")
    elif local_only or backbone_path.is_dir():
        if not backbone_path.is_dir():
            errors.append(
                f"DINOv2 local directory is missing: {backbone_path}. "
                "Copy a complete Hugging Face save_pretrained directory there."
            )
        else:
            config_path = backbone_path / "config.json"
            weight_names = (
                "model.safetensors",
                "pytorch_model.bin",
                "model.safetensors.index.json",
                "pytorch_model.bin.index.json",
            )
            if not config_path.is_file():
                errors.append(f"DINOv2 config is missing: {config_path}")
            else:
                try:
                    with config_path.open("r", encoding="utf-8") as handle:
                        model_type = str(json.load(handle).get("model_type", ""))
                    if model_type != "dinov2":
                        errors.append(f"DINOv2 config has model_type={model_type!r}, expected 'dinov2': {config_path}")
                except (OSError, json.JSONDecodeError) as error:
                    errors.append(f"DINOv2 config cannot be read: {config_path} ({error})")
            if not any((backbone_path / name).is_file() and (backbone_path / name).stat().st_size > 0 for name in weight_names):
                errors.append(
                    f"DINOv2 weights are missing under {backbone_path}; expected one of {', '.join(weight_names)}"
                )
    else:
        errors.append(
            f"DINOv2 backbone {backbone_name!r} is neither a local .pth checkpoint nor a local model directory. "
            "Use weights/dinov2_vitb14_pretrain.pth so training does not depend on network access."
        )

    if model_cfg.get("resnet_pretrained", False):
        filename = Path(ResNet18_Weights.DEFAULT.url).name
        resnet_path = Path(model_cfg.get("resnet_weights", Path("weights") / filename)).expanduser()
        # Persist the resolved path in run_metadata/checkpoints and pass the
        # same explicit asset to the model instead of relying on torch.hub's
        # user-global cache.
        model_cfg["resnet_weights"] = str(resnet_path)
        if not resnet_path.is_file() or resnet_path.stat().st_size == 0:
            print(f"Preflight: downloading ImageNet ResNet-18 weights to {resnet_path}", flush=True)
            try:
                resnet_path.parent.mkdir(parents=True, exist_ok=True)
                torch.hub.load_state_dict_from_url(
                    ResNet18_Weights.DEFAULT.url,
                    model_dir=str(resnet_path.parent),
                    file_name=resnet_path.name,
                    progress=True,
                    check_hash=True,
                )
            except Exception as error:
                errors.append(
                    f"ImageNet ResNet-18 weights are missing and download failed: {resnet_path} ({error})"
                )
            if not resnet_path.is_file() or resnet_path.stat().st_size == 0:
                errors.append(f"ImageNet ResNet-18 download did not create a usable file: {resnet_path}")

    if errors:
        message = "Preflight failed; training has not started:\n" + "\n".join(f"- {error}" for error in errors)
        raise FileNotFoundError(message)

    print(f"Preflight passed: DINOv2={backbone_path}; ResNet-18={'required' if model_cfg.get('resnet_pretrained', False) else 'disabled'}")


def main() -> None:
    parser = argparse.ArgumentParser("Image-only EReCu training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None, help="resumable checkpoint_last.pth or checkpoint_best.pth")
    parser.add_argument("--dino-model", default=None, help="local DINOv2 model directory or official .pth checkpoint")
    parser.add_argument("--train-images", default=None, help="optional override for data.train_images")
    parser.add_argument("--epochs", type=int, default=None, help="epochs to run in this invocation")
    parser.add_argument("--seed", type=int, default=None, help="override the experiment random seed")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.dino_model:
        config["model"]["backbone"] = args.dino_model
    if args.train_images:
        config["data"]["train_images"] = args.train_images
    preflight_pretrained_assets(config)
    seed_everything(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output or config["experiment"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    dataset = ImageOnlyDataset(
        config["data"]["train_images"],
        image_size=config["model"]["image_size"],
        limit=config["data"].get("train_limit"),
        hflip=config["data"].get("hflip", 0.5),
    )
    expected_count = config["data"].get("expected_count")
    if expected_count is not None and len(dataset) != int(expected_count):
        raise RuntimeError(f"Training image count mismatch: expected={expected_count}, found={len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=config["training"]["workers"] > 0,
        collate_fn=train_collate,
    )
    model = DinoV2EReCuModel(
        image_size=config["model"]["image_size"],
        resnet_pretrained=config["model"].get("resnet_pretrained", False),
        resnet_weights_path=config["model"].get("resnet_weights"),
        mnp_sigma1=config["mnp"]["sigma1"],
        mnp_sigma2=config["mnp"]["sigma2"],
        mnp_threshold=config["mnp"]["threshold"],
        mnp_dilation=config["mnp"].get("dilation", 5),
        mnp_samples=config["mnp"].get("samples", 5),
        mnp_patch_size=config["mnp"].get("patch_size", 15),
        mnp_grid_patch_size=config["mnp"].get("grid_patch_size"),
        tas_seed_blend=config["pseudo"].get("tas_seed_blend", 0.65),
        tas_temperature=config["pseudo"].get("tas_temperature", 0.15),
        layers=tuple(config["model"]["student_layers"]),
        backbone_name=config["model"]["backbone"],
        backbone_local_files_only=config["model"].get("backbone_local_files_only", False),
        backbone_frozen=config["model"].get("backbone_frozen", False),
        backbone_trainable_last_blocks=config["model"].get("backbone_trainable_last_blocks"),
    )
    save_json(
        {
            "config": config,
            "device": str(device),
            "training_images": len(dataset),
            "backbone": config["model"]["backbone"],
            "backbone_frozen": all(not parameter.requires_grad for parameter in model.backbone.parameters()),
            "backbone_trainable_parameters": sum(
                parameter.numel() for parameter in model.backbone.parameters() if parameter.requires_grad
            ),
            "physical_batch_size": config["training"]["batch_size"],
            "gradient_accumulation": config["training"]["grad_accumulation"],
            "effective_batch_size": config["training"]["batch_size"] * config["training"]["grad_accumulation"],
            "supervision": "images_only",
        },
        output / "run_metadata.json",
    )
    print(f"device={device}; images={len(dataset)}; output={output}")
    trainer = DinoV2Trainer(model, config, device, output)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loader, max_epochs=args.epochs)


if __name__ == "__main__":
    main()
