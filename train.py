from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from erecu.backbone import load_dino_weights
from erecu.data import ImageOnlyDataset, train_collate
from erecu.model import EReCuModel
from erecu.trainer import Trainer
from erecu.utils import load_yaml, save_json, seed_everything, sha256


def main() -> None:
    parser = argparse.ArgumentParser("Image-only EReCu training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", default=None, help="full checkpoint_epoch_*.pth to resume; not teacher_last.pth")
    parser.add_argument("--dino-weights", default=None, help="optional server-side override for model.dino_weights")
    parser.add_argument("--train-images", default=None, help="optional override for data.train_images")
    parser.add_argument("--epochs", type=int, default=None, help="epochs to run in this invocation")
    parser.add_argument("--save-every", type=int, default=None, help="override checkpoint interval for this invocation")
    args = parser.parse_args()
    config = load_yaml(args.config)
    if args.dino_weights:
        config["model"]["dino_weights"] = args.dino_weights
    if args.train_images:
        config["data"]["train_images"] = args.train_images
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
    model = EReCuModel(
        image_size=config["model"]["image_size"],
        resnet_pretrained=config["model"].get("resnet_pretrained", False),
        mnp_sigma1=config["mnp"]["sigma1"],
        mnp_sigma2=config["mnp"]["sigma2"],
        mnp_threshold=config["mnp"]["threshold"],
        mnp_samples=config["mnp"].get("samples", 5),
        mnp_patch_size=config["mnp"].get("patch_size", 15),
        tas_seed_blend=config["pseudo"].get("tas_seed_blend", 0.65),
        tas_temperature=config["pseudo"].get("tas_temperature", 0.15),
        layers=tuple(config["model"]["student_layers"]),
    )
    load_dino_weights(model.backbone, config["model"]["dino_weights"])
    save_json(
        {
            "config": config,
            "device": str(device),
            "training_images": len(dataset),
            "dino_sha256": sha256(config["model"]["dino_weights"]),
            "supervision": "images_only",
        },
        output / "run_metadata.json",
    )
    print(f"device={device}; images={len(dataset)}; output={output}")
    trainer = Trainer(model, config, device, output)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(loader, max_epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
