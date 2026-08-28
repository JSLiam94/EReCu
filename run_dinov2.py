
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TESTS = {"CAMO": 250, "CHAMELEON": 76, "COD10K": 2026, "NC4K": 4121}


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    data_root = Path(args.data_root)
    train_images = Path(args.train_images) if args.train_images else data_root / "TrainDataset" / "Imgs"
    test_root = Path(args.test_root) if args.test_root else data_root / "TestDataset"
    output = Path(args.output)
    return train_images, test_root, output


def check(args: argparse.Namespace, training: bool, testing: bool) -> None:
    train_images, test_root, _ = paths(args)
    command = [sys.executable, "tools/check_data.py", "--train-root", str(train_images), "--test-root", str(test_root)]
    if training and not testing:
        command.append("--train-only")
    elif testing and not training:
        command.append("--skip-train")
    if testing and args.datasets:
        command.extend(["--datasets", *args.datasets])
    run(command)


def train_one(args: argparse.Namespace, seed: int | None = None) -> Path:
    train_images, _, output = paths(args)
    if seed is not None:
        output = output / f"seed_{seed}"
    run([
        sys.executable, "train_dinov2.py", "--config", args.config, "--train-images", str(train_images),
        "--output", str(output),
        *( ["--dino-model", args.dino_model] if args.dino_model else [] ),
        *( ["--seed", str(seed)] if seed is not None else [] ),
        *( ["--resume", args.resume] if args.resume else [] ),
    ])
    return output


def train(args: argparse.Namespace) -> list[Path]:
    check(args, training=True, testing=False)
    return [train_one(args, seed) for seed in (args.seeds or [None])]


def test_one(args: argparse.Namespace, output: Path, checkpoint: Path | None = None) -> None:
    _, test_root, _ = paths(args)
    checkpoint = checkpoint or (
        Path(args.checkpoint) if args.checkpoint else output / f"checkpoint_{args.checkpoint_kind}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}. Train first or pass --checkpoint.")
    prediction_root = output / "benchmark" / "predictions"
    metrics_root = output / "benchmark" / "metrics"
    prediction_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict] = {}
    tests = {name: TESTS[name] for name in (args.datasets or tuple(TESTS))}
    for name, count in tests.items():
        prediction_dir = prediction_root / name
        metric_path = metrics_root / f"{name}.json"
        infer_command = [
            sys.executable, "infer_dinov2.py", "--config", args.config, "--checkpoint", str(checkpoint),
            "--images", str(test_root / name / "Imgs"), "--expected-count", str(count),
            "--output", str(prediction_dir), "--batch-size", str(args.eval_batch_size),
            *( ["--dino-model", args.dino_model] if args.dino_model else [] ),
        ]
        run(infer_command)
        run([
            sys.executable, "evaluate.py", "--predictions", str(prediction_dir),
            "--gt", str(test_root / name / "GT"), "--dataset", name,
            "--expected-count", str(count), "--output", str(metric_path),
        ])
        metrics[name] = json.loads(metric_path.read_text(encoding="utf-8"))
    raw_keys = ("Sm", "Fm_max", "Fm_mean", "wFm", "Fm_adaptive", "Em_max", "Em_mean", "Em_adaptive", "MAE")
    summary = {
        "checkpoint": str(checkpoint),
        "binary_method": "otsu",
        "datasets": metrics,
        "average": {key: sum(value[key] for value in metrics.values()) / len(metrics) for key in raw_keys},
    }
    summary_path = output / "benchmark" / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {summary_path}")


def test(args: argparse.Namespace, outputs: list[Path] | None = None) -> None:
    check(args, training=False, testing=True)
    if args.seeds and args.checkpoint:
        raise ValueError("--checkpoint cannot be combined with --seeds; each seed uses its own selected checkpoint kind.")
    if outputs is not None:
        targets = outputs
    elif args.seeds:
        targets = [paths(args)[2] / f"seed_{seed}" for seed in args.seeds]
    else:
        targets = [paths(args)[2]]
    for output in targets:
        test_one(args, output)


def main() -> None:
    parser = argparse.ArgumentParser("EReCu launcher")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "train", "test", "all"),
        default="all",
        help="default: all",
    )
    parser.add_argument("--config", default="configs/erecu_dinov2.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--dino-model", default=None, help="optional local DINOv2 model directory or official .pth checkpoint override")
    parser.add_argument("--output", default="outputs/erecu_dinov2")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--checkpoint-kind",
        choices=("best", "last"),
        default="best",
        help="checkpoint used with --seeds when --checkpoint is omitted; default: best",
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", choices=tuple(TESTS), default=None, help="default: all four test datasets")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="seeds to run (default: 2026)")
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    if args.seeds is None and args.command in {"train", "all"}:
        args.seeds = [2026]
    elif args.seeds is None and args.command == "test" and args.checkpoint is None:
        # Match the default single-seed training output while preserving the
        # existing explicit-checkpoint behaviour.
        args.seeds = [2026]
    if args.command == "check":
        check(args, training=True, testing=True)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    else:
        outputs = train(args)
        test(args, outputs)


if __name__ == "__main__":
    main()
