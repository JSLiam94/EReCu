
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


def paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    data_root = Path(args.data_root)
    train_images = Path(args.train_images) if args.train_images else data_root / "TrainDataset" / "Imgs"
    test_root = Path(args.test_root) if args.test_root else data_root / "TestDataset"
    dino = Path(args.dino_weights)
    output = Path(args.output)
    return train_images, test_root, dino, output


def check(args: argparse.Namespace, training: bool, testing: bool) -> None:
    train_images, test_root, _, _ = paths(args)
    command = [sys.executable, "tools/check_data.py", "--train-root", str(train_images), "--test-root", str(test_root)]
    if training and not testing:
        command.append("--train-only")
    elif testing and not training:
        command.append("--skip-train")
    if testing and args.datasets:
        command.extend(["--datasets", *args.datasets])
    run(command)


def train(args: argparse.Namespace) -> None:
    check(args, training=True, testing=False)
    train_images, _, dino, output = paths(args)
    run([
        sys.executable, "train.py", "--config", args.config, "--train-images", str(train_images),
        "--dino-weights", str(dino), "--output", str(output),
        *( ["--resume", args.resume] if args.resume else [] ),
    ])


def test(args: argparse.Namespace) -> None:
    check(args, training=False, testing=True)
    _, test_root, _, output = paths(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / "checkpoint_best.pth"
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
            sys.executable, "infer.py", "--config", args.config, "--checkpoint", str(checkpoint),
            "--images", str(test_root / name / "Imgs"), "--expected-count", str(count),
            "--output", str(prediction_dir), "--batch-size", str(args.eval_batch_size),
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


def main() -> None:
    parser = argparse.ArgumentParser("EReCu launcher")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("check", "train", "test", "all"),
        default="all",
        help="default: all",
    )
    parser.add_argument("--config", default="configs/erecu.yaml")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-images", default=None)
    parser.add_argument("--test-root", default=None)
    parser.add_argument("--dino-weights", default="weights/dino_deitsmall8_pretrain.pth")
    parser.add_argument("--output", default="outputs/erecu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--datasets", nargs="+", choices=tuple(TESTS), default=None, help="default: all four test datasets")
    args = parser.parse_args()
    if args.eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be positive")
    if args.command == "check":
        check(args, training=True, testing=True)
    elif args.command == "train":
        train(args)
    elif args.command == "test":
        test(args)
    else:
        train(args)
        test(args)


if __name__ == "__main__":
    main()
