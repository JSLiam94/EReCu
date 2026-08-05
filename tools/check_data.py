from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from erecu.utils import discover_images


EXPECTED_TRAIN = 4040
EXPECTED_TESTS = {"COD10K": 2026, "CAMO": 250, "CHAMELEON": 76, "NC4K": 4121}


def indexed(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    files = discover_images(root)
    result = {path.stem: path for path in files}
    if len(result) != len(files):
        raise RuntimeError(f"Duplicate file stems under {root}: files={len(files)}, unique_stems={len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser("Strict dataset preflight")
    parser.add_argument("--train-root", default="data/TrainDataset/Imgs")
    parser.add_argument("--test-root", default="data/TestDataset")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--images-only", action="store_true", help="check test Imgs without touching GT paths")
    parser.add_argument("--datasets", nargs="+", choices=tuple(EXPECTED_TESTS), default=None)
    args = parser.parse_args()
    if args.train_only and args.skip_train:
        raise ValueError("--train-only and --skip-train cannot be combined")

    report: dict[str, object] = {"tests": {}}
    if not args.skip_train:
        train = indexed(Path(args.train_root))
        if len(train) != EXPECTED_TRAIN:
            raise RuntimeError(f"Training count mismatch: expected={EXPECTED_TRAIN}, found={len(train)}")
        report["train"] = len(train)

    if not args.train_only:
        test_root = Path(args.test_root)
        for name in args.datasets or EXPECTED_TESTS:
            expected = EXPECTED_TESTS[name]
            images = indexed(test_root / name / "Imgs")
            if len(images) != expected:
                raise RuntimeError(f"{name} image count mismatch: expected={expected}, found={len(images)}")
            if args.images_only:
                report["tests"][name] = {"images": len(images)}
                continue
            masks = indexed(test_root / name / "GT")
            if len(masks) != expected:
                raise RuntimeError(f"{name} GT count mismatch: expected={expected}, found={len(masks)}")
            missing_gt = sorted(set(images) - set(masks))
            missing_image = sorted(set(masks) - set(images))
            if missing_gt or missing_image:
                raise RuntimeError(
                    f"{name} image/GT stems differ: missing_gt={len(missing_gt)}, "
                    f"missing_image={len(missing_image)}"
                )
            report["tests"][name] = {"images": len(images), "GT": len(masks)}
    print(json.dumps(report, indent=2))
    print("PASS: all requested dataset counts and image/GT stems are exact")


if __name__ == "__main__":
    main()
