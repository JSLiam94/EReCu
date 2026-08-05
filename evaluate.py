from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BINARY_METHOD = "otsu"


def metric_classes():
    try:
        from py_sod_metrics import (
            Emeasure,
            FmeasureHandler,
            FmeasureV2,
            MAE,
            Smeasure,
            WeightedFmeasure,
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "Missing py_sod_metrics. Install it with "
            "`python -m pip install pysodmetrics`."
        ) from error
    return Emeasure, FmeasureHandler, FmeasureV2, MAE, Smeasure, WeightedFmeasure


def discover_images(root: str) -> list[Path]:
    return sorted(
        (path for path in Path(root).rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.lower(),
    )


def index_by_stem(root: str) -> dict[str, Path]:
    files = discover_images(root)
    indexed = {path.stem: path for path in files}
    if len(indexed) != len(files):
        raise RuntimeError(f"Duplicate stems in {root}")
    return indexed


def otsu_threshold(prediction: np.ndarray) -> int:
    """Return the per-image Otsu threshold for an 8-bit prediction map."""
    histogram = np.bincount(prediction.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram / max(float(histogram.sum()), 1.0)
    omega = np.cumsum(probability)
    means = np.cumsum(probability * np.arange(256, dtype=np.float64))
    total_mean = means[-1]
    between = (total_mean * omega - means) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.argmax(between))


def binarize_otsu(prediction: np.ndarray) -> np.ndarray:
    """Apply Otsu to continuous maps, preserving maps already saved as binary."""
    values = np.unique(prediction)
    if values.size <= 2 and np.all(np.isin(values, (0, 255))):
        return prediction >= 128
    return prediction >= otsu_threshold(prediction)


def new_metric_set(Emeasure, FmeasureHandler, FmeasureV2, MAE, Smeasure, WeightedFmeasure) -> dict[str, object]:

    fm = FmeasureV2(
        metric_handlers={
            "fm": FmeasureHandler(
                with_dynamic=True,
                with_adaptive=True,
                beta=0.3,
            )
        }
    )
    return {
        "sm": Smeasure(),
        "fm": fm,
        "wfm": WeightedFmeasure(),
        "em": Emeasure(),
        "mae": MAE(),
    }


def metric_result(metrics: dict[str, object]) -> dict[str, float]:
    """Return the paper-style COD metrics in the requested display order."""
    em_result = metrics["em"].get_results()["em"]
    fm_result = metrics["fm"].get_results()["fm"]
    return {
        "Sm": float(metrics["sm"].get_results()["sm"]),
        "Fm_max": float(fm_result["dynamic"].max()),
        "Fm_mean": float(fm_result["dynamic"].mean()),
        "wFm": float(metrics["wfm"].get_results()["wfm"]),
        "Fm_adaptive": float(fm_result["adaptive"]),
        "Em_max": float(em_result["curve"].max()),
        "Em_mean": float(em_result["curve"].mean()),
        "Em_adaptive": float(em_result["adp"]),
        "MAE": float(metrics["mae"].get_results()["mae"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser("EReCu COD evaluation; GT is read only here")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--dataset", default=None, help="optional dataset name recorded before metrics")
    parser.add_argument("--output", default=None)
    parser.add_argument("--ids", default=None, help="optional image manifest; evaluates only its file stems")
    parser.add_argument("--expected-count", type=int, default=None, help="require this many predictions and GT masks")
    args = parser.parse_args()

    preds, gts = index_by_stem(args.predictions), index_by_stem(args.gt)
    if args.ids:
        requested = {Path(line.strip()).stem for line in Path(args.ids).read_text(encoding="utf-8").splitlines() if line.strip()}
        gts = {stem: path for stem, path in gts.items() if stem in requested}
        preds = {stem: path for stem, path in preds.items() if stem in requested}
    if args.expected_count is not None and (len(preds) != args.expected_count or len(gts) != args.expected_count):
        raise RuntimeError(f"Evaluation count mismatch: expected={args.expected_count}, predictions={len(preds)}, GT={len(gts)}")
    missing, extra = sorted(set(gts) - set(preds)), sorted(set(preds) - set(gts))
    if missing or extra:
        raise RuntimeError(f"Prediction/GT mismatch: missing={len(missing)}, extra={len(extra)}")

    Emeasure, FmeasureHandler, FmeasureV2, MAE, Smeasure, WeightedFmeasure = metric_classes()
    metric_types = (Emeasure, FmeasureHandler, FmeasureV2, MAE, Smeasure, WeightedFmeasure)
    metrics = new_metric_set(*metric_types)

    for stem in tqdm(sorted(gts), desc="Evaluate", unit="image", dynamic_ncols=True):
        prediction = np.asarray(Image.open(preds[stem]).convert("L"))
        target = np.asarray(Image.open(gts[stem]).convert("L"))
        if prediction.shape != target.shape:
            prediction = np.asarray(Image.fromarray(prediction).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR))
        binary_uint8 = binarize_otsu(prediction).astype(np.uint8) * 255
        for metric in metrics.values():
            metric.step(binary_uint8, target)

    result = {
        **({"dataset": args.dataset} if args.dataset else {}),
        "images": len(gts),
        "binary_method": BINARY_METHOD,
        **metric_result(metrics),
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
