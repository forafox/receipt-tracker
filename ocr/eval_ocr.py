from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence

import cv2
import numpy as np

from common.ocr import OcrEngine
from ocr.config import OcrSettings
from ocr.dataset_det import build_detection_dataset
from ocr.dataset_rec import build_recognition_dataset
from ocr.factory import build_ocr_engine
from ocr.geometry import Box, axis_aligned, iou
from ocr.metrics import character_error_rate, word_error_rate
from ocr.ordering import blocks_to_text

RECOGNITION_UPSCALE = 3.0


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the OCR evaluation tool.

    Returns:
        Parsed evaluation configuration.
    """
    parser = argparse.ArgumentParser(description="Evaluate an OCR engine on the receipt dataset.")
    parser.add_argument("--engine", choices=("tesseract", "easyocr"), default="tesseract")
    parser.add_argument("--task", choices=("det", "rec"), default="det")
    parser.add_argument("--source", choices=("eng", "ru"), default="eng")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    return parser.parse_args()


def evaluate_detection(
    engine: OcrEngine, source: str, split: str, limit: int, seed: int, iou_threshold: float
) -> dict[str, float]:
    """Evaluate box detection against ground-truth polygons.

    Args:
        engine: OCR engine under evaluation.
        source: Dataset source identifier, either ``eng`` or ``ru``.
        split: Dataset split to evaluate on.
        limit: Maximum number of images to evaluate.
        seed: Seed for deterministic image sampling.
        iou_threshold: Minimum IoU for a prediction to count as a true positive.

    Returns:
        Detection metrics: true positives, false positives, false negatives, precision,
        recall, F1 and mean IoU of matched boxes.
    """
    dataset = build_detection_dataset(source, split)
    indices = _sample_indices(len(dataset), limit, seed)
    true_positives = false_positives = false_negatives = 0
    matched_ious: list[float] = []
    for index in indices:
        _, ground_truth = dataset[index]
        result = engine.recognize_blocks(dataset.image_paths[index].read_bytes())
        predicted = [axis_aligned(block.bbox) for block in result.blocks]
        image_tp, image_fp, image_fn, image_ious = _match_boxes(
            predicted, ground_truth, iou_threshold
        )
        true_positives += image_tp
        false_positives += image_fp
        false_negatives += image_fn
        matched_ious.extend(image_ious)
    return _detection_metrics(true_positives, false_positives, false_negatives, matched_ious)


def evaluate_recognition(
    engine: OcrEngine, source: str, split: str, limit: int, seed: int
) -> dict[str, float]:
    """Evaluate recognition on cropped text lines.

    Args:
        engine: OCR engine under evaluation.
        source: Dataset source identifier, either ``eng`` or ``ru``.
        split: Dataset split to evaluate on.
        limit: Maximum number of crops to evaluate.
        seed: Seed for deterministic crop sampling.

    Returns:
        Recognition metrics: mean character and word error rates and the exact-match ratio.
    """
    dataset = build_recognition_dataset(source, split)
    indices = _sample_indices(len(dataset), limit, seed)
    character_rates: list[float] = []
    word_rates: list[float] = []
    exact_matches = 0
    for index in indices:
        crop, reference = dataset[index]
        result = engine.recognize_blocks(_encode_crop(crop))
        hypothesis = _normalize(blocks_to_text(result))
        expected = _normalize(reference)
        character_rates.append(character_error_rate(expected, hypothesis))
        word_rates.append(word_error_rate(expected, hypothesis))
        exact_matches += int(expected == hypothesis)
    count = max(1, len(indices))
    return {
        "crops": float(len(indices)),
        "cer": sum(character_rates) / count,
        "wer": sum(word_rates) / count,
        "exact_match": exact_matches / count,
    }


def main() -> None:
    """Run the OCR evaluation tool."""
    args = parse_args()
    settings = OcrSettings(engine=args.engine, min_confidence=args.min_confidence)
    engine = build_ocr_engine(settings)
    if args.task == "det":
        metrics = evaluate_detection(
            engine, args.source, args.split, args.limit, args.seed, args.iou_threshold
        )
    else:
        metrics = evaluate_recognition(engine, args.source, args.split, args.limit, args.seed)
    summary = {"engine": args.engine, "task": args.task, "source": args.source, "split": args.split}
    summary.update(metrics)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


def _sample_indices(total: int, limit: int, seed: int) -> list[int]:
    if total == 0:
        return []
    count = min(max(0, limit), total)
    return sorted(random.Random(seed).sample(range(total), count))


def _match_boxes(
    predicted: Sequence[Box], ground_truth: Sequence[Box], iou_threshold: float
) -> tuple[int, int, int, list[float]]:
    pairs = sorted(
        (
            (iou(prediction, truth), prediction_index, truth_index)
            for prediction_index, prediction in enumerate(predicted)
            for truth_index, truth in enumerate(ground_truth)
        ),
        reverse=True,
    )
    used_predictions: set[int] = set()
    used_truths: set[int] = set()
    matched_ious: list[float] = []
    for score, prediction_index, truth_index in pairs:
        if score < iou_threshold:
            break
        if prediction_index in used_predictions or truth_index in used_truths:
            continue
        used_predictions.add(prediction_index)
        used_truths.add(truth_index)
        matched_ious.append(score)
    true_positives = len(matched_ious)
    false_positives = len(predicted) - true_positives
    false_negatives = len(ground_truth) - true_positives
    return true_positives, false_positives, false_negatives, matched_ious


def _detection_metrics(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    matched_ious: list[float],
) -> dict[str, float]:
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return {
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def _encode_crop(crop: np.ndarray) -> bytes:
    scaled = cv2.resize(
        crop,
        None,
        fx=RECOGNITION_UPSCALE,
        fy=RECOGNITION_UPSCALE,
        interpolation=cv2.INTER_CUBIC,
    )
    success, buffer = cv2.imencode(".png", scaled)
    if not success:
        raise RuntimeError("Cannot encode recognition crop")
    return buffer.tobytes()


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


if __name__ == "__main__":
    main()
