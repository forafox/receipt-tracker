from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common.ocr import OcrResult
from ocr.config import OcrSettings
from ocr.dataset_det import Box, build_detection_dataset
from ocr.factory import build_ocr_engine
from ocr.geometry import axis_aligned

PREVIEW_DIR = Path(__file__).resolve().parent / "_preview"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the OCR preview tool.

    Returns:
        Parsed preview configuration.
    """
    parser = argparse.ArgumentParser(description="Render OCR output over a dataset image.")
    parser.add_argument("--engine", choices=("tesseract", "easyocr"), default="tesseract")
    parser.add_argument("--source", choices=("eng", "ru"), default="eng")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def render(image: np.ndarray, boxes: list[Box], result: OcrResult) -> np.ndarray:
    """Draw ground-truth boxes and OCR predictions on an image copy.

    Args:
        image: Source receipt image.
        boxes: Ground-truth boxes in ``(x1, y1, x2, y2)`` pixel form.
        result: OCR result to draw.

    Returns:
        Image copy with ground truth in green and predictions in red.
    """
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
    for block in result.blocks:
        polygon = np.array(axis_aligned(block.bbox), dtype=np.int32)
        cv2.rectangle(
            canvas,
            (polygon[0], polygon[1]),
            (polygon[2], polygon[3]),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            canvas,
            block.text,
            (polygon[0], max(12, polygon[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    """Run the OCR preview tool."""
    args = parse_args()
    settings = OcrSettings(engine=args.engine, min_confidence=args.min_confidence)
    engine = build_ocr_engine(settings)

    dataset = build_detection_dataset(args.source, args.split)
    total = len(dataset)
    if total == 0:
        raise SystemExit(f"Dataset {args.source}/{args.split} is empty")
    index = args.index % total
    image, boxes = dataset[index]
    image_bytes = dataset.image_paths[index].read_bytes()
    result = engine.recognize_blocks(image_bytes)

    output = args.output or PREVIEW_DIR / f"{args.engine}_{args.source}_{args.split}_{index}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), render(image, boxes, result))
    print(
        f"{args.engine}: {len(result.blocks)} blocks, {len(boxes)} ground-truth boxes -> {output}"
    )
    for block in result.blocks:
        confidence = "-" if block.confidence is None else f"{block.confidence:.2f}"
        print(f"  [{confidence}] {block.text}")


if __name__ == "__main__":
    main()
