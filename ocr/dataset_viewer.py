from __future__ import annotations

import argparse
import random

import cv2
import numpy as np

from ocr.dataset_det import DetectionDataset, build_detection_dataset
from ocr.dataset_rec import RecognitionDataset, build_recognition_dataset

WINDOW = "receipt-dataset-viewer"
HEADER = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview receipt dataset items with OpenCV.")
    parser.add_argument("--task", choices=("det", "rec"), default="det")
    parser.add_argument("--source", choices=("eng", "ru"), default="eng")
    parser.add_argument("--split", default="test")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--keep-arabic",
        action="store_true",
        help="Do not filter Arabic receipts from the eng detection dataset.",
    )
    parser.add_argument(
        "--drop-unknown",
        action="store_true",
        help="Drop eng detection receipts that have no OCR label to classify language.",
    )
    return parser.parse_args()


def render_detection(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    canvas = image.copy()
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
    return canvas


def render_recognition(image: np.ndarray, text: str) -> np.ndarray:
    height, width = image.shape[:2]
    canvas = np.full((height + HEADER, width, 3), 255, dtype=np.uint8)
    canvas[HEADER:, :] = image
    max_chars = max(1, width // 12)
    label = text if len(text) <= max_chars else text[: max_chars - 1] + "\u2026"
    cv2.putText(
        canvas,
        label,
        (6, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return canvas


def clamp_window(
    canvas: np.ndarray, max_width: int = 1200, max_height: int = 900
) -> tuple[int, int]:
    height, width = canvas.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    return int(width * scale), int(height * scale)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset: DetectionDataset | RecognitionDataset
    if args.task == "det":
        dataset = build_detection_dataset(
            args.source,
            args.split,
            exclude_arabic=not args.keep_arabic,
            drop_unknown=args.drop_unknown,
        )
    else:
        dataset = build_recognition_dataset(args.source, args.split)

    total = len(dataset)
    if total == 0:
        raise SystemExit(f"Dataset {args.source}/{args.split}/{args.task} is empty")

    indices = rng.sample(range(total), min(args.count, total))
    title = f"{args.task} {args.source}/{args.split}"
    window = f"{WINDOW} - {title}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        for position, index in enumerate(indices, start=1):
            image, payload = dataset[index]
            if args.task == "det":
                canvas = render_detection(image, payload)  # type: ignore[arg-type]
                print(f"[{position}/{len(indices)}] #{index} boxes={len(payload)}")
            else:
                canvas = render_recognition(image, payload)  # type: ignore[arg-type]
                print(f"[{position}/{len(indices)}] #{index} text={payload!r}")

            cv2.imshow(window, canvas)
            # cv2.resizeWindow(window, *clamp_window(canvas))
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
