from __future__ import annotations

import json
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from ocr.dataset_rec import contains_arabic

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

Box = tuple[int, int, int, int]
_BoxN = tuple[float, float, float, float]
_Item = tuple[Path, list[_BoxN]]

_ENG_IMAGE_DIRS = {
    "test": Path("detection/eng_arab/test/images"),
    "val": Path("detection/eng_arab/val/val/images"),
}
_ENG_SPLITS = tuple(_ENG_IMAGE_DIRS)
_RU_SPLITS = ("test", "train", "validation")


def _to_pixels(box: _BoxN, width: int, height: int) -> Box | None:
    x1 = max(0, min(width, int(round(box[0] * width))))
    y1 = max(0, min(height, int(round(box[1] * height))))
    x2 = max(0, min(width, int(round(box[2] * width))))
    y2 = max(0, min(height, int(round(box[3] * height))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2, y2)


def _polygon_to_box(bbox: Any) -> _BoxN | None:
    if not isinstance(bbox, list) or not bbox:
        return None
    xs = [float(point[0]) for point in bbox if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in bbox if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _ru_split_dir(split: str) -> str:
    return "validation" if split == "val" else split


def receipt_uuid(stem: str) -> str:
    """Extract a receipt identifier from a recognition crop stem.

    Args:
        stem: Crop filename stem such as ``<uuid>_line_7`` or ``<uuid>_date``.

    Returns:
        Receipt UUID prefix shared by related crops.
    """
    return stem.split("_", 1)[0]


def _parse_ocr_text(txt_path: Path) -> str:
    try:
        data: Any = json.loads(txt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(data, list):
        return " ".join(str(part) for part in data).strip()
    return str(data).strip()


@lru_cache(maxsize=4)
def load_ocr_index(root: Path) -> dict[str, list[str]]:
    """Map receipt UUIDs to OCR line texts across all recognition splits.

    The CORU recognition crops share receipt UUIDs with detection images, so
    they are the only local signal for a receipt's language.

    Args:
        root: Root directory containing recognition datasets.

    Returns:
        OCR line texts grouped by receipt UUID.
    """
    index: dict[str, list[str]] = {}
    for split in ("test", "train", "val"):
        directory = root / "recognition/eng_arab" / split / split
        if not directory.is_dir():
            continue
        for txt_path in directory.glob("*.txt"):
            text = _parse_ocr_text(txt_path)
            if text:
                index.setdefault(receipt_uuid(txt_path.stem), []).append(text)
    return index


def arabic_ratio(texts: list[str]) -> float:
    """Calculate the share of OCR lines containing Arabic characters.

    Args:
        texts: OCR line texts to inspect.

    Returns:
        Fraction of lines containing at least one Arabic character.
    """
    if not texts:
        return 0.0
    return sum(1 for text in texts if contains_arabic(text)) / len(texts)


class DetectionDataset(ABC):
    """Base class for receipt text-detection datasets.

    An item is returned as ``(image, boxes)`` where ``image`` is a BGR
    ``numpy`` array and ``boxes`` is a list of absolute pixel
    ``(x1, y1, x2, y2)`` tuples.
    """

    def __init__(self, split: str, root: Path | None = None) -> None:
        self.split: str = split
        self.root: Path = root if root is not None else DATASET_ROOT
        self._items: list[_Item] = self._load()

    @abstractmethod
    def _load(self) -> list[_Item]:
        """Return ``(image_path, normalized_boxes)`` entries."""

    def __len__(self) -> int:
        return len(self._items)

    @property
    def image_paths(self) -> list[Path]:
        """Return image paths represented by this dataset.

        Returns:
            Paths in dataset iteration order.
        """
        return [path for path, _ in self._items]

    def __getitem__(self, index: int) -> tuple[np.ndarray, list[Box]]:
        image_path, normalized = self._items[index]
        image: np.ndarray | None = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        boxes = [
            box for box in (_to_pixels(raw, width, height) for raw in normalized) if box is not None
        ]
        return image, boxes


class EngDetectionDataset(DetectionDataset):
    """CORU English/Arabic detection (COCO JSON, absolute ``[x, y, w, h]``).

    Arabic receipts are filtered using the linked OCR line labels (matched by
    receipt UUID) when ``exclude_arabic`` is set. Receipts without OCR labels
    cannot be classified and are kept unless ``drop_unknown`` is set.
    """

    def __init__(
        self,
        split: str,
        root: Path | None = None,
        *,
        exclude_arabic: bool = True,
        arabic_threshold: float = 0.25,
        drop_unknown: bool = False,
    ) -> None:
        self.exclude_arabic = exclude_arabic
        self.arabic_threshold = arabic_threshold
        self.drop_unknown = drop_unknown
        super().__init__(split, root)

    def _load(self) -> list[_Item]:
        if self.split not in _ENG_IMAGE_DIRS:
            raise ValueError(f"Unsupported eng split {self.split!r}; expected one of {_ENG_SPLITS}")

        images_dir = self.root / _ENG_IMAGE_DIRS[self.split]
        json_path = self.root / "detection/eng_arab" / f"{self.split}.json"
        with json_path.open(encoding="utf-8") as handle:
            data: Any = json.load(handle)

        raw_images = cast(list[dict[str, Any]], data["images"])
        raw_annotations = cast(list[dict[str, Any]], data["annotations"])

        sizes: dict[int, tuple[str, int, int]] = {}
        for raw_image in raw_images:
            sizes[int(raw_image["id"])] = (
                str(raw_image["file_name"]),
                int(raw_image["width"]),
                int(raw_image["height"]),
            )

        grouped: dict[int, list[_BoxN]] = {}
        for annotation in raw_annotations:
            image_id = int(annotation["image_id"])
            if image_id not in sizes:
                continue
            _, width, height = sizes[image_id]
            if width <= 0 or height <= 0:
                continue
            x, y, w, h = (float(value) for value in annotation["bbox"])
            grouped.setdefault(image_id, []).append(
                (x / width, y / height, (x + w) / width, (y + h) / height)
            )

        ocr_index: dict[str, list[str]] = load_ocr_index(self.root) if self.exclude_arabic else {}

        items: list[_Item] = []
        for image_id, (file_name, _, _) in sizes.items():
            image_path = images_dir / file_name
            if not image_path.is_file():
                continue
            if self.exclude_arabic:
                texts = ocr_index.get(receipt_uuid(image_path.stem))
                if not texts:
                    if self.drop_unknown:
                        continue
                elif arabic_ratio(texts) > self.arabic_threshold:
                    continue
            items.append((image_path, grouped.get(image_id, [])))
        return items


class RuDetectionDataset(DetectionDataset):
    """cdek-ocr Russian detection (JSONL, normalized 4-point polygons)."""

    def _load(self) -> list[_Item]:
        split = _ru_split_dir(self.split)
        if split not in _RU_SPLITS:
            raise ValueError(f"Unsupported ru split {self.split!r}; expected one of {_RU_SPLITS}")

        base = self.root / "detection/ru"
        jsonl_path = base / "annotations" / f"{split}.jsonl"

        items: list[_Item] = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = cast(dict[str, Any], json.loads(line))
                image_ref = record.get("image")
                if not isinstance(image_ref, str):
                    continue
                image_path = base / image_ref
                if not image_path.is_file():
                    continue
                items.append((image_path, _ru_record_boxes(record)))
        return items


def _ru_record_boxes(record: dict[str, Any]) -> list[_BoxN]:
    boxes: list[_BoxN] = []
    for field in ("seller", "inn", "date", "total"):
        value = record.get(field)
        if isinstance(value, dict):
            box = _polygon_to_box(value.get("bbox"))
            if box is not None:
                boxes.append(box)
    items = record.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                box = _polygon_to_box(item.get("bbox"))
                if box is not None:
                    boxes.append(box)
    return boxes


def build_detection_dataset(
    source: str,
    split: str,
    root: Path | None = None,
    *,
    exclude_arabic: bool = True,
    arabic_threshold: float = 0.25,
    drop_unknown: bool = False,
) -> DetectionDataset:
    """Build a receipt text-detection dataset for a supported source.

    Args:
        source: Dataset source identifier, either ``eng`` or ``ru``.
        split: Dataset split to load.
        root: Optional dataset root overriding the repository default.
        exclude_arabic: Whether to exclude Arabic receipts from the English source.
        arabic_threshold: Maximum allowed ratio of Arabic OCR lines.
        drop_unknown: Whether to drop receipts without OCR labels for language detection.

    Returns:
        Detection dataset configured for the requested source and split.
    """
    if source == "eng":
        return EngDetectionDataset(
            split,
            root,
            exclude_arabic=exclude_arabic,
            arabic_threshold=arabic_threshold,
            drop_unknown=drop_unknown,
        )
    if source == "ru":
        return RuDetectionDataset(split, root)
    raise ValueError(f"Unknown detection source: {source!r}")
