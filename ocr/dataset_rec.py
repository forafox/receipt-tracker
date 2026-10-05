from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NamedTuple, cast

import cv2
import numpy as np

DATASET_ROOT = Path(__file__).resolve().parents[1] / "dataset"

ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

_BoxN = tuple[float, float, float, float]
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def contains_arabic(text: str) -> bool:
    return ARABIC_RE.search(text) is not None


class _RecItem(NamedTuple):
    image_path: Path
    text: str
    box: _BoxN | None


def _ru_split_dir(split: str) -> str:
    return "validation" if split == "val" else split


def _crop(image: np.ndarray, box: _BoxN) -> np.ndarray:
    height, width = image.shape[:2]
    x1 = max(0, min(width, int(round(box[0] * width))))
    y1 = max(0, min(height, int(round(box[1] * height))))
    x2 = max(0, min(width, int(round(box[2] * width))))
    y2 = max(0, min(height, int(round(box[3] * height))))
    return image[y1:y2, x1:x2].copy()


class RecognitionDataset(ABC):
    """Base class for receipt text-recognition datasets.

    An item is returned as ``(image, text)`` where ``image`` is a BGR
    ``numpy`` array (a crop for Russian samples) and ``text`` is its label.
    Arabic-annotated samples are excluded.
    """

    def __init__(self, split: str, root: Path | None = None) -> None:
        self.split: str = split
        self.root: Path = root if root is not None else DATASET_ROOT
        self._items: list[_RecItem] = self._load()
        self._cached_path: Path | None = None
        self._cached_image: np.ndarray | None = None

    @abstractmethod
    def _load(self) -> list[_RecItem]:
        """Return ``(image_path, text, normalized_crop)`` entries."""

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        item = self._items[index]
        image = self._read(item.image_path)
        if item.box is None:
            return image.copy(), item.text
        return _crop(image, item.box), item.text

    def _read(self, path: Path) -> np.ndarray:
        if self._cached_path != path or self._cached_image is None:
            image: np.ndarray | None = cv2.imread(str(path))
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {path}")
            self._cached_path = path
            self._cached_image = image
        return self._cached_image


class EngRecognitionDataset(RecognitionDataset):
    """CORU English/Arabic OCR crops (``<name>.jpg`` + ``<name>.txt``)."""

    def _load(self) -> list[_RecItem]:
        base = self.root / "recognition/eng_arab" / self.split / self.split
        if not base.is_dir():
            raise ValueError(f"Recognition split directory not found: {base}")

        items: list[_RecItem] = []
        for txt_path in sorted(base.glob("*.txt")):
            image_path = _sibling_image(txt_path)
            if image_path is None:
                continue
            text = _parse_eng_text(txt_path)
            if not text or contains_arabic(text):
                continue
            items.append(_RecItem(image_path, text, None))
        return items


class RuRecognitionDataset(RecognitionDataset):
    """Russian recognition synthesized by cropping text boxes of the ru detection set."""

    def _load(self) -> list[_RecItem]:
        split = _ru_split_dir(self.split)
        base = self.root / "detection/ru"
        jsonl_path = base / "annotations" / f"{split}.jsonl"

        items: list[_RecItem] = []
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
                items.extend(_ru_record_items(record, image_path))
        return items


def _sibling_image(txt_path: Path) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = txt_path.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    return None


def _parse_eng_text(txt_path: Path) -> str:
    try:
        data: Any = json.loads(txt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if isinstance(data, list):
        return " ".join(str(part) for part in data if str(part).strip()).strip()
    return str(data).strip()


def _ru_record_items(record: dict[str, Any], image_path: Path) -> list[_RecItem]:
    entries: list[_RecItem] = []
    for field in ("seller", "inn", "date", "total"):
        value = record.get(field)
        if isinstance(value, dict):
            _append_entry(entries, image_path, value)
    raw_items = record.get("items")
    if isinstance(raw_items, list):
        for item in raw_items:
            if isinstance(item, dict):
                _append_entry(entries, image_path, item)
    return entries


def _append_entry(entries: list[_RecItem], image_path: Path, entry: dict[str, Any]) -> None:
    box = _polygon_to_box(entry.get("bbox"))
    text = entry.get("text")
    if box is None or not isinstance(text, str):
        return
    text = text.strip()
    if not text or contains_arabic(text):
        return
    entries.append(_RecItem(image_path, text, box))


def _polygon_to_box(bbox: Any) -> _BoxN | None:
    if not isinstance(bbox, list) or not bbox:
        return None
    xs = [float(point[0]) for point in bbox if isinstance(point, list) and len(point) >= 2]
    ys = [float(point[1]) for point in bbox if isinstance(point, list) and len(point) >= 2]
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def build_recognition_dataset(
    source: str, split: str, root: Path | None = None
) -> RecognitionDataset:
    if source == "eng":
        return EngRecognitionDataset(split, root)
    if source == "ru":
        return RuRecognitionDataset(split, root)
    raise ValueError(f"Unknown recognition source: {source!r}")
