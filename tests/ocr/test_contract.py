from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.ocr import OcrResult, TextBlock
from ocr.errors import ImageDecodeError
from ocr.geometry import axis_aligned, iou
from ocr.image import decode_image
from ocr.ordering import blocks_to_text


def _block(text: str, x1: float, y1: float, x2: float, y2: float) -> TextBlock:
    return TextBlock(text=text, bbox=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)])


def test_text_block_requires_four_points() -> None:
    with pytest.raises(ValidationError):
        TextBlock(text="x", bbox=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


def test_text_block_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TextBlock(text="x", bbox=[(0, 0), (1, 0), (1, 1), (0, 1)], extra=1)  # type: ignore[call-arg]


def test_text_block_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValidationError):
        TextBlock(text="x", bbox=[(0, 0), (1, 0), (1, 1), (0, 1)], confidence=1.5)


def test_axis_aligned_covers_polygon() -> None:
    box = axis_aligned([(5.0, 8.0), (1.0, 2.0), (4.0, 9.0)])
    assert box == (1.0, 2.0, 5.0, 9.0)


def test_axis_aligned_rejects_empty_polygon() -> None:
    with pytest.raises(ValueError):
        axis_aligned([])


def test_iou_identical_and_disjoint() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0
    assert iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_half_overlap() -> None:
    assert iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest.approx(50 / 150)


def test_blocks_to_text_orders_lines_top_to_bottom() -> None:
    result = OcrResult(
        blocks=[
            _block("ПЯТЁРОЧКА", 0, 20, 80, 30),
            _block("ИТОГО", 0, 0, 40, 10),
            _block("250,00", 50, 0, 90, 10),
        ],
        image_width=100,
        image_height=40,
    )

    assert blocks_to_text(result) == "ИТОГО 250,00\nПЯТЁРОЧКА"


def test_blocks_to_text_separates_distant_blocks() -> None:
    result = OcrResult(blocks=[_block("верх", 0, 0, 20, 10), _block("низ", 0, 50, 20, 60)])

    assert blocks_to_text(result) == "верх\nниз"


def test_decode_image_rejects_garbage() -> None:
    with pytest.raises(ImageDecodeError):
        decode_image(b"not an image")
