from __future__ import annotations

from itertools import permutations

import pytest

from common.ocr import TextBlock
from ocr.core.postprocess import PostprocessSettings, postprocess_blocks


def _block(
    text: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float | None = None,
) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        confidence=confidence,
    )


def test_overlapping_blocks_keep_highest_confidence() -> None:
    stronger = _block("ИТОГО", 0, 0, 10, 10, 0.9)
    weaker = _block("ИТОГО", 0, 0, 10, 10, 0.8)

    assert postprocess_blocks([weaker, stronger], PostprocessSettings()) == [stronger]


def test_non_overlapping_blocks_are_preserved() -> None:
    first = _block("верх", 0, 0, 10, 10, 0.9)
    second = _block("низ", 0, 20, 10, 30, 0.8)

    assert postprocess_blocks([second, first], PostprocessSettings()) == [first, second]


def test_iou_equal_to_threshold_is_not_suppressed() -> None:
    first = _block("верх", 0, 0, 10, 10, 0.9)
    second = _block("низ", 0, 0, 10, 5, 0.8)
    settings = PostprocessSettings(nms_iou_threshold=0.5)

    assert len(postprocess_blocks([first, second], settings)) == 2


def test_iou_above_threshold_is_suppressed() -> None:
    first = _block("верх", 0, 0, 10, 10, 0.9)
    second = _block("низ", 0, 0, 10, 5, 0.8)
    settings = PostprocessSettings(nms_iou_threshold=0.49)

    assert postprocess_blocks([first, second], settings) == [first]


def test_tiny_boxes_are_filtered_and_boundary_is_kept() -> None:
    tiny = _block("шум", 0, 0, 2, 2, 0.9)
    boundary = _block("ok", 10, 10, 13, 13, 0.9)

    assert postprocess_blocks([tiny, boundary], PostprocessSettings()) == [boundary]


def test_empty_text_is_filtered() -> None:
    empty = _block(" ", 0, 0, 10, 10, 0.9)
    text = _block("текст", 20, 0, 30, 10, 0.9)

    assert postprocess_blocks([empty, text], PostprocessSettings()) == [text]


def test_blocks_are_sorted_into_reading_order() -> None:
    bottom = _block("низ", 0, 20, 10, 30, 0.9)
    right = _block("право", 20, 0, 30, 10, 0.9)
    left = _block("лево", 0, 0, 10, 10, 0.9)

    assert postprocess_blocks([bottom, right, left], PostprocessSettings()) == [
        left,
        right,
        bottom,
    ]


def test_missing_confidence_is_kept() -> None:
    block = _block("текст", 0, 0, 10, 10, None)

    assert postprocess_blocks([block], PostprocessSettings()) == [block]


def test_result_is_independent_of_input_order() -> None:
    blocks = [
        _block("низ", 0, 20, 10, 30, 0.9),
        _block("право", 20, 0, 30, 10, 0.9),
        _block("лево", 0, 0, 10, 10, 0.9),
        _block("дубль", 20, 0, 30, 10, 0.7),
    ]
    expected = postprocess_blocks(blocks, PostprocessSettings())

    for ordering in permutations(blocks):
        assert postprocess_blocks(ordering, PostprocessSettings()) == expected


def test_empty_input_returns_empty_list() -> None:
    assert postprocess_blocks([], PostprocessSettings()) == []


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        PostprocessSettings(nms_iou_threshold=0)
