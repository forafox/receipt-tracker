from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from common.ocr import TextBlock
from ocr.core.geometry import axis_aligned, iou
from ocr.core.ordering import group_blocks_into_lines


@dataclass(frozen=True)
class PostprocessSettings:
    """Settings for shared OCR block postprocessing.

    Attributes:
        min_box_area: Minimum axis-aligned box area in square pixels to keep a block.
        nms_iou_threshold: Maximum IoU with a stronger block before a block is suppressed.
    """

    min_box_area: float = 9.0
    nms_iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.min_box_area < 0:
            raise ValueError("min_box_area must be non-negative")
        if not 0 < self.nms_iou_threshold <= 1:
            raise ValueError("nms_iou_threshold must be in the (0, 1] range")


def postprocess_blocks(
    blocks: Sequence[TextBlock], settings: PostprocessSettings
) -> list[TextBlock]:
    """Filter tiny and overlapping blocks and sort the rest into reading order.

    Blocks with empty text or a box smaller than ``min_box_area`` are dropped. Among
    overlapping blocks, the one with the highest confidence is kept. The result is ordered
    top to bottom and left to right.

    Args:
        blocks: Blocks produced by an OCR engine.
        settings: Filtering and ordering thresholds.

    Returns:
        Kept blocks in reading order.
    """
    sizable = [
        block
        for block in blocks
        if block.text.strip() and _box_area(block) >= settings.min_box_area
    ]
    kept = _suppress_overlaps(sizable, settings.nms_iou_threshold)
    return [block for line in group_blocks_into_lines(kept) for block in line]


def _box_area(block: TextBlock) -> float:
    left, top, right, bottom = axis_aligned(block.bbox)
    return max(0.0, right - left) * max(0.0, bottom - top)


def _suppress_overlaps(blocks: Sequence[TextBlock], threshold: float) -> list[TextBlock]:
    ranked = sorted(
        blocks,
        key=lambda block: (-_confidence(block), *_top_left(block), block.text),
    )
    kept: list[TextBlock] = []
    for block in ranked:
        rect = axis_aligned(block.bbox)
        if all(iou(rect, axis_aligned(other.bbox)) <= threshold for other in kept):
            kept.append(block)
    return kept


def _confidence(block: TextBlock) -> float:
    return block.confidence if block.confidence is not None else 1.0


def _top_left(block: TextBlock) -> tuple[float, float]:
    left, top, _, _ = axis_aligned(block.bbox)
    return top, left
