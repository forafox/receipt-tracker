from dataclasses import dataclass
from math import isfinite

from common.ocr import OcrResult, Point, TextBlock


@dataclass(frozen=True)
class LayoutLine:
    """A physical receipt row retaining original OCR block references.

    Attributes:
        blocks: Original fragments ordered from left to right, with geometry and confidence.
        x_min: Left edge of the row's axis-aligned bounds at reconstruction time.
        y_min: Top edge of the row's axis-aligned bounds at reconstruction time.
        x_max: Right edge of the row's axis-aligned bounds at reconstruction time.
        y_max: Bottom edge of the row's axis-aligned bounds at reconstruction time.
    """

    blocks: tuple[TextBlock, ...]
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def text(self) -> str:
        """Return the original fragment texts joined by one space."""
        return " ".join(block.text for block in self.blocks)


@dataclass(frozen=True)
class _PositionedBlock:
    block: TextBlock
    x_min: float
    y_min: float
    x_max: float
    y_max: float


def reconstruct_layout(result: OcrResult) -> list[LayoutLine]:
    """Group OCR fragments into physical rows without discarding their geometry.

    Every pair of fragments in a row must overlap vertically by at least half
    the taller fragment's height. Pairwise comparison prevents an expanding
    row envelope from bridging neighboring rows. This relative heuristic handles
    modest offsets, not severe skew or blocks spanning multiple physical rows.

    Fragments are considered top to bottom and assigned to the earliest compatible
    row. Equal positions are ordered by bounds, text, polygon points and confidence.
    Zero-height blocks form separate rows; zero-width blocks still use their Y spans.
    Source blocks are retained by reference and are never modified or split.

    Args:
        result: Recognized receipt fragments, in any engine order.

    Returns:
        Rows ordered top to bottom, each containing fragments ordered left to right.

    Raises:
        ValueError: If a bounding-box coordinate is not finite.
    """
    positioned = [_position_block(block) for block in result.blocks]
    rows: list[list[_PositionedBlock]] = []
    for block in sorted(positioned, key=lambda block: (block.y_min, _block_order_key(block))):
        row = next(
            (row for row in rows if all(_vertically_aligned(block, other) for other in row)),
            None,
        )
        if row is None:
            rows.append([block])
        else:
            row.append(block)
    return [_make_line(row) for row in rows]


def _position_block(block: TextBlock) -> _PositionedBlock:
    if not all(isfinite(coordinate) for point in block.bbox for coordinate in point):
        raise ValueError("Bounding-box coordinates must be finite")
    return _PositionedBlock(
        block=block,
        x_min=min(x for x, _ in block.bbox),
        y_min=min(y for _, y in block.bbox),
        x_max=max(x for x, _ in block.bbox),
        y_max=max(y for _, y in block.bbox),
    )


def _block_order_key(
    positioned: _PositionedBlock,
) -> tuple[float, float, float, float, str, tuple[Point, ...], float]:
    block = positioned.block
    return (
        positioned.x_min,
        positioned.y_min,
        positioned.x_max,
        positioned.y_max,
        block.text,
        tuple(block.bbox),
        -1.0 if block.confidence is None else block.confidence,
    )


def _vertically_aligned(first: _PositionedBlock, second: _PositionedBlock) -> bool:
    first_height = first.y_max - first.y_min
    second_height = second.y_max - second.y_min
    if first_height == 0 or second_height == 0:
        return False
    overlap = min(first.y_max, second.y_max) - max(first.y_min, second.y_min)
    return overlap >= 0.5 * max(first_height, second_height)


def _make_line(row: list[_PositionedBlock]) -> LayoutLine:
    return LayoutLine(
        blocks=tuple(positioned.block for positioned in sorted(row, key=_block_order_key)),
        x_min=min(block.x_min for block in row),
        y_min=min(block.y_min for block in row),
        x_max=max(block.x_max for block in row),
        y_max=max(block.y_max for block in row),
    )
