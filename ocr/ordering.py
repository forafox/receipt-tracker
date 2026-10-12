from __future__ import annotations

from common.ocr import OcrResult, TextBlock
from ocr.geometry import Box, axis_aligned


def blocks_to_text(result: OcrResult, *, line_overlap: float = 0.5) -> str:
    """Assemble recognized blocks into reading-order text.

    Blocks whose vertical spans overlap by at least ``line_overlap`` of a block height are
    joined into the same line and separated by a space. Lines are ordered top to bottom and
    blocks inside a line left to right.

    Args:
        result: OCR result whose blocks should be assembled.
        line_overlap: Minimum vertical overlap ratio for two blocks to share a line.

    Returns:
        Recognized text with one line per assembled line of the receipt.
    """
    rects = [(block, axis_aligned(block.bbox)) for block in result.blocks]
    lines: list[list[tuple[TextBlock, Box]]] = []
    for block, rect in sorted(rects, key=lambda item: (item[1][1], item[1][0])):
        line = _matching_line(lines, rect, line_overlap)
        if line is None:
            lines.append([(block, rect)])
        else:
            line.append((block, rect))

    ordered = sorted(lines, key=lambda line: min(rect[1] for _, rect in line))
    return "\n".join(
        " ".join(block.text for block, _ in sorted(line, key=lambda item: item[1][0]))
        for line in ordered
    )


def _matching_line(
    lines: list[list[tuple[TextBlock, Box]]], rect: Box, line_overlap: float
) -> list[tuple[TextBlock, Box]] | None:
    best: list[tuple[TextBlock, Box]] | None = None
    best_overlap = 0.0
    for line in lines:
        top = min(item[1][1] for item in line)
        bottom = max(item[1][3] for item in line)
        overlap = min(bottom, rect[3]) - max(top, rect[1])
        height = rect[3] - rect[1]
        ratio = overlap / height if height > 0 else 0.0
        if ratio >= line_overlap and ratio > best_overlap:
            best = line
            best_overlap = ratio
    return best
