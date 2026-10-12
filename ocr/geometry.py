from __future__ import annotations

from collections.abc import Sequence

from common.ocr import Point

Box = tuple[float, float, float, float]


def axis_aligned(bbox: Sequence[Point]) -> Box:
    """Convert a quadrilateral to its axis-aligned bounding rectangle.

    Args:
        bbox: Polygon points in ``(x, y)`` form.

    Returns:
        Rectangle ``(x1, y1, x2, y2)`` covering all polygon points.

    Raises:
        ValueError: If ``bbox`` is empty.
    """
    if not bbox:
        raise ValueError("bbox must contain at least one point")
    xs = [point[0] for point in bbox]
    ys = [point[1] for point in bbox]
    return (min(xs), min(ys), max(xs), max(ys))


def iou(first: Box, second: Box) -> float:
    """Compute the intersection-over-union of two rectangles.

    Args:
        first: First rectangle ``(x1, y1, x2, y2)``.
        second: Second rectangle ``(x1, y1, x2, y2)``.

    Returns:
        Intersection-over-union in the ``0..1`` range, or ``0.0`` for empty rectangles.
    """
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection == 0.0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union
