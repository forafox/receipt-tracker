import re

from common.contracts import Item
from extract._text import _AMOUNT_PATTERN, _parse_decimal
from extract.item_region import _select_item_candidate_indices
from extract.items import _is_item_name, _parse_simple_item_line
from extract.layout import LayoutLine

_QUANTITY_PATTERN = re.compile(
    rf"(?<![\w.,+\-])(?P<quantity>\d+(?:[.,]\d+)?)\s*"
    rf"(?:кг|kg|г|g|шт\.?|л|l)?\s*[xх×*]\s*"
    rf"(?P<price>{_AMOUNT_PATTERN.pattern})(?:\s*=\s*|\s+)"
    rf"(?P<total>{_AMOUNT_PATTERN.pattern})$",
    re.IGNORECASE,
)
_NUMERIC_FRAGMENT = re.compile(r"[\d\s.,]+")
_PERCENT_FRAGMENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")


def _extract_layout_items(lines: list[LayoutLine], *, store: str | None) -> list[Item]:
    """Associate adjacent descriptions with explicit amounts, without deriving values.

    Description fragments and named rows must align within one text height;
    vertical gaps must be nonnegative and no larger than one text height. A
    nameless quantity row must start beneath the description, and a bare amount
    must lie to its right. Metadata and unsupported numeric rows break continuity.
    Overlapping blocks and multiple amounts without a multiplication label are skipped.
    An aligned unpriced fragment can still resemble a separate product missing
    its amount; this heuristic cannot resolve that ambiguity reliably.

    Args:
        lines: Receipt rows reconstructed from OCR geometry.
        store: Previously identified merchant header, if available.

    Returns:
        Items in reading order. All numeric fields come from explicit OCR text;
        weight quantities retain the printed numeric value without unit conversion.
    """
    candidates = set(_select_item_candidate_indices([line.text for line in lines], store=store))
    pending: list[LayoutLine] = []
    items: list[Item] = []
    for index, line in enumerate(lines):
        if index not in candidates or not _has_usable_geometry(line):
            pending.clear()
            continue
        if pending and not _nearby(pending[-1], line):
            pending.clear()
        text = line.text.strip()
        quantity_match = _QUANTITY_PATTERN.search(text)
        if quantity_match is not None:
            name = text[: quantity_match.start()].strip()
            if name and not (_is_item_name(name) or _PERCENT_FRAGMENT.fullmatch(name)):
                pending.clear()
                continue
            name = _complete_name(name, pending, line)
            quantity = _parse_decimal(quantity_match.group("quantity"))
            if _is_item_name(name) and quantity > 0:
                items.append(
                    Item(
                        name=name,
                        quantity=quantity,
                        price=_parse_decimal(quantity_match.group("price")),
                        sum=_parse_decimal(quantity_match.group("total")),
                    )
                )
        elif item := _simple_row_item(line):
            item.name = _complete_name(item.name, pending, line)
            items.append(item)
        elif _AMOUNT_PATTERN.fullmatch(text):
            if pending and line.x_min >= max(fragment.x_max for fragment in pending):
                name = " ".join(fragment.text.strip() for fragment in pending)
                if _is_item_name(name):
                    items.append(Item(name=name, sum=_parse_decimal(text)))
        elif _is_item_name(text) or _PERCENT_FRAGMENT.fullmatch(text):
            if pending and not _left_aligned(pending[0], line):
                pending.clear()
            pending.append(line)
            continue
        pending.clear()
    return items


def _has_usable_geometry(line: LayoutLine) -> bool:
    previous_right = float("-inf")
    for block in line.blocks:
        left = min(x for x, _ in block.bbox)
        right = max(x for x, _ in block.bbox)
        top = min(y for _, y in block.bbox)
        bottom = max(y for _, y in block.bbox)
        if right <= left or bottom <= top or left < previous_right:
            return False
        previous_right = right
    return bool(line.blocks)


def _nearby(first: LayoutLine, second: LayoutLine) -> bool:
    height = min(first.y_max - first.y_min, second.y_max - second.y_min)
    return 0 <= second.y_min - first.y_max <= height


def _left_aligned(first: LayoutLine, second: LayoutLine) -> bool:
    height = min(first.y_max - first.y_min, second.y_max - second.y_min)
    return abs(first.x_min - second.x_min) <= height


def _complete_name(name: str, pending: list[LayoutLine], line: LayoutLine) -> str:
    if not pending:
        return name
    if name and not _left_aligned(pending[0], line):
        return name
    if not name and not pending[0].x_min <= line.x_min <= pending[0].x_max:
        return name
    return " ".join([fragment.text.strip() for fragment in pending] + ([name] if name else []))


def _simple_row_item(line: LayoutLine) -> Item | None:
    if any(_NUMERIC_FRAGMENT.fullmatch(block.text.strip()) for block in line.blocks[:-1]):
        return None
    return _parse_simple_item_line(line.text)
