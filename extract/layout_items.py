import re

from common.contracts import Item
from extract._text import _AMOUNT_PATTERN, _parse_decimal
from extract.item_region import _has_unresolved_item_candidates, _select_item_candidate_indices
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


def _extract_layout_items(
    lines: list[LayoutLine], *, store: str | None
) -> tuple[list[Item], bool, bool]:
    """Associate adjacent descriptions with explicit amounts, without deriving values.

    Description fragments must align within one text height;
    vertical gaps must be nonnegative and no larger than one text height. A
    nameless quantity row must start beneath the description, and a bare amount
    must lie to its right. Metadata and unsupported numeric rows break continuity.
    Overlapping blocks and multiple amounts without a multiplication label are skipped.
    Rows with their own name and amount never absorb preceding descriptions: an
    unpriced fragment could be a separate product whose amount was lost by OCR.
    Nameless amount rows still use nearby descriptions, which remains a heuristic.

    Args:
        lines: Receipt rows reconstructed from OCR geometry.
        store: Previously identified merchant header, if available.

    Returns:
        Items in reading order, whether every candidate independently produced a named item,
        and whether item candidates were skipped. Fully consumed multiline descriptions are
        not skipped, but conservatively leave reconciliation completeness unproven.
        All values come from explicit text; weight quantities retain their printed numeric values.
    """
    texts = [line.text for line in lines]
    candidates = set(_select_item_candidate_indices(texts, store=store))
    standalone_rows: set[int] = set()
    covered_rows: set[int] = set()
    pending: list[int] = []
    items: list[Item] = []
    for index, line in enumerate(lines):
        if index not in candidates or not _has_usable_geometry(line):
            pending.clear()
            continue
        if pending and not _nearby(lines[pending[-1]], line):
            pending.clear()
        text = line.text.strip()
        quantity_match = _QUANTITY_PATTERN.search(text)
        if quantity_match is not None:
            name = text[: quantity_match.start()].strip()
            if name and not (_is_item_name(name) or _PERCENT_FRAGMENT.fullmatch(name)):
                pending.clear()
                continue
            has_name = bool(name)
            uses_pending = not _is_item_name(name)
            name = _complete_name(name, [lines[row] for row in pending], line)
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
                covered_rows.add(index)
                if uses_pending:
                    covered_rows.update(pending)
                if has_name and len(text.splitlines()) == 1:
                    standalone_rows.add(index)
        elif item := _simple_row_item(line):
            items.append(item)
            covered_rows.add(index)
            standalone_rows.add(index)
        elif _AMOUNT_PATTERN.fullmatch(text):
            if pending and line.x_min >= max(lines[row].x_max for row in pending):
                name = " ".join(lines[row].text.strip() for row in pending)
                if _is_item_name(name):
                    items.append(Item(name=name, sum=_parse_decimal(text)))
                    covered_rows.update([*pending, index])
        elif _is_item_name(text) or _PERCENT_FRAGMENT.fullmatch(text):
            if pending and not _left_aligned(lines[pending[0]], line):
                pending.clear()
            pending.append(index)
            continue
        pending.clear()
    return (
        items,
        candidates.issubset(standalone_rows),
        _has_unresolved_item_candidates(texts, candidates, covered_rows),
    )


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
    if not pending or _is_item_name(name):
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
