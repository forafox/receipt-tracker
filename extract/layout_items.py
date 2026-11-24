import re

from common.contracts import Item
from extract._text import _AMOUNT_PATTERN, _TOTAL_LABEL_PATTERN, _parse_decimal
from extract.item_region import (
    _has_unresolved_item_candidates,
    _select_item_candidate_indices,
    _TotalBoundary,
)
from extract.items import _is_item_name, _parse_simple_item_line
from extract.layout import LayoutLine, _has_usable_geometry, _nearby

# Fiscal letters and short tax codes can survive OCR without a readable rate.
# A decimal rate requires %, so an additional monetary amount cannot become an annotation.
_ITEM_SUFFIX = (
    r"(?:[ \t]*_?[A-ZА-Я])?"
    r"(?:[ \t]*[A-ZА-Я])?"
    r"(?:[ \t]+[A-ZА-Я]{2,3}(?:[ \t]*(?:\d{1,3}(?:[.,]\d+)?[ \t]*%|\d{1,3}))?)?$"
)
_QUANTITY_PATTERN = re.compile(
    rf"(?<![\w.,+\-−])(?P<quantity>\d+(?:[.,]\d+)?)\s*"
    rf"(?P<unit>кг|kg|г|g|шт\.?|л|l)?\s*[xх×*ж]\s*"
    rf"(?P<price>{_AMOUNT_PATTERN.pattern})(?:\s*=\s*|\s+)"
    rf"(?P<total>{_AMOUNT_PATTERN.pattern}){_ITEM_SUFFIX}",
    re.IGNORECASE,
)
_PRICE_QUANTITY_PATTERN = re.compile(
    rf"(?<![\w.,+\-−])(?P<price>{_AMOUNT_PATTERN.pattern})\s*"
    rf"[xх×*«»ж]\s*"
    rf"(?P<quantity>\d+(?:[.,]\d+)?)\s*"
    rf"(?P<unit>кг|kg|г|g|шт\.?|л|l)?\s*"
    rf"(?:=\s*|\s+)(?P<total>{_AMOUNT_PATTERN.pattern}){_ITEM_SUFFIX}",
    re.IGNORECASE,
)
_NUMERIC_FRAGMENT = re.compile(r"[\d\s.,]+")
_PERCENT_FRAGMENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
_SEPARATE_TOTAL = re.compile(rf"=\s*(?P<amount>{_AMOUNT_PATTERN.pattern})")
_TAX_ONLY = re.compile(r"^(?:н[дaа]с|нас)(?:\b|\s)", re.IGNORECASE)
_BARCODE_LINE = re.compile(r"^\[м\]\s*\d{8,14}\s*", re.IGNORECASE)
_ITEM_DESC_AMOUNT = re.compile(
    r"^(?P<name>.+?)\s+(?P<price>\d+(?:[.,]\d{2}))\s*\*\s*(?P<quantity>\d+(?:[.,]\d+)?)\s*(?P<unit>шт\.?)?\s*$",
    re.IGNORECASE,
)
_ITEM_DESC_PRICE = re.compile(r"^(?P<name>.+?)\s+(?P<price>\d+(?:[.,]\d{2}))\s*$", re.IGNORECASE)


def _extract_layout_items(
    lines: list[LayoutLine], *, store: str | None, totals: tuple[_TotalBoundary, ...] = ()
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
        totals: Recognized total boundaries retaining original row indices.

    Returns:
        Items in reading order, whether every candidate independently produced a named sum,
        and whether candidates were skipped or quantity/price roles remain ambiguous.
        Fully consumed multiline descriptions are
        not skipped, but conservatively leave reconciliation completeness unproven.
        All values come from explicit text; weight quantities retain their printed numeric values.
    """
    texts = [_normalize_table_markup(line.text) for line in lines]
    candidates = set(_select_item_candidate_indices(texts, store=store, totals=totals))
    standalone_rows: set[int] = set()
    covered_rows: set[int] = set()
    pending: list[int] = []
    items: list[Item] = []
    ambiguous_items = False
    for index, line in enumerate(lines):
        if index not in candidates or not _has_usable_geometry(line):
            pending.clear()
            continue
        if pending and not _nearby(lines[pending[-1]], line):
            pending.clear()
        text = texts[index].strip()
        if _TAX_ONLY.match(text):
            pending.clear()
            continue
        if text.casefold() in {"итог", "позиции"}:
            pending.clear()
            continue
        if (
            "|" in line.text
            and _TOTAL_LABEL_PATTERN.fullmatch(text.strip(" |#").strip()) is not None
        ):
            pending.clear()
            continue
        description_match = _ITEM_DESC_AMOUNT.fullmatch(text)
        if description_match and index + 1 < len(lines):
            next_text = texts[index + 1].strip()
            price_total = re.fullmatch(rf"=\s*(?P<amount>{_AMOUNT_PATTERN.pattern})", next_text)
            if price_total:
                name = description_match.group("name").strip()
                if _is_item_name(name):
                    items.append(Item(name=name, sum=_parse_decimal(price_total.group("amount"))))
                    covered_rows.update((index, index + 1))
                    continue
        description_price = _ITEM_DESC_PRICE.fullmatch(text)
        if description_price and index + 1 < len(lines):
            next_line = texts[index + 1].strip()
            if re.fullmatch(r"\d+(?:[.,]\d+)?\s*\*\s*\d+(?:[.,]\d+)?\s*(?:шт\.?\s*)?", next_line):
                amount_match = _AMOUNT_PATTERN.search(next_line)
                if amount_match:
                    name = description_price.group("name").strip()
                    if _is_item_name(name):
                        items.append(Item(name=name, sum=_parse_decimal(amount_match.group())))
                        covered_rows.update((index, index + 1))
                        continue
        quantity_match = _QUANTITY_PATTERN.search(text) or _PRICE_QUANTITY_PATTERN.search(text)
        if quantity_match is not None:
            name = text[: quantity_match.start()].strip()
            if _BARCODE_LINE.match(name):
                name = _BARCODE_LINE.sub("", name).strip()
            if name and not (_is_item_name(name) or _PERCENT_FRAGMENT.fullmatch(name)):
                pending.clear()
                continue
            has_name = bool(name)
            uses_pending = not _is_item_name(name)
            name = _complete_name(name, [lines[row] for row in pending], line)
            quantity = _parse_decimal(quantity_match.group("quantity"))
            price = _parse_decimal(quantity_match.group("price"))
            # Two nonnegative money-shaped operands have no intrinsic quantity/price order.
            # Retain the printed sum but report the unresolved roles as incomplete.
            ambiguous_roles = (
                quantity_match.group("unit") is None
                and _AMOUNT_PATTERN.fullmatch(quantity_match.group("quantity")) is not None
                and price >= 0
            )
            if _is_item_name(name) and (quantity > 0 or ambiguous_roles):
                items.append(
                    Item(
                        name=name,
                        quantity=None if ambiguous_roles else quantity,
                        price=None if ambiguous_roles else price,
                        sum=_parse_decimal(quantity_match.group("total")),
                    )
                )
                covered_rows.add(index)
                if uses_pending:
                    covered_rows.update(pending)
                ambiguous_items |= ambiguous_roles
                if has_name and len(text.splitlines()) == 1:
                    standalone_rows.add(index)
        elif separate_total := _SEPARATE_TOTAL.fullmatch(text):
            if pending:
                pending_names = []
                for row in pending:
                    fragment = _normalize_table_markup(lines[row].text).strip()
                    fragment = _BARCODE_LINE.sub("", fragment).strip()
                    if fragment:
                        pending_names.append(fragment)
                name = " ".join(pending_names)
                if _is_item_name(name):
                    items.append(
                        Item(name=name, sum=_parse_decimal(separate_total.group("amount")))
                    )
                    covered_rows.update([*pending, index])
                    pending.clear()
        elif item := _simple_row_item(line):
            items.append(item)
            covered_rows.add(index)
            standalone_rows.add(index)
        elif _AMOUNT_PATTERN.fullmatch(text):
            if pending and line.x_min >= max(lines[row].x_max for row in pending):
                name = " ".join(_normalize_table_markup(lines[row].text).strip() for row in pending)
                if _is_item_name(name):
                    items.append(Item(name=name, sum=_parse_decimal(text)))
                    covered_rows.update([*pending, index])
        elif _BARCODE_LINE.match(text):
            product_name = _BARCODE_LINE.sub("", text).strip()
            if product_name and _is_item_name(product_name) and index + 1 < len(lines):
                pending.append(index)
            else:
                pending.clear()
            continue
        elif (
            pending
            and _BARCODE_LINE.match(_normalize_table_markup(lines[pending[-1]].text).strip())
            and re.fullmatch(r"\d+(?:[.,]\d+)?\s*\*", text)
        ):
            pending.append(index)
            continue
        elif _is_item_name(text) or _PERCENT_FRAGMENT.fullmatch(text):
            if pending and not _left_aligned(lines[pending[0]], line):
                pending.clear()
            pending.append(index)
            continue
        pending.clear()
    return (
        items,
        candidates.issubset(standalone_rows),
        ambiguous_items or _has_unresolved_item_candidates(texts, candidates, covered_rows),
    )


def _left_aligned(first: LayoutLine, second: LayoutLine) -> bool:
    height = min(first.y_max - first.y_min, second.y_max - second.y_min)
    return abs(first.x_min - second.x_min) <= height


def _normalize_table_markup(text: str) -> str:
    """Turn Markdown-like OCR column separators into spaces before row parsing.

    Returns:
        Text with empty table columns omitted and remaining cells space-separated.
    """
    if "|" not in text:
        return text
    return " ".join(part.strip() for part in text.strip().strip("|").split("|") if part.strip())


def _complete_name(name: str, pending: list[LayoutLine], line: LayoutLine) -> str:
    if not pending or _is_item_name(name):
        return name
    if name and not _left_aligned(pending[0], line):
        return name
    if not name and not pending[0].x_min <= line.x_min <= pending[0].x_max:
        return name
    return " ".join(
        [_normalize_table_markup(fragment.text).strip() for fragment in pending]
        + ([name] if name else [])
    )


def _simple_row_item(line: LayoutLine) -> Item | None:
    if any(_NUMERIC_FRAGMENT.fullmatch(block.text.strip()) for block in line.blocks[:-1]):
        return None
    return _parse_simple_item_line(_normalize_table_markup(line.text))
