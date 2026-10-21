import re
from datetime import datetime, tzinfo

from common.contracts import Receipt
from extract._text import (
    _DATETIME_PATTERN,
    SERVICE_LINE_PREFIXES,
    _contains_total_label,
    _extract_total,
    _has_service_prefix,
)
from extract.item_region import _select_item_candidate_lines
from extract.items import _parse_simple_item_line

STORE_HEADER_LINE_LIMIT = 10

STORE_NOT_FOUND = "store_not_found"
DATETIME_NOT_FOUND = "datetime_not_found"
TOTAL_NOT_FOUND = "total_not_found"

_NUMBER_ONLY_PATTERN = re.compile(r"[\d\s.,:+\-№#]+")


def parse_receipt(text: str, *, timezone: tzinfo) -> Receipt:
    """Parse OCR-like receipt text into a structured receipt.

    Args:
        text: Raw text recognized from a receipt.
        timezone: Timezone to attach to a parsed local receipt datetime.

    Returns:
        Receipt containing extracted fields and non-fatal parsing warnings.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    store = _extract_store(lines)
    receipt_datetime = _extract_datetime(text, timezone)
    total = _extract_total(lines)
    items = [
        item
        for line in _select_item_candidate_lines(lines, store=store)
        if (item := _parse_simple_item_line(line)) is not None
    ]

    warnings: list[str] = []
    if store is None:
        warnings.append(STORE_NOT_FOUND)
    if receipt_datetime is None:
        warnings.append(DATETIME_NOT_FOUND)
    if total is None:
        warnings.append(TOTAL_NOT_FOUND)

    return Receipt(
        store=store,
        datetime=receipt_datetime,
        total=total,
        items=items,
        warnings=warnings,
    )


def _extract_store(lines: list[str]) -> str | None:
    for line in lines[:STORE_HEADER_LINE_LIMIT]:
        if _is_plausible_store_line(line):
            return line
    return None


def _is_plausible_store_line(line: str) -> bool:
    if _DATETIME_PATTERN.search(line) or _contains_total_label(line):
        return False
    if _NUMBER_ONLY_PATTERN.fullmatch(line):
        return False

    folded_line = line.casefold()
    if any(_has_service_prefix(folded_line, prefix) for prefix in SERVICE_LINE_PREFIXES):
        return False
    return any(character.isalpha() for character in line)


def _extract_datetime(text: str, timezone: tzinfo) -> datetime | None:
    for match in _DATETIME_PATTERN.finditer(text):
        value = " ".join(match.group(1).split())
        date_format = "%d.%m.%Y %H:%M:%S" if value.count(":") == 2 else "%d.%m.%Y %H:%M"
        try:
            parsed = datetime.strptime(value, date_format)
        except ValueError:
            continue
        return parsed.replace(tzinfo=timezone)
    return None
