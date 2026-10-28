import re
from datetime import datetime, tzinfo

from common.contracts import Receipt
from common.ocr import OcrResult
from extract._text import (
    _AMOUNT_PATTERN,
    _DATETIME_PATTERN,
    _extract_total,
    _is_total_line,
)
from extract.item_region import (
    _can_reconcile_item_region,
    _has_unresolved_item_candidates,
    _is_metadata_line,
    _select_item_candidate_indices,
)
from extract.items import _parse_simple_item_line
from extract.layout import reconstruct_layout
from extract.layout_items import _extract_layout_items
from extract.validation import _validate_arithmetic

STORE_NOT_FOUND = "store_not_found"
DATETIME_NOT_FOUND = "datetime_not_found"
TOTAL_NOT_FOUND = "total_not_found"
ITEMS_INCOMPLETE = "items_incomplete"

_NUMBER_ONLY_PATTERN = re.compile(r"[\d\s.,:+\-№#]+")


def parse_receipt(text: str, *, timezone: tzinfo) -> Receipt:
    """Parse OCR-like receipt text into a structured receipt.

    Args:
        text: Raw text recognized from a receipt.
        timezone: Timezone to attach to a parsed local receipt datetime.

    Returns:
        Receipt containing extracted fields and non-fatal parsing warnings.
        Skipped item candidates are reported as items_incomplete; Item sums are always known.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    receipt = _parse_receipt_fields(text, lines, timezone)
    parsed_candidates = {
        index: _parse_simple_item_line(lines[index])
        for index in _select_item_candidate_indices(lines, store=receipt.store)
    }
    receipt.items = [item for item in parsed_candidates.values() if item is not None]
    covered_rows = {index for index, item in parsed_candidates.items() if item is not None}
    if _has_unresolved_item_candidates(lines, set(parsed_candidates), covered_rows):
        receipt.warnings.append(ITEMS_INCOMPLETE)
    _validate_arithmetic(
        receipt,
        extraction_complete=all(item is not None for item in parsed_candidates.values())
        and _can_reconcile_item_region(lines, store=receipt.store),
    )
    return receipt


def parse_ocr_result(result: OcrResult, *, timezone: tzinfo) -> Receipt:
    """Parse receipt items using reconstructed OCR rows and their spatial relationships.

    Args:
        result: Recognized fragments with absolute-pixel bounding boxes.
        timezone: Timezone to attach to a parsed local receipt datetime.

    Returns:
        Receipt with geometry-associated items, field warnings and applicable arithmetic warnings.

    Raises:
        ValueError: If a bounding-box coordinate is not finite.
    """
    layout = reconstruct_layout(result)
    text = "\n".join(line.text for line in layout)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    receipt = _parse_receipt_fields(text, lines, timezone)
    receipt.items, extraction_complete, items_incomplete = _extract_layout_items(
        layout, store=receipt.store
    )
    if items_incomplete:
        receipt.warnings.append(ITEMS_INCOMPLETE)
    _validate_arithmetic(
        receipt,
        extraction_complete=extraction_complete
        and _can_reconcile_item_region(lines, store=receipt.store),
    )
    return receipt


def _parse_receipt_fields(text: str, lines: list[str], timezone: tzinfo) -> Receipt:
    store = _extract_store(lines)
    receipt_datetime = _extract_datetime(text, timezone)
    total = _extract_total(lines)

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
        warnings=warnings,
    )


def _extract_store(lines: list[str]) -> str | None:
    # Only a unique header before a date or priced row is identifiable as a merchant.
    candidates: list[str] = []
    for line in lines:
        if _DATETIME_PATTERN.search(line) or _is_total_line(line) or _AMOUNT_PATTERN.search(line):
            break
        if _is_plausible_store_line(line):
            candidates.append(line)
    return candidates[0] if len(candidates) == 1 else None


def _is_plausible_store_line(line: str) -> bool:
    if _is_metadata_line(line):
        return False
    if _NUMBER_ONLY_PATTERN.fullmatch(line):
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
