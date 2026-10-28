import re

from extract._text import (
    _AMOUNT_PATTERN,
    _DATETIME_PATTERN,
    SERVICE_LINE_PREFIXES,
    _extract_total,
    _has_service_prefix,
    _is_total_line,
)

_NUMBERED_SERVICE_PREFIXES = ("касса", "смена", "чек")
_HEADER_PREFIXES = (
    "адрес",
    "телефон",
    "место расчетов",
    "место расчётов",
    "кассовый чек",
    "рн ккт",
)
_PAYMENT_PREFIXES = ("картой", "наличными", "сдача", "безналичными", "безналичный расчет")
_FOOTER_PREFIXES = ("спасибо за покупку", "сайт фнс", "www.nalog.ru")
_ADJUSTMENT_PREFIXES = (
    "скидка",
    "сумма скидки",
    "наценка",
    "коррекция",
    "округление",
    "бонусы",
    "списано бонусов",
    "возврат",
)
_DATE_OR_TIME_PATTERN = re.compile(r"(?:\d{2}\.\d{2}\.\d{4}|\d{2}:\d{2}(?::\d{2})?)")
_SUMMARY_PATTERN = re.compile(
    r"(?:скидка|сумма скидки|ндс|сумма ндс|без ндс)"
    r"(?:\s*[:=]?\s*[\d\s.,:%=+\-]+(?:\s*руб\.?)?)?",
    re.IGNORECASE,
)


def _select_item_candidate_lines(lines: list[str], *, store: str | None) -> list[str]:
    """Keep unclassified lines before the final total or first payment line.

    Only the supplied merchant and recognizable metadata are removed; dates do
    not define the start of the region. Unclassified text is retained without
    requiring prices, preserving possible multiline items. Plain text cannot
    reliably distinguish an unlabeled header/footer from an item fragment.

    Args:
        lines: Receipt text split into separate lines.
        store: Previously identified merchant header, if available.

    Returns:
        Cleaned candidate lines in their original order, without joining fragments.
    """
    return [lines[index].strip() for index in _select_item_candidate_indices(lines, store=store)]


def _select_item_candidate_indices(lines: list[str], *, store: str | None) -> list[int]:
    cleaned_lines = [(index, line.strip()) for index, line in enumerate(lines) if line.strip()]
    region_end = len(cleaned_lines)
    for index in range(len(cleaned_lines) - 1, -1, -1):
        _, line = cleaned_lines[index]
        if _is_total_line(line) and _extract_total([line]) is not None:
            region_end = index
            break

    candidates: list[int] = []
    merchant_removed = False
    for index, line in cleaned_lines[:region_end]:
        folded_line = line.casefold()
        if store is not None and not merchant_removed and folded_line == store.strip().casefold():
            merchant_removed = True
            continue
        if any(_has_service_prefix(folded_line, prefix) for prefix in _PAYMENT_PREFIXES):
            break
        if _is_metadata_line(line) or not any(character.isalnum() for character in line):
            continue
        candidates.append(index)
    return candidates


def _can_reconcile_item_region(lines: list[str], *, store: str | None) -> bool:
    """Require a clear total boundary without unrepresented monetary summaries.

    Args:
        lines: Receipt rows, including metadata and summaries excluded from candidates.
        store: Identified merchant; an amount in this header may indicate a lost item.

    Returns:
        Whether structural cues permit reconciliation, subject to actual candidate coverage.
        Multiple total labels, early payment boundaries, adjustments and even tax summaries
        conservatively disable the check, as do unclassified monetary rows after the total.
        This cannot detect rows omitted entirely by OCR.
    """
    cleaned = [line.strip() for line in lines if line.strip()]
    totals = [index for index, line in enumerate(cleaned) if _is_total_line(line)]
    if len(totals) != 1:
        return False
    total_index = totals[0]
    if _extract_total([cleaned[total_index]]) is None:
        return False
    if store is not None and _AMOUNT_PATTERN.search(store):
        return False
    footer_candidates = _select_item_candidate_lines(cleaned[total_index + 1 :], store=None)
    if any(_AMOUNT_PATTERN.search(line) for line in footer_candidates):
        return False
    if any(
        _SUMMARY_PATTERN.fullmatch(line)
        or any(_has_service_prefix(line.casefold(), prefix) for prefix in _ADJUSTMENT_PREFIXES)
        for line in cleaned
    ):
        return False
    return not any(
        _has_service_prefix(line.casefold(), prefix)
        for line in cleaned[:total_index]
        for prefix in _PAYMENT_PREFIXES
    )


def _is_metadata_line(line: str) -> bool:
    folded_line = line.casefold()
    for prefix in _NUMBERED_SERVICE_PREFIXES:
        if _has_service_prefix(folded_line, prefix):
            suffix = folded_line[len(prefix) :].lstrip(" :=№#")
            return not suffix or suffix[0].isdigit()
    if any(
        _has_service_prefix(folded_line, prefix)
        for prefix in SERVICE_LINE_PREFIXES + _HEADER_PREFIXES + _FOOTER_PREFIXES
    ):
        return True
    if _is_total_line(line) or _SUMMARY_PATTERN.fullmatch(line):
        return True
    for prefix in ("дата", "время"):
        if _has_service_prefix(folded_line, prefix):
            line = line[len(prefix) :].lstrip(" :")
            break
    return bool(_DATETIME_PATTERN.fullmatch(line) or _DATE_OR_TIME_PATTERN.fullmatch(line))
