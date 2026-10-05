import re
from datetime import datetime, tzinfo
from decimal import Decimal, InvalidOperation

from common.contracts import Receipt

STORE_HEADER_LINE_LIMIT = 10
SERVICE_LINE_PREFIXES = (
    "инн",
    "ккт",
    "фн",
    "фд",
    "фп",
    "фпд",
    "касса",
    "смена",
    "чек",
    "кассир",
)
TOTAL_LABEL_PATTERNS = (
    r"\bк\s+оплате\b",
    r"\bитого\b",
    r"\bитог\b",
)

STORE_NOT_FOUND = "store_not_found"
DATETIME_NOT_FOUND = "datetime_not_found"
TOTAL_NOT_FOUND = "total_not_found"

_DATETIME_PATTERN = re.compile(r"(?<!\d)(\d{2}\.\d{2}\.\d{4}[ \t]+\d{2}:\d{2}(?::\d{2})?)(?![:\d])")
_AMOUNT_PATTERN = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:[ \u00a0]+\d{3})+|\d+)[,.]\d{2}(?!\d)")
_NUMBER_ONLY_PATTERN = re.compile(r"[\d\s.,:+\-№#]+")
_TOTAL_LABEL_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in TOTAL_LABEL_PATTERNS)


def parse_receipt(text: str, *, timezone: tzinfo) -> Receipt:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

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
        items=[],
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


def _has_service_prefix(line: str, prefix: str) -> bool:
    if line == prefix:
        return True
    if not line.startswith(prefix):
        return False
    next_character = line[len(prefix)]
    return next_character.isspace() or next_character in ":=№#"


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


def _extract_total(lines: list[str]) -> Decimal | None:
    for line in reversed(lines):
        if not _contains_total_label(line):
            continue

        for match in reversed(list(_AMOUNT_PATTERN.finditer(line))):
            try:
                return _parse_decimal(match.group())
            except InvalidOperation:
                continue
    return None


def _contains_total_label(line: str) -> bool:
    return any(pattern.search(line) for pattern in _TOTAL_LABEL_REGEXES)


def _parse_decimal(value: str) -> Decimal:
    normalized = value.replace(" ", "").replace("\u00a0", "").replace(",", ".")
    return Decimal(normalized)
