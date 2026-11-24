import re
from decimal import Decimal, InvalidOperation

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
_DATETIME_PATTERN = re.compile(
    r"(?<!\d)(\d{2}\.\d{2}\.(?:\d{2}|\d{4})[ \t]+\d{2}:\d{2}(?::\d{2})?)(?![:\d])"
)
_AMOUNT_PATTERN = re.compile(
    r"(?<![\d.,+\-−])(?:[+\-−][ \u00a0]*)?"
    r"(?:\d{1,3}(?:[ \u00a0]+\d{3})+|\d+)[,.]\d{2}(?!\d)"
)
_TOTAL_LABEL_PATTERN = re.compile(
    r"(?<!\w)(?:(?:всего\s+)?к\s+оплате|итог(?:о)?)\b",
    re.IGNORECASE,
)


def _has_service_prefix(line: str, prefix: str) -> bool:
    if line == prefix:
        return True
    if not line.startswith(prefix):
        return False
    next_character = line[len(prefix)]
    return next_character.isspace() or next_character in ":=№#"


def _extract_total(lines: list[str]) -> Decimal | None:
    for line in reversed(lines):
        if not _is_total_line(line):
            continue

        for match in reversed(list(_AMOUNT_PATTERN.finditer(line))):
            try:
                return _parse_decimal(match.group())
            except InvalidOperation:
                continue
    return None


def _is_total_line(line: str) -> bool:
    # A total can start the line or be the second column of a summary row:
    # "ОКРУГЛЕНИЕ: 0.00 ИТОГ: 1399.37". The delimiter after the label keeps
    # product names such as "НАПИТОК ИТОГО 90.00" from becoming totals.
    stripped = line.strip()
    if _TOTAL_LABEL_PATTERN.match(stripped) is not None:
        return True
    return any(
        match.start() > 0 and stripped[match.end() :].lstrip().startswith((":", "="))
        for match in _TOTAL_LABEL_PATTERN.finditer(stripped)
    )


def _parse_decimal(value: str) -> Decimal:
    normalized = value.replace(" ", "").replace("\u00a0", "").replace(",", ".").replace("−", "-")
    return Decimal(normalized)
