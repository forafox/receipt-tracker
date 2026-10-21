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
TOTAL_LABEL_PATTERNS = (
    r"\bк\s+оплате\b",
    r"\bитого\b",
    r"\bитог\b",
)

_DATETIME_PATTERN = re.compile(r"(?<!\d)(\d{2}\.\d{2}\.\d{4}[ \t]+\d{2}:\d{2}(?::\d{2})?)(?![:\d])")
_AMOUNT_PATTERN = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:[ \u00a0]+\d{3})+|\d+)[,.]\d{2}(?!\d)")
_TOTAL_LABEL_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in TOTAL_LABEL_PATTERNS)


def _has_service_prefix(line: str, prefix: str) -> bool:
    if line == prefix:
        return True
    if not line.startswith(prefix):
        return False
    next_character = line[len(prefix)]
    return next_character.isspace() or next_character in ":=№#"


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
