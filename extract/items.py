import re

from common.contracts import Item
from extract._text import _AMOUNT_PATTERN, _parse_decimal

_SIMPLE_ITEM_PATTERN = re.compile(rf"(?P<name>.+?)\s+(?P<amount>{_AMOUNT_PATTERN.pattern})")
_MULTIPLICATION_PATTERN = re.compile(r"\d\s*[*xх×](?=\s|$)", re.IGNORECASE)


def _parse_simple_item_line(line: str) -> Item | None:
    """Parse a name and one trailing amount; skip explicit multiplication rows.

    Extra standalone amounts are ambiguous and are left for more advanced
    parsing. Decimal numbers attached to units or percentages remain in names.

    Args:
        line: One candidate line, independent of receipt layout.

    Returns:
        A simple item with quantity and price unset, or None for unsupported lines.
    """
    match = _SIMPLE_ITEM_PATTERN.fullmatch(line.strip())
    if match is None:
        return None
    name = match.group("name").strip()
    if not _is_item_name(name):
        return None
    return Item(name=name, sum=_parse_decimal(match.group("amount")))


def _is_item_name(name: str) -> bool:
    if not any(character.isalpha() for character in name) or name.endswith(("-", "+")):
        return False
    if _MULTIPLICATION_PATTERN.search(name):
        return False
    return not any(
        amount.end() == len(name) or name[amount.end()].isspace()
        for amount in _AMOUNT_PATTERN.finditer(name)
    )
