from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item
from extract import parse_receipt
from extract.items import _parse_simple_item_line


@pytest.mark.parametrize(
    ("line", "name", "amount"),
    [
        ("  КАПУЧИНО              250.00  ", "КАПУЧИНО", "250.00"),
        ("КРУАССАН\t150,00", "КРУАССАН", "150.00"),
        ("Кофе в зёрнах 1 234,50", "Кофе в зёрнах", "1234.50"),
        ("Кофе в зёрнах 1\u00a0234.50", "Кофе в зёрнах", "1234.50"),
        ("Молоко 3,2% 1Л 89,99", "Молоко 3,2% 1Л", "89.99"),
        ("Вода 0.33Л 50.00", "Вода 0.33Л", "50.00"),
        ("7UP 0.5Л 99.90", "7UP 0.5Л", "99.90"),
        ("Рамка 10x15 150.00", "Рамка 10x15", "150.00"),
        ("Подарок 0.00", "Подарок", "0.00"),
        ("ТОВАР -12.99", "ТОВАР", "-12.99"),
        ("ТОВАР - 12.99", "ТОВАР", "-12.99"),
    ],
)
def test_simple_item_line_preserves_name_and_parses_decimal(
    line: str, name: str, amount: str
) -> None:
    item = _parse_simple_item_line(line)

    assert item is not None
    assert item == Item(name=name, sum=Decimal(amount))
    assert item.quantity is None
    assert item.price is None


@pytest.mark.parametrize(
    "line",
    [
        "",
        "СЫР РОССИЙСКИЙ",
        "45% 200Г",
        "179.90",
        "123 179.90",
        "ТОВАР 12.999",
        "ТОВАР 12,9",
        "ТОВАР 12..99",
        "ТОВАР --12.99",
        "ТОВАР 12.99 комментарий",
        "1 x 179.90              179.90",
        "1 х 179.90",
        "ТОВАР 2 * 50 100.00",
        "ТОВАР 50.00 100.00",
    ],
)
def test_unsupported_lines_do_not_produce_items(line: str) -> None:
    assert _parse_simple_item_line(line) is None


def test_public_parser_uses_region_preserves_order_and_duplicates_without_total() -> None:
    receipt = parse_receipt(
        "Лавка у реки\nИНН 1234567890\nКАССА 2\n"
        "Чай 50,00\nПеченье 80.00\nЧай 50,00\n"
        "05.10.2026 18:20\nНДС 20% 30,00\nСКИДКА 5,00\n"
        "КАРТОЙ 180,00\nТерминал 123.00",
        timezone=ZoneInfo("Europe/Moscow"),
    )

    assert receipt.items == [
        Item(name="Чай", sum=Decimal("50.00")),
        Item(name="Печенье", sum=Decimal("80.00")),
        Item(name="Чай", sum=Decimal("50.00")),
    ]
    assert receipt.store == "Лавка у реки"
    assert receipt.total is None
    assert receipt.warnings == ["total_not_found"]


def test_public_parser_skips_multiline_items_and_keeps_later_simple_items() -> None:
    receipt = parse_receipt(
        "Лавка\nСЫР РОССИЙСКИЙ\n45% 200Г\n1 x 179.90 179.90\n"
        "ХЛЕБ 50.00\nИТОГО 229.90\nКАРТОЙ 229.90",
        timezone=ZoneInfo("Europe/Moscow"),
    )

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.total == Decimal("229.90")
    # Without a date or fiscal separator, the merchant and unpriced description are ambiguous.
    assert receipt.store is None
    assert receipt.warnings == ["store_not_found", "datetime_not_found", "items_incomplete"]
