from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from extract.item_region import _select_item_candidate_lines
from extract.parser import parse_receipt


def test_unknown_merchant_items_and_public_parser_compatibility() -> None:
    text = """
КОФЕЙНЯ У ДЯДИ ВАСИ
ИНН 1234567890
КАССА 2

05.10.2026 18:20

КАПУЧИНО              250.00
КРУАССАН              150.00

ИТОГО                 400.00
КАРТОЙ                400.00
ФН 123456789
"""
    timezone = ZoneInfo("Europe/Moscow")
    receipt = parse_receipt(text, timezone=timezone)

    assert _select_item_candidate_lines(text.splitlines(), store=receipt.store) == [
        "КАПУЧИНО              250.00",
        "КРУАССАН              150.00",
    ]
    assert receipt.store == "КОФЕЙНЯ У ДЯДИ ВАСИ"
    assert receipt.datetime == datetime(2026, 10, 5, 18, 20, tzinfo=timezone)
    assert receipt.total == Decimal("400.00")
    assert receipt.items == []
    assert receipt.warnings == []


def test_metadata_is_filtered_without_a_fixed_header_length() -> None:
    lines = [
        "ИНН: 1234567890",
        "ККТ №123",
        "РН ККТ 123456789",
        "ФН 123",
        "ФД 456",
        "ФП 789",
        "ФПД 789",
        "КАССА 2",
        "СМЕНА 3",
        "ЧЕК №4",
        "КАССОВЫЙ ЧЕК",
        "КАССИР Иванова",
        "  Лавка у реки  ",
        "Адрес: ул. Садовая, 1",
        "Телефон: 1234567",
        "  ХЛЕБ    50,00  ",
        "Дата: 05.10.2026",
        "Время: 18:20:30",
        "ИТОГО 50,00",
    ]

    assert _select_item_candidate_lines(lines, store="Лавка у реки") == ["ХЛЕБ    50,00"]


@pytest.mark.parametrize("total", [[], ["ИТОГО 50,00"]])
def test_late_datetime_and_payment_boundary_with_or_without_total(total: list[str]) -> None:
    lines = [
        "Пекарня",
        "ХЛЕБ 50,00",
        "05.10.2026 18:20",
        *total,
        "НАЛИЧНЫМИ 100,00",
        "СДАЧА 50,00",
        "Терминал 123",
        "До новых встреч!",
    ]

    assert _select_item_candidate_lines(lines, store="Пекарня") == ["ХЛЕБ 50,00"]


def test_final_recognized_total_bounds_region_and_all_total_labels_are_filtered() -> None:
    lines = [
        "Чай 100,00",
        "ИТОГО 100,00",
        "Печенье 80,00",
        "К   ОПЛАТЕ: 180,00",
        "Служебный текст",
        "ИТОГО сумма не распознана",
    ]

    assert _select_item_candidate_lines(lines, store=None) == ["Чай 100,00", "Печенье 80,00"]


def test_multiline_item_and_percentage_are_preserved_separately() -> None:
    lines = [
        "СЫР РОССИЙСКИЙ",
        "45% 200Г",
        "1 x 179.90              179.90",
        "СКИДКА 10% 17.99",
        "НДС 20% = 29.98",
        "СУММА НДС: 29.98",
        "БЕЗ НДС",
        "ИТОГ 161.91",
    ]

    assert _select_item_candidate_lines(lines, store=None) == lines[:3]


def test_similar_words_in_product_names_are_not_metadata() -> None:
    lines = [
        "КАССАВА 70,00",
        "КАССА ИГРУШЕЧНАЯ 500,00",
        "ЧЕКОВАЯ ЛЕНТА 60,00",
        "ЧЕК ЛЕНТА 50,00",
        "СКИДКА НА СЫР 80,00",
        "НАПИТОК ИТОГО 90,00",
        "ИТОГО 350,00",
    ]

    assert _select_item_candidate_lines(lines, store=None) == lines[:-1]


@pytest.mark.parametrize(
    ("lines", "store"),
    [
        ([], None),
        (["  ", "\t", "---"], None),
        (["Лавка", "ИНН 123", "05.10.2026 18:20", "ИТОГО 0,00"], "Лавка"),
        (["ККТ 123", "СКИДКА 0,00", "НДС 0%", "КАРТОЙ 0,00"], None),
    ],
)
def test_empty_or_metadata_only_receipt_has_no_candidates(
    lines: list[str], store: str | None
) -> None:
    assert _select_item_candidate_lines(lines, store=store) == []
