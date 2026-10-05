from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from extract.parser import parse_receipt

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


def test_parse_receipt_extracts_unknown_merchant_without_brand_whitelist() -> None:
    text = """
        КОФЕЙНЯ У ДЯДИ ВАСИ
        ИНН 1234567890

        05.10.2026 18:20

        КАПУЧИНО              250.00

        К ОПЛАТЕ              250.00
    """

    receipt = parse_receipt(text, timezone=MOSCOW_TIMEZONE)

    assert receipt.store == "КОФЕЙНЯ У ДЯДИ ВАСИ"
    assert receipt.datetime == datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.datetime.tzinfo is MOSCOW_TIMEZONE
    assert receipt.total == Decimal("250.00")
    assert receipt.items == []
    assert receipt.warnings == []


def test_store_extraction_skips_service_metadata() -> None:
    receipt = parse_receipt(
        "ИНН 1234567890\nККТ 00012345\nФН 123\nФД 456\nКАССА 2\nСМЕНА 42\n"
        "МАГАЗИН ХОРОШИХ ВЕЩЕЙ\n05.10.2026 10:15:31\nИТОГО 350,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.store == "МАГАЗИН ХОРОШИХ ВЕЩЕЙ"
    assert receipt.datetime == datetime(2026, 10, 5, 10, 15, 31, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("350.00")
    assert receipt.warnings == []


def test_store_preserves_cleaned_original_line() -> None:
    receipt = parse_receipt(
        "  Семейная пекарня №7  \n05.10.2026 15:41\nИТОГО 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.store == "Семейная пекарня №7"


def test_datetime_with_seconds_is_supported() -> None:
    receipt = parse_receipt(
        "МАГНИТ\nДата: 05.10.2026 15:41:27\nИТОГ 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.datetime == datetime(2026, 10, 5, 15, 41, 27, tzinfo=MOSCOW_TIMEZONE)


def test_invalid_datetime_does_not_crash_parsing() -> None:
    receipt = parse_receipt(
        "ЛЕНТА\n35.15.2026 25:90\nИТОГО 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.datetime is None
    assert receipt.warnings == ["datetime_not_found"]


def test_datetime_extraction_skips_invalid_candidate() -> None:
    receipt = parse_receipt(
        "ПЕРЕКРЁСТОК\n35.15.2026 25:90\nДата 05.10.2026 15:41\nИТОГО 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.datetime == datetime(2026, 10, 5, 15, 41, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.warnings == []


@pytest.mark.parametrize(
    ("total_line", "expected"),
    [
        ("ИТОГО 144.98", Decimal("144.98")),
        ("ИТОГ: 144,98", Decimal("144.98")),
        ("К ОПЛАТЕ 1 244,50", Decimal("1244.50")),
        ("к оплате: 2\u00a0030.00", Decimal("2030.00")),
        ("К   ОПЛАТЕ     842,00", Decimal("842.00")),
    ],
)
def test_total_formats_are_supported(total_line: str, expected: Decimal) -> None:
    receipt = parse_receipt(total_line, timezone=MOSCOW_TIMEZONE)

    assert receipt.total == expected


def test_bottom_most_valid_total_is_used() -> None:
    receipt = parse_receipt(
        "ИТОГО 200,00\nПромежуточные строки\nК ОПЛАТЕ 180,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total == Decimal("180.00")


def test_total_extraction_continues_after_unusable_bottom_line() -> None:
    receipt = parse_receipt(
        "ИТОГО 200,00\nК ОПЛАТЕ сумма не распознана",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total == Decimal("200.00")


def test_trailing_amount_is_preferred_on_total_line() -> None:
    receipt = parse_receipt("ИТОГО НДС 20% 144,98", timezone=MOSCOW_TIMEZONE)

    assert receipt.total == Decimal("144.98")


def test_product_prices_are_not_used_as_total() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\n05.10.2026 15:41\nМОЛОКО 89.99\nХЛЕБ 54.99",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total is None
    assert receipt.warnings == ["total_not_found"]


@pytest.mark.parametrize(
    ("text", "expected_warning"),
    [
        ("ИНН 1234567890\n05.10.2026 15:41\nИТОГО 144,98", "store_not_found"),
        ("ПЯТЁРОЧКА\nИТОГО 144,98", "datetime_not_found"),
        ("ПЯТЁРОЧКА\n05.10.2026 15:41\nМОЛОКО 144,98", "total_not_found"),
    ],
)
def test_missing_field_produces_exact_warning(text: str, expected_warning: str) -> None:
    receipt = parse_receipt(text, timezone=MOSCOW_TIMEZONE)

    assert receipt.warnings == [expected_warning]


def test_partial_extraction_returns_receipt() -> None:
    receipt = parse_receipt("К ОПЛАТЕ 842,00", timezone=MOSCOW_TIMEZONE)

    assert receipt.store is None
    assert receipt.datetime is None
    assert receipt.total == Decimal("842.00")
    assert receipt.items == []
    assert receipt.warnings == ["store_not_found", "datetime_not_found"]


def test_empty_input_returns_all_warnings_in_deterministic_order() -> None:
    receipt = parse_receipt(" \n\t", timezone=MOSCOW_TIMEZONE)

    assert receipt.store is None
    assert receipt.datetime is None
    assert receipt.total is None
    assert receipt.items == []
    assert receipt.warnings == [
        "store_not_found",
        "datetime_not_found",
        "total_not_found",
    ]
