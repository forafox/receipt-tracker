from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item
from common.ocr import OcrResult, TextBlock
from extract.parser import parse_ocr_result, parse_receipt

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


def _layout_result(text: str) -> OcrResult:
    width, height = 1200, 1600
    raw_lines = text.splitlines()
    row_height = max(1, height // max(2 * len(raw_lines), 1))
    blocks = []
    for index, line in enumerate(raw_lines):
        if not line.strip():
            continue
        y_min = min(height - 1, round(index * height / max(len(raw_lines), 1)))
        y_max = min(height, max(y_min + 1, y_min + row_height))
        blocks.append(
            TextBlock(
                text=line.strip(),
                bbox=[(0, y_min), (width, y_min), (width, y_max), (0, y_max)],
            )
        )
    return OcrResult(blocks=blocks, image_width=width, image_height=height)


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
    assert receipt.items == [Item(name="КАПУЧИНО", sum=Decimal("250.00"))]
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


def test_ocr_layout_extracts_legal_store_and_stops_at_payment_block() -> None:
    text = """
ООО "ЛЕНТА"
РОССИЯ, 197350, Г. САНКТ-ПЕТЕРБУРГ,
УЛ. ПАРАШЮТНАЯ, Д. 61, К.4, СТР.1, ПОМ.221-Н
КАССА:0005 КАССИР:
ДОКУМЕНТ НА ПРОДАЖУ 150 (ПРИХОД) СМЕНА N: 0156
СЛИВКИ БОЛЬШ КРУЖКА У/ 124.99 *1 =124.99 НДС 10%
ХЛЕБЕЦ ЭНЕР ЗДОР ПОД 3 89.99 *1 =89.99 НДС 10%
БАНКОВСКИЕ ОПЛАТЫ
03.11.25 20:50 ЧЕК 0131
ПАО СБЕРБАНК ОПЛАТА
ТЕРМИНАЛ: 39334622 МЕРЧАНТ:991000225979
MIR (E4)*5765 A0000006581010
СУММА(РУБ): 214.98
ИТОГО К ОПЛАТЕ С УЧЕТОМ СКИДКИ =214.98
СУММА НДС 10% =19.54
БАЛАНС БАЛЛОВ НА 03-11-2025 20:50 0.00
ЧЕК 0145 ИТОГ = 214.98
    """

    receipt = parse_ocr_result(_layout_result(text), timezone=MOSCOW_TIMEZONE)

    assert receipt.store == 'ООО "ЛЕНТА"'
    assert receipt.datetime == datetime(2025, 11, 3, 20, 50, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("214.98")
    assert receipt.items == [
        Item(
            name="СЛИВКИ БОЛЬШ КРУЖКА У/",
            quantity=Decimal("1"),
            price=Decimal("124.99"),
            sum=Decimal("124.99"),
        ),
        Item(
            name="ХЛЕБЕЦ ЭНЕР ЗДОР ПОД 3",
            quantity=Decimal("1"),
            price=Decimal("89.99"),
            sum=Decimal("89.99"),
        ),
    ]
    assert receipt.warnings == []


def test_ocr_layout_parses_pipe_delimited_item_table_before_payment() -> None:
    text = """
ООО "ЛЕНТА"
РОССИЯ, 197350, Г. САНКТ-ПЕТЕРБУРГ,
УЛ. ПАРАШЮТНАЯ Д. 60, ЛИТЕРА А
ДОКУМЕНТ НА ПРОДАЖУ
| КАРТОФЕЛЬ ЧИК-ЧИК СВЕЖ | 74.99 *1.000 | =74.99 НДС 10% |
|------------------------|-------------|---------------|
| ПАКЕТ ЛЕНТА 12КГ       | 12.99 *1     | =12.99 НДС 20% |
BANKORPAYMENTENTITY
КАССОВЫЙ ЧЕК
СИСТЕМА БЫСТРЫХ ПЛАТЕЖЕЙ 05.11.2025 11:52
ИТОГО К ОПЛАТЕ С УЧЕТОМ СКИДКИ =87.98
СУММА НДС 10% =6.82
ЧЕК 0091 ИТОГ =87.98
    """

    receipt = parse_ocr_result(_layout_result(text), timezone=MOSCOW_TIMEZONE)

    assert receipt.store == 'ООО "ЛЕНТА"'
    assert receipt.datetime == datetime(2025, 11, 5, 11, 52, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("87.98")
    assert receipt.items == [
        Item(
            name="КАРТОФЕЛЬ ЧИК-ЧИК СВЕЖ",
            quantity=Decimal("1.000"),
            price=Decimal("74.99"),
            sum=Decimal("74.99"),
        ),
        Item(
            name="ПАКЕТ ЛЕНТА 12КГ",
            quantity=Decimal("1"),
            price=Decimal("12.99"),
            sum=Decimal("12.99"),
        ),
    ]
    assert receipt.warnings == []


def test_store_preserves_cleaned_original_line() -> None:
    receipt = parse_receipt(
        "  Семейная пекарня №7  \n05.10.2026 15:41\nИТОГО 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.store == "Семейная пекарня №7"


def test_amount_bearing_product_is_not_used_as_merchant() -> None:
    receipt = parse_receipt("05.10.2026 18:20\nХЛЕБ 50.00\nИТОГО 50.00", timezone=MOSCOW_TIMEZONE)

    assert receipt.store is None
    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.warnings == ["store_not_found"]


def test_merchant_can_follow_more_than_ten_fiscal_header_lines() -> None:
    metadata = [f"ФД {number}" for number in range(12)]
    receipt = parse_receipt(
        "\n".join([*metadata, "МАГАЗИН У РЕКИ", "05.10.2026 18:20", "ИТОГО 50.00"]),
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.store == "МАГАЗИН У РЕКИ"
    assert receipt.warnings == []


def test_timestamp_with_ocr_punctuation_does_not_become_another_merchant() -> None:
    receipt = parse_receipt(
        "МАГАЗИН\nДата: .05.10.2026 18:20\nХЛЕБ 50.00\nИТОГО 50.00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.store == "МАГАЗИН"
    assert receipt.datetime == datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.total == Decimal("50.00")
    assert receipt.warnings == []


def test_datetime_with_seconds_is_supported() -> None:
    receipt = parse_receipt(
        "МАГНИТ\nДата: 05.10.2026 15:41:27\nИТОГ 10,00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.datetime == datetime(2026, 10, 5, 15, 41, 27, tzinfo=MOSCOW_TIMEZONE)


def test_two_digit_year_is_supported() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\nкассир 04.01.22 14:04\nИТОГ: 1399.37",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.datetime == datetime(2022, 1, 4, 14, 4, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("1399.37")


def test_mid_line_total_label_is_supported() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\n04.01.2022 14:04\nОКРУГЛЕНИЕ: 0.00 ИТОГ: 1399.37",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total == Decimal("1399.37")


def test_subtotal_label_is_not_a_total() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\n04.01.2022 14:04\nСКИДКА: 0.00 ПОДЫТОГ: 1399.37",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total is None


def test_bank_sum_is_used_when_total_amount_is_missing() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\n04.01.2022 14:04\n"
        "ОКРУГЛЕНИЕ: ИТОГ:\n"
        "умна (Руб) : 1399.37, Комиссия за операцию 0 Руб",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total == Decimal("1399.37")


def test_bank_sum_does_not_override_total_label() -> None:
    receipt = parse_receipt(
        "ПЯТЁРОЧКА\n04.01.2022 14:04\nИТОГО 100.00\nСумма (Руб): 200.00",
        timezone=MOSCOW_TIMEZONE,
    )

    assert receipt.total == Decimal("100.00")


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
        ("ИТОГО 772.00,", Decimal("772.00")),
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


@pytest.mark.parametrize("line", ["ИТОГО НДС 20% 144,98", "ИТОГО НДС 20% 24,16 144,98"])
def test_trailing_amount_is_preferred_on_total_line(line: str) -> None:
    receipt = parse_receipt(line, timezone=MOSCOW_TIMEZONE)

    assert receipt.total == Decimal("144.98")


@pytest.mark.parametrize("amount", ["--12.99", "12.999", "12.99.00", "12.99, 00"])
def test_malformed_total_amount_is_not_partially_accepted(amount: str) -> None:
    receipt = parse_receipt(f"МАГАЗИН\n05.10.2026 18:20\nИТОГО {amount}", timezone=MOSCOW_TIMEZONE)

    assert receipt.total is None
    assert receipt.items == []
    assert receipt.warnings == ["total_not_found"]


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
