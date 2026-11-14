from datetime import datetime
from decimal import Decimal
from random import Random
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item, Receipt
from common.ocr import OcrResult, TextBlock
from extract import parse_ocr_result, parse_receipt

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


def _make_ocr(*fragments: tuple[str, int, int, int], shuffled: bool = False) -> OcrResult:
    result = OcrResult(
        image_width=360,
        blocks=[
            TextBlock(
                text=text,
                bbox=[(left, top), (right, top), (right, top + 12), (left, top + 12)],
                confidence=0.98,
            )
            for text, left, top, right in fragments
        ],
    )
    if shuffled:
        original = result.blocks.copy()
        Random(2026).shuffle(result.blocks)
        assert result.blocks != original
    return result


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
def test_unpriced_items_are_skipped_with_one_incomplete_warning(ocr: bool) -> None:
    text = "МАГАЗИН\n05.10.2026 18:20\nМОЛОКО\nЯЙЦА\nХЛЕБ 50.00\nИТОГО 150.00"
    result = _make_ocr(
        ("МАГАЗИН", 10, 10, 110),
        ("05.10.2026 18:20", 10, 30, 180),
        ("МОЛОКО", 10, 60, 100),
        ("ЯЙЦА", 10, 80, 100),
        ("ХЛЕБ", 10, 100, 100),
        ("50.00", 275, 100, 335),
        ("ИТОГО 150.00", 10, 130, 180),
        shuffled=True,
    )
    receipt = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )

    assert receipt == Receipt(
        store="МАГАЗИН",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("150.00"),
        items=[Item(name="ХЛЕБ", sum=Decimal("50.00"))],
        warnings=["items_incomplete"],
    )


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
def test_ambiguous_header_keeps_store_unknown(ocr: bool) -> None:
    text = "ДОБРО ПОЖАЛОВАТЬ\nМАГАЗИН У РЕКИ\n05.10.2026 18:20\nМОЛОКО 100.00\nИТОГО 100.00"
    result = _make_ocr(
        ("ДОБРО ПОЖАЛОВАТЬ", 10, 10, 220),
        ("МАГАЗИН У РЕКИ", 10, 30, 200),
        ("05.10.2026 18:20", 10, 50, 180),
        ("МОЛОКО", 10, 80, 100),
        ("100.00", 275, 80, 335),
        ("ИТОГО 100.00", 10, 110, 180),
        shuffled=True,
    )
    actual = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )

    assert actual == Receipt(
        store=None,
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("100.00"),
        items=[Item(name="МОЛОКО", sum=Decimal("100.00"))],
        warnings=["store_not_found"],
    )


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize(
    "date_line", ["05.10 18:20", "Дата: 05.10.2026", "Дата: 05.10 18:20", "05.10.26 18:20"]
)
def test_incomplete_date_is_not_completed_from_current_time(ocr: bool, date_line: str) -> None:
    text = f"МАГАЗИН\n{date_line}\nХЛЕБ 50.00\nИТОГО 50.00"
    result = _make_ocr(
        ("МАГАЗИН", 10, 10, 110),
        (date_line, 10, 30, 200),
        ("ХЛЕБ", 10, 60, 100),
        ("50.00", 275, 60, 335),
        ("ИТОГО 50.00", 10, 90, 180),
    )
    actual = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )

    assert actual == Receipt(
        store="МАГАЗИН",
        datetime=None,
        total=Decimal("50.00"),
        items=[Item(name="ХЛЕБ", sum=Decimal("50.00"))],
        warnings=["datetime_not_found"],
    )


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize("amount", ["-100.00", "- 100,00", "−100.00"])
def test_refund_preserves_negative_item_and_final_total(ocr: bool, amount: str) -> None:
    text = (
        "МАГАЗИН\n05.10.2026 18:20\nВОЗВРАТ ПРИХОДА\n"
        f"МОЛОКО {amount}\nИТОГ -120.00\nК ОПЛАТЕ {amount}\nКАРТОЙ {amount}"
    )
    result = _make_ocr(
        ("МАГАЗИН", 10, 10, 110),
        ("05.10.2026 18:20", 10, 30, 180),
        ("ВОЗВРАТ ПРИХОДА", 10, 50, 200),
        ("МОЛОКО", 10, 80, 100),
        (amount, 255, 80, 335),
        ("ИТОГ -120.00", 10, 110, 180),
        ("К ОПЛАТЕ", 10, 130, 160),
        (amount, 255, 130, 335),
        (f"КАРТОЙ {amount}", 10, 150, 220),
        shuffled=True,
    )
    actual = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )

    assert actual == Receipt(
        store="МАГАЗИН",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("-100.00"),
        items=[Item(name="МОЛОКО", sum=Decimal("-100.00"))],
        warnings=[],
    )


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize("label", ["ВСЕГО К ОПЛАТЕ", "всего\t к\u00a0оплате:"])
@pytest.mark.parametrize(
    ("amount", "warnings"), [("100,00", []), ("99,00", ["items_total_mismatch"])]
)
def test_total_label_is_shared_by_fields_item_region_and_validation(
    ocr: bool, label: str, amount: str, warnings: list[str]
) -> None:
    text = f"МАГАЗИН\nМОЛОКО 100.00\n{label} {amount}"
    result = _make_ocr(
        ("МАГАЗИН", 10, 10, 110),
        ("МОЛОКО", 10, 40, 100),
        ("100.00", 275, 40, 335),
        (label, 10, 70, 210),
        (amount, 275, 70, 335),
        shuffled=True,
    )
    expected = Receipt(
        store="МАГАЗИН",
        total=Decimal(amount.replace(",", ".")),
        items=[Item(name="МОЛОКО", sum=Decimal("100.00"))],
        warnings=["datetime_not_found", *warnings],
    )

    actual = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )
    assert actual == expected


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize("total", [None, "89.99"])
def test_total_word_inside_product_name_neither_supplies_nor_hides_a_total(
    ocr: bool,
    total: str | None,
) -> None:
    text = "МАГАЗИН\nНАПИТОК ИТОГО 90.00"
    result = _make_ocr(
        ("МАГАЗИН", 10, 10, 110),
        ("НАПИТОК ИТОГО", 10, 40, 180),
        ("90.00", 275, 40, 335),
    )
    if total is not None:
        text += f"\nИТОГО {total}"
        result.blocks.append(
            TextBlock(text=f"ИТОГО {total}", bbox=[(10, 70), (335, 70), (335, 82), (10, 82)])
        )
    expected = Receipt(
        store="МАГАЗИН",
        total=Decimal(total) if total is not None else None,
        items=[Item(name="НАПИТОК ИТОГО", sum=Decimal("90.00"))],
        warnings=[
            "datetime_not_found",
            "items_total_mismatch" if total is not None else "total_not_found",
        ],
    )

    actual = (
        parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
        if ocr
        else parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    )
    assert actual == expected


@pytest.mark.parametrize("shuffled", [False, True], ids=["ordered", "shuffled"])
def test_ocr_simple_items_preserve_order_and_input(shuffled: bool) -> None:
    result = _make_ocr(
        ("КОФЕЙНЯ У ДЯДИ ВАСИ", 10, 10, 230),
        ("ИНН 1234567890", 10, 30, 160),
        ("КАССА 2", 10, 50, 90),
        ("05.10.2026 18:20", 10, 70, 180),
        ("КАПУЧИНО", 10, 100, 110),
        ("250.00", 275, 101, 335),
        ("КРУАССАН", 10, 124, 110),
        ("150.00", 275, 123, 335),
        ("ИТОГО", 10, 154, 90),
        ("400.00", 275, 154, 335),
        ("КАРТОЙ 400.00", 10, 178, 180),
        ("ФН 123456789", 10, 202, 150),
        shuffled=shuffled,
    )
    original = result.model_dump()

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="КОФЕЙНЯ У ДЯДИ ВАСИ",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("400.00"),
        items=[
            Item(name="КАПУЧИНО", sum=Decimal("250.00")),
            Item(name="КРУАССАН", sum=Decimal("150.00")),
        ],
        warnings=[],
    )
    assert result.model_dump() == original


def test_ocr_quantity_and_unit_price_are_associated_with_the_product() -> None:
    result = _make_ocr(
        ("ПЕКАРНЯ ЛУННЫЙ КОЛОС", 10, 10, 240),
        ("06.10.2026 08:15", 10, 30, 180),
        ("БУЛОЧКА С КОРИЦЕЙ", 10, 60, 155),
        ("2 x 79.90", 170, 60, 255),
        ("159.80", 275, 60, 335),
        ("ЧАЙ ЧЕРНЫЙ", 10, 84, 130),
        ("45.00", 275, 84, 335),
        ("ИТОГО 204.80", 10, 114, 190),
        ("НАЛИЧНЫМИ 250.00", 10, 138, 190),
        ("СДАЧА 45.20", 10, 162, 150),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="ПЕКАРНЯ ЛУННЫЙ КОЛОС",
        datetime=datetime(2026, 10, 6, 8, 15, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("204.80"),
        items=[
            Item(
                name="БУЛОЧКА С КОРИЦЕЙ",
                quantity=Decimal("2"),
                price=Decimal("79.90"),
                sum=Decimal("159.80"),
            ),
            Item(name="ЧАЙ ЧЕРНЫЙ", sum=Decimal("45.00")),
        ],
        warnings=[],
    )


def test_ocr_weighted_item_uses_monetary_rounding() -> None:
    result = _make_ocr(
        ("ФРУКТОВАЯ ЛАВКА НА ХОЛМЕ", 10, 10, 260),
        ("07.10.2026 10:42:33", 10, 30, 210),
        ("ЯБЛОКИ ГАЛА", 10, 60, 140),
        ("0,734 кг × 129,99", 155, 60, 260),
        ("95,41", 275, 60, 335),
        ("ПАКЕТ БУМАЖНЫЙ", 10, 84, 160),
        ("5,00", 275, 84, 335),
        ("ИТОГО 100,41", 10, 114, 180),
        ("КАРТОЙ 100,41", 10, 138, 180),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="ФРУКТОВАЯ ЛАВКА НА ХОЛМЕ",
        datetime=datetime(2026, 10, 7, 10, 42, 33, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("100.41"),
        items=[
            Item(
                name="ЯБЛОКИ ГАЛА",
                quantity=Decimal("0.734"),
                price=Decimal("129.99"),
                sum=Decimal("95.41"),
            ),
            Item(name="ПАКЕТ БУМАЖНЫЙ", sum=Decimal("5.00")),
        ],
        warnings=[],
    )


@pytest.mark.parametrize("shuffled", [False, True], ids=["ordered", "shuffled"])
def test_ocr_multiline_description_stays_with_its_quantity_row(shuffled: bool) -> None:
    result = _make_ocr(
        ("СЫРНАЯ ЛАВКА У МОСТА", 10, 10, 240),
        ("08.10.2026 17:05", 10, 30, 180),
        ("СЫР РОССИЙСКИЙ", 10, 60, 170),
        ("45% 200Г", 10, 80, 110),
        ("1 x 179.90", 10, 100, 130),
        ("179.90", 275, 100, 335),
        ("ХЛЕБ ЗЕРНОВОЙ", 10, 124, 155),
        ("64.50", 275, 124, 335),
        ("ИТОГО 244.40", 10, 154, 180),
        ("КАРТОЙ 244.40", 10, 178, 180),
        shuffled=shuffled,
    )
    original = result.model_dump()

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="СЫРНАЯ ЛАВКА У МОСТА",
        datetime=datetime(2026, 10, 8, 17, 5, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("244.40"),
        items=[
            Item(
                name="СЫР РОССИЙСКИЙ 45% 200Г",
                quantity=Decimal("1"),
                price=Decimal("179.90"),
                sum=Decimal("179.90"),
            ),
            Item(name="ХЛЕБ ЗЕРНОВОЙ", sum=Decimal("64.50")),
        ],
        warnings=[],
    )
    assert result.model_dump() == original


def test_ocr_partial_extraction_skips_unsupported_item_without_total_warning() -> None:
    result = _make_ocr(
        ("МАГАЗИН ТИХИЙ ДВОР", 10, 10, 230),
        ("ИНН 7701234567", 10, 30, 160),
        ("МОЛОКО", 10, 60, 100),
        ("100.00", 275, 60, 335),
        ("СЫР", 10, 84, 60),
        ("2 x 50.00 ???", 150, 84, 260),
        ("100.00", 275, 84, 335),
        ("ХЛЕБ", 10, 108, 80),
        ("50.00", 275, 108, 335),
        ("ИТОГО 250.00", 10, 138, 180),
        ("КАРТОЙ 250.00", 10, 162, 180),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="МАГАЗИН ТИХИЙ ДВОР",
        datetime=None,
        total=Decimal("250.00"),
        items=[
            Item(name="МОЛОКО", sum=Decimal("100.00")),
            Item(name="ХЛЕБ", sum=Decimal("50.00")),
        ],
        warnings=["datetime_not_found", "items_incomplete"],
    )


def test_ocr_separate_discount_does_not_create_an_item_or_total_mismatch() -> None:
    result = _make_ocr(
        ("КОФЕЙНЯ СЕВЕРНЫЙ ВЕТЕР", 10, 10, 250),
        ("10.10.2026 15:30", 10, 30, 180),
        ("КОФЕ В ЗЕРНАХ", 10, 60, 155),
        ("300.00", 275, 60, 335),
        ("ПЕЧЕНЬЕ ОВСЯНОЕ", 10, 84, 180),
        ("50.00", 275, 84, 335),
        ("СКИДКА", 10, 108, 90),
        ("-50.00", 275, 108, 335),
        ("ИТОГО 300.00", 10, 138, 180),
        ("КАРТОЙ 300.00", 10, 162, 180),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="КОФЕЙНЯ СЕВЕРНЫЙ ВЕТЕР",
        datetime=datetime(2026, 10, 10, 15, 30, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("300.00"),
        items=[
            Item(name="КОФЕ В ЗЕРНАХ", sum=Decimal("300.00")),
            Item(name="ПЕЧЕНЬЕ ОВСЯНОЕ", sum=Decimal("50.00")),
        ],
        warnings=[],
    )


def test_ocr_incorrect_item_arithmetic_preserves_sums_and_deduplicates_warning() -> None:
    result = _make_ocr(
        ("МОЛОЧНАЯ ЛАВКА НА ПЛОЩАДИ", 10, 10, 285),
        ("ИНН 7704567890", 10, 30, 160),
        ("МОЛОКО", 10, 60, 100),
        ("2 x 89.99", 170, 60, 255),
        ("179.97", 275, 60, 335),
        ("КЕФИР", 10, 84, 90),
        ("2 x 79.90", 170, 84, 255),
        ("159.79", 275, 84, 335),
        ("ПАКЕТ", 10, 108, 80),
        ("5.00", 275, 108, 335),
        ("ИТОГО 344.76", 10, 138, 180),
        ("КАРТОЙ 344.76", 10, 162, 180),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="МОЛОЧНАЯ ЛАВКА НА ПЛОЩАДИ",
        datetime=None,
        total=Decimal("344.76"),
        items=[
            Item(
                name="МОЛОКО", quantity=Decimal("2"), price=Decimal("89.99"), sum=Decimal("179.97")
            ),
            Item(
                name="КЕФИР", quantity=Decimal("2"), price=Decimal("79.90"), sum=Decimal("159.79")
            ),
            Item(name="ПАКЕТ", sum=Decimal("5.00")),
        ],
        warnings=["datetime_not_found", "item_amount_mismatch"],
    )


def test_ocr_complete_item_region_reports_incorrect_receipt_total() -> None:
    result = _make_ocr(
        ("ЛАВКА КАНЦТОВАРОВ У ПАРКА", 10, 10, 275),
        ("12.10.2026 12:45", 10, 30, 180),
        ("ТЕТРАДЬ", 10, 60, 85),
        ("В КЛЕТКУ", 90, 60, 180),
        ("89.90", 275, 60, 335),
        ("РУЧКА", 10, 84, 65),
        ("СИНЯЯ", 70, 84, 150),
        ("35.00", 275, 84, 335),
        ("ИТОГО", 10, 114, 90),
        ("124.89", 275, 114, 335),
        ("КАРТОЙ 124.89", 10, 138, 180),
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="ЛАВКА КАНЦТОВАРОВ У ПАРКА",
        datetime=datetime(2026, 10, 12, 12, 45, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("124.89"),
        items=[
            Item(name="ТЕТРАДЬ В КЛЕТКУ", sum=Decimal("89.90")),
            Item(name="РУЧКА СИНЯЯ", sum=Decimal("35.00")),
        ],
        warnings=["items_total_mismatch"],
    )


def test_text_simple_items_remain_supported() -> None:
    text = """КОФЕЙНЯ У ДЯДИ ВАСИ
ИНН 1234567890
05.10.2026 18:20
КАПУЧИНО 250.00
КРУАССАН 150.00
ИТОГО 400.00
КАРТОЙ 400.00"""

    assert parse_receipt(text, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="КОФЕЙНЯ У ДЯДИ ВАСИ",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("400.00"),
        items=[
            Item(name="КАПУЧИНО", sum=Decimal("250.00")),
            Item(name="КРУАССАН", sum=Decimal("150.00")),
        ],
        warnings=[],
    )


def test_text_quantity_rows_are_skipped_without_inferring_missing_values() -> None:
    text = """ПЕКАРНЯ ЛУННЫЙ КОЛОС
06.10.2026 08:15
БУЛОЧКА С КОРИЦЕЙ 2 x 79.90 159.80
ЧАЙ ЧЕРНЫЙ 45.00
ИТОГО 204.80
НАЛИЧНЫМИ 250.00
СДАЧА 45.20"""

    assert parse_receipt(text, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="ПЕКАРНЯ ЛУННЫЙ КОЛОС",
        datetime=datetime(2026, 10, 6, 8, 15, tzinfo=MOSCOW_TIMEZONE),
        total=Decimal("204.80"),
        items=[Item(name="ЧАЙ ЧЕРНЫЙ", sum=Decimal("45.00"))],
        warnings=["items_incomplete"],
    )


def test_text_partial_receipt_preserves_missing_field_warning_and_printed_total() -> None:
    text = """МАГАЗИН ТИХИЙ ДВОР
ИНН 7701234567
МОЛОКО 100.00
СЫР 2 x 50.00 ??? 100.00
ХЛЕБ 50.00
ИТОГО 250.00
КАРТОЙ 250.00"""

    assert parse_receipt(text, timezone=MOSCOW_TIMEZONE) == Receipt(
        store="МАГАЗИН ТИХИЙ ДВОР",
        datetime=None,
        total=Decimal("250.00"),
        items=[
            Item(name="МОЛОКО", sum=Decimal("100.00")),
            Item(name="ХЛЕБ", sum=Decimal("50.00")),
        ],
        warnings=["datetime_not_found", "items_incomplete"],
    )
