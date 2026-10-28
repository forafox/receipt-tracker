from decimal import ROUND_DOWN, Decimal, Inexact, Rounded, localcontext
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item, Receipt
from common.ocr import OcrResult, TextBlock
from extract import parse_ocr_result, parse_receipt
from extract.validation import _validate_arithmetic

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")
HEADER = "Лавка у реки\n05.10.2026 18:20\n"


def _parse(text: str, *, ocr: bool) -> Receipt:
    if not ocr:
        return parse_receipt(text, timezone=MOSCOW_TIMEZONE)
    blocks = []
    for index, line in enumerate(text.splitlines()):
        if line.strip():
            y = index * 20
            blocks.append(
                TextBlock(text=line, bbox=[(10, y), (300, y), (300, y + 12), (10, y + 12)])
            )
    return parse_ocr_result(OcrResult(blocks=blocks[::-1]), timezone=MOSCOW_TIMEZONE)


@pytest.mark.parametrize(
    ("quantity", "price", "amount", "warnings"),
    [
        ("2", "89.99", "179.98", []),
        ("2", "89.99", "179.97", ["item_amount_mismatch"]),
        ("0.734", "129.99", "95.41", []),
        ("0.5", "0.01", "0.01", []),
        ("0.5", "0.01", "0.00", ["item_amount_mismatch"]),
        (None, None, "250.00", []),
        ("2", None, "250.00", []),
        (None, "89.99", "250.00", []),
    ],
)
def test_item_validation_rounds_half_up_and_preserves_extracted_values(
    quantity: str | None, price: str | None, amount: str, warnings: list[str]
) -> None:
    item = Item(
        name="ТОВАР",
        quantity=Decimal(quantity) if quantity is not None else None,
        price=Decimal(price) if price is not None else None,
        sum=Decimal(amount),
    )
    receipt = Receipt(items=[item])
    original = receipt.model_dump_json(exclude={"warnings"})

    _validate_arithmetic(receipt, extraction_complete=False)

    assert receipt.warnings == warnings
    assert receipt.model_dump_json(exclude={"warnings"}) == original
    assert receipt.items[0] is item


@pytest.mark.parametrize(
    ("total", "has_item", "complete", "warnings"),
    [
        ("50.00", True, True, []),
        ("49.99", True, True, ["items_total_mismatch"]),
        ("49.99", True, False, []),
        (None, True, True, []),
        ("49.99", False, True, []),
    ],
)
def test_total_validation_requires_complete_extraction_total_and_items(
    total: str | None, has_item: bool, complete: bool, warnings: list[str]
) -> None:
    receipt = Receipt(
        total=Decimal(total) if total is not None else None,
        items=[Item(name="ХЛЕБ", sum=Decimal("50.00"))] if has_item else [],
    )
    original = receipt.model_dump_json(exclude={"warnings"})

    _validate_arithmetic(receipt, extraction_complete=complete)

    assert receipt.warnings == warnings
    assert receipt.model_dump_json(exclude={"warnings"}) == original


def test_warning_order_is_stable_and_repeated_validation_does_not_duplicate_codes() -> None:
    receipt = Receipt(
        items=[
            Item(name=name, quantity=Decimal("2"), price=Decimal("89.99"), sum=Decimal("179.97"))
            for name in ("МОЛОКО", "КЕФИР")
        ],
        total=Decimal("0.00"),
        warnings=["store_not_found", "datetime_not_found"],
    )

    _validate_arithmetic(receipt, extraction_complete=True)
    _validate_arithmetic(receipt, extraction_complete=True)

    assert receipt.warnings == [
        "store_not_found",
        "datetime_not_found",
        "item_amount_mismatch",
        "items_total_mismatch",
    ]


def test_validation_is_independent_of_ambient_decimal_context() -> None:
    receipt = Receipt(
        items=[
            Item(
                name="ЯБЛОКИ",
                quantity=Decimal("0.734"),
                price=Decimal("129.99"),
                sum=Decimal("95.41"),
            ),
            Item(
                name="Большая сумма OCR",
                quantity=Decimal("2"),
                price=Decimal("12345678901234567890123456789.99"),
                sum=Decimal("24691357802469135780246913579.98"),
            ),
        ]
    )

    with localcontext() as context:
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        context.traps[Rounded] = True
        _validate_arithmetic(receipt, extraction_complete=False)
        assert context.prec == 3
        assert context.rounding == ROUND_DOWN
        assert context.traps[Inexact]
        assert context.traps[Rounded]

    assert receipt.warnings == []


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize(
    ("total", "warnings"), [("400.00", []), ("399.00", ["items_total_mismatch"])]
)
def test_both_entry_points_validate_a_fully_parsed_item_region(
    ocr: bool, total: str, warnings: list[str]
) -> None:
    receipt = _parse(HEADER + f"КАПУЧИНО 250.00\nКРУАССАН 150.00\nИТОГО {total}", ocr=ocr)

    assert receipt.items == [
        Item(name="КАПУЧИНО", sum=Decimal("250.00")),
        Item(name="КРУАССАН", sum=Decimal("150.00")),
    ]
    assert receipt.total == Decimal(total)
    assert receipt.warnings == warnings


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize(
    "unresolved", ["UNRECOGNIZED ITEM LINE", "СЫР 2 x 50.00 ??? 100.00", "100.00"]
)
def test_unresolved_candidates_prevent_definitive_total_warnings(
    ocr: bool, unresolved: str
) -> None:
    receipt = _parse(HEADER + f"МОЛОКО 100.00\n{unresolved}\nХЛЕБ 50.00\nИТОГО 250.00", ocr=ocr)

    assert [item.sum for item in receipt.items] == [Decimal("100.00"), Decimal("50.00")]
    assert receipt.total == Decimal("250.00")
    assert receipt.warnings == ["items_incomplete"]


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize(
    "body",
    [
        "КОФЕ 300.00\nСКИДКА -50.00\nИТОГО 250.00",
        "КОФЕ 300.00\nИТОГО 250.00\nОКРУГЛЕНИЕ -0.01",
        "КОФЕ 300.00\nНДС 20% 50.00\nИТОГО 250.00",
        "КОФЕ 300.00\nКАРТОЙ 250.00\nХЛЕБ 50.00\nИТОГО 250.00",
        "КОФЕ 300.00\nИТОГО 300.00\nК ОПЛАТЕ 250.00",
        "КОФЕ 300.00\nИТОГО 250.00\nНЕРАСПОЗНАННЫЙ ИТОГ 200.00",
        "КОФЕ 300.00\nИТОГО 250.00\nНЕРАСПОЗНАННАЯ СТРОКА 200.00",
    ],
)
def test_summaries_and_unclear_boundaries_disable_total_reconciliation(
    ocr: bool, body: str
) -> None:
    assert _parse(HEADER + body, ocr=ocr).warnings == []


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
def test_percentage_in_product_name_does_not_disable_total_validation(ocr: bool) -> None:
    receipt = _parse(HEADER + "СЫР 45% 200Г 300.00\nИТОГО 250.00", ocr=ocr)

    assert receipt.warnings == ["items_total_mismatch"]


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
@pytest.mark.parametrize("operation", ["ПРИХОД", "ВОЗВРАТ ПРИХОДА"])
def test_operation_labels_do_not_count_as_unparsed_items(ocr: bool, operation: str) -> None:
    receipt = _parse(HEADER + f"{operation}\nХЛЕБ 50.00\nИТОГО 50.00", ocr=ocr)

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.warnings == []


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
def test_products_without_merchant_are_kept_and_reconciled(ocr: bool) -> None:
    receipt = _parse("КАПУЧИНО 250.00\nКРУАССАН 150.00\nИТОГО 399.00", ocr=ocr)

    assert receipt.store is None
    assert receipt.items == [
        Item(name="КАПУЧИНО", sum=Decimal("250.00")),
        Item(name="КРУАССАН", sum=Decimal("150.00")),
    ]
    assert receipt.warnings == ["store_not_found", "datetime_not_found", "items_total_mismatch"]


def test_ocr_item_warnings_follow_missing_field_warnings_even_when_total_is_unavailable() -> None:
    receipt = _parse("Лавка\nМОЛОКО 2 x 89.99 179.97", ocr=True)

    assert receipt.warnings == [
        "datetime_not_found",
        "total_not_found",
        "item_amount_mismatch",
    ]


@pytest.mark.parametrize("ocr", [False, True], ids=["text", "ocr"])
def test_incomplete_warning_follows_missing_fields(ocr: bool) -> None:
    receipt = _parse("Лавка\nХЛЕБ 50.00\nСЫР", ocr=ocr)

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.warnings == ["datetime_not_found", "total_not_found", "items_incomplete"]


def test_incomplete_warning_precedes_item_arithmetic_and_disables_total_check() -> None:
    receipt = _parse(HEADER + "МОЛОКО 2 x 89.99 179.97\nСЫР\nИТОГО 100.00", ocr=True)

    assert receipt.items == [
        Item(name="МОЛОКО", quantity=Decimal("2"), price=Decimal("89.99"), sum=Decimal("179.97"))
    ]
    assert receipt.total == Decimal("100.00")
    assert receipt.warnings == ["items_incomplete", "item_amount_mismatch"]


def test_multiline_ocr_items_get_item_checks_but_not_definitive_total_comparison() -> None:
    receipt = _parse(HEADER + "МОЛОКО\n2 x 89.99 179.97\nИТОГО 100.00", ocr=True)

    assert receipt.items == [
        Item(name="МОЛОКО", quantity=Decimal("2"), price=Decimal("89.99"), sum=Decimal("179.97"))
    ]
    assert receipt.total == Decimal("100.00")
    assert receipt.warnings == ["item_amount_mismatch"]
