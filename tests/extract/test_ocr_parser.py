from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common.contracts import Item
from common.ocr import OcrResult, TextBlock
from extract import parse_ocr_result, parse_receipt
from extract.layout import reconstruct_layout

MOSCOW_TIMEZONE = ZoneInfo("Europe/Moscow")


def _block(text: str, x: float, y: float, width: float = 100) -> TextBlock:
    return TextBlock(
        text=text,
        bbox=[(x, y), (x + width, y), (x + width, y + 12), (x, y + 12)],
        confidence=None,
    )


def _result(blocks: list[TextBlock], *, total: str | None = None) -> OcrResult:
    header = [_block("Лавка у реки", 10, 0, 150), _block("05.10.2026 18:20", 10, 20, 180)]
    footer = []
    if total is not None:
        bottom = max(y for block in blocks for _, y in block.bbox) + 20
        footer = [_block(f"ИТОГО {total}", 10, bottom, 200)]
    return OcrResult(blocks=[*footer, *reversed(blocks), *header])


def test_simple_items_use_shuffled_blocks_and_preserve_existing_receipt_fields() -> None:
    result = _result(
        [
            _block("КАПУЧИНО", 10, 60),
            _block("250.00", 210, 61, 60),
            _block("КРУАССАН", 10, 85),
            _block("150.00", 210, 84, 60),
        ],
        total="400.00",
    )
    original = result.model_dump()

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(name="КАПУЧИНО", sum=Decimal("250.00")),
        Item(name="КРУАССАН", sum=Decimal("150.00")),
    ]
    assert receipt.store == "Лавка у реки"
    assert receipt.datetime == datetime(2026, 10, 5, 18, 20, tzinfo=MOSCOW_TIMEZONE)
    assert receipt.total == Decimal("400.00")
    assert receipt.warnings == []
    text = "\n".join(line.text for line in reconstruct_layout(result))
    assert receipt.model_dump(exclude={"items"}) == parse_receipt(
        text, timezone=MOSCOW_TIMEZONE
    ).model_dump(exclude={"items"})
    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == receipt
    assert result.model_dump() == original
    result.blocks.reverse()
    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE) == receipt


def test_multi_block_row_preserves_explicit_values_with_arithmetic_warnings() -> None:
    result = _result(
        [
            _block("МОЛОКО", 10, 60, 60),
            _block("2", 100, 60, 10),
            _block("x", 120, 60, 10),
            _block("50.00", 140, 60, 50),
            _block("99.00", 210, 60, 60),
        ],
        total="300.00",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(name="МОЛОКО", quantity=Decimal("2"), price=Decimal("50.00"), sum=Decimal("99.00"))
    ]
    assert receipt.total == Decimal("300.00")
    assert receipt.warnings == ["item_amount_mismatch", "items_total_mismatch"]


def test_simple_name_fragments_preserve_product_numbers_and_duplicate_items() -> None:
    result = _result(
        [
            _block(text, x, y, width)
            for y in (60, 80)
            for text, x, width in [
                ("КОФЕ", 10, 40),
                ("МОЛОТЫЙ 0.33КГ", 60, 130),
                ("250.00", 210, 60),
            ]
        ]
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE).items == [
        Item(name="КОФЕ МОЛОТЫЙ 0.33КГ", sum=Decimal("250.00")),
        Item(name="КОФЕ МОЛОТЫЙ 0.33КГ", sum=Decimal("250.00")),
    ]


@pytest.mark.parametrize(
    ("expression", "quantity", "price", "amount"),
    [
        ("2 * 50.00 = 100.00", "2", "50.00", "100.00"),
        ("0,532 кг × 799,90 425,55", "0.532", "799.90", "425.55"),
        ("0.25kg х 1 200.00 = 300.00", "0.25", "1200.00", "300.00"),
        ("1 x -100,00 = -100,00", "1", "-100.00", "-100.00"),
    ],
)
def test_explicit_quantity_and_weight_in_an_entire_row_block(
    expression: str, quantity: str, price: str, amount: str
) -> None:
    result = _result([_block(f"ТОВАР {expression}", 10, 60, 270)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(name="ТОВАР", quantity=Decimal(quantity), price=Decimal(price), sum=Decimal(amount))
    ]
    assert receipt.warnings == ["total_not_found"]
    text = "\n".join(line.text for line in reconstruct_layout(result))
    assert parse_receipt(text, timezone=MOSCOW_TIMEZONE).items == []


@pytest.mark.parametrize("scale", [0.5, 3.0])
def test_multiline_descriptions_and_quantity_rows_use_relative_geometry(scale: float) -> None:
    result = _result(
        [
            _block("СЫР РОССИЙСКИЙ", 10, 60, 160),
            _block("45% 200Г", 10, 80, 100),
            _block("1 x 179.90", 10, 100, 120),
            _block("179.90", 210, 100, 60),
            _block("КОФЕ МОЛОТЫЙ", 10, 140, 150),
            _block("АРАБИКА", 10, 160, 100),
            _block("250.00", 210, 160, 60),
        ]
    )
    for block in result.blocks:
        block.bbox = [(x * scale, y * scale) for x, y in block.bbox]

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="СЫР РОССИЙСКИЙ 45% 200Г",
            quantity=Decimal("1"),
            price=Decimal("179.90"),
            sum=Decimal("179.90"),
        ),
        # A named, priced row cannot prove that the preceding fragment belongs to it.
        Item(name="АРАБИКА", sum=Decimal("250.00")),
    ]
    assert receipt.warnings == ["total_not_found", "items_incomplete"]


@pytest.mark.parametrize("expression", ["50.00", "1 x 50.00 50.00"])
def test_complete_named_item_does_not_absorb_a_preceding_unpriced_item(expression: str) -> None:
    result = _result(
        [
            _block("МОЛОКО", 10, 60),
            _block("ХЛЕБ", 10, 80),
            _block(expression, 130, 80, 140),
        ],
        total="50.00",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="ХЛЕБ",
            quantity=Decimal("1") if "x" in expression else None,
            price=Decimal("50.00") if "x" in expression else None,
            sum=Decimal("50.00"),
        )
    ]
    assert receipt.total == Decimal("50.00")
    assert receipt.warnings == ["items_incomplete"]


@pytest.mark.parametrize(
    ("x", "y", "expected"), [(210, 80, True), (10, 80, False), (210, 140, False)]
)
def test_bare_amount_requires_nearby_description_to_its_left(
    x: float, y: float, expected: bool
) -> None:
    result = _result([_block("ХЛЕБ", 10, 60), _block("50.00", x, y, 60)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == ([Item(name="ХЛЕБ", sum=Decimal("50.00"))] if expected else [])
    assert receipt.warnings == ["total_not_found", *([] if expected else ["items_incomplete"])]


@pytest.mark.parametrize(
    "unsupported",
    [
        "2 x 50.00",
        "0 x 50.00 0.00",
        "-2 x 50.00 100.00",
        "50.00 100.00",
        "1 x ??? 50.00",
        "2 1 x 50.00 50.00",
    ],
)
def test_incomplete_or_ambiguous_values_are_not_inferred_or_carried_forward(
    unsupported: str,
) -> None:
    result = _result(
        [
            _block("БЕЗ ЦЕНЫ", 10, 60),
            _block(unsupported, 10, 80, 260),
            _block("ХЛЕБ 50.00", 10, 100, 260),
        ]
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)
    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.warnings == ["total_not_found", "items_incomplete"]


def test_metadata_breaks_multiline_association_and_payment_ends_item_region() -> None:
    result = _result(
        [
            _block("ИНН 1234567890", 10, 40, 170),
            _block("БЕЗ ЦЕНЫ", 10, 60),
            _block("НДС 20% 10.00", 10, 80, 180),
            _block("50.00", 210, 100, 60),
            _block("ХЛЕБ 50.00", 10, 120, 260),
            _block("СКИДКА 5.00", 10, 140, 260),
            _block("КАРТОЙ 50.00", 10, 160, 260),
            _block("СЛУЖЕБНЫЙ ТЕКСТ 123.00", 10, 180, 260),
        ]
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.warnings == ["total_not_found", "items_incomplete"]


def test_overlapping_or_competing_numeric_blocks_are_skipped() -> None:
    result = _result(
        [
            _block("ТОВАР", 10, 60),
            _block("50.00", 80, 60, 60),
            _block("ТОВАР", 10, 80),
            _block("50.00", 150, 80, 50),
            _block("100.00", 210, 80, 60),
        ]
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE).items == []


def test_empty_ocr_result_preserves_existing_missing_field_warnings() -> None:
    assert parse_ocr_result(OcrResult(), timezone=MOSCOW_TIMEZONE) == parse_receipt(
        "", timezone=MOSCOW_TIMEZONE
    )
