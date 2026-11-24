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


@pytest.mark.parametrize(
    ("expression", "quantity", "price", "amount"),
    [
        ("5.49 * 2 = 10.98", "2", "5.49", "10.98"),
        ("84.99 * 1.500 = 127.49", "1.500", "84.99", "127.49"),
        ("0.00*3 0.00", "3", "0.00", "0.00"),
        ("115.00 * 1шт. = 115.00", "1", "115.00", "115.00"),
        ("140.00 ж 3 = 420.00", "3", "140.00", "420.00"),
    ],
)
def test_price_times_quantity_rows_are_parsed(
    expression: str, quantity: str, price: str, amount: str
) -> None:
    result = _result([_block(f"ТОВАР {expression} НДС 20%", 10, 60, 300)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(name="ТОВАР", quantity=Decimal(quantity), price=Decimal(price), sum=Decimal(amount))
    ]


def test_multiline_description_uses_price_times_quantity_row() -> None:
    result = _result(
        [
            _block("картофель печёный", 10, 60, 160),
            _block("115.00 * 1шт. = 115.00", 10, 80, 300),
        ]
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="картофель печёный",
            quantity=Decimal("1"),
            price=Decimal("115.00"),
            sum=Decimal("115.00"),
        )
    ]


def test_bank_sum_line_is_not_an_item() -> None:
    result = _result(
        [
            _block("МОЛОКО", 10, 60),
            _block("100.00", 210, 60, 60),
            _block("Сумма (Руб): 2100.08", 10, 80, 220),
        ],
        total="2100.08",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="МОЛОКО", sum=Decimal("100.00"))]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ИТОГО 100.00\nНДС 20% 16.67", Decimal("100.00")),
        ("ИТОГО 100.00\nКАРТОЙ 200.00", Decimal("100.00")),
        ("ХЛЕБ 50.00\nИТОГО:", None),
    ],
)
def test_total_in_multiline_ocr_block_uses_logical_text_lines(
    text: str, expected: Decimal | None
) -> None:
    result = _result([_block(text, 10, 60, 300)])
    original = result.model_dump()

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.total == expected
    assert receipt.total == parse_receipt(text, timezone=MOSCOW_TIMEZONE).total
    assert result.model_dump() == original


@pytest.mark.parametrize(
    ("label", "amount", "item_amount", "expected", "warnings"),
    [
        ("ИТОГО:", "50.00", "50.00", "50.00", []),
        ("К ОПЛАТЕ", "−50,00", "-50.00", "-50.00", []),
        ("ИТОГО К ОПЛАТЕ", "=49.99", "50.00", "49.99", ["items_total_mismatch"]),
    ],
)
def test_adjacent_total_amount_is_shared_by_fields_region_and_reconciliation(
    label: str, amount: str, item_amount: str, expected: str, warnings: list[str]
) -> None:
    result = _result(
        [
            _block(f"ХЛЕБ {item_amount}", 10, 60, 260),
            _block(label, 10, 100),
            _block(amount, 210, 120, 60),
        ]
    )
    original = result.model_dump()

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.total == Decimal(expected)
    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal(item_amount))]
    assert receipt.warnings == warnings
    assert result.model_dump() == original
    text = "\n".join(line.text for line in reconstruct_layout(result))
    assert parse_receipt(text, timezone=MOSCOW_TIMEZONE).total is None


@pytest.mark.parametrize(
    ("amount", "x", "y"),
    [
        ("50.00", 10, 120),
        ("50.00", 210, 160),
        ("50.00 70.00", 210, 120),
        ("НДС 50.00", 210, 120),
        ("50.00 руб 70.00", 210, 120),
    ],
)
def test_split_total_rejects_unrelated_or_ambiguous_amounts(
    amount: str, x: float, y: float
) -> None:
    result = _result(
        [
            _block("ХЛЕБ 50.00", 10, 60, 260),
            _block("ИТОГО", 110, 100),
            _block(amount, x, y, 100),
        ]
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE).total is None


def test_two_nearby_amount_rows_do_not_identify_a_unique_total() -> None:
    result = _result(
        [
            _block("ХЛЕБ 50.00", 10, 60, 260),
            _block("ИТОГО", 10, 100),
            TextBlock(text="50.00", bbox=[(210, 112), (270, 112), (270, 116), (210, 116)]),
            _block("70.00", 210, 116, 60),
        ]
    )

    assert parse_ocr_result(result, timezone=MOSCOW_TIMEZONE).total is None


def test_fuel_item_uses_printed_price_quantity_and_amount_across_ocr_blocks() -> None:
    result = _result(
        [
            TextBlock(text=text, bbox=[(left, top), (right, top), (right, bottom), (left, bottom)])
            for text, left, top, right, bottom in [
                ("ТРК #2 Бензин неэтилированный АИ-95-К5", 126, 279, 731, 311),
                ("40. 90 x 24.440 Л", 126, 314, 381, 364),
                ("=999.60_A", 751, 339, 893, 365),
            ]
        ]
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="ТРК #2 Бензин неэтилированный АИ-95-К5",
            quantity=Decimal("24.440"),
            price=Decimal("40.90"),
            sum=Decimal("999.60"),
        )
    ]
    assert receipt.warnings == ["total_not_found"]


@pytest.mark.parametrize("suffix", [" А", "_A"])
def test_quantity_first_expression_accepts_a_single_fiscal_letter(suffix: str) -> None:
    result = _result([_block(f"ТОВАР 10.000 x 65.00 = 650.00{suffix}", 10, 60, 300)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="ТОВАР", quantity=Decimal("10.000"), price=Decimal("65.00"), sum=Decimal("650.00")
        )
    ]


@pytest.mark.parametrize(
    "tail",
    ["700.00", "А 700.00", "НДС 20% 700.00", "НДС 20.00", ".99", "0"],
)
def test_price_first_expression_does_not_ignore_extra_financial_values(tail: str) -> None:
    separator = "" if tail in {".99", "0"} else " "
    result = _result([_block(f"ТОВАР 65.00 x 10 = 650.00{separator}{tail}", 10, 60, 400)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == []
    assert receipt.warnings == ["total_not_found", "items_incomplete"]


@pytest.mark.parametrize(
    ("expression", "suffix", "quantity", "price", "amount"),
    [
        ("65.00 x 10 = 650.00", "_A", "10", "65.00", "650.00"),
        ("10 x 65.00 = 650.00", " НДС 20 %", "10", "65.00", "650.00"),
        ("-65.00 x 10 л = -650.00", " HAC 20%", "10", "-65.00", "-650.00"),
        ("65.00 x 10 = 650.00", " НИС 102", "10", "65.00", "650.00"),
        ("65.00 x 10 = 650.00", " НД", "10", "65.00", "650.00"),
    ],
)
def test_financial_expression_suffixes_preserve_explicit_values(
    expression: str, suffix: str, quantity: str, price: str, amount: str
) -> None:
    result = _result([_block(f"ТОВАР {expression}{suffix}", 10, 60, 400)])

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(name="ТОВАР", quantity=Decimal(quantity), price=Decimal(price), sum=Decimal(amount))
    ]
    assert receipt.warnings == ["total_not_found"]


@pytest.mark.parametrize("suffix", ["", " НДС 20%"])
@pytest.mark.parametrize("amount", ["650.00", "649.00"])
def test_two_monetary_operands_do_not_assign_quantity_roles(amount: str, suffix: str) -> None:
    result = _result(
        [_block(f"БЕНЗИН 65.00 x 10.00 = {amount}{suffix}", 10, 60, 300)], total=amount
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="БЕНЗИН", sum=Decimal(amount))]
    assert receipt.warnings == ["items_incomplete"]


@pytest.mark.parametrize("split_total", [False, True])
def test_unknown_operand_roles_do_not_hide_a_receipt_total_mismatch(split_total: bool) -> None:
    blocks = [_block("БЕНЗИН 65.00 x 10.00 = 650.00", 10, 60, 300)]
    if split_total:
        blocks += [_block("ИТОГО", 10, 100), _block("700.00", 210, 120, 60)]
    result = _result(blocks, total=None if split_total else "700.00")

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="БЕНЗИН", sum=Decimal("650.00"))]
    assert receipt.total == Decimal("700.00")
    assert receipt.warnings == ["items_incomplete", "items_total_mismatch"]


@pytest.mark.parametrize("expression", ["10.00 л x 65.00", "65.00 x 10.00 л"])
def test_explicit_unit_resolves_two_monetary_operands(expression: str) -> None:
    result = _result([_block(f"БЕНЗИН {expression} = 650.00", 10, 60, 300)], total="650.00")

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="БЕНЗИН", quantity=Decimal("10.00"), price=Decimal("65.00"), sum=Decimal("650.00")
        )
    ]
    assert receipt.warnings == []


def test_truncated_product_name_ending_in_hyphen_keeps_its_explicit_amounts() -> None:
    result = _result(
        [_block("Сельдь МЕРИДИАН филе- 169.99 * 1 =169.99 HAC 10%", 10, 60, 400)],
        total="169.99",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [
        Item(
            name="Сельдь МЕРИДИАН филе-",
            quantity=Decimal("1"),
            price=Decimal("169.99"),
            sum=Decimal("169.99"),
        )
    ]
    assert receipt.warnings == []


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


def test_nameless_quantity_row_outside_description_column_is_not_associated() -> None:
    result = _result(
        [
            _block("СЫР", 10, 60),
            _block("1 x 179.90 179.90", 210, 80, 190),
            _block("ХЛЕБ", 10, 110),
            _block("50.00", 210, 110, 60),
        ],
        total="50.00",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.total == Decimal("50.00")
    assert receipt.warnings == ["items_incomplete"]


def test_unaligned_descriptions_are_not_merged_for_a_bare_amount() -> None:
    result = _result(
        [
            _block("МОЛОКО", 10, 60),
            _block("СЫР", 150, 80),
            _block("179.90", 300, 100, 60),
        ],
        total="179.90",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="СЫР", sum=Decimal("179.90"))]
    assert receipt.total == Decimal("179.90")
    assert receipt.warnings == ["items_incomplete"]


@pytest.mark.parametrize("quantity", ["-2", "- 2", "−2", "− 2"])
def test_negative_named_quantity_is_not_reinterpreted_as_positive(quantity: str) -> None:
    result = _result(
        [
            _block(f"МОЛОКО {quantity} x 50.00 -100.00", 10, 60, 300),
            _block("ХЛЕБ", 10, 90),
            _block("50.00", 210, 90, 60),
        ],
        total="-50.00",
    )

    receipt = parse_ocr_result(result, timezone=MOSCOW_TIMEZONE)

    assert receipt.items == [Item(name="ХЛЕБ", sum=Decimal("50.00"))]
    assert receipt.total == Decimal("-50.00")
    assert receipt.warnings == ["items_incomplete"]


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
