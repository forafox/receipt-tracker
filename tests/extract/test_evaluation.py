from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from common.contracts import Item, Receipt
from common.ocr import OcrResult, TextBlock
from extract.evaluation import (
    FieldAccuracy,
    ItemMetrics,
    ReceiptMismatch,
    evaluate_ocr_results,
    evaluate_receipts,
)


def _item(name: str = "Milk", amount: str = "50.00") -> Item:
    return Item(name=name, quantity=Decimal("1"), price=Decimal(amount), sum=Decimal(amount))


def test_perfect_match_ignores_item_order_and_warnings_without_mutating_inputs() -> None:
    reference = Receipt(
        store="Shop",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=UTC),
        total=Decimal("70.00"),
        items=[_item(), _item("Bread", "20")],
    )
    prediction = reference.model_copy(deep=True)
    prediction.items.reverse()
    prediction.warnings.append("items_incomplete")
    before = prediction.model_dump(), reference.model_dump()

    result = evaluate_receipts(iter([(prediction, reference)]))

    assert result.receipt_count == 1
    assert result.store == result.datetime == result.total == FieldAccuracy(1, 1)
    assert result.store.accuracy == 1.0
    assert result.items == ItemMetrics(2, 2, 2)
    assert result.items.precision == result.items.recall == result.items.f1 == 1.0
    assert result.mismatches == ()
    assert (prediction.model_dump(), reference.model_dump()) == before


def test_micro_metrics_count_missing_extra_and_duplicate_occurrences_per_receipt() -> None:
    result = evaluate_receipts(
        [
            (Receipt(items=[_item()]), Receipt(items=[_item(), _item()])),
            (Receipt(items=[_item(), _item(), _item()]), Receipt(items=[_item()])),
            (Receipt(items=[_item("Bread", "20")]), Receipt()),
            (Receipt(), Receipt(items=[_item("Bread", "20")])),
        ]
    )

    assert result.receipt_count == 4
    assert result.items == ItemMetrics(2, 5, 4)
    assert result.items.precision == pytest.approx(2 / 5)
    assert result.items.recall == pytest.approx(2 / 4)
    assert result.items.f1 == pytest.approx(4 / 9)
    assert result.mismatches == (
        ReceiptMismatch(0, (), 1, 0),
        ReceiptMismatch(1, (), 0, 2),
        ReceiptMismatch(2, (), 0, 1),
        ReceiptMismatch(3, (), 1, 0),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "milk"),
        ("name", "Milk "),
        ("quantity", Decimal("2")),
        ("quantity", None),
        ("price", Decimal("50.001")),
        ("price", None),
        ("sum", Decimal("49.99")),
    ],
)
def test_each_item_field_requires_exact_equality(field: str, value: str | Decimal | None) -> None:
    prediction = _item().model_copy(update={field: value})

    result = evaluate_receipts([(Receipt(items=[prediction]), Receipt(items=[_item()]))])

    assert result.items == ItemMetrics(0, 1, 1)
    assert result.items.precision == result.items.recall == result.items.f1 == 0.0
    assert result.mismatches == (ReceiptMismatch(0, (), 1, 1),)


def test_decimal_scale_and_equivalent_datetime_instants_match() -> None:
    reference = Receipt(
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=timezone(timedelta(hours=3))),
        total=Decimal("50.00"),
        items=[_item(), Item(name="Bread", sum=Decimal("0.00"))],
    )
    prediction = Receipt(
        datetime=datetime(2026, 10, 5, 15, 20, tzinfo=UTC),
        total=Decimal("50"),
        items=[
            Item(name="Milk", quantity=Decimal("1.00"), price=Decimal("50"), sum=Decimal("50")),
            Item(name="Bread", sum=Decimal("0")),
        ],
    )

    result = evaluate_receipts([(prediction, reference)])

    assert result.datetime.accuracy == result.total.accuracy == result.items.f1 == 1.0
    assert result.mismatches == ()


def test_scalar_denominators_include_every_reference_and_compare_absent_values() -> None:
    reference = Receipt(
        store="Shop", datetime=datetime(2026, 10, 5, tzinfo=UTC), total=Decimal("0")
    )
    wrong = Receipt(store="shop", datetime=datetime(2026, 10, 6, tzinfo=UTC), total=Decimal("0.01"))

    result = evaluate_receipts(
        [
            (reference, reference),
            (Receipt(), reference),
            (wrong, reference),
            (wrong, Receipt()),
            (Receipt(), Receipt(store="Shop")),
        ]
    )

    assert result.store == FieldAccuracy(1, 5)
    assert result.datetime == result.total == FieldAccuracy(2, 5)
    assert result.store.accuracy == 0.2
    assert result.total.accuracy == 0.4
    assert result.mismatches == (
        ReceiptMismatch(1, ("store", "datetime", "total"), 0, 0),
        ReceiptMismatch(2, ("store", "datetime", "total"), 0, 0),
        ReceiptMismatch(3, ("store", "datetime", "total"), 0, 0),
        ReceiptMismatch(4, ("store",), 0, 0),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("store", "НЕСУЩЕСТВУЮЩИЙ МАГАЗИН"),
        ("datetime", datetime(2026, 10, 5, tzinfo=UTC)),
        ("total", Decimal("100")),
    ],
)
def test_predicted_value_against_none_reference_is_a_mismatch(
    field: str, value: str | datetime | Decimal
) -> None:
    prediction = Receipt().model_copy(update={field: value})

    result = evaluate_receipts([(prediction, Receipt())])

    assert getattr(result, field) == FieldAccuracy(0, 1)
    assert getattr(result, field).accuracy == 0.0
    assert result.mismatches == (ReceiptMismatch(0, (field,), 0, 0),)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_empty_inputs_and_empty_receipts(count: int) -> None:
    result = evaluate_receipts((Receipt(), Receipt()) for _ in range(count))

    assert result.receipt_count == count
    assert result.store == result.datetime == result.total == FieldAccuracy(count, count)
    assert result.store.accuracy == (1.0 if count else None)
    assert result.items == ItemMetrics(0, 0, 0)
    assert result.items.precision == result.items.recall == result.items.f1 == 0.0
    assert result.mismatches == ()


@pytest.mark.parametrize("prediction_empty", [True, False])
def test_one_sided_empty_items(prediction_empty: bool) -> None:
    empty, purchase = Receipt(), Receipt(items=[_item()])
    pair = (empty, purchase) if prediction_empty else (purchase, empty)

    result = evaluate_receipts([pair])

    assert result.items.matched_count == 0
    assert result.items.precision == result.items.recall == result.items.f1 == 0.0
    assert result.mismatches == (
        ReceiptMismatch(0, (), int(prediction_empty), int(not prediction_empty)),
    )


def test_existing_ocr_results_are_parsed_against_independent_reference() -> None:
    local_timezone = timezone(timedelta(hours=3))
    lines = ["Лавка", "05.10.2026 18:20", "МОЛОКО 50.00", "ИТОГО 50.00"]
    ocr_result = OcrResult(
        blocks=[
            TextBlock(
                text=text,
                bbox=[(0, y * 20), (200, y * 20), (200, y * 20 + 12), (0, y * 20 + 12)],
            )
            for y, text in enumerate(lines, start=1)
        ]
    )
    reference = Receipt(
        store="Лавка",
        datetime=datetime(2026, 10, 5, 18, 20, tzinfo=local_timezone),
        total=Decimal("50.00"),
        items=[Item(name="МОЛОКО", sum=Decimal("50.00"))],
    )
    before = ocr_result.model_dump(), reference.model_dump()

    result = evaluate_ocr_results(
        iter([(ocr_result, reference), (OcrResult(), reference)]), timezone=local_timezone
    )

    assert result.receipt_count == 2
    assert result.store == result.datetime == result.total == FieldAccuracy(1, 2)
    assert result.items == ItemMetrics(1, 1, 2)
    assert result.mismatches == (ReceiptMismatch(1, ("store", "datetime", "total"), 1, 0),)
    assert (ocr_result.model_dump(), reference.model_dump()) == before
