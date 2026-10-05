from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from common.contracts import Item, Receipt


def test_item_and_receipt_accept_complete_data() -> None:
    item = Item(
        name="МОЛОКО",
        quantity=Decimal("2"),
        price=Decimal("80.50"),
        sum=Decimal("161.00"),
    )
    receipt_datetime = datetime(2026, 10, 5, 15, 41, tzinfo=UTC)

    receipt = Receipt(
        store="Пятёрочка",
        datetime=receipt_datetime,
        total=Decimal("161.00"),
        items=[item],
    )

    assert receipt.store == "Пятёрочка"
    assert receipt.datetime == receipt_datetime
    assert receipt.total == Decimal("161.00")
    assert receipt.items == [item]


def test_item_quantity_and_price_are_optional() -> None:
    item = Item(name="ХЛЕБ", sum=Decimal("54.99"))

    assert item.quantity is None
    assert item.price is None


def test_receipt_fields_are_optional() -> None:
    receipt = Receipt()

    assert receipt.store is None
    assert receipt.datetime is None
    assert receipt.total is None


def test_receipt_list_defaults_are_independent() -> None:
    first = Receipt()
    second = Receipt()

    first.items.append(Item(name="ХЛЕБ", sum=Decimal("54.99")))
    first.warnings.append("example")

    assert len(first.items) == 1
    assert second.items == []
    assert first.warnings == ["example"]
    assert second.warnings == []


def test_empty_item_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Item(name="", sum=Decimal("10.00"))


def test_receipt_datetime_must_be_timezone_aware() -> None:
    with pytest.raises(ValidationError):
        Receipt(datetime=datetime(2026, 10, 5, 15, 41))

    aware_datetime = datetime(2026, 10, 5, 15, 41, tzinfo=UTC)
    assert Receipt(datetime=aware_datetime).datetime == aware_datetime


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Receipt.model_validate({"total": Decimal("100.00"), "currency": "RUB"})


def test_receipt_json_round_trip() -> None:
    receipt = Receipt(
        store="Пятёрочка",
        datetime=datetime(2026, 10, 5, 15, 41, tzinfo=UTC),
        total=Decimal("144.98"),
        items=[
            Item(
                name="ХЛЕБ ДАРНИЦКИЙ",
                quantity=Decimal("1"),
                price=Decimal("54.99"),
                sum=Decimal("54.99"),
            ),
            Item(name="МОЛОКО 2.5%", sum=Decimal("89.99")),
        ],
        warnings=["example"],
    )

    serialized = receipt.model_dump_json()
    restored = Receipt.model_validate_json(serialized)

    assert restored == receipt
