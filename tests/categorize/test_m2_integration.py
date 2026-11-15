"""End-to-end contract checks without loading the neural model."""

from datetime import UTC, datetime
from decimal import Decimal

from categorize.service import CategoryService
from common import CategoryPrediction, CategoryScore, Item, Receipt
from common.contracts import Item as M2Item
from common.contracts import Receipt as M2Receipt
from common.models import ClassifiedItem, ClassifiedReceipt


class FakeClassifier:
    def predict(self, item_name: str) -> CategoryPrediction:
        return CategoryPrediction(categories=[CategoryScore(category="food", confidence=0.9)])


def test_m2_is_the_only_receipt_contract() -> None:
    assert Item is M2Item
    assert Receipt is M2Receipt


def test_receipt_preserves_every_field_and_optional_values() -> None:
    incoming = M2Receipt(
        store="Магнит",
        datetime=datetime(2026, 10, 28, 19, 2, tzinfo=UTC),
        total=Decimal("218.43"),
        items=[
            M2Item(name="Молоко 2.5%", quantity=None, price=None, sum=Decimal("129.99")),
            M2Item(
                name="Хлеб",
                quantity=Decimal("2"),
                price=Decimal("44.22"),
                sum=Decimal("88.44"),
            ),
        ],
        warnings=["unknown_tax_id"],
    )
    result = CategoryService(FakeClassifier()).classify_receipt(incoming)
    assert isinstance(result, ClassifiedReceipt)
    assert all(isinstance(item, ClassifiedItem) for item in result.items)
    assert result.store == incoming.store
    assert result.datetime == incoming.datetime
    assert result.total == incoming.total
    assert result.warnings == incoming.warnings
    assert [item.model_dump(exclude={"categories"}) for item in result.items] == [
        item.model_dump() for item in incoming.items
    ]
    assert result.items[0].categories[0].category == "food"
    assert result.items[0].quantity is None
    assert result.items[0].price is None


def test_empty_m2_receipt_can_be_classified() -> None:
    assert CategoryService(FakeClassifier()).classify_receipt(M2Receipt()).items == []
