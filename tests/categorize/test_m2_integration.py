"""End-to-end contract checks without loading the neural model."""

from datetime import UTC, datetime
from decimal import Decimal

from categorize.ollama import CategoryValidation
from categorize.service import CategoryService
from common import CategoryPrediction, CategoryScore, Item, Receipt
from common.contracts import Item as M2Item
from common.contracts import Receipt as M2Receipt
from common.models import ClassifiedItem, ClassifiedReceipt


class FakeClassifier:
    category_names = ["food", "household"]

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
    assert [
        item.model_dump(exclude={"categories", "category_validation", "suggested_categories"})
        for item in result.items
    ] == [item.model_dump(exclude={"categories"}) for item in incoming.items]
    assert result.items[0].categories[0].category == "food"
    assert result.items[0].quantity is None
    assert result.items[0].price is None


def test_empty_m2_receipt_can_be_classified() -> None:
    assert CategoryService(FakeClassifier()).classify_receipt(M2Receipt()).items == []


def test_local_validator_is_reported_without_replacing_classifier_categories() -> None:
    class FakeValidator:
        def validate(
            self, item_name: str, predicted: list[str], allowed: list[str]
        ) -> CategoryValidation:
            assert item_name == "Молоко"
            assert predicted == ["food"]
            assert allowed == ["food", "household"]
            return CategoryValidation(status="disagreed", categories=["household"])

    result = CategoryService(FakeClassifier(), validator=FakeValidator()).classify_item(
        M2Item(name="Молоко", sum=Decimal("100"))
    )
    assert result.categories[0].category == "food"
    assert result.category_validation == "disagreed"
    assert result.suggested_categories == ["household"]


def test_reclassification_replaces_categories_already_stored_on_item() -> None:
    item = M2Item(name="Молоко", sum=Decimal("100"), categories=["household"])

    result = CategoryService(FakeClassifier()).classify_item(item)

    assert [score.category for score in result.categories] == ["food"]


def test_uncertain_validation_keeps_previous_m3_category() -> None:
    class UncertainValidator:
        def validate(
            self, item_name: str, predicted: list[str], allowed: list[str]
        ) -> CategoryValidation:
            return CategoryValidation(status="uncertain", categories=[])

    result = CategoryService(FakeClassifier(), validator=UncertainValidator()).classify_item(
        M2Item(name="Неясное OCR имя", sum=Decimal("10"))
    )
    assert [score.category for score in result.categories] == ["food"]
    assert result.category_validation == "uncertain"
    assert result.suggested_categories == []
