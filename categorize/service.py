"""M2→M3 service using the canonical common.contracts input models."""

from typing import Protocol

from categorize.classifier import CategoryClassifier
from categorize.ollama import CategoryValidation
from common.contracts import Item, Receipt
from common.models import CategoryPrediction, CategoryScore, ClassifiedItem, ClassifiedReceipt


class CategoryValidator(Protocol):
    """Second-opinion interface for category labels."""

    def validate(
        self, item_name: str, predicted: list[str], allowed: list[str]
    ) -> CategoryValidation:
        """Return an independent category assessment."""
        ...


class CategoryService:
    """Append category predictions while preserving all fields from M2."""

    def __init__(
        self,
        classifier: CategoryClassifier,
        validator: CategoryValidator | None = None,
    ) -> None:
        self._classifier = classifier
        self._validator = validator

    def classify_item(self, item: Item) -> ClassifiedItem:
        prediction = self._classifier.predict(item.name)
        return self._classified_item(item, prediction)

    def _classified_item(self, item: Item, prediction: CategoryPrediction) -> ClassifiedItem:
        validation = (
            self._validate_category(item.name, prediction.categories)
            if self._validator is not None
            else CategoryValidation(status="uncertain", categories=[])
        )
        return ClassifiedItem(
            **item.model_dump(exclude={"categories"}),
            categories=prediction.categories,
            category_validation=validation.status,
            suggested_categories=validation.categories,
        )

    def _validate_category(
        self, item_name: str, categories: list[CategoryScore]
    ) -> CategoryValidation:
        labels = getattr(self._classifier, "category_names", None)
        predicted = [score.category for score in categories]
        allowed = list(labels) if labels else predicted
        return self._validator.validate(item_name, predicted, allowed)

    def classify_items(self, items: list[Item]) -> list[ClassifiedItem]:
        predict_many = getattr(self._classifier, "predict_many", None)
        if callable(predict_many):
            predictions = predict_many([item.name for item in items])
            if len(predictions) != len(items):
                raise ValueError("batch classifier returned a different number of predictions")
            return [
                self._classified_item(item, prediction)
                for item, prediction in zip(items, predictions, strict=True)
            ]
        return [self.classify_item(item) for item in items]

    def classify_receipt(self, receipt: Receipt) -> ClassifiedReceipt:
        return ClassifiedReceipt(
            **receipt.model_dump(exclude={"items"}),
            items=self.classify_items(receipt.items),
        )
