"""M2→M3 service using the canonical common.contracts input models."""

from categorize.classifier import CategoryClassifier
from common.contracts import Item, Receipt
from common.models import ClassifiedItem, ClassifiedReceipt


class CategoryService:
    """Append category predictions while preserving all fields from M2."""

    def __init__(self, classifier: CategoryClassifier) -> None:
        self._classifier = classifier

    def classify_item(self, item: Item) -> ClassifiedItem:
        prediction = self._classifier.predict(item.name)
        return ClassifiedItem(**item.model_dump(), categories=prediction.categories)

    def classify_items(self, items: list[Item]) -> list[ClassifiedItem]:
        return [self.classify_item(item) for item in items]

    def classify_receipt(self, receipt: Receipt) -> ClassifiedReceipt:
        return ClassifiedReceipt(
            **receipt.model_dump(exclude={"items"}),
            items=self.classify_items(receipt.items),
        )
