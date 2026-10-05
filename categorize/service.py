from categorize.classifier import CategoryClassifier
from common import ClassifiedItem, Item


class CategoryService:
    def __init__(self, classifier: CategoryClassifier) -> None:
        self._classifier = classifier

    def classify_item(self, item: Item) -> ClassifiedItem:
        prediction = self._classifier.predict(item.name)
        return ClassifiedItem(
            name=item.name,
            quantity=item.quantity,
            total=item.total,
            category=prediction.category,
            confidence=prediction.confidence,
        )

    def classify_items(self, items: list[Item]) -> list[ClassifiedItem]:
        return [self.classify_item(item) for item in items]
