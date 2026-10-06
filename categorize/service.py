from categorize.classifier import CategoryClassifier
from common import ClassifiedItem, ClassifiedReceipt, Item, Receipt


class CategoryService:
    def __init__(self, classifier: CategoryClassifier) -> None:
        self._classifier = classifier

    def classify_item(self, item: Item) -> ClassifiedItem:
        prediction = self._classifier.predict(item.name)
        return ClassifiedItem(
            name=item.name,
            quantity=item.quantity,
            total=item.total,
            categories=prediction.categories,
        )

    def classify_items(self, items: list[Item]) -> list[ClassifiedItem]:
        return [self.classify_item(item) for item in items]

    def classify_receipt(self, receipt: Receipt) -> ClassifiedReceipt:
        return ClassifiedReceipt(
            receipt_id=receipt.receipt_id,
            purchased_at=receipt.purchased_at,
            items=self.classify_items(receipt.items),
        )
