"""CategoryService wrapping a CategoryClassifier for M2→M3 transformation.

PDF requirements:
- CategoryService(classifier): takes a CategoryClassifier instance
- classify_item(item: Item) -> ClassifiedItem: classifies a single item
- classify_items(items: list[Item]) -> list[ClassifiedItem]: classifies multiple items
- classify_receipt(receipt: Receipt) -> ClassifiedReceipt: classifies all items in a receipt
"""
from categorize.classifier import CategoryClassifier
from common.models import ClassifiedItem, ClassifiedReceipt, Item, Receipt


class CategoryService:
    """Service for classifying receipt items and receipts using a CategoryClassifier.
    
    PDF flow: Item(name, quantity, total) → predict(name) → ClassifiedItem(categories=[...])
    """
    
    def __init__(self, classifier: CategoryClassifier) -> None:
        self._classifier = classifier

    def classify_item(self, item: Item) -> ClassifiedItem:
        """Classify a single item and return a ClassifiedItem.
        
        PDF flow: predict(item.name) → categories → ClassifiedItem
        """
        prediction = self._classifier.predict(item.name)
        return ClassifiedItem(
            name=item.name,
            quantity=item.quantity,
            total=item.total,
            categories=prediction.categories,
        )

    def classify_items(self, items: list[Item]) -> list[ClassifiedItem]:
        """Classify multiple items."""
        return [self.classify_item(item) for item in items]

    def classify_receipt(self, receipt: Receipt) -> ClassifiedReceipt:
        """Classify all items in a receipt (M2→M3 transformation)."""
        return ClassifiedReceipt(
            receipt_id=receipt.receipt_id,
            purchased_at=receipt.purchased_at,
            items=self.classify_items(receipt.items),
        )