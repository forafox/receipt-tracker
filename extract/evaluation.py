"""Evaluate extraction against independent receipt annotations without changing the parser."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import tzinfo
from decimal import Decimal

from common.contracts import Item, Receipt
from common.ocr import OcrResult
from extract.parser import parse_ocr_result


@dataclass(frozen=True)
class FieldAccuracy:
    """Exact matches for an annotated receipt field.

    Attributes:
        matches: Number of predictions equal to the reference, including None == None.
        reference_count: Number of fully annotated reference receipts evaluated.
    """

    matches: int
    reference_count: int

    @property
    def accuracy(self) -> float | None:
        """Return matches / reference_count, or None when there are no references."""
        return self.matches / self.reference_count if self.reference_count else None


@dataclass(frozen=True)
class ItemMetrics:
    """Corpus-level counts for micro-averaged exact item matching.

    Attributes:
        matched_count: Number of one-to-one exact item matches.
        predicted_count: Number of predicted items, including duplicates.
        reference_count: Number of annotated items, including duplicates.
    """

    matched_count: int
    predicted_count: int
    reference_count: int

    @property
    def precision(self) -> float:
        """Return matches / predictions, or 0.0 when there are no predictions."""
        return self.matched_count / self.predicted_count if self.predicted_count else 0.0

    @property
    def recall(self) -> float:
        """Return matches / references, or 0.0 when there are no references."""
        return self.matched_count / self.reference_count if self.reference_count else 0.0

    @property
    def f1(self) -> float:
        """Return 2 * matches / (predictions + references), or 0.0 if both are empty."""
        count = self.predicted_count + self.reference_count
        return 2 * self.matched_count / count if count else 0.0


@dataclass(frozen=True)
class ReceiptMismatch:
    """Concise differences for one fully annotated input pair.

    Attributes:
        index: Zero-based position of the pair in the input iterable.
        fields: Names of mismatched annotated scalar fields, in contract order.
        missing_items: Unmatched reference item count, including duplicate occurrences.
        extra_items: Unmatched predicted item count, including duplicate occurrences.
    """

    index: int
    fields: tuple[str, ...]
    missing_items: int
    extra_items: int


@dataclass(frozen=True)
class ExtractionEvaluation:
    """Aggregate extraction quality and differences in input order.

    Attributes:
        receipt_count: Total number of prediction/reference pairs evaluated.
        store: Exact store name accuracy, without text normalization.
        datetime: Exact datetime accuracy using aware datetime equality (same instant).
        total: Exact total accuracy using Decimal equality, without rounding.
        items: Micro-averaged item counts and derived scores.
        mismatches: Only pairs with scored differences; warnings are ignored.
    """

    receipt_count: int
    store: FieldAccuracy
    datetime: FieldAccuracy
    total: FieldAccuracy
    items: ItemMetrics
    mismatches: tuple[ReceiptMismatch, ...]


def evaluate_receipts(pairs: Iterable[tuple[Receipt, Receipt]]) -> ExtractionEvaluation:
    """Compare predictions with independent references, without parsing or loading data.

    References must be fully annotated. None means an absent scalar value and is compared
    for equality: None matches None, and any non-None value mismatches None in either
    direction. Each scalar denominator is the receipt count. An empty reference item
    list means no purchases, not missing labels. Partial annotations are not supported.
    Items match as multisets of (name, quantity, price, sum), ignoring order. Names are
    case/whitespace sensitive; numeric values use Decimal equality (1.0 equals 1.00),
    without tolerance or rounding. Optional item values of None match only None.
    Each occurrence matches at most once, within its receipt. Warnings are ignored.

    Args:
        pairs: Iterable of (predicted receipt, reference receipt) pairs.

    Returns:
        Counts, scalar accuracies, micro item metrics and concise mismatches. Empty input
        yields zero counts, None scalar accuracies and 0.0 item scores.
    """
    fields = ("store", "datetime", "total")
    matches: Counter[str] = Counter()
    receipt_count = matched_count = predicted_count = reference_count = 0
    mismatches: list[ReceiptMismatch] = []
    for index, (prediction, reference) in enumerate(pairs):
        receipt_count += 1
        wrong_fields = []
        for field in fields:
            expected = getattr(reference, field)
            if getattr(prediction, field) == expected:
                matches[field] += 1
            else:
                wrong_fields.append(field)
        predicted_items = Counter(_item_key(item) for item in prediction.items)
        reference_items = Counter(_item_key(item) for item in reference.items)
        matched = (predicted_items & reference_items).total()
        missing = len(reference.items) - matched
        extra = len(prediction.items) - matched
        matched_count += matched
        predicted_count += len(prediction.items)
        reference_count += len(reference.items)
        if wrong_fields or missing or extra:
            mismatches.append(ReceiptMismatch(index, tuple(wrong_fields), missing, extra))
    return ExtractionEvaluation(
        receipt_count=receipt_count,
        store=FieldAccuracy(matches["store"], receipt_count),
        datetime=FieldAccuracy(matches["datetime"], receipt_count),
        total=FieldAccuracy(matches["total"], receipt_count),
        items=ItemMetrics(matched_count, predicted_count, reference_count),
        mismatches=tuple(mismatches),
    )


def evaluate_ocr_results(
    pairs: Iterable[tuple[OcrResult, Receipt]], *, timezone: tzinfo
) -> ExtractionEvaluation:
    """Parse existing OCR results and evaluate against independent receipt annotations.

    Args:
        pairs: Iterable of (OCR result, independently annotated reference receipt) pairs.
        timezone: Timezone passed to parse_ocr_result for local receipt datetimes.

    Returns:
        The same metrics and mismatch information as evaluate_receipts.

    Raises:
        ValueError: If the parser encounters a non-finite bounding-box coordinate.
    """
    return evaluate_receipts(
        (parse_ocr_result(result, timezone=timezone), reference) for result, reference in pairs
    )


def _item_key(item: Item) -> tuple[str, Decimal | None, Decimal | None, Decimal]:
    return item.name, item.quantity, item.price, item.sum
