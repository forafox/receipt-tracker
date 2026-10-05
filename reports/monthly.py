from collections import defaultdict
from decimal import Decimal

from common import ClassifiedItem, MonthlyReport, Receipt, ReceiptCategorySummary


def build_monthly_report(receipts: list[Receipt]) -> MonthlyReport:
    if not receipts:
        msg = "monthly report requires at least one receipt"
        raise ValueError(msg)

    year = receipts[0].purchased_at.year
    month = receipts[0].purchased_at.month
    totals: defaultdict[str, Decimal] = defaultdict(lambda: Decimal("0"))
    counts: defaultdict[str, int] = defaultdict(int)

    for receipt in receipts:
        if receipt.purchased_at.year != year or receipt.purchased_at.month != month:
            msg = "all receipts must belong to the same month"
            raise ValueError(msg)
        for item in receipt.items:
            if not isinstance(item, ClassifiedItem):
                msg = "receipt items must be classified before reporting"
                raise TypeError(msg)
            totals[item.category] += item.total
            counts[item.category] += 1

    categories = [
        ReceiptCategorySummary(category=category, total=total, items_count=counts[category])
        for category, total in sorted(totals.items(), key=lambda value: value[1], reverse=True)
    ]
    return MonthlyReport(
        year=year,
        month=month,
        total=sum(totals.values(), Decimal("0")),
        categories=categories,
    )
