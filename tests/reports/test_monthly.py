from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common import ClassifiedItem, Item, Receipt
from reports import build_monthly_report


def test_build_monthly_report_groups_totals_by_category() -> None:
    receipt = Receipt(
        receipt_id="r-1",
        purchased_at=datetime(2026, 10, 1, tzinfo=ZoneInfo("Europe/Moscow")),
        items=[
            ClassifiedItem(
                name="молоко",
                quantity=Decimal("1"),
                total=Decimal("80"),
                category="food",
                confidence=0.9,
            ),
            ClassifiedItem(
                name="шампунь",
                quantity=Decimal("1"),
                total=Decimal("150"),
                category="personal_care",
                confidence=0.8,
            ),
        ],
    )

    report = build_monthly_report([receipt])

    assert report.total == Decimal("230")
    assert [category.category for category in report.categories] == ["personal_care", "food"]


def test_build_monthly_report_rejects_unclassified_items() -> None:
    receipt = Receipt(
        receipt_id="r-1",
        purchased_at=datetime(2026, 10, 1, tzinfo=ZoneInfo("Europe/Moscow")),
        items=[Item(name="молоко", quantity=Decimal("1"), total=Decimal("80"))],
    )

    with pytest.raises(TypeError):
        build_monthly_report([receipt])
