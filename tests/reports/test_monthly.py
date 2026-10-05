from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from common import ClassifiedItem, ClassifiedReceipt
from reports import build_monthly_report


def test_build_monthly_report_groups_totals_by_category() -> None:
    receipt = ClassifiedReceipt(
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


def test_build_monthly_report_rejects_receipts_from_different_months() -> None:
    first = ClassifiedReceipt(
        receipt_id="r-1",
        purchased_at=datetime(2026, 10, 1, tzinfo=ZoneInfo("Europe/Moscow")),
        items=[
            ClassifiedItem(
                name="молоко",
                quantity=Decimal("1"),
                total=Decimal("80"),
                category="food",
                confidence=0.9,
            )
        ],
    )
    second = ClassifiedReceipt(
        receipt_id="r-2",
        purchased_at=datetime(2026, 11, 1, tzinfo=ZoneInfo("Europe/Moscow")),
        items=[
            ClassifiedItem(
                name="хлеб",
                quantity=Decimal("1"),
                total=Decimal("60"),
                category="food",
                confidence=0.9,
            )
        ],
    )

    with pytest.raises(ValueError):
        build_monthly_report([first, second])
