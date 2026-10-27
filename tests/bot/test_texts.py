from datetime import date
from decimal import Decimal

import pytest

from bot.texts import (
    EN,
    ITEMS_TOTAL_MISMATCH,
    RU,
    format_history,
    format_monthly_summary,
    format_receipt,
    texts_for,
)
from common.contracts import Item, Receipt
from extract import validation
from pipeline import MonthlySummary


@pytest.mark.parametrize(
    ("language_code", "expected"),
    [("ru", RU), ("ru-RU", RU), ("RU", RU), ("en", EN), ("en-GB", EN), ("de", EN), (None, EN)],
)
def test_texts_for_picks_language(language_code: str | None, expected: object) -> None:
    assert texts_for(language_code) is expected


def test_format_receipt_marks_missing_fields() -> None:
    receipt = Receipt(store="ЛЕНТА")

    assert format_receipt(receipt, RU) == (
        "Магазин: ЛЕНТА\nДата: не распознано\nСумма: не распознано"
    )


def test_format_monthly_summary_without_receipts() -> None:
    summary = MonthlySummary(month=date(2026, 10, 1), receipt_count=0, total=Decimal(0))

    assert format_monthly_summary(summary, EN) == EN.month_empty


def test_format_history_without_receipts() -> None:
    assert format_history([], RU) == RU.history_empty


def test_format_receipt_lists_items() -> None:
    receipt = Receipt(
        store="ЛЕНТА",
        total=Decimal("134.90"),
        items=[
            Item(name="МОЛОКО", sum=Decimal("89.9")),
            Item(name="ХЛЕБ", sum=Decimal("45")),
        ],
    )

    assert format_receipt(receipt, RU) == (
        "Магазин: ЛЕНТА\nДата: не распознано\nСумма: 134.90\n"
        "\nПокупки:\n- МОЛОКО: 89.90\n- ХЛЕБ: 45.00"
    )


def test_format_receipt_warns_when_items_do_not_add_up() -> None:
    receipt = Receipt(
        store="ЛЕНТА",
        total=Decimal("100.00"),
        items=[Item(name="ХЛЕБ", sum=Decimal("45"))],
        warnings=["items_total_mismatch"],
    )

    assert format_receipt(receipt, EN).endswith(f"\n\n{EN.items_mismatch}")


def test_mismatch_warning_matches_extraction_module() -> None:
    assert ITEMS_TOTAL_MISMATCH == validation.ITEMS_TOTAL_MISMATCH
