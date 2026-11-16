from datetime import date
from decimal import Decimal

import pytest

from bot.texts import (
    EN,
    RU,
    format_history,
    format_monthly_summary,
    format_receipt,
    texts_for,
)
from common.contracts import Item, Receipt, ReceiptWarning
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
        warnings=[ReceiptWarning.ITEMS_TOTAL_MISMATCH],
    )

    assert format_receipt(receipt, EN).endswith(f"\n\n{EN.items_mismatch}")


def test_format_receipt_warns_when_items_are_incomplete() -> None:
    receipt = Receipt(
        store="ЛЕНТА",
        items=[Item(name="ХЛЕБ", sum=Decimal("45"))],
        warnings=[ReceiptWarning.ITEMS_INCOMPLETE],
    )

    assert format_receipt(receipt, RU).endswith(f"\n\n{RU.items_incomplete}")


def test_format_receipt_shows_item_categories() -> None:
    receipt = Receipt(
        store="ЛЕНТА",
        items=[
            Item(name="МОЛОКО", sum=Decimal("89.9"), category="food"),
            Item(name="ХЛЕБ", sum=Decimal("45")),
            Item(name="ЗОНТ", sum=Decimal("500"), category="unknown_code"),
        ],
    )

    assert format_receipt(receipt, RU).endswith(
        "Покупки:\n- МОЛОКО: 89.90 (продукты)\n- ХЛЕБ: 45.00\n- ЗОНТ: 500.00 (unknown_code)"
    )


def test_every_language_names_every_category() -> None:
    assert RU.category_names.keys() == EN.category_names.keys()


def test_format_monthly_summary_lists_categories_from_largest() -> None:
    summary = MonthlySummary(
        month=date(2026, 10, 1),
        receipt_count=2,
        total=Decimal("190"),
        by_category={"personal_care": Decimal("50"), None: Decimal("10"), "food": Decimal("130")},
    )

    assert format_monthly_summary(summary, RU) == (
        "Расходы за 10.2026: 190.00\nЧеков: 2\n"
        "\nПо категориям:\n- продукты: 130.00\n- уход: 50.00\n- без категории: 10.00"
    )


def test_format_monthly_summary_hides_breakdown_without_categories() -> None:
    summary = MonthlySummary(
        month=date(2026, 10, 1),
        receipt_count=1,
        total=Decimal("10"),
        by_category={None: Decimal("10")},
    )

    assert format_monthly_summary(summary, EN) == "Spending for 10.2026: 10.00\nReceipts: 1"
