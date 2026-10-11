import pytest

from bot.texts import EN, RU, format_receipt, texts_for
from common.contracts import Receipt


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
