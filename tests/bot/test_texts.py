import pytest

from bot.texts import EN, RU, texts_for


@pytest.mark.parametrize(
    ("language_code", "expected"),
    [("ru", RU), ("ru-RU", RU), ("RU", RU), ("en", EN), ("en-GB", EN), ("de", EN), (None, EN)],
)
def test_texts_for_picks_language(language_code: str | None, expected: object) -> None:
    assert texts_for(language_code) is expected
