from __future__ import annotations

import random

from categorize.text import add_noise, denoise, shorten


def test_denoise_removes_vat_and_quantities() -> None:
    assert denoise("НДС 20% Молоко 3,2% 1 шт") == "молоко 3,2%"


def test_denoise_fixes_mixed_alphabet_and_case() -> None:
    assert denoise("MолOKO 3,2%") == "молоко 3,2%"


def test_denoise_keeps_percent_in_product_name() -> None:
    assert denoise("Кефир 1%") == "кефир 1%"


def test_denoise_collapses_whitespace_and_punctuation() -> None:
    assert denoise("  Сыр   Российский *  ") == "сыр российский"


def test_add_noise_is_deterministic() -> None:
    first = add_noise("стиральный порошок автомат", random.Random(7))
    second = add_noise("стиральный порошок автомат", random.Random(7))
    assert first == second
    assert first != "стиральный порошок автомат"


def test_shorten_keeps_first_words() -> None:
    assert shorten("Молоко 3,2% пастеризованное ультрапастеризованное", max_words=3) == (
        "молоко 3,2% пастеризованное"
    )
