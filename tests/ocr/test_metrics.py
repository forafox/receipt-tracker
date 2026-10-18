from __future__ import annotations

import pytest

from ocr.core.metrics import character_error_rate, levenshtein, word_error_rate


def test_levenshtein_known_distances() -> None:
    assert levenshtein("", "") == 0
    assert levenshtein("abc", "abc") == 0
    assert levenshtein("", "abc") == 3
    assert levenshtein("kitten", "sitting") == 3


def test_character_error_rate_perfect_match() -> None:
    assert character_error_rate("итого 250,00", "итого 250,00") == 0.0


def test_character_error_rate_substitution() -> None:
    assert character_error_rate("abcd", "abce") == pytest.approx(0.25)


def test_character_error_rate_empty_reference() -> None:
    assert character_error_rate("", "") == 0.0
    assert character_error_rate("", "text") == 1.0


def test_word_error_rate_uses_words() -> None:
    assert word_error_rate("итого двести", "итого 200") == pytest.approx(0.5)


def test_word_error_rate_empty_reference() -> None:
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "text") == 1.0
