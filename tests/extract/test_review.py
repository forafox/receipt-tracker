from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from pytest import MonkeyPatch

from extract import review

MOSCOW = ZoneInfo("Europe/Moscow")
TEXT = (
    "ПЯТЁРОЧКА\n05.10.2026 18:20\nМОЛОКО 3,2% 89.99\n"
    "ХЛЕБ БОРОДИНСКИЙ 45.00\nИТОГО 134.99"
)


class _Response:
    def __init__(self, content: str) -> None:
        self._content = content

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps({"message": {"content": self._content}}).encode()


def _fake_reply(monkeypatch: MonkeyPatch, corrected: str, verdict: str = "correct") -> None:
    content = json.dumps(
        {
            "verdict": verdict,
            "summary": "Чек согласован",
            "issues": [],
            "corrected_text": corrected,
        },
        ensure_ascii=False,
    )
    monkeypatch.setattr(review, "urlopen", lambda *args, **kwargs: _Response(content))


def test_review_returns_model_verdict_and_deterministic_parse(monkeypatch) -> None:
    _fake_reply(monkeypatch, TEXT)

    result = review.review_receipt_text(TEXT, timezone=MOSCOW)

    assert result.verdict == "correct"
    assert result.safe_to_apply
    assert result.numeric_tokens_preserved
    assert result.original_receipt.total == Decimal("134.99")
    assert result.original_receipt.datetime == datetime(
        2026, 10, 5, 18, 20, tzinfo=MOSCOW
    )


def test_review_rejects_suggestion_that_changes_digits(monkeypatch) -> None:
    _fake_reply(monkeypatch, TEXT.replace("89.99", "99.99"))

    result = review.review_receipt_text(TEXT, timezone=MOSCOW)

    assert not result.numeric_tokens_preserved
    assert not result.safe_to_apply


def test_uncertain_model_cannot_mark_text_safe(monkeypatch) -> None:
    _fake_reply(monkeypatch, TEXT, verdict="uncertain")

    result = review.review_receipt_text(TEXT, timezone=MOSCOW)

    assert not result.safe_to_apply


def test_arithmetic_mismatch_overrides_model_verdict(monkeypatch: MonkeyPatch) -> None:
    bad_text = TEXT.replace("134.99", "139.99")
    _fake_reply(monkeypatch, bad_text, verdict="correct")

    result = review.review_receipt_text(bad_text, timezone=MOSCOW)

    assert result.model_verdict == "correct"
    assert result.verdict == "incorrect"
    assert "не совпадает" in result.summary
    assert not result.safe_to_apply
