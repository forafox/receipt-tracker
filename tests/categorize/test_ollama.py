from __future__ import annotations

import json

import pytest

from categorize.ollama import OllamaCategoryValidator


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self._body = body

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self._body).encode()


def _body(categories: list[str]) -> dict[str, object]:
    return {"message": {"content": json.dumps({"categories": categories})}}


def test_validator_reports_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    from categorize import ollama

    monkeypatch.setattr(ollama, "urlopen", lambda *args, **kwargs: _Response(_body(["food"])))
    result = OllamaCategoryValidator().validate("Молоко", ["food"], ["food", "household"])
    assert result.status == "agreed"
    assert result.categories == ["food"]


def test_validator_reports_disagreement_without_mutating_m3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from categorize import ollama

    monkeypatch.setattr(ollama, "urlopen", lambda *args, **kwargs: _Response(_body(["household"])))
    result = OllamaCategoryValidator().validate("Губка для посуды", ["food"], ["food", "household"])
    assert result.status == "disagreed"
    assert result.categories == ["household"]


def test_validator_rejects_unknown_category(monkeypatch: pytest.MonkeyPatch) -> None:
    from categorize import ollama

    monkeypatch.setattr(ollama, "urlopen", lambda *args, **kwargs: _Response(_body(["unknown"])))
    result = OllamaCategoryValidator().validate("Молоко", ["food"], ["food", "household"])
    assert result.status == "uncertain"
    assert result.categories == []
