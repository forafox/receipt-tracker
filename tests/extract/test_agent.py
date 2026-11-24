import json
from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from pytest import MonkeyPatch

from common.contracts import Item, Receipt, ReceiptWarning
from common.ocr import OcrResult, TextBlock
from extract.agent import parse_with_agent
from extract.parser import parse_ocr_result

MOSCOW = ZoneInfo("Europe/Moscow")


def _ocr_result() -> OcrResult:
    return OcrResult(
        blocks=[
            TextBlock(text="МАГАЗИН", bbox=[(0, 0), (100, 0), (100, 10), (0, 10)]),
            TextBlock(
                text="05.10.2026 18:20",
                bbox=[(0, 20), (120, 20), (120, 30), (0, 30)],
            ),
            TextBlock(text="ХЛЕБ 50.00", bbox=[(0, 40), (100, 40), (100, 50), (0, 50)]),
            TextBlock(text="ИТОГО 50.00", bbox=[(0, 60), (100, 60), (100, 70), (0, 70)]),
        ]
    )


def test_agent_parser_uses_existing_parser_without_token(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("KILO_API_KEY", raising=False)
    monkeypatch.delenv("KILO_TOKEN", raising=False)

    assert parse_with_agent(_ocr_result(), timezone=MOSCOW) == parse_ocr_result(
        _ocr_result(), timezone=MOSCOW
    )


def test_agent_parser_accepts_valid_structured_response(monkeypatch: MonkeyPatch) -> None:
    from extract import agent

    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "store": "МАГАЗИН",
                            "datetime": "2026-10-05T18:20:00+03:00",
                            "total": "50.00",
                            "items": [{"name": "ХЛЕБ", "sum": "50.00"}],
                        }
                    )
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    monkeypatch.setenv("KILO_API_KEY", "test-token")
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: FakeResponse())

    receipt = parse_with_agent(_ocr_result(), timezone=MOSCOW)

    assert receipt.store == "МАГАЗИН"
    assert receipt.datetime == datetime.fromisoformat("2026-10-05T18:20:00+03:00")
    assert receipt.total == Decimal("50.00")
    assert receipt.items[0].sum == Decimal("50.00")
    assert receipt.warnings == []


def test_agent_parser_rejects_non_reconciling_response(monkeypatch: MonkeyPatch) -> None:
    from extract import agent

    response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "store": "OTHER",
                            "datetime": None,
                            "total": "99.00",
                            "items": [{"name": "ХЛЕБ", "sum": "50.00"}],
                        }
                    )
                }
            }
        ]
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(response).encode()

    monkeypatch.setenv("KILO_API_KEY", "test-token")
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: FakeResponse())

    assert parse_with_agent(_ocr_result(), timezone=MOSCOW) == parse_ocr_result(
        _ocr_result(), timezone=MOSCOW
    )


def test_local_parser_fallback_accepts_validated_structured_result(
    monkeypatch: MonkeyPatch,
) -> None:
    from extract import agent

    monkeypatch.delenv("KILO_API_KEY", raising=False)
    monkeypatch.delenv("KILO_TOKEN", raising=False)
    monkeypatch.setattr(agent, "_local_model_available", True)
    fallback = Receipt(warnings=[ReceiptWarning.ITEMS_INCOMPLETE])
    improved_data = {
        "store": "МАГАЗИН",
        "datetime": "2026-10-05T18:20:00+03:00",
        "total": "50.00",
        "items": [{"name": "ХЛЕБ", "sum": "50.00"}],
        "warnings": [],
    }

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": json.dumps(improved_data)}}).encode()

    monkeypatch.setattr(agent, "parse_ocr_result", lambda *args, **kwargs: fallback)
    monkeypatch.setattr(agent, "urlopen", lambda *args, **kwargs: FakeResponse())

    result = parse_with_agent(_ocr_result(), timezone=MOSCOW)
    assert result.store == "МАГАЗИН"
    assert result.total == Decimal("50.00")
    assert [item.name for item in result.items] == ["ХЛЕБ"]


def test_numeric_guard_allows_date_separators_but_rejects_digit_changes() -> None:
    from extract.review import _number_tokens

    assert _number_tokens("03-11-2025") == _number_tokens("03.11.2025")
    assert _number_tokens("Сумма 50.00") != _number_tokens("Сумма 5000")


def test_local_model_evidence_checks_rejects_address_as_store_and_discount_as_item() -> None:
    from extract.agent import (
        _split_ocr_text,
        _supported_item,
        _supported_store,
        _supported_total,
    )

    lines = ["ООО ЛЕНТА", "Санкт-Петербург", "Скидка по карте ЛЕНТА 30.00"]
    assert _supported_store("ЛЕНТА", lines)
    assert not _supported_store("ПЕТЕРБУРГ", lines)
    assert not _supported_store("БУРГ", lines)
    assert not _supported_item(Item(name="ЛЕНТА", sum=Decimal("30.00")), lines)
    assert _supported_total(
        Decimal("1678.38"), ["10000702652 CYMMA (РУБ.); 1678.38"], {Decimal("1678.38")}
    )
    assert not _supported_total(Decimal("439.99"), ["БЕКОН 439.99"], {Decimal("439.99")})
    chunks = _split_ocr_text("строка1\nстрока2\nстрока3\nстрока4", max_chars=13)
    assert all(len(chunk) <= 13 for chunk in chunks)
