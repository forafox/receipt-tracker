import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from common.contracts import Receipt
from common.ocr import OcrResult
from extract.samples import SampleLoadError, load_evaluation_samples


def _record(sample_id: str = "receipt-001") -> dict[str, Any]:
    return {
        "id": sample_id,
        "ocr": {"blocks": [], "image_width": None, "image_height": None},
        "expected": {
            "store": None,
            "datetime": None,
            "total": None,
            "items": [],
            "warnings": [],
        },
    }


def _write(path: Path, *records: dict[str, Any]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_load_preserves_order_ids_and_explicit_nulls(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    _write(path, _record(" z "), _record("a"))

    samples = load_evaluation_samples(str(path))

    assert [sample.id for sample in samples] == [" z ", "a"]
    for sample in samples:
        assert sample.ocr == OcrResult()
        assert sample.expected == Receipt()
        assert sample.expected.model_fields_set == set(Receipt.model_fields)


def test_load_validates_contracts_without_inferring_or_normalizing(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    record = _record()
    record["ocr"] = {
        "blocks": [{"text": " ИТОГО 99.99 ", "bbox": [[0, 0], [20, 0], [20, 10], [0, 10]]}],
        "image_width": 30,
        "image_height": 40,
    }
    record["expected"] = {
        "store": " Лавка ",
        "datetime": "2026-10-05T18:20:00+03:00",
        "total": "0.12345678901234567890123456789",
        "items": [{"name": " Milk ", "quantity": None, "price": None, "sum": "-1.00"}],
        "warnings": ["independent_annotation"],
    }
    # Exercise a JSON number beyond float precision.
    record["expected"]["total"] = Decimal("0.12345678901234567890123456789")

    path.write_text(
        json.dumps(record, ensure_ascii=False, default=str).replace(
            '"total": "0.12345678901234567890123456789"',
            '"total": 0.12345678901234567890123456789',
        )
        + "\n",
        encoding="utf-8",
    )
    before = path.read_bytes()

    sample = load_evaluation_samples(path)[0]

    assert sample.ocr.blocks[0].text == " ИТОГО 99.99 "
    assert sample.ocr.blocks[0].bbox == [(0, 0), (20, 0), (20, 10), (0, 10)]
    assert (sample.ocr.image_width, sample.ocr.image_height) == (30, 40)
    assert sample.expected.store == " Лавка "
    assert sample.expected.datetime == datetime.fromisoformat("2026-10-05T18:20:00+03:00")
    assert sample.expected.total == Decimal("0.12345678901234567890123456789")
    assert sample.expected.items[0].model_dump() == {
        "name": " Milk ",
        "quantity": None,
        "price": None,
        "sum": Decimal("-1.00"),
    }
    assert sample.expected.warnings == ["independent_annotation"]
    assert path.read_bytes() == before


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    assert load_evaluation_samples(path) == []


@pytest.mark.parametrize("line", ['{"id":', "\n", "null", "[]", '{"id": NaN}'])
def test_invalid_records_report_path_and_physical_line(tmp_path: Path, line: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(_record()) + "\n" + line, encoding="utf-8")

    with pytest.raises(SampleLoadError, match=rf"{path}:2:"):
        load_evaluation_samples(path)


@pytest.mark.parametrize("field", list(Receipt.model_fields))
def test_missing_receipt_annotations_are_rejected(tmp_path: Path, field: str) -> None:
    record = _record()
    del record["expected"][field]
    path = tmp_path / "missing.jsonl"
    _write(path, record)

    with pytest.raises(SampleLoadError, match=rf"{path}:1:") as caught:
        load_evaluation_samples(path)
    assert field in str(caught.value)


@pytest.mark.parametrize("field", ["name", "quantity", "price", "sum"])
def test_missing_item_annotations_are_rejected(tmp_path: Path, field: str) -> None:
    record = _record()
    item = {"name": "Milk", "quantity": None, "price": None, "sum": "1"}
    del item[field]
    record["expected"]["items"] = [item]
    path = tmp_path / "missing-item.jsonl"
    _write(path, record)

    with pytest.raises(SampleLoadError) as caught:
        load_evaluation_samples(path)
    assert f"{path}:1:" in str(caught.value)
    assert field in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ocr", {"image_width": 0}),
        ("ocr", {"blocks": [{"text": "Milk", "bbox": []}]}),
        ("expected", {**_record()["expected"], "datetime": "2026-10-05T18:20:00"}),
        ("expected", {**_record()["expected"], "total": "not money"}),
        ("expected", {**_record()["expected"], "items": None}),
        ("extra", True),
        ("id", ""),
        ("id", " \t"),
        ("id", None),
        ("id", 1),
    ],
)
def test_invalid_contracts_and_ids(tmp_path: Path, field: str, value: Any) -> None:
    record = {**_record(), field: value}
    path = tmp_path / "invalid.jsonl"
    _write(path, record)

    with pytest.raises(SampleLoadError, match=rf"{path}:1:"):
        load_evaluation_samples(path)


@pytest.mark.parametrize("field", ["id", "ocr", "expected"])
def test_missing_envelope_fields(tmp_path: Path, field: str) -> None:
    record = _record()
    del record[field]
    path = tmp_path / "missing.jsonl"
    _write(path, record)

    with pytest.raises(SampleLoadError, match=rf"{path}:1:"):
        load_evaluation_samples(path)


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    _write(path, _record(), _record("other"), _record())

    with pytest.raises(SampleLoadError, match=rf"{path}:3: duplicate sample ID: 'receipt-001'"):
        load_evaluation_samples(path)
