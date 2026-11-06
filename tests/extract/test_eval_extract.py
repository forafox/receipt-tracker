import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from extract.eval_extract import main


def _record(sample_id: str, *, matching: bool = True) -> dict[str, Any]:
    return {
        "id": sample_id,
        "ocr": {
            "blocks": [
                {
                    "text": text,
                    "bbox": [[0, y * 20], [200, y * 20], [200, y * 20 + 12], [0, y * 20 + 12]],
                }
                for y, text in enumerate(
                    ["Лавка", "05.10.2026 18:20", "МОЛОКО 50.00", "ИТОГО 50.00"], start=1
                )
            ]
        },
        "expected": {
            "store": "Лавка" if matching else "Other shop",
            "datetime": "2026-10-05T18:20:00+03:00",
            "total": "50.00" if matching else None,
            "items": [{"name": "МОЛОКО", "quantity": None, "price": None, "sum": "50.00"}]
            if matching
            else [],
            "warnings": [],
        },
    }


def _write(path: Path, *records: dict[str, Any]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_module_execution_reports_metrics_and_original_mismatch_ids(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    _write(
        path,
        _record("matching"),
        _record(" z receipt ", matching=False),
        _record("a-receipt", matching=False),
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "extract.eval_extract",
            "--input",
            str(path),
            "--timezone",
            "Europe/Moscow",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.splitlines() == [
        "Receipts: 3",
        "store accuracy: 0.3333 (1/3)",
        "datetime accuracy: 1.0000 (3/3)",
        "total accuracy: 0.3333 (1/3)",
        "item precision: 0.3333",
        "item recall: 1.0000",
        "item F1: 0.5000",
        "Mismatches: 2",
        "  ' z receipt ': fields=store, total; missing_items=0; extra_items=1",
        "  'a-receipt': fields=store, total; missing_items=0; extra_items=1",
    ]


def test_perfect_match(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "sample.jsonl"
    _write(path, _record("receipt"))

    main(["--input", str(path), "--timezone", "Europe/Moscow"])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "store accuracy: 1.0000 (1/1)" in captured.out
    assert "datetime accuracy: 1.0000 (1/1)" in captured.out
    assert "total accuracy: 1.0000 (1/1)" in captured.out
    assert "item F1: 1.0000" in captured.out
    assert "Mismatches: 0" in captured.out


@pytest.mark.parametrize("timezone", ["Unknown/Timezone", "", "/absolute/path"])
def test_invalid_timezone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], timezone: str
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--input", str(tmp_path / "unused.jsonl"), "--timezone", timezone])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "invalid timezone" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("content", [None, b'{"id":', b"{}", b"\xff"])
def test_invalid_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes | None
) -> None:
    path = tmp_path / "invalid.jsonl"
    if content is not None:
        path.write_bytes(content)

    with pytest.raises(SystemExit) as caught:
        main(["--input", str(path), "--timezone", "Europe/Moscow"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert f"cannot load {path}" in captured.err
    if content in (b'{"id":', b"{}"):
        assert f"{path}:1:" in captured.err
    assert captured.out == ""


def test_empty_dataset_reports_undefined_accuracy_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        main(["--input", str(path), "--timezone", "Europe/Moscow"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert f"empty dataset: {path}" in captured.err
    assert captured.out.splitlines() == [
        "Receipts: 0",
        "store accuracy: N/A (0/0)",
        "datetime accuracy: N/A (0/0)",
        "total accuracy: N/A (0/0)",
        "item precision: 0.0000",
        "item recall: 0.0000",
        "item F1: 0.0000",
        "Mismatches: 0",
    ]


def test_evaluation_failure_is_reported_without_partial_metrics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid-coordinate.jsonl"
    record = _record("receipt")
    record["ocr"]["blocks"][0]["bbox"][0][0] = "NaN"
    _write(path, record)

    with pytest.raises(SystemExit) as caught:
        main(["--input", str(path), "--timezone", "Europe/Moscow"])

    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert f"cannot evaluate {path}" in captured.err
    assert captured.out == ""
