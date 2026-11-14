"""Load independently annotated OCR/reference pairs from JSONL."""

import json
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from common.contracts import Item, Receipt
from common.ocr import OcrResult


class SampleLoadError(ValueError):
    """An invalid evaluation record, with its source path and line number."""


class EvaluationSample(BaseModel):
    """An OCR result paired with an independently annotated reference.

    Attributes:
        id: Stable, nonblank sample identifier, preserved verbatim.
        ocr: OCR output in engine order.
        expected: Fully annotated receipt, including explicit null values.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    ocr: OcrResult
    expected: Receipt

    @field_validator("id")
    @classmethod
    def _nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sample ID must not be empty or whitespace-only")
        return value

    @field_validator("expected")
    @classmethod
    def _fully_annotated(cls, value: Receipt) -> Receipt:
        missing = Receipt.model_fields.keys() - value.model_fields_set
        if missing:
            raise ValueError(f"expected is missing annotations: {', '.join(sorted(missing))}")
        for index, item in enumerate(value.items):
            missing = Item.model_fields.keys() - item.model_fields_set
            if missing:
                raise ValueError(
                    f"expected.items[{index}] is missing annotations: {', '.join(sorted(missing))}"
                )
        return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def load_evaluation_samples(path: str | Path) -> list[EvaluationSample]:
    """Load UTF-8 JSONL records with ``id``, ``ocr`` and ``expected`` fields.

    Every Receipt and Item field must be explicitly annotated, including nullable
    values and warnings. OCR fields follow the existing OcrResult defaults. Values
    are validated by the shared contracts without normalization or inference; JSON
    decimal numbers are decoded as Decimal to preserve precision. Blank lines are
    invalid records. IDs are unique within the file and are never stripped.

    Args:
        path: JSONL file containing one independently annotated receipt per line.

    Returns:
        Validated samples in file order, or an empty list for an empty file.

    Raises:
        SampleLoadError: If a record is invalid or an ID is duplicated. The message
            includes the file path and one-based line number.
        OSError: If the file cannot be opened or read.
    """
    samples: list[EvaluationSample] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line, parse_float=Decimal, parse_constant=_reject_constant)
                sample = EvaluationSample.model_validate(record)
                if sample.id in seen:
                    raise ValueError(f"duplicate sample ID: {sample.id!r}")
            except ValueError as error:
                raise SampleLoadError(f"{path}:{line_number}: {error}") from error
            seen.add(sample.id)
            samples.append(sample)
    return samples
