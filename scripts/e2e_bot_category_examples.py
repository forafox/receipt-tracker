"""Exercise bot photo and /month handlers against the checked-in receipt examples.

OCR is replayed from the transcripts produced by HunyuanOCR; the real M2 parser,
M3 transformer, SQLite storage, bot handlers, and response formatting are exercised.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

from bot.handlers import handle_month, handle_photo
from bot.texts import RU
from categorize.service import CategoryService
from categorize.transformer import TransformerCategoryClassifier
from common.contracts import Receipt
from common.ocr import OcrResult, TextBlock
from extract import agent as extraction_agent
from pipeline import ReceiptPipeline
from storage import SqliteReceiptStorage


class TranscriptReplayRecognizer:
    """Return the saved Hunyuan transcript associated with each example image."""

    name = "hunyuan_transcript_replay"

    def __init__(self, transcripts: dict[str, str]) -> None:
        self._transcripts = transcripts

    def recognize_blocks(self, image: bytes) -> OcrResult:
        """Adapt the cached Hunyuan transcript to OCR blocks.

        Returns:
            OCR blocks in the transcript's reading order.
        """
        transcript = self._transcripts[hashlib.sha256(image).hexdigest()]
        lines = [line.strip() for line in transcript.splitlines() if line.strip()]
        width = 2000
        height = max(1, len(lines) * 20)
        blocks = [
            TextBlock(
                text=line,
                bbox=[
                    (0, index * 20),
                    (width, index * 20),
                    (width, index * 20 + 10),
                    (0, index * 20 + 10),
                ],
            )
            for index, line in enumerate(lines)
        ]
        return OcrResult(blocks=blocks, image_width=width, image_height=height)

    def recognize(self, image: bytes) -> str:
        """Return the cached transcript as plain text."""
        return "\n".join(block.text for block in self.recognize_blocks(image).blocks)


def _message(sent_at: datetime) -> MagicMock:
    message = MagicMock()
    message.from_user.id = 707070
    message.chat.id = 707070
    message.date = sent_at
    message.photo = [MagicMock()]
    message.answer = AsyncMock()
    return message


async def _run(model_path: Path, examples_path: Path, report_path: Path | None) -> dict[str, Any]:
    os.environ.pop("KILO_API_KEY", None)
    os.environ.pop("KILO_TOKEN", None)
    extraction_agent._local_model_available = False

    example_files = [
        path
        for path in sorted(examples_path.glob("*.json"))
        if path.with_suffix(".jpg").exists() and path.with_suffix(".txt").exists()
    ]
    transcripts: dict[str, str] = {}
    examples: list[tuple[str, bytes, Receipt]] = []
    for path in example_files:
        image = path.with_suffix(".jpg").read_bytes()
        transcript = path.with_suffix(".txt").read_text(encoding="utf-8")
        transcripts[hashlib.sha256(image).hexdigest()] = transcript
        expected = json.loads(path.read_text(encoding="utf-8"))
        examples.append((path.stem, image, Receipt.model_validate(expected["receipt"])))

    classifier = TransformerCategoryClassifier.load(model_path)
    category_service = CategoryService(classifier)
    with tempfile.TemporaryDirectory(prefix="receipt-category-e2e-") as directory:
        storage = SqliteReceiptStorage(Path(directory) / "receipts.sqlite")
        pipeline = ReceiptPipeline(
            TranscriptReplayRecognizer(transcripts),
            storage,
            ZoneInfo("Europe/Moscow"),
            category_service=category_service,
        )

        photo_responses = []
        for example_id, image, expected in examples:
            sent_at = expected.datetime or datetime(2024, 12, 1, tzinfo=UTC)
            message = _message(sent_at)

            async def download(_file: object, destination: BytesIO, content: bytes = image) -> None:
                destination.write(content)

            bot = MagicMock()
            bot.download = AsyncMock(side_effect=download)
            bot.send_chat_action = AsyncMock()
            await handle_photo(message, bot, pipeline, RU)
            message.answer.assert_awaited_once()
            response = message.answer.await_args.args[0]
            if response == RU.error or response == RU.unreadable:
                raise AssertionError(f"{example_id}: bot returned {response!r}")
            saved_receipt = next(
                receipt
                for receipt in storage.list_recent(707070, len(examples))
                if receipt.datetime == expected.datetime and receipt.total == expected.total
            )
            photo_responses.append(
                {
                    "id": example_id,
                    "response": response,
                    "receipt": saved_receipt.model_dump(mode="json"),
                }
            )

        receipts = storage.list_recent(707070, len(examples))
        if len(receipts) != len(examples):
            raise AssertionError(
                f"saved {len(receipts)} of {len(examples)} submitted receipt images"
            )

        months: dict[str, datetime] = {}
        for receipt in receipts:
            moment = receipt.datetime or datetime(2024, 12, 1, tzinfo=UTC)
            local_month = moment.astimezone(ZoneInfo("Europe/Moscow")).strftime("%Y-%m")
            months.setdefault(local_month, moment)

        month_responses = []
        for month_key, moment in sorted(months.items()):
            message = _message(moment)
            await handle_month(message, pipeline, RU)
            message.answer.assert_awaited_once()
            response = message.answer.await_args.args[0]
            summary = await pipeline.monthly_summary(707070, moment)
            local_start = moment.astimezone(ZoneInfo("Europe/Moscow")).replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
            local_end = (local_start + timedelta(days=32)).replace(day=1)
            month_receipts = storage.list_between(707070, local_start, local_end)
            expected_count = len(month_receipts)
            expected_total = sum(
                (receipt.total for receipt in month_receipts if receipt.total is not None),
                Decimal(0),
            )
            expected_categories: defaultdict[str, Decimal] = defaultdict(Decimal)
            for receipt in month_receipts:
                for item in receipt.items:
                    if item.categories:
                        share = (item.sum / len(item.categories)).quantize(
                            Decimal("0.01"), rounding=ROUND_DOWN
                        )
                        remainder = item.sum - share * len(item.categories)
                        for index, category in enumerate(item.categories):
                            expected_categories[category] += share + (
                                remainder if index == 0 else Decimal(0)
                            )
            if summary.receipt_count != expected_count or summary.total != expected_total:
                raise AssertionError(f"{month_key}: monthly summary does not match stored receipts")
            if summary.category_totals != (dict(expected_categories) or None):
                raise AssertionError(
                    f"{month_key}: category aggregation does not match stored items"
                )
            if f"Чеков: {expected_count}" not in response:
                raise AssertionError(f"{month_key}: receipt count missing or wrong: {response!r}")
            if f"{expected_total:.2f}" not in response:
                raise AssertionError(f"{month_key}: monthly total missing or wrong: {response!r}")
            expected_category_lines = {
                f"- {name}: {amount:.2f}" for name, amount in expected_categories.items()
            }
            actual_category_lines = {
                line for line in response.splitlines() if line.startswith("- ")
            }
            if actual_category_lines != expected_category_lines:
                raise AssertionError(
                    f"{month_key}: category totals mismatch; expected "
                    f"{sorted(expected_category_lines)}, got {sorted(actual_category_lines)}"
                )
            month_responses.append({"month": month_key, "response": response})

        result = {
            "e2e": (
                "photo handler -> Hunyuan transcript replay -> M2 -> M3 transformer -> SQLite "
                "-> /month handler"
            ),
            "receipts_submitted": len(examples),
            "receipts_saved": len(receipts),
            "category_items": sum(len(receipt.items) for receipt in receipts),
            "photo_responses": photo_responses,
            "month_responses": month_responses,
        }
        if report_path is not None:
            report_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return result


def main() -> None:
    """Run the complete offline bot flow on ten example receipts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("categorize/models/category_transformer")
    )
    parser.add_argument("--examples", type=Path, default=Path("docs/examples"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = asyncio.run(_run(args.model_path, args.examples, args.report))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
