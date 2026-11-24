"""Run HunyuanOCR and deterministic M2 on ten annotated Russian receipts."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from common.ocr import OcrResult, TextBlock
from extract.layout import reconstruct_layout
from extract.parser import parse_ocr_result
from ocr.engines.hunyuan_vision import HunyuanVisionEngine

TIMEZONE = ZoneInfo("Europe/Moscow")
ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=cdek-ocr/receipt-ocr-ru"
    "&config=default&split=test&offset=0&length=30"
)


def main() -> None:
    """Evaluate the first ten annotated receipt images; resume saved runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reparse",
        action="store_true",
        help="re-run M2 on saved transcripts without calling Hunyuan again",
    )
    parser.add_argument("--output", type=Path, default=Path("/tmp/hunyuan-e2e10.json"))
    parser.add_argument("--image-cache", type=Path, default=Path("/tmp/receipt-e2e10-images"))
    parser.add_argument("--seed-ocr-json", type=Path)
    parser.add_argument(
        "--url",
        default=os.getenv("OCR_HUNYUAN_OCR_URL", "http://localhost:8231/v1/chat/completions"),
    )
    parser.add_argument("--model", default=os.getenv("OCR_HUNYUAN_OCR_MODEL", "HunyuanOCR"))
    args = parser.parse_args()

    with urllib.request.urlopen(ROWS_URL, timeout=45) as response:
        rows = json.loads(response.read())["rows"]
    rows = [
        wrapped["row"]
        for wrapped in rows
        if wrapped["row"].get("date_text")
        and wrapped["row"].get("total_text")
        and wrapped["row"]["total_text"].replace(".", "", 1).isdigit()
    ][: args.limit]
    args.image_cache.mkdir(parents=True, exist_ok=True)

    previous = {}
    if args.resume and args.output.exists():
        previous = {row["id"]: row for row in json.loads(args.output.read_text("utf-8"))}
    engine = HunyuanVisionEngine(model=args.model, url=args.url)
    results = []

    for index, target in enumerate(rows, start=1):
        image_id = target["image_id"]
        cached = previous.get(image_id)
        if cached is not None and not args.reparse:
            result = cached
        else:
            image_path = args.image_cache / f"{image_id}.jpg"
            if image_path.exists():
                image = image_path.read_bytes()
            else:
                with urllib.request.urlopen(target["image"]["src"], timeout=60) as response:
                    image = response.read()
                image_path.write_bytes(image)

            started = time.perf_counter()
            if cached is not None:
                transcript = str(cached["transcript"])
                elapsed = float(cached["seconds"])
                ocr_result = _synthetic_blocks(image, transcript)
            else:
                seeded = _seed_transcript(args.seed_ocr_json, image_id)
                if seeded is None:
                    ocr_result = engine.recognize_blocks(image)
                    elapsed = time.perf_counter() - started
                    transcript = "\n".join(line.text for line in reconstruct_layout(ocr_result))
                else:
                    transcript, elapsed = seeded
                    ocr_result = _synthetic_blocks(image, transcript)

            receipt = parse_ocr_result(ocr_result, timezone=TIMEZONE)
            target_date = _parse_date(target["date_text"])
            target_total = Decimal(target["total_text"])
            expected_sums = {Decimal(value) for value in target["item_sums"]}
            store_target = _normalize_store(target["seller_text"])
            parsed_store = _normalize_store(receipt.store or "")
            result = {
                "id": image_id,
                "seconds": round(elapsed, 2),
                "ocr_blocks": len(ocr_result.blocks),
                "transcript": transcript,
                "target": {
                    "store": target["seller_text"],
                    "date": target["date_text"],
                    "total": str(target_total),
                    "items": target["item_texts"],
                    "item_sums": target["item_sums"],
                },
                "m2": receipt.model_dump(mode="json"),
                "match": {
                    "store": bool(store_target and store_target in parsed_store),
                    "date": bool(receipt.datetime and receipt.datetime.date() == target_date),
                    "total": receipt.total == target_total,
                    "item_sum_hits": sum(item.sum in expected_sums for item in receipt.items),
                    "item_rows": len(receipt.items),
                },
            }
        results.append(result)
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "index": index,
                    "id": result["id"],
                    "seconds": result["seconds"],
                    "match": result["match"],
                    "m2": result["m2"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    print(json.dumps(_summary(results), ensure_ascii=False, indent=2), flush=True)


def _seed_transcript(path: Path | None, image_id: str) -> tuple[str, float] | None:
    if path is None or image_id != "IMG_2040_enh" or not path.exists():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return saved["content"], float(saved["seconds"])


def _synthetic_blocks(image: bytes, transcript: str) -> OcrResult:
    decoded = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("cached receipt image could not be decoded")
    height, width = decoded.shape[:2]
    raw_lines = transcript.splitlines()
    rows = [(index, line.strip()) for index, line in enumerate(raw_lines) if line.strip()]
    row_height = max(1, height // max(2 * len(raw_lines), 1))
    blocks = []
    for raw_index, line in rows:
        y_min = min(height - 1, round(raw_index * height / max(len(raw_lines), 1)))
        y_max = min(height, max(y_min + 1, y_min + row_height))
        blocks.append(
            TextBlock(
                text=line,
                bbox=[(0, y_min), (width, y_min), (width, y_max), (0, y_max)],
                confidence=None,
            )
        )
    return OcrResult(blocks=blocks, image_width=width, image_height=height)


def _parse_date(value: str) -> date:
    token = value.split()[0]
    for date_format in ("%d.%m.%y", "%d.%m.%Y"):
        try:
            return datetime.strptime(token, date_format).date()
        except ValueError:
            continue
    raise ValueError(f"unsupported annotated date: {value}")


def _normalize_store(value: str) -> str:
    normalized = " ".join(value.casefold().replace('"', "").split())
    for legal_form in ("ооо", "ип", "пао", "ао", "зао", "оао"):
        normalized = normalized.replace(legal_form, " ")
    return " ".join(normalized.split())


def _summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    matches = [row["match"] for row in rows]
    metrics = [match for match in matches if isinstance(match, dict)]
    return {
        "receipts": len(rows),
        "fields": {
            name: sum(bool(match[name]) for match in metrics) for name in ("store", "date", "total")
        },
        "item_sum_hits": sum(int(match["item_sum_hits"]) for match in metrics),
        "item_rows": sum(int(match["item_rows"]) for match in metrics),
        "mean_seconds": round(sum(float(row["seconds"]) for row in rows) / len(rows), 2)
        if rows
        else None,
        "warnings": sum(bool(row["m2"]["warnings"]) for row in rows),
    }


if __name__ == "__main__":
    main()
