# Running receipt-tracker locally

This is the end-to-end setup guide for running the whole project on one machine:
the Telegram bot (OCR, receipt parsing, SQLite storage) and the M3 category
classifier API.

> Verified on Ubuntu 22.04 with Python 3.14.3, uv 0.9.30 and Tesseract 4.1.1.
> CI targets Python 3.12. The project requires Python 3.12 or newer.

## 1. What runs where

| Process | Command | Responsibility |
| --- | --- | --- |
| Bot (M4) | `uv run python -m bot` | Telegram polling, M1 OCR, M2 extraction, SQLite storage |
| Categorizer (M3) | `uv run --with uvicorn uvicorn categorize.api:app ...` | FastAPI `/classify`, `/classify/items`, `/classify/receipt`, `/train` |

At the time of writing, the bot pipeline does **not** call the M3 API. Running
both processes gives you a working bot and a working classifier service, but
the bot's replies do not contain categories yet. M3 is used over HTTP by other
clients (or manually through Swagger UI).

The trained M3 model is committed to the repository at
`categorize/models/category_transformer/`, so no training is required for a
normal run.

## 2. Prerequisites

- 64-bit Linux or macOS. On Windows, use WSL2.
- About 10 GB of free disk space. PyTorch makes the virtual environment large
  (about 6.2 GB in this checkout), and uv also keeps a download cache.
- 4 GB of RAM (more is better for M3 inference).
- Network access for the first `uv sync` and for Telegram.
- A Telegram bot token.

Check the basics:

```bash
uv --version
tesseract --version
tesseract --list-langs
df -h .
```

## 3. Install uv

Linux/macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# Restart the shell, or load the environment in the current one:
source "$HOME/.local/bin/env"
uv --version
```

On macOS you can also use `brew install uv`.

## 4. Clone and install dependencies

```bash
git clone https://github.com/forafox/receipt-tracker.git
cd receipt-tracker
```

Install Python and all dependency groups:

```bash
# Optional, but matches CI:
uv python install 3.12

uv sync --locked --group dev --group ocr
# To force the CI interpreter, add: --python 3.12
```

What the groups mean:

- `dev` — pytest, ruff, mypy, and the rest of the development tooling.
- `ocr` — `pytesseract`, `easyocr`, OpenCV and their dependencies. The bot
  cannot recognize receipts without this group.

`make setup` is only the dev part (`uv sync --locked --group dev`). If you run
it after the full sync above, it prunes the `ocr` group. Use
`uv sync --locked --group dev --group ocr` for a full setup.

The first sync downloads PyTorch and the other heavy dependencies, so expect a
few minutes and several gigabytes. Later syncs are fast.

Verify the environment:

```bash
uv run python --version
uv run python -c "import aiogram, cv2, fastapi, pytesseract, torch, transformers; print('imports ok')"
```

## 5. Install Tesseract with Russian support

The default OCR engine is Tesseract and the default language string is
`rus+eng`. Tesseract without the Russian data fails with
`Failed loading language 'rus'`.

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
tesseract --list-langs
```

Expected output includes:

```text
eng
osd
rus
```

macOS:

```bash
brew install tesseract tesseract-lang
```

If you cannot install the system language pack, keep `rus.traineddata` and
`eng.traineddata` in a directory of your choice and point Tesseract at it:

```bash
mkdir -p "$HOME/.local/share/tessdata"
curl -fL -o "$HOME/.local/share/tessdata/rus.traineddata" \
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/rus.traineddata
curl -fL -o "$HOME/.local/share/tessdata/eng.traineddata" \
  https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/eng.traineddata
export TESSDATA_PREFIX="$HOME/.local/share/tessdata"
tesseract --list-langs
```

Add the `export` line to `~/.bashrc` or `~/.zshrc` to keep it after a restart.

### Adaptive page segmentation

The bot runs Tesseract with PSM 3, 4 and 6 and keeps the parsed candidate with
the best date/total/items/arithmetic score. This is enabled in `.env`:

```dotenv
OCR_ADAPTIVE=true
OCR_ADAPTIVE_PSM_MODES=[3,4,6]
```

Set `OCR_ADAPTIVE=false` to do a single pass. `OCR_TESSERACT_CONFIG` may still
contain extra flags; an existing `--psm` is replaced by each candidate mode.
Adaptive mode costs up to three Tesseract runs per photo but handles different
receipt layouts without changing configuration.

### Alternative: EasyOCR (no Tesseract binary)

EasyOCR is installed by the `ocr` group and does not need the system Tesseract
binary. Make it the default for your local checkout in `.env`:

```dotenv
OCR_ENGINE=easyocr
OCR_EASYOCR_LANGUAGES=["ru","en"]
OCR_GPU=false
```

Set `OCR_GPU=true` to use CUDA; check it first with
`uv run python -c "import torch; print(torch.cuda.is_available())"`.

The first recognition downloads the detection and recognition weights to
`~/.EasyOCR/model` (roughly 100–200 MB); later runs reuse the cache. Verify the
settings:

```bash
uv run python -c "from ocr import OcrSettings; s = OcrSettings(); print(s.engine, s.easyocr_languages, s.gpu)"
# easyocr ['ru', 'en'] False
```

Then run the OCR smoke test from step 7. Keep Tesseract installed only if you
also run the Tesseract slow tests; the bot itself no longer uses it.

Precedence: a shell variable (`OCR_ENGINE=easyocr uv run python -m bot`)
overrides `.env`, and an explicit `OcrSettings(engine=...)` argument overrides
both. If the engine still reports `tesseract`, check `echo $OCR_ENGINE` and make
sure you run from the repository root.

EasyOCR is not automatically more accurate than Tesseract. In a clean synthetic
test it read `12:30` as `12.30`, so the receipt datetime was not parsed; test on
real receipt photos before relying on it.

If you want EasyOCR to be the fallback even when neither `.env` nor the
environment selects an engine, change `engine: str = "easyocr"` in
`ocr/core/config.py`, update the default-engine assertion in
`tests/ocr/test_service.py`, and run `make check`. This is a code change; the
`.env` switch above is enough for normal use.

## 6. Create `.env`

```bash
cp .env.example .env
```

Open `.env` and set at least the token:

```dotenv
BOT_TOKEN=123456789:AA...your-token-from-BotFather...
BOT_TIMEZONE=Europe/Moscow
BOT_DATABASE_PATH=receipts.sqlite

OCR_ENGINE=tesseract
OCR_TESSERACT_LANGUAGES=rus+eng
OCR_TESSERACT_CONFIG=
OCR_ADAPTIVE=true
OCR_ADAPTIVE_PSM_MODES=[3,4,6]
```

Get the token from [@BotFather](https://t.me/BotFather): send `/newbot`, choose
a name and a username ending in `bot`, then copy the token.

`.env` is listed in `.gitignore`. Never commit it or paste the real token into
an issue, chat or screenshot.

Check that the bot settings are readable without printing the secret:

```bash
uv run python -c "from bot.config import BotSettings; s = BotSettings(); print('token:', bool(s.token.get_secret_value()), 'tz:', s.timezone, 'db:', s.database_path)"
```

Expected:

```text
token: True tz: Europe/Moscow db: receipts.sqlite
```

Both the bot and OCR read the same `.env`; each module ignores variables that
belong to the other one.

## 7. Verify the installation before starting

Run the project checks:

```bash
make check
```

This runs formatting, Ruff, mypy and pytest. In this checkout the result is:

```text
448 passed, 1 skipped, 1 deselected
```

Slow tests that need real OCR engines or network are excluded by default. Run
them separately if needed:

```bash
make test-slow
```

### OCR smoke test

Use any photo of a receipt:

```bash
uv run python - <<'PY'
from pathlib import Path

from ocr import build_default_ocr_service

text = build_default_ocr_service().recognize(Path("receipt.jpg").read_bytes())
print(text)
PY
```

Expected: Russian text from the receipt. If you get
`ImageQualityError`, retake the photo with better light/focus, or temporarily
set `OCR_QUALITY_CHECK_ENABLED=false` in `.env` to confirm the engine works.

### Full pipeline without Telegram

This exercises M1 OCR, M2 extraction and SQLite storage without a bot token:

```bash
uv run python - <<'PY'
import asyncio
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from ocr import build_default_ocr_service
from pipeline import ReceiptPipeline
from storage import SqliteReceiptStorage

async def main() -> None:
    pipeline = ReceiptPipeline(
        build_default_ocr_service(),
        SqliteReceiptStorage(Path(".smoke.sqlite")),
        ZoneInfo("Europe/Moscow"),
    )
    receipt = await pipeline.process_photo(
        1,
        Path("receipt.jpg").read_bytes(),
        datetime.now(ZoneInfo("Europe/Moscow")),
    )
    print(receipt.model_dump_json(indent=2))

asyncio.run(main())
PY
rm -f .smoke.sqlite
```

Expected: JSON with `store`, `datetime`, `total`, `items` and `warnings`.

## 8. Start the M3 categorizer API

Open a second terminal in the repository root:

```bash
uv run --with uvicorn uvicorn categorize.api:app --host 127.0.0.1 --port 8000
```

The model is loaded from `categorize/models/category_transformer/`. Wait for:

```text
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

> **Why this command?** `categorize/api.py` only defines the ASGI `app`; it does
> not start a server. The project does not list an ASGI server in its
> dependencies, so `uv run fastapi ...` prints “To use the fastapi command,
> please install fastapi[standard]” and exits. `uv run --with uvicorn ...`
> supplies uvicorn inside the project environment for this one command without
> changing `pyproject.toml`. Without `--with`, `uv run uvicorn` can fall back to
> a globally installed uvicorn running a different Python and fail to import the
> project. Add `--reload` for auto-reload during development.

Verify the API:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8000/docs
# 200
```

Classify one item:

```bash
curl -s -X POST http://127.0.0.1:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"name":"молоко 3.2%","quantity":"1","total":"89.90"}'
```

Expected shape:

```json
{
  "name": "молоко 3.2%",
  "quantity": "1",
  "total": "89.90",
  "categories": [{"category": "food", "confidence": 0.9896}]
}
```

Classify a receipt:

```bash
curl -s -X POST http://127.0.0.1:8000/classify/receipt \
  -H 'Content-Type: application/json' \
  -d '{"receipt_id":"r-1","purchased_at":"2026-11-24T12:00:00+03:00","items":[{"name":"парацетамол","quantity":"1","total":"50"}]}'
```

Swagger UI with all endpoints and request schemas:

```text
http://127.0.0.1:8000/docs
```

Notes:

- If the model directory is missing, the API trains it automatically from
  `categorize/data/training.csv`. That downloads `cointegrated/rubert-tiny` and
  takes much longer than a normal startup.
- `POST /train` retrains and overwrites the model. Do not call it casually.
- Do not run the API and `make test-slow` at the same time if memory is tight.

## 9. Start the Telegram bot

In the first terminal (repository root, where `.env` lives):

```bash
uv run python -m bot
```

Expected: aiogram logs the polling start. Open your bot in Telegram, send
`/start`, then send a receipt photo.

Available commands:

- `/start` — greeting;
- `/help` — help;
- `/month` — current month's total and receipt count;
- `/history` — last five receipts;
- `/undo` — delete the most recently saved receipt;
- a photo or an image file — recognize, parse and save a receipt.

Tips:

- Send the receipt as a **file** to avoid Telegram compression; OCR is more
  accurate on the original image.
- Files larger than 20 MB cannot be downloaded through the Bot API.
- The database is created at `BOT_DATABASE_PATH` (`receipts.sqlite` by default)
  on the first saved receipt.
- Stop the bot with `Ctrl+C`. Data stays in SQLite. Pending duplicate
  confirmations are kept in memory and are lost on restart, so resend the photo.

## 10. Stop everything

`Ctrl+C` in both terminals. To reset local data:

```bash
rm -f receipts.sqlite
```

The bot must be stopped before deleting the database.

## 11. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `uv: command not found` | Install uv and restart the shell (step 3). |
| `requires-python >=3.12` / no interpreter | `uv python install 3.12`, then rerun the sync with `--python 3.12`. |
| `ModuleNotFoundError: pytesseract` or OCR unavailable | The `ocr` group is missing: `uv sync --locked --group dev --group ocr`. |
| `TesseractNotFoundError` / `tesseract binary not found` | Install `tesseract-ocr` (step 5). |
| `Failed loading language 'rus'` | Install `tesseract-ocr-rus` or set `TESSDATA_PREFIX` to a directory containing `rus.traineddata` and `eng.traineddata`. Check `tesseract --list-langs`. |
| `ImageQualityError` | Photo is blurry, flat or has glare. Retake it, send it as a file, or lower the quality gate in `.env`. |
| Date or total is “not recognized” on a column-style receipt | Adaptive PSM is enabled by default. If it was disabled, set `OCR_ADAPTIVE=true`. To test one mode, set `OCR_ADAPTIVE=false` and `OCR_TESSERACT_CONFIG=--psm 4` (or `--psm 6`). |
| `pydantic ... token Field required` | `.env` is missing, empty, or you started the bot from another directory. Run `cp .env.example .env`, set `BOT_TOKEN`, and run from the repository root. |
| `Unauthorized` from Telegram | Token is wrong or revoked. Create a new one with BotFather. |
| Telegram connection timeouts | Network/firewall blocks `api.telegram.org`. Check proxy/VPN settings. |
| `uv run fastapi ...` asks for `fastapi[standard]` | Known dependency gap. Use the `uv run --with uvicorn ...` command from step 8. |
| API returns 503 | The model service did not initialize. Check the API terminal for loading/training errors. |
| `make check` is slow the first time | mypy and pytest import heavy modules. Later runs are faster. |
| Out of disk space during sync | Free at least ~10 GB; the virtual environment is about 6.2 GB and uv keeps a download cache. |
| Bot replies “recognition temporarily unavailable” | Tesseract or its language data is broken. Check the bot log; the error is not shown to the user. |

## 12. Docker (optional, not a full-project runner)

The provided `Dockerfile` only serves M3. It installs the main dependency group
and runs `uv run fastapi run ...`; because `fastapi-cli` is not installed, that
container command exits with the `fastapi[standard]` hint. It also does not
include Tesseract or the bot.

Use the native setup above for the full project. If you only need an M3
container, update the Dockerfile to install `uvicorn` (or `fastapi[standard]`)
and change the `CMD` accordingly, then build and run it. That path is not
covered by this guide.

## 13. What was verified in this checkout

| Step | Result |
| --- | --- |
| `make check` (format, Ruff, mypy, pytest) | Passed: 470 passed, 1 skipped, 1 deselected |
| OCR with Tesseract 4.1.1 + `rus`/`eng` data | Clean test receipt recognized and parsed into store, date, total and two items with no warnings |
| Adaptive PSM 3/4/6 selection | Picked the best candidate on Лента, Автокафе, Пятёрочка and three public Russian sample receipts |
| Full pipeline + SQLite | Receipt saved, monthly summary and history returned, undo worked |
| M3 API startup and `/docs` | HTTP 200 |
| `/classify`, `/classify/items`, `/classify/receipt` | Returned expected `food`, `personal_care`, `cafe`, `medicine` categories |
| `python -m categorize eval` | `accuracy=0.955 micro_f1=0.955 examples=110` |
| Bot configuration and startup path | Token/config loading verified; live polling was not run because `api.telegram.org` is unreachable from the development sandbox |

The remaining step that genuinely needs your machine and your account is a live
Telegram session: put a real token in `.env`, run `uv run python -m bot`, and
send `/start` plus a receipt photo.
