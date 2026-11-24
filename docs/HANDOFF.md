# HANDOFF

## 1. Что это за проект

`receipt-tracker` — учебный трекер расходов по фотографиям чеков. Пользователь
отправляет фото Telegram-боту, система распознаёт чек, разбирает его в структуру,
классифицирует товары и сохраняет расходы. Проект делится на четыре модуля:

| Модуль | Зона ответственности | Основные файлы |
|---|---|---|
| M1 OCR | фото → текст с координатами | `ocr/` |
| M2 Extract | OCR-блоки → `Receipt` с товарами | `extract/` |
| M3 Categorize | товары → категории | `categorize/` |
| M4 Bot/Pipeline | Telegram, оркестрация, SQLite | `bot/`, `pipeline/`, `storage/` |

Общие контракты лежат в `common/`. Модули не должны импортировать внутренности
друг друга: только `common/`.

## 2. Текущее состояние

- Ветка: `uvusibuneka`; Python 3.12; пакетный менеджер `uv`.
- Проверки: `ruff` и `mypy` проходят, `pytest tests -q` — **439 passed, 7 skipped**.
- M3 по умолчанию использует `categorize/models/category_transformer` (m3e).
- M3-эксперимент с BPE-dropout и OCR-шумом (m3f) лучше на шумных строках, но
  хуже по overall exact, поэтому **не является дефолтным** и в поставку не входит.
- Основной датасет `categorize/data/training_final.csv` генерируется и в git не
  входит; после добавления шума в нём 45 768 строк.
- E2E на реальных фото чеков сейчас **не проходит** по полям M2 (см. раздел 5).

## 3. Структура и вспомогательные файлы

### Корень

| Файл | Назначение |
|---|---|
| `README.md` | обзор проекта, модулей, запуск бота |
| `CONTRIBUTING.md` | правила веток, кода, тестов и коммитов |
| `Makefile` | `setup`, `format`, `lint`, `typecheck`, `test`, `check` |
| `Dockerfile` | production-образ FastAPI-сервиса M3 |
| `.env.example` | переменные бота и OCR |
| `pyproject.toml` | зависимости, ruff, mypy, pytest |
| `uv.lock` | lock-файл зависимостей |
| `AGENTS.md` | локальные инструкции coding-агентам, в git не входит |
| `server-access.md` | локальные заметки о сервере, в git не входит |

### `common/`

| Файл | Назначение |
|---|---|
| `contracts.py` | `Item`, `Receipt` — контракт M2 → M3/M4 |
| `models.py` | `CategoryScore`, `CategoryPrediction`, `ClassifiedItem`, `ClassifiedReceipt` |
| `ocr.py` | `TextBlock`, `OcrResult`, протокол `OcrEngine` |

### `ocr/`

| Файл / папка | Назначение |
|---|---|
| `service.py` | `OcrService`: лениво выбирает и создаёт движок |
| `core/config.py` | настройки `OCR_*` |
| `core/errors.py` | ошибки OCR |
| `core/image.py` | декодирование, лимиты 20 МиБ / 25 млн пикселей |
| `core/ordering.py` | сортировка блоков и сборка текста |
| `core/geometry.py`, `core/metrics.py` | геометрия и CER/WER/IoU |
| `engines/tesseract.py` | Tesseract через `pytesseract` |
| `engines/easyocr.py` | EasyOCR |
| `engines/factory.py` | выбор движка по конфигу |
| `datasets/` | загрузчики CORU и русского detection/recognition |
| `tools/preview.py`, `tools/eval_ocr.py` | превью боксов и оценка OCR |
| `README.md`, `diagram.png` | описание M1 и схема |

### `extract/` (M2)

| Файл | Назначение |
|---|---|
| `parser.py` | `parse_receipt` / `parse_ocr_result` |
| `layout.py` | восстановление строк из OCR-боксов |
| `layout_items.py` | извлечение товаров из layout |
| `item_region.py` | поиск области товаров, метаданные |
| `items.py` | разбор простой строки «название + сумма» |
| `validation.py` | проверка арифметики и предупреждения |
| `evaluation.py`, `eval_extract.py` | оценка M2 на размеченных примерах |
| `samples.py` | примеры/фикстуры |

### `categorize/` (M3)

| Файл | Назначение |
|---|---|
| `api.py` | FastAPI: `/classify`, `/classify/items`, `/classify/receipt`, `/train` |
| `service.py` | сервисный слой M3 |
| `transformer.py` | загрузка модели, `denoise`, sigmoid, пороги |
| `text.py` | `denoise`, `add_noise`, профили `ocr/layout/receipt/words/severe` |
| `augment.py` | синтетические пары/тройки, шумовые варианты, `group_id` |
| `bpe.py` | byte-level BPE и BPE-dropout только для аугментации |
| `prepare_dataset.py` | сборка `training_final.csv` |
| `training.py` | group-aware split, oversampling, обучение, пороги, манифест |
| `dataset.py`, `schemas.py` | чтение CSV и `TrainingExample` |
| `manifest.py` | `model_manifest.json` |
| `external_multilabel.py` | извлечение multilabel-строк из Open Food Facts/Beauty |
| `external_dataset.py` | legacy-сборка из HF |
| `baseline.py` | TF-IDF + LogisticRegression |
| `error_analysis.py`, `report.py` | разбор ошибок, метрики, CSV/графики |
| `data_audit.py` | аудит датасетов |
| `eval.py` | быстрый exact/micro-F1 |
| `data/` | исходные и сгенерированные CSV |
| `models/` | веса и metadata моделей |

### `pipeline/`, `storage/`, `bot/`

| Файл | Назначение |
|---|---|
| `pipeline/receipt.py` | OCR → M2 → сохранение, повторы, месячные суммы |
| `storage/receipts.py` | SQLite-хранилище |
| `bot/__main__.py` | сборка и запуск aiogram-бота |
| `bot/handlers.py` | команды, фото, кнопки повторов |
| `bot/texts.py` | русские/английские тексты |
| `bot/middleware.py` | выбор языка |
| `bot/config.py` | `BOT_*` настройки |

### `tests/`

| Папка | Что покрывает |
|---|---|
| `tests/ocr/` | движки, сервис, контракты, метрики, лимиты |
| `tests/extract/` | парсер, layout, товары, валидация, регрессии |
| `tests/categorize/` | API, сервис, модель, текст, BPE, датасеты |
| `tests/pipeline/` | полный путь фото → чек, ошибки, месяцы |
| `tests/storage/` | SQLite |
| `tests/bot/` | handlers, тексты, middleware |

### `docs/`

| Файл | Назначение |
|---|---|
| `architecture.jpg`, `architecture.pdf` | схема архитектуры |
| `m3_categorize.md` | описание M3 |
| `m3_evaluation.md` | метрики и разбор M3 |
| `m4/README.md` | история итераций M4 |
| `HANDOFF.md` | этот файл |

## 4. Быстрые команды

```bash
make setup
make check

uv sync --group ocr
uv run pytest
uv run python -m categorize eval \
  --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer

uv run python -m categorize.prepare_dataset
uv run python -m categorize train \
  --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer \
  --epochs 15

cp .env.example .env
uv run python -m bot
```

## 5. E2E на реальных фото чеков

### Датасет

Использован **`cdek-ocr/receipt-ocr-ru`** (Hugging Face, MIT, русские фото
чеков, test split — 151 чек). Аннотации содержат `seller_text`, `date_text`,
`total_text`, `item_texts`, `item_sums`. Из test взяты 5 чеков через
`datasets-server`; изображения сохранены во временную папку.

### Как воспроизвести

```bash
# 1. Tesseract без sudo: скачать .deb и распаковать
apt-get download tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus \
  libtesseract5 liblept5
mkdir -p /tmp/receipt-e2e/tesseract
for f in *.deb; do dpkg-deb -x "$f" /tmp/receipt-e2e/tesseract; done

# 2. Python-обёртка
uv pip install 'pytesseract>=0.3.13'

# 3. Запуск (пути из примера)
export LD_LIBRARY_PATH=/tmp/receipt-e2e/tesseract/usr/lib/x86_64-linux-gnu
export TESSDATA_PREFIX=/tmp/receipt-e2e/tesseract/usr/share/tesseract-ocr/5/tessdata
export PYTHONPATH=.
uv run python /tmp/receipt-e2e/run_e2e.py
```

### Результат на 5 реальных фото

| # | Чек | OCR-блоков | store | date | total | items | warnings |
|---:|---|---:|---|---|---|---:|---|
| 0 | ЛЕНТА, 03.11.25, 214.98 | 45 | — | — | — | 1 | store/date/total/items |
| 1 | ЛЕНТА, 05.11.25, 1678.38 | 34 | — | да | — | 2 | store/total/items |
| 2 | Лента, 05.12.24, 464.00 | 27 | — | — | да | 0 | store/date/items |
| 3 | ИП Дейнека, 17.10.25, 493.00 | 50 | — | — | — | 0 | все |
| 4 | ИП Петраков, 04.10.25, 535.00 | 29 | — | — | — | 0 | все |

Итог: **0/5 store, 1/5 date, 1/5 total, 0/5 полный список товаров**. Один
извлечённый товар (`БАЛАНС BANOB...`) классифицирован как `personal_care`,
то есть M3 получил мусорную строку. Полный путь фото → поля → категории на
реальных чеках пока не работает.

Основные причины:

1. `extract/_text.py` принимает только дату `dd.mm.yyyy`; в датасете —
   `dd.mm.yy` (`03.11.25`).
2. Итог ищется по меткам `итого/к оплате`; в части чеков сумма подписана
   `СУММА`, `10000702652 CYMMA (РУБ.)` и т. п.
3. Магазин определяется только если до первой даты/суммы ровно одна
   подходящая строка; в реальных чеках несколько строк шапки.
4. Layout-парсер товаров чувствителен к качеству OCR и не восстановил
   товарные строки на 4 из 5 чеков.
5. Tesseract на этих фото даёт шумные блоки; `enh`-версии не гарантируют
   чистый OCR.

Это дефекты M2/OCR, а не M3. Чтобы честно оценить M3 на реальных данных,
нужно отдельно прогнать классификатор по `item_texts` из аннотаций, не
смешивая с ошибками OCR/парсера.

## 6. Промпты для e2e-тестирования

### Русский

```text
Ты — QA-инженер проекта receipt-tracker. Проведи e2e-тест на реальных фото
чеков, не меняя код без отдельного разрешения.

Датасет: Hugging Face `cdek-ocr/receipt-ocr-ru`, split `test` (русские чеки,
MIT). Возьми 5–10 чеков. Для каждого:
1. Скачай изображение и аннотации (`seller_text`, `date_text`, `total_text`,
   `item_texts`, `item_sums`).
2. Прогони полный путь: bytes → `OcrService`/`TesseractEngine` →
   `extract.parse_ocr_result` → `categorize.TransformerCategoryClassifier`.
3. Зафиксируй: file_name, число OCR-блоков, store/date/total/items,
   warnings, категории и confidence для каждого товара.
4. Сравни с аннотациями: store exact, date exact, total exact, item recall,
   item sum recall. Отдельно отметь, где ошибка OCR, где M2, где M3.
5. В конце дай таблицу pass/fail, top-3 причины и минимальные фиксы.
Запускай команды и приводи реальный вывод; не выдумывай результаты.
```

### English

```text
You are a QA engineer for the receipt-tracker project. Run an end-to-end test on
real receipt photos. Do not change code without explicit permission.

Dataset: Hugging Face `cdek-ocr/receipt-ocr-ru`, `test` split (Russian receipts,
MIT). Pick 5–10 receipts. For each:
1. Download the image and annotations (`seller_text`, `date_text`, `total_text`,
   `item_texts`, `item_sums`).
2. Run the full path: bytes → `OcrService`/`TesseractEngine` →
   `extract.parse_ocr_result` → `categorize.TransformerCategoryClassifier`.
3. Record: file name, OCR block count, store/date/total/items, warnings,
   categories and confidence for each item.
4. Compare with annotations: store exact, date exact, total exact, item recall,
   item-sum recall. Attribute each failure to OCR, M2, or M3.
5. Finish with a pass/fail table, top-3 root causes, and minimal fixes.
Run the commands and report real output; never fabricate results.
```

## 7. Известные ограничения

- M3 обучен на русских названиях товаров; английские/смешанные строки
  классифицируются хуже.
- m3f (BPE + OCR-noise) — эксперимент, не default; в поставку не входит.
- `training_final.csv` и веса `category_transformer_*` — генерируемые/
  локальные артефакты; модель по умолчанию хранится в git.
- `report.py` использует `matplotlib`, но зависимость не объявлена.
- `docs/m3_categorize.md` местами устарел (старые 24k/0.892); актуальные
  цифры — в `categorize/PROGRESS.md` и `docs/m3_evaluation.md`.
- E2E на реальных фото требует Tesseract/EasyOCR; в `make check` они не входят.
