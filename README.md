# receipt-tracker

Трекер расходов по фотографиям чеков.

Сфотографировал чек, отправил Telegram-боту и получил ответ: где была покупка, на какую сумму и что именно куплено, с категорией у каждой позиции. Команда `/month` показывает расходы за месяц с разбивкой по категориям. Никаких таблиц и ручного ввода, только фото чека.

Проект учебный, его делает команда из четырёх человек осенью 2026 года.

## Как это работает

```
фото в Telegram
  → M4 бот и пайплайн
  → M1 распознавание текста (Tesseract, EasyOCR или Hunyuan OCR)
  → M2 разбор: магазин, дата, итог, позиции
  → M3 категории позиций (дообученный rubert-tiny)
  → M4 сохранение в SQLite и ответ пользователю
```

Модули обмениваются данными через общие типы из `common/`: `OcrResult` (результат распознавания), `Receipt` и `Item` (чек и его позиции), `ReceiptWarning` (коды предупреждений).

## Модули

| Модуль | Ответственный | Пакеты | Документация |
|---|---|---|---|
| M1, распознавание | Артём Зайцев | `ocr/` | [ocr/README.md](ocr/README.md) |
| M2, разбор чека | Даниил Горляков | `extract/` | [extract/README.md](extract/README.md) |
| M3, категории | Даниил Качанов | `categorize/` | [categorize/README.md](categorize/README.md) |
| M4, бот, пайплайн, хранилище | Андрей Карабанов | `bot/`, `pipeline/`, `storage/` | [bot/README.md](bot/README.md) |

Первая схема системы, нарисованная до начала разработки: [docs/architecture.jpg](docs/architecture.jpg), подробнее в [docs/architecture.pdf](docs/architecture.pdf).

## Структура репозитория

```
bot/          Telegram-бот (M4)
pipeline/     порядок вызова модулей, повторы, итоги за месяц (M4)
storage/      хранение чеков в SQLite (M4)
ocr/          движки распознавания и их настройка (M1)
extract/      разбор текста чека в Receipt (M2)
categorize/   классификатор категорий, его данные, модель и API (M3)
common/       общие контракты между модулями
tests/        тесты, по папке на модуль
scripts/      сквозные прогоны на наборах реальных чеков (нужны OCR и модели)
docs/         инструкции, отчёты, журнал M4; оглавление в docs/README.md
```

Обученная модель M3 лежит в репозитории (`categorize/models/category_transformer/`), обучать её для запуска не нужно.

## Быстрый старт

```bash
make setup                                    # зависимости для разработки
uv sync --locked --group dev --group ocr      # плюс движки OCR
brew install tesseract tesseract-lang         # Tesseract с русским языком (macOS)
cp .env.example .env                          # вписать BOT_TOKEN
uv run python -m bot
```

Подробная инструкция для Linux и macOS, проверка установки и решение проблем: [docs/RUN.md](docs/RUN.md).

Перед первым запуском проверьте `.env`. В шаблоне включён Hunyuan OCR (`OCR_HUNYUAN_OCR_ENABLED=true`). Без локального сервера Hunyuan бот каждый раз сначала пытается до него достучаться и только потом переходит к Tesseract. Если сервера нет, поставьте `false`.

## Проверки и тесты

```bash
make check     # форматирование, ruff, mypy, pytest
make test      # только тесты
uv run pytest tests/bot tests/pipeline tests/storage   # тесты одного модуля
```

Весь набор тестов проходит примерно за 10 секунд и не требует сети, Telegram, Tesseract или языковых моделей. Тесты, которым нужны настоящие движки OCR или модели, помечены `@pytest.mark.slow` и в обычный прогон не входят.

| Модуль | Тестов |
|---|---:|
| M1, `tests/ocr` | 86 |
| M2, `tests/extract` | 347 |
| M3, `tests/categorize` | 68 |
| M4, `tests/bot`, `tests/pipeline`, `tests/storage` | 77 |
| контракты, `tests/common` | 11 |

## Известные проблемы (25.11.2026)

- `make check` на `main` не проходит. 23 теста M2 падают после коммита `9d16295 Add Hunyuan OCR receipt parsing`, который поменял разбор M2. Кроме того, mypy находит ошибки в `categorize/` и `common/models.py`. Чинить должны владельцы M2 и M3.
- В `common/` два описания чека: `contracts.py` (M2 и M4) и `models.py` (API M3).
- `Dockerfile` собирает только API M3, а его команда запуска не работает без `uvicorn`, см. раздел 12 в [docs/RUN.md](docs/RUN.md).

## Документация

Оглавление всех документов: [docs/README.md](docs/README.md). Ход разработки M4 по итерациям: [docs/m4/README.md](docs/m4/README.md).

## Как участвовать

Правила разработки, оформления кода и коммитов описаны в [CONTRIBUTING.md](CONTRIBUTING.md).
