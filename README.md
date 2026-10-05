# receipt-tracker

`receipt-tracker` — учебный сервис для учета расходов по фотографиям чеков.
Пользователь фотографирует чек, отправляет его Telegram-боту, а система
распознает текст, извлекает структуру чека, классифицирует товары по категориям
и собирает месячные отчеты.

Главная идея проекта: убрать ручной ввод расходов. Пользователь работает только
с фото чека и командами бота, а пайплайн сам превращает изображение в понятные
категории трат.

## Возможности

- прием фотографии чека через Telegram-бота;
- OCR-распознавание текста с изображения;
- извлечение структуры чека: дата, сумма, позиции;
- классификация товарных позиций по категориям;
- confidence для каждой предсказанной категории;
- месячная агрегация расходов по категориям;
- HTTP API для модуля классификации;
- CLI для сборки датасета, обучения и оценки модели;
- обученная transformer-модель в репозитории;
- тесты, типизация, линтинг и Dockerfile.

## Архитектура

Проект разделен на четыре модуля:

| Модуль | Пакеты | Ответственность |
|---|---|---|
| M1 OCR | `ocr/` | Принимает фото чека и возвращает распознанный текст с координатами. |
| M2 Extract | `extract/` | Превращает OCR-текст в JSON-чек: дата, сумма, товары. |
| M3 Categorize | `categorize/`, `reports/` | Классифицирует товары по категориям и строит месячные отчеты. |
| M4 Bot/Pipeline/Storage | `bot/`, `pipeline/`, `storage/` | Общается с пользователем, вызывает M1-M3 и сохраняет данные. |

Общие модели данных лежат в `common/`. Модули должны общаться через эти
контракты, а не импортировать внутренние реализации друг друга.

## Структура ветки

Ветка `uvusibuneka` содержит реализованный модуль M3 и общие контракты,
необходимые для его работы:

```text
categorize/                  # классификатор категорий, API, обучение, eval
common/                      # Pydantic-контракты Receipt, Item, MonthlyReport
dataset/category/            # seed-набор и собранный training.csv
models/category_transformer/ # обученная transformer-модель
reports/                     # месячная агрегация расходов
tests/                       # unit/API tests для M3
Dockerfile                   # запуск M3 API в контейнере
pyproject.toml               # зависимости и настройки инструментов
uv.lock                      # lock-файл uv
```

## Контракты данных

Основные модели находятся в `common/models.py`.

`Item` — позиция чека:

```json
{
  "name": "молоко цельное",
  "quantity": "1",
  "total": "89.90"
}
```

`Receipt` — чек:

```json
{
  "receipt_id": "r-1",
  "purchased_at": "2026-10-03T10:00:00+03:00",
  "items": [
    {
      "name": "молоко цельное",
      "quantity": "1",
      "total": "89.90"
    }
  ]
}
```

`ClassifiedItem` — позиция после классификации:

```json
{
  "name": "молоко цельное",
  "quantity": "1",
  "total": "89.90",
  "category": "food",
  "confidence": 0.93
}
```

Деньги хранятся через `Decimal`, даты — timezone-aware `datetime`.

## M3: классификатор категорий

M3 реализован в пакете `categorize/`.

Компоненты:

- `classifier.py` — протокол `CategoryClassifier`;
- `transformer.py` — inference обученной transformer-модели;
- `training.py` — fine-tuning transformer-модели;
- `external_dataset.py` — загрузка внешнего датасета с Hugging Face;
- `dataset.py` — чтение CSV-датасета;
- `service.py` — сервис классификации позиций;
- `api.py` — FastAPI-приложение;
- `eval.py` — расчет accuracy;
- `__main__.py` — CLI.

Используемая базовая модель: `cointegrated/rubert-tiny`.

Обученная модель сохранена в:

```text
models/category_transformer/
```

В каталоге модели лежат:

- `model.safetensors` — веса fine-tuned transformer;
- `config.json` — конфигурация модели;
- `tokenizer.json` и `tokenizer_config.json` — tokenizer;
- `category_metadata.json` — соответствие id категории и имени категории;
- `training_args.bin` — параметры обучения.

## Категории

Текущие категории M3:

| Категория | Значение |
|---|---|
| `food` | продукты, напитки, бакалея, мясо, молочные товары |
| `household` | бытовая химия и товары для дома |
| `personal_care` | гигиена и уход |
| `medicine` | аптека и медицинские товары |
| `cafe` | кафе, рестораны, готовая еда |

Категории можно расширять через датасет и повторное обучение.

## Датасет

Для обучения используется CSV:

```text
dataset/category/training.csv
```

Он собирается из двух источников:

1. Локальный seed-набор чековых позиций:

   ```text
   dataset/category/items.csv
   ```

2. Открытый русский датасет Hugging Face:

   ```text
   IvanTatarkin/ru-product-taxonomy
   ```

Внешний датасет содержит русскоязычную taxonomy товаров. В M3 он используется
как источник реальных товарных названий для усиления категории `food`, а
локальный seed-набор добавляет категории, которых нет в продуктовой taxonomy:
`household`, `personal_care`, `medicine`, `cafe`.

Собрать датасет заново:

```bash
uv run python -m categorize build-dataset --limit 300
```

Параметры:

- `--output` — путь для итогового CSV, по умолчанию
  `dataset/category/training.csv`;
- `--seed` — локальный seed CSV, по умолчанию `dataset/category/items.csv`;
- `--limit` — сколько строк взять из внешнего датасета.

## Обучение

Переобучить transformer:

```bash
uv run python -m categorize train \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

По умолчанию обучение использует:

- базовую модель `cointegrated/rubert-tiny`;
- `6` эпох;
- `max_length=64`;
- train/test split `80/20`;
- метрику `accuracy`.

Последний прогон обучения дал:

```text
eval_accuracy=0.9385
train_loss=0.2914
```

Проверка сохраненной модели на собранном CSV:

```bash
uv run python -m categorize eval \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

Результат последней проверки:

```text
accuracy=0.960 examples=322
```

## API

M3 поднимается как FastAPI-сервис.

Локальный запуск:

```bash
uv run fastapi dev categorize/api.py
```

Production-like запуск:

```bash
uv run fastapi run categorize/api.py --host 0.0.0.0 --port 8000
```

### `POST /classify`

Классифицирует одну позицию.

Запрос:

```json
{
  "name": "молоко цельное",
  "quantity": "1",
  "total": "89.90"
}
```

Ответ:

```json
{
  "name": "молоко цельное",
  "quantity": "1",
  "total": "89.90",
  "category": "food",
  "confidence": 0.93
}
```

### `POST /classify/items`

Классифицирует список позиций.

Запрос:

```json
{
  "items": [
    {
      "name": "шампунь",
      "quantity": "1",
      "total": "210"
    },
    {
      "name": "хлеб",
      "quantity": "1",
      "total": "55"
    }
  ]
}
```

Ответ:

```json
{
  "items": [
    {
      "name": "шампунь",
      "quantity": "1",
      "total": "210",
      "category": "personal_care",
      "confidence": 0.88
    },
    {
      "name": "хлеб",
      "quantity": "1",
      "total": "55",
      "category": "food",
      "confidence": 0.94
    }
  ]
}
```

### `POST /reports/monthly`

Классифицирует позиции чеков и собирает месячный отчет.

Запрос:

```json
[
  {
    "receipt_id": "r-1",
    "purchased_at": "2026-10-03T10:00:00+03:00",
    "items": [
      {
        "name": "шампунь",
        "quantity": "1",
        "total": "210"
      },
      {
        "name": "хлеб",
        "quantity": "1",
        "total": "55"
      }
    ]
  }
]
```

Ответ:

```json
{
  "year": 2026,
  "month": 10,
  "total": "265",
  "categories": [
    {
      "category": "personal_care",
      "total": "210",
      "items_count": 1
    },
    {
      "category": "food",
      "total": "55",
      "items_count": 1
    }
  ]
}
```

### `POST /train`

Переобучает модель на указанном датасете и перезагружает сервис.

Запрос:

```json
{
  "dataset_path": "dataset/category/training.csv",
  "model_path": "models/category_transformer"
}
```

Успешный ответ: `204 No Content`.

## Отчеты

Месячные отчеты реализованы в `reports/monthly.py`.

`build_monthly_report()` принимает список чеков с уже классифицированными
позициями, проверяет, что все чеки относятся к одному месяцу, и возвращает:

- год;
- месяц;
- общую сумму;
- суммы по категориям;
- количество позиций в каждой категории.

Если передать неклассифицированные позиции, функция выбросит `TypeError`.

## Docker

Сборка образа:

```bash
docker build -t receipt-tracker-m3 .
```

Запуск:

```bash
docker run --rm -p 8000:8000 receipt-tracker-m3
```

После запуска API будет доступно на:

```text
http://localhost:8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

Примечание: `torch` может подтянуть крупные platform-specific wheels. На
машинах с ограниченным свободным местом Docker build может потребовать очистки
Docker cache или отдельной настройки CPU-only PyTorch index.

## Установка и запуск

Требования:

- Python 3.12+;
- `uv`;
- доступ к интернету для повторной сборки датасета или скачивания базовой
  transformer-модели;
- Docker, если нужен контейнерный запуск.

Установка зависимостей:

```bash
uv sync
```

Запуск API:

```bash
uv run fastapi dev categorize/api.py
```

Проверка модели:

```bash
uv run python -m categorize eval
```

## Тесты и качество

Форматирование:

```bash
uv run ruff format
```

Линтер:

```bash
uv run ruff check
```

Типизация:

```bash
uv run mypy common categorize reports tests
```

Тесты:

```bash
uv run pytest
```

Последний полный прогон:

```text
ruff check: passed
mypy: Success, no issues found in 18 source files
pytest: 6 passed, 1 warning
```

## Git workflow

Правила разработки описаны в [CONTRIBUTING.md](CONTRIBUTING.md).

Коротко:

- одна задача — одна ветка;
- коммиты на английском в формате Conventional Commits;
- перед PR должны проходить `ruff format`, `ruff check`, `mypy`, `pytest`;
- изменения контрактов в `common/` должны быть согласованы с владельцами
  затронутых модулей.

## Команда

- Андрей Карабанов
- Даниил Качанов
- Артем Зайцев
- Даниил Горляков

## Статус

В ветке `uvusibuneka` полностью реализован M3:

- есть обученный transformer-классификатор;
- есть рабочее API;
- есть CLI для датасета, обучения и eval;
- есть месячные отчеты;
- есть тесты;
- есть Dockerfile;
- модель и датасет добавлены в репозиторий.
