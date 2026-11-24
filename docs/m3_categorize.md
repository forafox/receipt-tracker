# M3 Categorize

M3 отвечает только за категоризацию товарных позиций. Модуль не принимает фото,
не запускает OCR и не парсит сырой текст чека. Ожидаемый поток такой:

1. M1 OCR распознает текст на изображении.
2. M2 Extract превращает OCR-результат в структурированный `Receipt`.
3. M3 Categorize принимает `Receipt` или отдельные `Item` через FastAPI.
4. M3 возвращает `ClassifiedReceipt` с категориями для каждой позиции.

## Компоненты

- `categorize/api.py` — FastAPI-приложение и HTTP-контракт модуля.
- `categorize/service.py` — сервисный слой классификации.
- `categorize/transformer.py` — загрузка обученной модели и inference.
- `categorize/lexical.py` — доменная корректировка русских товарных названий.
- `categorize/training.py` — обучение transformer-модели, multilabel-сплит и oversampling.
- `categorize/manifest.py` — `model_manifest.json` с версией, хэшем датасета и метриками.
- `categorize/prepare_dataset.py` — сборка `training_final.csv` из основного CSV и curated multilabel.
- `categorize/baseline.py` — TF-IDF + logistic regression baseline.
- `categorize/error_analysis.py` — разбор ошибок на holdout и ручном наборе.
- `categorize/data_audit.py` — аудит датасетов (дубли, покрытие, multilabel).
- `categorize/external_dataset.py` — legacy-сборка CSV из Hugging Face.
- `categorize/dataset.py` — чтение CSV-датасета.
- `categorize/eval.py` — оценка качества модели.
- `categorize/__main__.py` — CLI-команды.

## Модель

Базовая модель: `cointegrated/rubert-tiny`.

Сохраненная обученная модель лежит в:

```text
categorize/models/category_transformer/
```

Классификация работает в мультилейбл-режиме: один товар может получить несколько
категорий. Например, строка с несколькими товарами может вернуть `food`,
`personal_care` и `medicine` одновременно.

Текущий набор категорий:

| Категория | Значение |
|---|---|
| `food` | продукты и напитки |
| `household` | бытовая химия и товары для дома |
| `personal_care` | гигиена и уход |
| `medicine` | аптека и медицинские товары |
| `cafe` | кафе, рестораны, готовая еда |

## Датасет

Основной сырой CSV:

```text
categorize/data/training_wb.csv   # 24 219 строк, 5 классов
```

`training.csv` (110 строк) остаётся smoke-набором, `training_full.csv` —
устаревший и для обучения не используется.

Датасет `training_wb.csv` собран из Wildberries (`nyuuzyou/wb-products`, CC0)
и русского среза Open Food Facts. `cafe` размечен вручную, потому что
публичного русского датасета готовых блюд нет.

Во всех исходных данных ровно одна метка на строку. Настоящие multilabel-примеры
добавлены отдельно в `categorize/data/multilabel_curated.csv` — только там, где
сам текст чека неоднозначен без контекста магазина (`сода` — food и household,
`мыло` — household и personal_care, `кофе` — food и cafe). Искусственные
комбинации не генерируются.

Собрать финальный датасет:

```bash
uv run python -m categorize.prepare_dataset
```

Скрипт дедуплицирует тексты, накладывает curated-метки, проверяет неизвестные
категории и печатает распределение. Результат —
`categorize/data/training_final.csv` (24 234 строки, 39 multilabel).

## Обучение

```bash
uv run python -m categorize.prepare_dataset
uv run python -m categorize train \
  --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer \
  --epochs 15
```

Дефолт CLI — 6.0 эпох (значение из PDF). В проекте принято 15 эпох: на 6 эпохах
модель недоучивается на коротких формах (`сыр → cafe`).

В обучении используется multilabel-конфигурация модели и
`binary_cross_entropy_with_logits` с `pos_weight` по каждому классу (вес ограничен
5.0, чтобы редкий `cafe` не доминировал). Split — 80/20 по целым наборам меток,
поэтому multilabel-строка не попадает одновременно в train и test. Oversampling
применяется только к train-fold: пример повторяется по самому редкому из своих
классов, но не более 4 раз. Validation-fold не дублируется.

Baseline и разбор ошибок:

```bash
uv run python -m categorize.baseline
uv run python -m categorize.error_analysis --split test
uv run python -m categorize.error_analysis \
  --dataset categorize/data/holdout_manual.csv --split all
```

Проверить сохраненную модель:

```bash
uv run python -m categorize eval \
  --dataset categorize/data/training.csv \
  --model categorize/models/category_transformer
```

Последний проверенный результат на честном 20 % holdout из
`training_final.csv`: `micro_f1=0.892`, `macro_f1=0.834`, `exact_match=0.888`.
Подробности, baseline и разбор ошибок — в `docs/m3_evaluation.md`.

## FastAPI

Запуск dev-сервера:

```bash
uv run fastapi dev categorize/api.py
```

Production-like запуск:

```bash
uv run fastapi run categorize/api.py --host 0.0.0.0 --port 8000
```

Swagger UI:

```text
http://localhost:8000/docs
```

### `POST /classify`

Категоризирует одну позицию.

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
  "categories": [
    {
      "category": "food",
      "confidence": 0.8
    }
  ]
}
```

### `POST /classify/items`

Категоризирует список позиций.

### `POST /classify/receipt`

Основной endpoint для взаимодействия с M2/M4. Принимает структурированный
`Receipt`, возвращает `ClassifiedReceipt`, где у каждой позиции есть список
категорий.

### `POST /train`

Переобучает модель и перезагружает сервис.

Запрос:

```json
{
  "dataset_path": "categorize/data/training.csv",
  "model_path": "categorize/models/category_transformer"
}
```

Успешный ответ: `204 No Content`.

## Проверки

```bash
uv run ruff format
uv run ruff check
uv run mypy .
uv run pytest
uv run python -m categorize eval --dataset categorize/data/training.csv --model categorize/models/category_transformer
```
