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
- `categorize/training.py` — обучение transformer-модели.
- `categorize/external_dataset.py` — сборка обучающего CSV из Hugging Face.
- `categorize/dataset.py` — чтение CSV-датасета.
- `categorize/eval.py` — оценка качества модели.
- `categorize/__main__.py` — CLI-команды.

## Модель

Базовая модель: `cointegrated/rubert-tiny`.

Сохраненная обученная модель лежит в:

```text
models/category_transformer/
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

Основной обучающий CSV:

```text
dataset/category/training.csv
```

Источники данных:

- локальный seed-набор `dataset/category/items.csv`;
- Hugging Face датасет `IvanTatarkin/ru-product-taxonomy`.

Собрать датасет заново:

```bash
uv run python -m categorize build-dataset --limit 300
```

## Обучение

```bash
uv run python -m categorize train \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

В обучении используется multilabel-конфигурация модели. Для борьбы с
несбалансированными классами oversampling применяется только к train-fold после
stratified split, чтобы validation-fold не дублировался и метрики не завышались.

Проверить сохраненную модель:

```bash
uv run python -m categorize eval \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

Последний проверенный результат:

```text
accuracy=0.782 examples=110
```

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
  "dataset_path": "dataset/category/training.csv",
  "model_path": "models/category_transformer"
}
```

Успешный ответ: `204 No Content`.

## Проверки

```bash
uv run ruff format
uv run ruff check
uv run mypy .
uv run pytest
uv run python -m categorize eval --dataset dataset/category/training.csv --model models/category_transformer
```
