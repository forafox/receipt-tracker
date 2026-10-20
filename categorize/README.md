# categorize — M3, классификация товарных позиций

Модуль принимает структурированные позиции чека (от M2) и возвращает для каждой
позиции список категорий с уверенностями. Фото, OCR и разбор полей — не его задачи.

Поток: `Item(name, quantity, total)` → `predict(name)` → `ClassifiedItem(categories=[...])`.

## Категории

| Категория       | Значение                        |
| --------------- | ------------------------------- |
| `food`          | продукты и напитки              |
| `household`     | бытовая химия и товары для дома |
| `personal_care` | гигиена и уход                  |
| `medicine`      | аптека и медицинские товары     |
| `cafe`          | кафе, рестораны, готовая еда    |

Классификация мультилейбл: одна строка (например, «молоко, шампунь и парацетамол»)
может вернуть несколько категорий сразу. В CSV мультилейбл записывается через `|`.

## Структура

| Файл / папка       | Назначение                                                                 |
| ------------------ | -------------------------------------------------------------------------- |
| `api.py`           | FastAPI-приложение, HTTP-контракт модуля, пути по умолчанию к данным/модели |
| `service.py`       | `CategoryService`: оборачивает `Item`/`Receipt` в классифицированные версии |
| `classifier.py`    | `CategoryClassifier` — Protocol интерфейса (`predict(item_name)`)           |
| `transformer.py`   | инференс на обученном трансформере                                   |
| `training.py`      | обучение (`train_transformer`), сплит, балансировка, метрики               |
| `dataset.py`       | чтение CSV (`load_training_examples`)                                      |
| `external_dataset.py` | сборка обучающего CSV из Hugging Face + seed                             |
| `schemas.py`       | `TrainingExample(text, category)`                                          |
| `eval.py`          | `evaluate`: exact-match accuracy по множествам категорий                   |
| `__main__.py`      | CLI: `train`, `build-dataset`, `eval`                                      |
| `data/`            | обучающие CSV: `items.csv` (seed, 104 строки), `training.csv` (110 строк)  |
| `models/`          | обученные веса `category_transformer` + `category_metadata.json`           |

Контракты обмена (`Item`, `Receipt`, `CategoryScore`, `ClassifiedItem`, …) живут
в общем пакете `common` и здесь не дублируются.

## Как работает предсказание

1. Токенизация названия (`max_length=64`), `sigmoid` по логитам каждой категории.
2. Порог `0.5`; если ни одна категория не прошла — берётся argmax.
3. Результат сортируется по уверенности, других эвристик нет — только модель.
4. Пустая строка — `ValueError`.

## Данные

Формат `data/*.csv`: `text,category` в UTF-8 с заголовком. Пустые строки пропускаются.

```csv
text,category
молоко 3.2%,food
капучино,cafe
```

Основной датасет — `data/training_wb.csv` (24 127 строк): реальные русские
названия товаров из Wildberries (`nyuuzyou/wb-products`, CC0) и русский срез
Open Food Facts (4 430 строк). Разметка по `subj_root_name` / категориям OFF:

| Категория | Строк |
| --------- | ----- |
| `food` | 6 000 |
| `household` | 6 000 |
| `personal_care` | 6 000 |
| `medicine` | 6 000 |
| `cafe` | 127 |

`cafe` (готовые блюда и напитки) — единственная категория без публичного русского
датасета, список собран вручную. `build-dataset` (seed + `IvanTatarkin/ru-product-taxonomy`)
оставлен для совместимости; этот HF-датасет на 99 % состоит из нерусских
(французских) названий, поэтому для обучения не используется.

## Обучение

База — `cointegrated/rubert-tiny`, `problem_type="multi_label_classification"`:

- `epochs=15` (на GPU), `lr=5e-5`, `batch=8`, `weight_decay=0.01`, `seed=42`;
- стратифицированный сплит 80/20 (минимум 1 пример класса в валидации);
- oversampling только train-фолда до мажоритарного класса;
- метрики валидации: `exact_match` + `micro_f1`.

Метрики на сбалансированном holdout (`data/training_wb.csv`, 4 825 строк):

| Категория | exact_match |
| --------- | ----------- |
| `food` | 0.9075 |
| `household` | 0.9575 |
| `personal_care` | 0.9250 |
| `medicine` | 0.9208 |
| `cafe` | 0.8800 (n=25) |
| **итого** | **0.9275** (micro_f1 0.9285) |

```bash
# из корня репозитория:
uv run python -m categorize build-dataset --limit 300
uv run python -m categorize train \
  --dataset categorize/data/training_wb.csv \
  --model categorize/models/category_transformer --epochs 15
uv run python -m categorize eval \
  --dataset categorize/data/training_wb.csv \
  --model categorize/models/category_transformer
```

Если файла модели нет, `eval` и `load_service` обучат её автоматически.

## FastAPI

```bash
uv run fastapi dev categorize/api.py
```

| Метод | Путь | Тело | Ответ |
| ----- | ---- | ---- | ----- |
| `POST` | `/classify` | `Item` | `ClassifiedItem` |
| `POST` | `/classify/items` | `{items: [Item]}` | `{items: [ClassifiedItem]}` |
| `POST` | `/classify/receipt` | `Receipt` (контракт M2→M3) | `ClassifiedReceipt` |
| `POST` | `/train` | `{dataset_path, model_path}` | `204`, сервис перезагружается |

Сервис создаётся в `lifespan` через `load_service()`. Если он не инициализирован —
`503`, ошибки обучения — `400`.

## Проверки

```bash
uv run ruff check categorize
uv run ruff format --check categorize
uv run mypy categorize
uv run pytest tests/categorize -q
```
