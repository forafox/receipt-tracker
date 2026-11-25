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

| Файл / папка             | Назначение                                                                 |
| ------------------------ | -------------------------------------------------------------------------- |
| `api.py`                 | FastAPI-приложение, HTTP-контракт модуля, пути по умолчанию к данным/модели |
| `service.py`             | `CategoryService`: оборачивает `Item`/`Receipt` в классифицированные версии |
| `classifier.py`          | `CategoryClassifier` — Protocol интерфейса (`predict(item_name)`)           |
| `transformer.py`         | инференс: `denoise`, токенизация, sigmoid, per-label пороги                 |
| `training.py`            | обучение, group-aware split, best checkpoint по eval loss, подбор порогов, метрики        |
| `dataset.py`             | чтение CSV (`load_training_examples`)                                      |
| `schemas.py`             | `TrainingExample(text, category, source, group_id)`                         |
| `text.py`                | `denoise`, `add_noise` (профили `ocr`/`layout`/`receipt`/`words`/`severe`), `shorten` |
| `bpe.py`                 | BPE-токенизатор для аугментации и BPE-dropout                              |
| `augment.py`             | синтетические пары/тройки, шумовые варианты, BPE-dropout, `group_id`        |
| `prepare_dataset.py`     | сборка `training_final.csv` из всех источников и аугментаций                |
| `external_multilabel.py` | извлечение мультилейбл-строк из Open Food Facts/Beauty                      |
| `external_dataset.py`    | legacy-сборка из Hugging Face + seed                                       |
| `baseline.py`            | TF-IDF + LogisticRegression на том же split                                |
| `error_analysis.py`      | ошибки через `TransformerCategoryClassifier`                               |
| `report.py`              | метрики, CSV и графики для `reports/m3/`                                   |
| `eval.py`                | `evaluate`: exact-match accuracy и micro-F1                                |
| `__main__.py`            | CLI: `train`, `build-dataset`, `eval`                                      |
| `data/`                  | обучающие CSV и контрпримеры                                               |
| `models/`                | веса `category_transformer` + `category_metadata.json`                     |

Контракты обмена (`Item`, `Receipt`, `CategoryScore`, `ClassifiedItem`, …) живут
в общем пакете `common` и здесь не дублируются.

## Как работает предсказание

1. `denoise()`: NFKC, VAT-префиксы, гомоглифы, хвостовые количества.
2. Токенизация (`max_length=64`), `sigmoid` по логитам каждой категории.
3. Порог берётся из `category_metadata.json` для каждого класса; fallback `0.5`.
4. Если ни одна категория не прошла порог — берётся argmax.
5. Результат сортируется по уверенности; пустая строка — `ValueError`.

## Данные

Формат `data/*.csv`: `text,category` в UTF-8 с заголовком. Пустые строки пропускаются.
`category` — одна или несколько меток через `|`.

```csv
text,category
молоко 3.2%,food
капучино,cafe|food
```

- `data/training_wb.csv` — 24 219 реальных русских названий: Wildberries
  (`nyuuzyou/wb-products`, CC0) и русский срез Open Food Facts. Одна метка на строку.
- `data/multilabel_curated.csv` — 39 вручную размеченных неоднозначных чековых форм.
- `data/multilabel_external.csv` — 785 строк из Open Food Facts/Beauty, где теги
  товара пересекают группы M3 (`food|medicine`, `food|household`,
  `household|personal_care`, `cafe|food`).
- `data/hard_negatives.csv` — 77 контрпримеров из разбора ошибок.
- `data/training_final.csv` — 45 768 строк, 9 982 мультилейбл. Генерируется
  `prepare_dataset`; в git не входит.
- `data/training.csv` — 110 строк для smoke-тестов.
- `data/holdout_manual.csv` — 41 короткая чековая форма для ручной проверки.

Аугментации в `prepare_dataset`:

- пары и тройки из реальных названий разных групп (`pairs_per_pair=500`,
  `triples=1000`);
- OCR-шум для 25 % строк (`noise_fraction=0.25`), по 2 варианта на строку
  (`noise_variants=2`);
- профили шума `ocr`, `layout`, `receipt`, `words`, `severe`: смешения
  символов, потеря границ слов, обрезка, НДС/количество/разделители;
- BPE-dropout для 35 % шумовых вариантов (`bpe.py`, vocab 2 000);
- `group_id` у всех строк, чтобы чистые, короткие, синтетические и шумовые
  варианты одного товара не попадали одновременно в train и test.

## Обучение

База — `cointegrated/rubert-tiny`, `problem_type="multi_label_classification"`:

- `epochs=15` (CLI по умолчанию 6, для полного датасета нужно `--epochs 15`),
  `lr=5e-5`, `batch=8`, `weight_decay=0.01`, `seed=42`;
- group-aware split: 72 % train, 8 % validation, 20 % test;
- oversampling train-фолда до мажоритарного класса, но не больше 4 проходов;
- best checkpoint по `eval_loss`, `save_total_limit=1`;
- после обучения пороги подбираются на validation по F1 каждого класса и
  сохраняются в metadata/manifest.

Метрики m3e на старом test (7 784 строки, 1 718 мультилейбл). Текущая
`training_final.csv` содержит 45 768 строк после добавления BPE/шума; эксперимент
m3f описан в `reports/m3f/REPORT.md`.

| Метрика | Значение |
|---|---:|
| Exact match | 0.7821 |
| Micro F1 | 0.8661 |
| Macro F1 | 0.8649 |
| Hamming loss | 0.0671 |
| Mean label recall | 0.8789 |
| Multilabel exact match | 0.5722 |

| Класс | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| cafe | 0.836 | 0.878 | 0.857 | 773 |
| food | 0.860 | 0.876 | 0.868 | 2 349 |
| household | 0.881 | 0.901 | 0.891 | 2 184 |
| medicine | 0.861 | 0.826 | 0.843 | 2 245 |
| personal_care | 0.873 | 0.859 | 0.866 | 2 201 |

Baseline TF-IDF + LogisticRegression на том же split: exact 0.7157, micro 0.8375,
macro 0.8347, hamming 0.0811. Transformer лучше по всем агрегатам.

Ручной набор из 41 короткой формы: exact 0.854, micro 0.907.

## CLI

```bash
# из корня репозитория:
python -m categorize.prepare_dataset
python -m categorize train \
  --dataset categorize/data/training_final.csv \
  --model categorize/models/category_transformer --epochs 15
python -m categorize eval \
  --dataset categorize/data/holdout_manual.csv \
  --model categorize/models/category_transformer
python -m categorize.report \
  --model categorize/models/category_transformer \
  --dataset categorize/data/training_final.csv \
  --log reports/m3/raw/train.log \
  --output reports/m3
```

Если файла модели нет, `eval` и `load_service` обучат её автоматически на
`DEFAULT_DATASET_PATH` (`data/training.csv`, smoke-набор).

## FastAPI

```bash
uv run --with uvicorn uvicorn categorize.api:app --host 127.0.0.1 --port 8000
```

Swagger UI: http://127.0.0.1:8000/docs. Бот к API не обращается: он загружает модель сам.

| Метод | Путь | Тело | Ответ |
| ----- | ---- | ---- | ----- |
| `POST` | `/classify` | `Item` | `ClassifiedItem` |
| `POST` | `/classify/items` | `{items: [Item]}` | `{items: [ClassifiedItem]}` |
| `POST` | `/classify/receipt` | `Receipt` (контракт M2→M3) | `ClassifiedReceipt` |
| `POST` | `/train` | `{dataset_path, model_path}` | `204`, сервис перезагружается |

Сервис создаётся в `lifespan` через `load_service()`. Если он не инициализирован —
`503`, ошибки обучения — `400`.

API дополнительно проверяет каждое предсказание локальной моделью Ollama
(`categorize/ollama.py`, модель `M3_VALIDATION_MODEL`). Категории классификатора не
меняются; в ответ добавляются `category_validation` (`agreed`, `disagreed` или
`uncertain`) и `suggested_categories` из известного модели списка категорий. Если
Ollama недоступна, проверка возвращает `uncertain`. Бот эту проверку не использует.

## Проверки

```bash
ruff check categorize
ruff format --check categorize
mypy categorize
pytest tests/categorize -q
```

Подробные метрики и разбор ошибок — [docs/m3_evaluation.md](../docs/m3_evaluation.md).
