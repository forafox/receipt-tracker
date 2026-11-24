# categorize — Progress Tracker

## PDF ↔ реализация

| Требование PDF | Статус | Где |
|---|---|---|
| `Item` → `ClassifiedItem` с категориями и уверенностями | готово | `service.py` |
| Мультилейбл: одна строка → несколько категорий | готово | `transformer.py`, `schemas.py` |
| 5 категорий: food, household, personal_care, medicine, cafe | готово | `schemas.py`, модель |
| `api.py` с `/classify`, `/classify/items`, `/classify/receipt`, `/train` | готово | `api.py` |
| `CategoryClassifier` Protocol | готово | `classifier.py` |
| `TransformerCategoryClassifier`: rubert-tiny, sigmoid, порог | готово | `transformer.py` |
| Пустая строка → `ValueError` | готово | `transformer.py` |
| `train_transformer`: multi-label, lr=5e-5, batch=8, weight_decay=0.01, seed=42 | готово | `training.py` |
| Стратифицированный split, минимум 1 пример класса в validation | готово, group-aware 72/8/20 | `training.py` |
| Oversampling train до мажоритарного класса | готово, не более 4 проходов | `training.py` |
| Метрики validation: exact_match + micro_f1 | готово, плюс macro/hamming/mean label recall | `training.py` |
| Автообучение, если модели нет | готово | `api.load_service` |
| F1 на test > 0.85 | micro 0.8661, macro 0.8649 | `reports/m3/REPORT.md` |
| CSV `text,category` UTF-8 с заголовком | готово, плюс optional `source`,`group_id` | `dataset.py` |
| Seed 104 строки + HF | legacy-путь сохранён | `external_dataset.py` |
| Модель в `categorize/models/` | готово | `category_transformer/` |
| Нет мёртвого кода | модули покрыты тестами | `tests/categorize` |

## Текущее состояние

### Данные

- `data/training_wb.csv` — 24 219 строк, одна метка: WB + русский Open Food Facts.
- `data/multilabel_curated.csv` — 39 вручную размеченных неоднозначных форм.
- `data/multilabel_external.csv` — 785 строк из Open Food Facts/Beauty с
  настоящими пересечениями групп.
- `data/hard_negatives.csv` — 77 контрпримеров из разбора ошибок.
- `data/training_final.csv` — 45 768 строк, 9 982 мультилейбл; генерируется
  `python -m categorize.prepare_dataset`, в git не входит.
- Шум: 2 варианта на 25 % строк, профили `ocr/layout/receipt/words/severe`,
  BPE-dropout для 35 % вариантов (`bpe.py`, vocab 2 000).
- `data/training.csv` — 110 строк для smoke-тестов.
- `data/holdout_manual.csv` — 41 короткая чековая форма.

### Обучение

- Модель `models/category_transformer` обучена на `training_final.csv`
  (15 эпох, RTX 6000, best checkpoint по eval loss).
- Group-aware split 72/8/20; `group_id` не даёт вариантам одного товара
  пересекаться между фолдами.
- `denoise()` применяется и в обучении, и в инференсе.
- Per-label пороги подобраны на validation: cafe 0.80, food 0.68,
  household 0.62, medicine 0.76, personal_care 0.68.
- Manifest хранит версию, sha256 датасета, гиперпараметры, пороги, метрики.
- Эксперимент `models/category_transformer_noise` (m3f) обучен на новой
  BPE/шумовой сборке; сравнение — `reports/m3f/REPORT.md`. Дефолтной остаётся
  m3e, потому что на новом (более шумном) test её exact выше.

### Метрики m3e test (7 784 строки, 1 718 мультилейбл, старый test)

| Метрика | Значение |
|---|---:|
| Exact match | 0.7821 |
| Micro F1 | 0.8661 |
| Macro F1 | 0.8649 |
| Hamming loss | 0.0671 |
| Mean label recall | 0.8789 |
| Multilabel exact match | 0.5722 |

Baseline TF-IDF + LogisticRegression: exact 0.7157, micro 0.8375, macro 0.8347.
Transformer лучше по всем агрегатам.

Ручной набор из 41 короткой формы: exact 0.854, micro 0.907.

### Ошибки

- `wrong_single` 611, `missed_label` 432, `extra_label` 440, `partial` 213.
- Основные путаницы: medicine ↔ food, personal_care ↔ medicine,
  personal_care ↔ household.
- Мультилейбл-тройки остаются сложными: exact 0.13–0.48.

## CLI

| Команда | Назначение |
|---|---|
| `python -m categorize.prepare_dataset` | Собрать `training_final.csv` |
| `python -m categorize train --dataset ... --model ... --epochs 15` | Обучение |
| `python -m categorize eval --dataset ... --model ...` | Exact match + micro-F1 |
| `python -m categorize.report ...` | Метрики, CSV и графики в `reports/m3/` |

## Проверки

| Проверка | Результат |
|---|---|
| `ruff check categorize tests/categorize` | чисто |
| `mypy categorize` | Success, 20 файлов |
| `pytest tests -q` | 436 passed, 7 skipped |

Последний прогон: 2026-11-16, полный `tests/`. Подробности — `reports/m3/REPORT.md`.
