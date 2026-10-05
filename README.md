# receipt-tracker

Трекер расходов по фотографиям чеков.

Сфотографировал чек, отправил Telegram-боту и через пару секунд получил ответ: где была покупка, на какую сумму и как траты делятся по категориям — продукты, бытовая химия и так далее. Если бот ошибся с категорией, её можно поправить одной кнопкой. А в конце месяца бот покажет, куда ушли деньги, какие категории самые затратные и как это соотносится с прошлым месяцем.

Никаких таблиц и ручного ввода — только фото чека.

Проект учебный, разрабатывается командой из четырёх человек осенью 2026 года.

## Команда

- Андрей Карабанов
- Даниил Качанов
- Артем Зайцев
- Даниил Горляков

## Как участвовать

Правила разработки, оформления кода и коммитов описаны в [CONTRIBUTING.md](CONTRIBUTING.md).

## M3: классификатор категорий

Модуль `categorize/` классифицирует позиции чека по категориям и отдает confidence.
Модель обучается на размеченном CSV-наборе `dataset/category/items.csv`, сохраняется в
`models/category_classifier.joblib` и автоматически создается при первом запуске API.

Команды:

```bash
uv run python -m categorize build-dataset --limit 2000
uv run python -m categorize train
uv run python -m categorize eval
uv run fastapi dev categorize/api.py
```

`build-dataset` подтягивает открытый русский датасет
`IvanTatarkin/ru-product-taxonomy` с Hugging Face и объединяет его с локальным
seed-набором чековых позиций. Для fine-tuning используется transformer
`cointegrated/rubert-tiny`, обученная модель сохраняется в
`models/category_transformer/`.

Docker:

```bash
docker build -t receipt-tracker-m3 .
docker run --rm -p 8000:8000 receipt-tracker-m3
```

API:

- `POST /classify` — классифицировать одну позицию;
- `POST /classify/items` — классифицировать список позиций;
- `POST /reports/monthly` — классифицировать позиции чеков и собрать месячный отчет;
- `POST /train` — переобучить модель на указанном датасете.

Тесты: `uv run pytest`.
