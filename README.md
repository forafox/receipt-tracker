# receipt-tracker

`receipt-tracker` — учебный проект для обработки чеков. Пользователь отправляет
фото чека в Telegram-бота, система распознает текст, извлекает структурированный
чек и классифицирует товарные позиции по категориям.

## Возможности

- прием фото чека через Telegram-бота;
- OCR-распознавание текста;
- извлечение структурированного чека отдельным модулем M2;
- мультилейбл-категоризация позиций чека модулем M3;
- confidence для каждой предсказанной категории;
- FastAPI-интерфейс M3 для взаимодействия с остальными модулями;
- CLI для сборки датасета, обучения и оценки модели;
- обученная transformer-модель в репозитории;
- тесты, линтер, типизация и Dockerfile.

## Архитектура

| Модуль | Пакеты | Ответственность |
|---|---|---|
| M1 OCR | `ocr/` | Распознает текст на фото чека и возвращает OCR-результат. |
| M2 Extract | `extract/` | Превращает OCR-результат в структурированный `Receipt`. |
| M3 Categorize | `categorize/` | Принимает `Receipt` через FastAPI и возвращает `ClassifiedReceipt`. |
| M4 Bot/Pipeline/Storage | `bot/`, `pipeline/`, `storage/` | Оркестрирует M1-M3, общается с пользователем и хранит данные. |

M3 не принимает фото и не общается с OCR напрямую. Между OCR и M3 должен быть
M2 Extract. Внешний протокол M3 — HTTP JSON через FastAPI.

## Структура

```text
categorize/
common/
dataset/category/
models/category_transformer/
tests/
Dockerfile
pyproject.toml
uv.lock
```

## Контракты

Контракты лежат в `common/models.py`.

`Item`:

```json
{
  "name": "молоко цельное",
  "quantity": "1",
  "total": "89.90"
}
```

`Receipt`, который M2 передает в M3:

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

`ClassifiedReceipt`, который M3 возвращает M2/M4:

```json
{
  "receipt_id": "r-1",
  "purchased_at": "2026-10-03T10:00:00+03:00",
  "items": [
    {
      "name": "молоко цельное",
      "quantity": "1",
      "total": "89.90",
      "categories": [
        {
          "category": "food",
          "confidence": 0.93
        }
      ]
    }
  ]
}
```

## M3 Categorize

Компоненты:

- `categorize/api.py` — FastAPI-приложение;
- `categorize/service.py` — сервис категоризации;
- `categorize/transformer.py` — загрузка модели и inference;
- `categorize/lexical.py` — доменная корректировка русских товарных названий;
- `categorize/training.py` — обучение transformer-модели;
- `categorize/external_dataset.py` — сборка обучающего CSV из Hugging Face;
- `categorize/dataset.py` — чтение CSV;
- `categorize/eval.py` — расчет accuracy;
- `categorize/__main__.py` — CLI.

Базовая модель: `cointegrated/rubert-tiny`.

Обученная модель:

```text
models/category_transformer/
```

Категории могут назначаться в мультилейбл-режиме. Текущий набор категорий:

| Категория | Значение |
|---|---|
| `food` | продукты и напитки |
| `household` | бытовая химия и товары для дома |
| `personal_care` | гигиена и уход |
| `medicine` | аптека и медицинские товары |
| `cafe` | кафе, рестораны, готовая еда |

## Датасет

Основной CSV:

```text
dataset/category/training.csv
```

Источники:

- локальный seed-набор `dataset/category/items.csv`;
- Hugging Face датасет `IvanTatarkin/ru-product-taxonomy`.

Собрать CSV заново:

```bash
uv run python -m categorize build-dataset --limit 300
```

## Обучение

```bash
uv run python -m categorize train \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

Проверить сохраненную модель:

```bash
uv run python -m categorize eval \
  --dataset dataset/category/training.csv \
  --model models/category_transformer
```

Последний eval:

```text
accuracy=0.782 examples=110
```

## FastAPI

Запуск:

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

Категоризирует одну позицию. Ответ может содержать несколько категорий.

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
      "confidence": 0.93
    }
  ]
}
```

### `POST /classify/items`

Категоризирует список позиций. Каждая позиция может получить несколько категорий.

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
      "categories": [
        {
          "category": "personal_care",
          "confidence": 0.88
        }
      ]
    },
    {
      "name": "хлеб",
      "quantity": "1",
      "total": "55",
      "categories": [
        {
          "category": "food",
          "confidence": 0.94
        }
      ]
    }
  ]
}
```

### `POST /classify/receipt`

Основной endpoint для M2/M4. Принимает структурированный `Receipt`, возвращает
`ClassifiedReceipt`, где у каждой позиции есть список категорий.

Запрос:

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

Ответ:

```json
{
  "receipt_id": "r-1",
  "purchased_at": "2026-10-03T10:00:00+03:00",
  "items": [
    {
      "name": "молоко цельное",
      "quantity": "1",
      "total": "89.90",
      "categories": [
        {
          "category": "food",
          "confidence": 0.93
        }
      ]
    }
  ]
}
```

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

## Docker

```bash
docker build -t receipt-tracker-m3 .
docker run --rm -p 8000:8000 receipt-tracker-m3
```

## Проверки

```bash
uv run ruff format
uv run ruff check
uv run mypy common categorize tests
uv run pytest
```

## Модули

- **M1 (OCR)**: принимает фото чека и возвращает распознанный текст с координатами.
- **M2 (разбор структуры)**: превращает текст в JSON-чек (дата, сумма, товары). Поля извлекаются правилами, без LLM.
- **M3 (классификатор)**: распределяет товары по категориям с помощью трансформера и строит отчёты за месяц.
- **M4 (бот, оркестратор, хранилище)**: общается с пользователем, вызывает M1, M2 и M3 по очереди и сохраняет данные.

## Архитектура

![Схема архитектуры](docs/architecture.jpg)

Подробное описание: [docs/architecture.pdf](docs/architecture.pdf).

## M4: бот

Telegram-бот на aiogram 3. Принимает от пользователя фото чека и команды, передаёт их в сервис обработки и отправляет ответ. Сам бот чеки не разбирает и не знает, как устроены остальные модули.

Команды:

- `/start` — приветствие;
- `/help` — справка;
- `/month` — расходы за текущий месяц;
- фото чека — бот скачивает его в максимальном разрешении и отдаёт сервису.

Бот работает с остальной системой через интерфейс `ReceiptService` (`bot/service.py`). Сейчас вместо настоящего пайплайна подключена заглушка `StubReceiptService`.

Язык ответов (русский или английский) выбирается по языку интерфейса Telegram у пользователя. Тексты лежат в `bot/texts.py`, меню команд тоже переведено.

Структура:

| Файл | Назначение |
|---|---|
| `bot/__main__.py` | точка входа, регистрация команд, запуск polling |
| `bot/handlers.py` | обработчики команд и фото |
| `bot/service.py` | интерфейс сервиса и заглушка |
| `bot/texts.py` | тексты на русском и английском |
| `bot/middleware.py` | выбор языка для каждого сообщения |
| `bot/config.py` | настройки из переменных окружения |

### Запуск

```bash
make setup
cp .env.example .env  # вписать токен бота в BOT_TOKEN
uv run python -m bot
```

Тесты: `make test`.

## Разработка

После клонирования настройте окружение и запустите все проверки:

```bash
make setup
make check
```

Применить каноническое форматирование: `make format`. Подробные правила разработки описаны в
[CONTRIBUTING.md](CONTRIBUTING.md).

## Команда

- Андрей Карабанов
- Даниил Качанов
- Артем Зайцев
- Даниил Горляков
