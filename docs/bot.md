# Запуск Telegram-бота

Бот уже запущен в WSL и доступен в Telegram: [@hunyanbot](https://t.me/hunyanbot).
Для распознавания чеков локальный HunyuanOCR-сервер должен отвечать на
`http://localhost:8231/v1/chat/completions`.

Чтобы поднять бота вручную из корня проекта в WSL, выполни:

```bash
export BOT_API_PROXY=http://127.0.0.1:12334 OCR_HUNYUAN_OCR_ENABLED=true OCR_HUNYUAN_OCR_MODEL=HunyuanOCR OCR_HUNYUAN_OCR_URL=http://localhost:8231/v1/chat/completions OCR_HUNYUAN_OCR_TIMEOUT_SECONDS=600
uv run python -m bot
```

Перед запуском укажи `BOT_TOKEN` в `.env` в корне проекта. Процесс работает в
текущем терминале; для остановки нажми `Ctrl+C`. `BOT_API_PROXY` нужен в этом
окружении, потому что WSL не выходит к Telegram напрямую, а локальный Windows
proxy доступен по `127.0.0.1:12334`. Если WSL имеет прямой доступ к Telegram,
эту переменную можно убрать.
