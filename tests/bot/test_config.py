from pathlib import Path

from bot.config import BotSettings


def test_settings_ignore_other_modules_variables(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BOT_TOKEN=secret\nBOT_TIMEZONE=Asia/Tokyo\nOCR_ENGINE=tesseract\n")

    settings = BotSettings(_env_file=env_file)

    assert settings.token.get_secret_value() == "secret"
    assert settings.timezone == "Asia/Tokyo"


def test_settings_load_telegram_api_proxy(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BOT_TOKEN=secret\nBOT_API_PROXY=http://127.0.0.1:12334\n")

    settings = BotSettings(_env_file=env_file)

    assert settings.api_proxy == "http://127.0.0.1:12334"
