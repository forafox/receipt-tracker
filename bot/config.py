from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Telegram bot settings loaded from environment variables.

    Attributes:
        token: Secret Telegram bot token read from ``BOT_TOKEN``.
        timezone: Timezone of receipt times read from ``BOT_TIMEZONE``.
        database_path: SQLite database file read from ``BOT_DATABASE_PATH``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BOT_")

    token: SecretStr
    timezone: str = "Europe/Moscow"
    database_path: Path = Path("receipts.sqlite")
