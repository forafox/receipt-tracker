from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotSettings(BaseSettings):
    """Telegram bot settings loaded from environment variables.

    Attributes:
        token: Secret Telegram bot token read from ``BOT_TOKEN``.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BOT_")

    token: SecretStr
