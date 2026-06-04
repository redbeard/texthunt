"""Local configuration and secrets.

Secrets come from the environment or a gitignored ``.env`` file — never from source. The Slack token
is held as a :class:`~pydantic.SecretStr` so it stays out of logs, reprs, and tracebacks.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    slack_token: SecretStr


def load_settings() -> Settings:
    """Load settings from the environment and ``.env``.

    The single construction point, so callers never instantiate ``Settings`` directly (and the one
    suppression for the env-populated fields lives here).
    """
    return Settings()  # ty: ignore[missing-argument]
