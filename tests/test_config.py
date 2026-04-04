from __future__ import annotations

from pathlib import Path

from config import load_settings


def test_load_settings_reads_dotenv_without_overriding_env(monkeypatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LOG_LEVEL=debug\n"
        "EMERGING_COOLDOWN_MINUTES=15\n"
        "WATCHLIST_COOLDOWN_MINUTES=15\n"
        "BTCDOM_EMA_PERIOD=7\n"
        "TELEGRAM_CHAT_ID=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("EMERGING_COOLDOWN_MINUTES", raising=False)
    monkeypatch.delenv("WATCHLIST_COOLDOWN_MINUTES", raising=False)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "from-env")

    settings = load_settings()

    assert settings.log_level == "DEBUG"
    assert settings.emerging_cooldown_minutes == 15
    assert settings.watchlist_cooldown_minutes == 15
    assert settings.btcdom_ema_period == 7
    assert settings.telegram_chat_id == "from-env"
