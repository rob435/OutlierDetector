from __future__ import annotations

import pytest

from outlier_detector.config import ConfigError, Settings, load_settings, read_dotenv


def test_defaults_are_valid() -> None:
    settings = Settings()
    assert settings.interval_ms == 900_000
    assert settings.history_bars == settings.lookback_bars + 1
    assert not settings.telegram_enabled


def test_from_env_parses_each_type() -> None:
    settings = Settings.from_env(
        {
            "SYMBOLS": " solusdt, BTCUSDT,solusdt ,",
            "LOOKBACK_BARS": "48",
            "MIN_ZSCORE": "2.5",
            "DETECT_BREAKDOWNS": "off",
            "LOG_LEVEL": "debug",
            "TELEGRAM_BOT_TOKEN": "t",
            "TELEGRAM_CHAT_ID": "c",
            "UNRELATED": "ignored",
            "MIN_RVOL": "  ",
        }
    )
    assert settings.symbols == ("SOLUSDT", "BTCUSDT")
    assert settings.lookback_bars == 48
    assert settings.min_zscore == 2.5
    assert settings.detect_breakdowns is False
    assert settings.log_level == "DEBUG"
    assert settings.min_rvol == Settings().min_rvol
    assert settings.telegram_enabled


@pytest.mark.parametrize(
    ("key", "value"),
    [("LOOKBACK_BARS", "lots"), ("EARLY_ALERTS", "maybe"), ("MIN_ZSCORE", "x")],
)
def test_from_env_rejects_unparseable_values(key: str, value: str) -> None:
    with pytest.raises(ConfigError, match=key):
        Settings.from_env({key: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"interval_minutes": 7},
        {"lookback_bars": 10},
        {"impulse_bars": 0},
        {"impulse_bars": 200},
        {"min_zscore": 0},
        {"min_elapsed_fraction": 0},
        {"close_grace_seconds": 0},
        {"close_grace_seconds": 451},
        {"cooldown_minutes": 14},
        {"log_level": "LOUD"},
    ],
)
def test_validation(overrides: dict[str, object]) -> None:
    with pytest.raises(ConfigError):
        Settings(**overrides)  # type: ignore[arg-type]


def test_read_dotenv(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text("# comment\n\nexport MIN_ZSCORE=4\nTELEGRAM_CHAT_ID=\"-100\"\nBROKEN LINE\nA='b=c'\n")
    assert read_dotenv(path) == {"MIN_ZSCORE": "4", "TELEGRAM_CHAT_ID": "-100", "A": "b=c"}
    assert read_dotenv(tmp_path / "missing") == {}


def test_environment_overrides_dotenv(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("MIN_ZSCORE=4\nMIN_RVOL=3\n")
    monkeypatch.setenv("MIN_ZSCORE", "5")
    settings = load_settings(path)
    assert settings.min_zscore == 5.0
    assert settings.min_rvol == 3.0
