"""Settings, loaded from environment variables (optionally via a `.env` file).

Every field maps to an upper-cased environment variable of the same name,
e.g. `min_zscore` <- `MIN_ZSCORE`. Values are parsed by the type of the
field's default and validated once at construction.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

BYBIT_INTERVALS = frozenset({1, 3, 5, 15, 30, 60, 120, 240, 360, 720})
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Settings:
    # Universe. Empty `symbols` = auto: the `universe_size` most-traded USDT
    # perpetuals with at least `min_turnover_usd` of 24h turnover.
    symbols: tuple[str, ...] = ()
    universe_size: int = 100
    min_turnover_usd: float = 10_000_000.0

    # Detection. A breakout needs every gate to pass:
    #   price beyond the `lookback_bars` high/low,
    #   |move over `impulse_bars`| >= `min_move_pct`,
    #   market-relative move >= `min_zscore` sigmas of the symbol's own volatility,
    #   bar turnover >= `min_rvol` x the typical bar (pro-rated while forming).
    interval_minutes: int = 15
    lookback_bars: int = 96
    impulse_bars: int = 4
    min_zscore: float = 3.0
    min_move_pct: float = 1.0
    min_rvol: float = 2.0
    min_elapsed_fraction: float = 0.2  # floor for pro-rating a young bar's volume
    detect_breakdowns: bool = True
    early_alerts: bool = True
    cooldown_minutes: int = 240

    # Runtime
    scan_interval_seconds: float = 0.5
    close_grace_seconds: float = 10.0
    stale_stream_seconds: float = 60.0
    bootstrap_concurrency: int = 10
    bybit_rest_url: str = "https://api.bybit.com"
    bybit_ws_url: str = "wss://stream.bybit.com/v5/public/linear"

    # Output
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    sqlite_path: str = "signals.sqlite3"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        checks = [
            (
                self.interval_minutes in BYBIT_INTERVALS,
                f"INTERVAL_MINUTES must be one of {sorted(BYBIT_INTERVALS)}",
            ),
            (self.lookback_bars >= 20, "LOOKBACK_BARS must be >= 20"),
            (1 <= self.impulse_bars <= self.lookback_bars, "IMPULSE_BARS must be in [1, LOOKBACK_BARS]"),
            (self.universe_size >= 2, "UNIVERSE_SIZE must be >= 2"),
            (self.min_turnover_usd >= 0, "MIN_TURNOVER_USD must be >= 0"),
            (self.min_zscore > 0, "MIN_ZSCORE must be > 0"),
            (self.min_move_pct >= 0, "MIN_MOVE_PCT must be >= 0"),
            (self.min_rvol >= 0, "MIN_RVOL must be >= 0"),
            (0 < self.min_elapsed_fraction <= 1, "MIN_ELAPSED_FRACTION must be in (0, 1]"),
            (self.cooldown_minutes >= 0, "COOLDOWN_MINUTES must be >= 0"),
            (self.scan_interval_seconds > 0, "SCAN_INTERVAL_SECONDS must be > 0"),
            (
                0 <= self.close_grace_seconds < self.interval_minutes * 60,
                "CLOSE_GRACE_SECONDS must be in [0, bar length)",
            ),
            (self.stale_stream_seconds > 0, "STALE_STREAM_SECONDS must be > 0"),
            (self.bootstrap_concurrency >= 1, "BOOTSTRAP_CONCURRENCY must be >= 1"),
            (
                self.log_level in {"DEBUG", "INFO", "WARNING", "ERROR"},
                "LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR",
            ),
        ]
        errors = [message for ok, message in checks if not ok]
        if errors:
            raise ConfigError("; ".join(errors))

    @property
    def interval_ms(self) -> int:
        return self.interval_minutes * 60_000

    @property
    def history_bars(self) -> int:
        """Closed bars needed before the bar under evaluation."""
        return self.lookback_bars + 1

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        for field in fields(cls):
            raw = env.get(field.name.upper())
            if raw is None or raw.strip() == "":
                continue
            values[field.name] = _parse(field.name.upper(), raw.strip(), field.default)
        return cls(**values)  # type: ignore[arg-type]


def _parse(key: str, raw: str, default: object) -> object:
    try:
        if isinstance(default, bool):
            lowered = raw.lower()
            if lowered in _TRUE:
                return True
            if lowered in _FALSE:
                return False
            raise ValueError(raw)
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, tuple):
            return tuple(dict.fromkeys(s.strip().upper() for s in raw.split(",") if s.strip()))
        return raw.upper() if key == "LOG_LEVEL" else raw
    except ValueError:
        raise ConfigError(f"{key}: cannot parse {raw!r} as {type(default).__name__}") from None


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Parse a minimal KEY=VALUE file. Missing file -> empty dict."""
    path = Path(path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_settings(dotenv_path: str | Path = ".env") -> Settings:
    """Real environment variables take precedence over the `.env` file."""
    return Settings.from_env({**read_dotenv(dotenv_path), **os.environ})
