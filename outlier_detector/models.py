"""Value types shared across the detector."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Stage(StrEnum):
    EARLY = "early"  # fired on the live, still-forming bar
    CONFIRMED = "confirmed"  # fired on a closed bar


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Candle:
    start_ms: int
    high: float
    low: float
    close: float
    turnover: float  # traded value in the quote currency (USDT)


@dataclass(frozen=True, slots=True)
class KlineUpdate:
    symbol: str
    candle: Candle
    closed: bool  # True once the exchange has finalised the bar


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    stage: Stage
    direction: Direction
    bar_start_ms: int
    detected_at_ms: int
    price: float
    move_pct: float  # % move over the impulse window
    market_move_pct: float  # median % move of the universe over the same window
    zscore: float  # market-relative move in units of the symbol's own volatility
    rvol: float  # bar turnover vs. a typical bar, pro-rated while the bar is forming
    level: float  # the range high (UP) or low (DOWN) that was broken
