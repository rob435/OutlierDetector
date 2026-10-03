"""Synthetic market data for tests."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from outlier_detector.models import Candle

INTERVAL_MS = 15 * 60_000


def make_candles(
    closes: Sequence[float],
    *,
    start_ms: int = 0,
    turnover: float = 1_000_000.0,
    spread: float = 0.001,
) -> list[Candle]:
    return [
        Candle(
            start_ms=start_ms + i * INTERVAL_MS,
            high=close * (1 + spread),
            low=close * (1 - spread),
            close=close,
            turnover=turnover,
        )
        for i, close in enumerate(closes)
    ]


def random_walk(bars: int, *, seed: int, vol: float = 0.003, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.exp(np.cumsum(rng.normal(0.0, vol, bars)))
