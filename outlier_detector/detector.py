"""Breakout scoring: pure, vectorised over the whole cross-section.

For each symbol, with sigma = stdev of its last `lookback_bars` bar returns:

    move     = log(price / close `impulse_bars` ago)
    zscore   = (move - median move of the universe) / (sigma * sqrt(horizon))
    rvol     = bar turnover / (median bar turnover * fraction of bar elapsed)

A breakout is a symbol trading beyond its `lookback_bars` high (or low) with
a big, market-relative, high-volume move. Subtracting the universe median
means a market-wide pump is not an outlier; a coin leaving the pack is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .market import Snapshot
from .models import Direction, Signal, Stage

MIN_SIGMA = 1e-4  # per-bar log-return floor so flat-lined symbols can't divide by ~0
MIN_CROSS_SECTION = 5  # below this the universe median is meaningless; use raw moves


@dataclass(frozen=True, slots=True)
class Features:
    move: np.ndarray
    market_move: float
    zscore: np.ndarray
    rvol: np.ndarray
    range_high: np.ndarray
    range_low: np.ndarray


def compute_features(snap: Snapshot, settings: Settings) -> Features:
    lookback, impulse = settings.lookback_bars, settings.impulse_bars
    log_close = np.log(snap.close[:, -(lookback + 1) :])
    sigma = np.maximum(np.diff(log_close, axis=1).std(axis=1, ddof=1), MIN_SIGMA)

    move = np.log(snap.price) - log_close[:, -impulse]
    market_move = float(np.median(move)) if len(snap) >= MIN_CROSS_SECTION else 0.0
    # The impulse window spans (impulse - 1) closed bars plus the current one;
    # flooring the current bar's share stops a seconds-old bar from inflating z.
    pace = max(snap.elapsed, settings.min_elapsed_fraction)
    zscore = (move - market_move) / (sigma * math.sqrt(impulse - 1 + pace))

    typical_turnover = np.median(snap.turnover[:, -lookback:], axis=1)
    rvol = snap.bar_turnover / (np.maximum(typical_turnover, 1e-9) * pace)

    return Features(
        move=move,
        market_move=market_move,
        zscore=zscore,
        rvol=rvol,
        range_high=snap.high[:, -lookback:].max(axis=1),
        range_low=snap.low[:, -lookback:].min(axis=1),
    )


def find_breakouts(
    snap: Snapshot, features: Features, settings: Settings, stage: Stage, now_ms: int
) -> list[Signal]:
    move_pct = np.expm1(features.move) * 100.0
    common = (np.abs(move_pct) >= settings.min_move_pct) & (features.rvol >= settings.min_rvol)
    up = common & (snap.price > features.range_high) & (features.zscore >= settings.min_zscore)
    down = common & (snap.price < features.range_low) & (features.zscore <= -settings.min_zscore)
    if not settings.detect_breakdowns:
        down[:] = False

    market_move_pct = math.expm1(features.market_move) * 100.0
    signals = [
        Signal(
            symbol=snap.symbols[row],
            stage=stage,
            direction=Direction.UP if up[row] else Direction.DOWN,
            bar_start_ms=snap.bar_start_ms,
            detected_at_ms=now_ms,
            price=float(snap.price[row]),
            move_pct=float(move_pct[row]),
            market_move_pct=market_move_pct,
            zscore=float(features.zscore[row]),
            rvol=float(features.rvol[row]),
            level=float(features.range_high[row] if up[row] else features.range_low[row]),
        )
        for row in np.flatnonzero(up | down)
    ]
    signals.sort(key=lambda signal: abs(signal.zscore), reverse=True)
    return signals


def detect(snap: Snapshot, settings: Settings, stage: Stage, now_ms: int) -> list[Signal]:
    return find_breakouts(snap, compute_features(snap, settings), settings, stage, now_ms)


def top_movers(snap: Snapshot, features: Features, count: int) -> list[tuple[str, float, float]]:
    """(symbol, move %, zscore) for the `count` largest absolute z-scores."""
    order = np.argsort(-np.abs(features.zscore))[:count]
    return [
        (snap.symbols[row], float(np.expm1(features.move[row]) * 100.0), float(features.zscore[row]))
        for row in order
    ]


class CooldownGate:
    """At most one alert per (symbol, direction, stage) per cooldown window.

    Early and confirmed alerts are tracked separately so an early alert never
    suppresses the confirmation of the same breakout.
    """

    def __init__(self, cooldown_ms: int, last_alerts: dict[tuple[str, Direction, Stage], int] | None = None):
        self.cooldown_ms = cooldown_ms
        self._last = dict(last_alerts or {})

    def allow(self, signal: Signal) -> bool:
        key = (signal.symbol, signal.direction, signal.stage)
        last = self._last.get(key)
        if last is not None and signal.detected_at_ms - last < self.cooldown_ms:
            return False
        self._last[key] = signal.detected_at_ms
        return True
