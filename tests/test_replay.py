from __future__ import annotations

from dataclasses import replace

import pytest
from support import make_candles, random_walk

from outlier_detector.config import Settings
from outlier_detector.models import Candle, Direction, Stage
from outlier_detector.replay import replay_history, summarize

SETTINGS = Settings(cooldown_minutes=60)
BARS = SETTINGS.history_bars + 1 + 20
BREAKOUT_BAR = SETTINGS.history_bars + 5


def history() -> dict[str, list[Candle]]:
    data = {f"S{i}USDT": make_candles(random_walk(BARS, seed=i).tolist()) for i in range(8)}
    candles = data["S0USDT"]
    # S0 rips 4% above everything it traded in the lookback, on 6x volume, then holds there.
    level = max(c.high for c in candles[:BREAKOUT_BAR]) * 1.04
    for index in range(BREAKOUT_BAR, BARS):
        candles[index] = replace(
            candles[index], close=level, high=level * 1.001, low=level * 0.999, turnover=6e6
        )
    return data


def test_replay_finds_the_breakout_once_per_cooldown() -> None:
    signals = replay_history(history(), SETTINGS)
    assert [s.symbol for s in signals] == ["S0USDT"]
    [signal] = signals
    assert signal.direction is Direction.UP
    assert signal.stage is Stage.CONFIRMED
    bar = history()["S0USDT"][BREAKOUT_BAR].start_ms
    assert signal.bar_start_ms == bar
    assert signal.detected_at_ms == bar + SETTINGS.interval_ms


def test_replay_rejects_misaligned_or_short_history() -> None:
    data = history()
    data["S1USDT"] = data["S1USDT"][1:] + data["S1USDT"][:1]
    with pytest.raises(ValueError, match="aligned"):
        replay_history(data, SETTINGS)
    short = {symbol: candles[: SETTINGS.history_bars + 1] for symbol, candles in history().items()}
    with pytest.raises(ValueError, match="need more"):
        replay_history(short, SETTINGS)
    assert replay_history({}, SETTINGS) == []


def test_summarize() -> None:
    assert summarize([], 2) == "No breakouts."
    text = summarize(replay_history(history(), SETTINGS), 2)
    assert text.startswith("1 breakouts over 2 days (0.5/day): 1 up, 0 down.")
    assert "S0USDT (1)" in text
