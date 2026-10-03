from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from support import random_walk

from outlier_detector.config import Settings
from outlier_detector.detector import CooldownGate, compute_features, detect, top_movers
from outlier_detector.market import Snapshot
from outlier_detector.models import Direction, Signal, Stage

SETTINGS = Settings()
TYPICAL_TURNOVER = 1_000_000.0


def universe(n: int = 10, elapsed: float = 1.0) -> Snapshot:
    """A quiet random-walk universe; the bar under evaluation is unchanged."""
    bars = SETTINGS.history_bars
    close = np.stack([random_walk(bars, seed=seed) for seed in range(n)])
    return Snapshot(
        symbols=tuple(f"S{i}USDT" for i in range(n)),
        bar_start_ms=1_000 * 900_000,
        elapsed=elapsed,
        close=close,
        high=close * 1.001,
        low=close * 0.999,
        turnover=np.full((n, bars), TYPICAL_TURNOVER),
        price=close[:, -1].copy(),
        bar_turnover=np.full(n, TYPICAL_TURNOVER),
    )


def with_move(snap: Snapshot, row: int, *, factor: float, turnover: float = 5 * TYPICAL_TURNOVER) -> Snapshot:
    """Push one symbol's price `factor` beyond its range high (or low if factor < 1)."""
    price, bar_turnover = snap.price.copy(), snap.bar_turnover.copy()
    level = snap.high[row].max() if factor > 1 else snap.low[row].min()
    price[row] = level * factor
    bar_turnover[row] = turnover
    return replace(snap, price=price, bar_turnover=bar_turnover)


def test_quiet_market_has_no_breakouts() -> None:
    assert detect(universe(), SETTINGS, Stage.CONFIRMED, 0) == []


def test_detects_breakout_with_all_fields() -> None:
    snap = with_move(universe(), 3, factor=1.03)
    [signal] = detect(snap, SETTINGS, Stage.EARLY, now_ms=42)
    assert signal.symbol == "S3USDT"
    assert signal.direction is Direction.UP
    assert signal.stage is Stage.EARLY
    assert signal.bar_start_ms == snap.bar_start_ms
    assert signal.detected_at_ms == 42
    assert signal.price == pytest.approx(snap.price[3])
    assert signal.level == pytest.approx(snap.high[3].max())
    expected_move = (snap.price[3] / snap.close[3, -SETTINGS.impulse_bars] - 1) * 100
    assert signal.move_pct == pytest.approx(expected_move)
    assert signal.zscore >= SETTINGS.min_zscore
    assert signal.rvol == pytest.approx(5.0)


def test_detects_breakdown_and_can_disable_it() -> None:
    snap = with_move(universe(), 2, factor=0.97)
    [signal] = detect(snap, SETTINGS, Stage.CONFIRMED, 0)
    assert signal.direction is Direction.DOWN
    assert signal.zscore <= -SETTINGS.min_zscore
    assert signal.level == pytest.approx(snap.low[2].min())
    assert detect(snap, replace(SETTINGS, detect_breakdowns=False), Stage.CONFIRMED, 0) == []


def test_market_wide_move_is_not_an_outlier() -> None:
    snap = universe()
    pumped = replace(snap, price=snap.close[:, -1] * 1.06, bar_turnover=snap.bar_turnover * 5)
    features = compute_features(pumped, SETTINGS)
    assert np.expm1(features.market_move) > 0.05
    assert np.count_nonzero(pumped.price > features.range_high) >= 5  # most broke their range...
    assert detect(pumped, SETTINGS, Stage.CONFIRMED, 0) == []  # ...but nobody left the pack
    assert len(detect(pumped, replace(SETTINGS, min_zscore=0.01), Stage.CONFIRMED, 0)) > 0


def test_small_universe_falls_back_to_raw_moves() -> None:
    snap = universe(n=3)
    pumped = replace(snap, price=snap.high.max(axis=1) * 1.03, bar_turnover=snap.bar_turnover * 5)
    assert compute_features(pumped, SETTINGS).market_move == 0.0
    assert len(detect(pumped, SETTINGS, Stage.CONFIRMED, 0)) == 3


def test_requires_volume() -> None:
    snap = with_move(universe(), 3, factor=1.03, turnover=1.5 * TYPICAL_TURNOVER)
    assert detect(snap, SETTINGS, Stage.CONFIRMED, 0) == []


def test_requires_minimum_move() -> None:
    snap = universe()
    calm = replace(snap, close=snap.close[:, :1] + 1e-6 * (snap.close - snap.close[:, :1]))
    calm = replace(calm, high=calm.close * 1.00001, low=calm.close * 0.99999, price=calm.close[:, -1].copy())
    # A 0.5% pop is many sigmas for an ultra-calm symbol but below MIN_MOVE_PCT.
    moved = with_move(calm, 0, factor=1.005)
    assert compute_features(moved, SETTINGS).zscore[0] > SETTINGS.min_zscore
    assert detect(moved, SETTINGS, Stage.CONFIRMED, 0) == []
    assert detect(moved, replace(SETTINGS, min_move_pct=0.4), Stage.CONFIRMED, 0) != []


def test_requires_price_beyond_the_range() -> None:
    snap = universe()
    high = snap.high.copy()
    high[3, 10] = snap.high[3].max() * 1.2  # an old spike sets a range high far away
    snap = with_move(replace(snap, high=high), 3, factor=1.0)
    snap = replace(snap, price=np.where(np.arange(len(snap)) == 3, snap.close[3, -1] * 1.05, snap.price))
    assert compute_features(snap, SETTINGS).zscore[3] > SETTINGS.min_zscore
    assert detect(snap, SETTINGS, Stage.CONFIRMED, 0) == []


def test_intrabar_volume_is_pro_rated() -> None:
    quarter = with_move(universe(elapsed=0.25), 3, factor=1.03, turnover=0.75 * TYPICAL_TURNOVER)
    assert compute_features(quarter, SETTINGS).rvol[3] == pytest.approx(3.0)
    assert [s.symbol for s in detect(quarter, SETTINGS, Stage.EARLY, 0)] == ["S3USDT"]
    full = replace(quarter, elapsed=1.0)
    assert detect(full, SETTINGS, Stage.CONFIRMED, 0) == []


def test_young_bar_is_floored() -> None:
    snap = with_move(universe(elapsed=0.0), 3, factor=1.03, turnover=0.4 * TYPICAL_TURNOVER)
    rvol = compute_features(snap, SETTINGS).rvol[3]
    assert rvol == pytest.approx(0.4 / SETTINGS.min_elapsed_fraction)


def test_signals_sorted_by_strength_and_top_movers() -> None:
    snap = with_move(with_move(universe(), 1, factor=1.02), 6, factor=1.06)
    signals = detect(snap, SETTINGS, Stage.CONFIRMED, 0)
    assert [s.symbol for s in signals] == ["S6USDT", "S1USDT"]
    movers = top_movers(snap, compute_features(snap, SETTINGS), 2)
    assert [symbol for symbol, _, _ in movers] == ["S6USDT", "S1USDT"]


def make_signal(stage: Stage = Stage.EARLY, direction: Direction = Direction.UP, at: int = 0) -> Signal:
    return Signal("S1USDT", stage, direction, 0, at, 1.0, 2.0, 0.0, 4.0, 3.0, 0.9)


def test_cooldown_gate() -> None:
    gate = CooldownGate(cooldown_ms=1_000)
    assert gate.allow(make_signal(at=0))
    assert not gate.allow(make_signal(at=999))
    assert gate.allow(make_signal(stage=Stage.CONFIRMED, at=999))  # separate stage
    assert gate.allow(make_signal(direction=Direction.DOWN, at=999))  # separate direction
    assert gate.allow(make_signal(at=1_000))


def test_cooldown_gate_restores_state() -> None:
    gate = CooldownGate(cooldown_ms=1_000, last_alerts={("S1USDT", Direction.UP, Stage.EARLY): 500})
    assert not gate.allow(make_signal(at=1_000))
    assert gate.allow(make_signal(at=1_500))
