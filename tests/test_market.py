from __future__ import annotations

import numpy as np
import pytest
from support import INTERVAL_MS, make_candles

from outlier_detector.market import GapError, MarketData
from outlier_detector.models import Candle, KlineUpdate

HISTORY = 4  # columns = HISTORY + 1


def loaded_market(symbols=("AAAUSDT", "BBBUSDT")) -> MarketData:
    market = MarketData(symbols, HISTORY, INTERVAL_MS)
    for offset, symbol in enumerate(symbols):
        market.load(symbol, make_candles([100.0 + offset + i for i in range(HISTORY + 3)]))
    return market


def bar(index: int, close: float, turnover: float = 5.0) -> Candle:
    return Candle(index * INTERVAL_MS, close + 1, close - 1, close, turnover)


def test_load_keeps_latest_window() -> None:
    market = loaded_market()
    snap = market.snapshot_closed((HISTORY + 2) * INTERVAL_MS)
    assert snap is not None
    assert snap.symbols == ("AAAUSDT", "BBBUSDT")
    assert snap.close.shape == (2, HISTORY)
    np.testing.assert_allclose(snap.close[0], [102, 103, 104, 105])
    np.testing.assert_allclose(snap.price, [106, 107])
    assert snap.elapsed == 1.0
    assert market.loaded_count == 2
    assert market.latest_closed_ms == (HISTORY + 2) * INTERVAL_MS


def test_load_validates_input() -> None:
    market = MarketData(["AAAUSDT"], HISTORY, INTERVAL_MS)
    with pytest.raises(ValueError, match="need"):
        market.load("AAAUSDT", make_candles([1.0] * HISTORY))
    gappy = make_candles([1.0] * (HISTORY + 1))
    gappy[2] = Candle(gappy[2].start_ms + 1, 1, 1, 1, 1)
    with pytest.raises(ValueError, match="contiguous"):
        market.load("AAAUSDT", gappy)
    with pytest.raises(ValueError, match="duplicate"):
        MarketData(["A", "A"], HISTORY, INTERVAL_MS)


def test_closed_update_rolls_the_window() -> None:
    market = loaded_market()
    assert market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 200.0), closed=True))
    snap = market.snapshot_closed((HISTORY + 3) * INTERVAL_MS)
    assert snap is not None and snap.symbols == ("AAAUSDT",)
    np.testing.assert_allclose(snap.close[0], [103, 104, 105, 106])
    assert snap.price[0] == 200.0
    assert snap.bar_turnover[0] == 5.0
    assert market.closed_count((HISTORY + 3) * INTERVAL_MS) == 1
    assert market.lagging((HISTORY + 3) * INTERVAL_MS) == ["BBBUSDT"]


def test_stale_duplicate_and_unknown_updates_are_ignored() -> None:
    market = loaded_market()
    assert not market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 2, 1.0), closed=True))
    assert not market.apply(KlineUpdate("AAAUSDT", bar(1, 1.0), closed=False))
    assert not market.apply(KlineUpdate("ZZZUSDT", bar(HISTORY + 3, 1.0), closed=True))


def test_skipped_bar_raises_gap() -> None:
    market = loaded_market()
    with pytest.raises(GapError) as info:
        market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 4, 1.0), closed=False))
    assert info.value.symbol == "AAAUSDT"


def test_invalidated_symbol_is_excluded_until_reloaded() -> None:
    market = loaded_market()
    market.invalidate("AAAUSDT")
    assert not market.is_loaded("AAAUSDT")
    assert not market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 1.0), closed=True))
    snap = market.snapshot_closed((HISTORY + 2) * INTERVAL_MS)
    assert snap is not None and snap.symbols == ("BBBUSDT",)
    market.load("AAAUSDT", make_candles([1.0] * (HISTORY + 3)))
    assert market.is_loaded("AAAUSDT")


def test_live_snapshot_uses_latest_tick_and_aligns_symbols() -> None:
    market = loaded_market(("AAAUSDT", "BBBUSDT", "CCCUSDT"))
    forming = (HISTORY + 3) * INTERVAL_MS
    market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 150.0, turnover=9.0), closed=False))
    market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 151.0, turnover=12.0), closed=False))
    # CCC closes the forming bar early, so it is no longer in sync for it.
    market.apply(KlineUpdate("CCCUSDT", bar(HISTORY + 3, 99.0), closed=True))

    snap = market.snapshot_live(forming, elapsed=0.5)
    assert snap is not None
    assert snap.symbols == ("AAAUSDT", "BBBUSDT")
    np.testing.assert_allclose(snap.close[0], [103, 104, 105, 106])
    np.testing.assert_allclose(snap.price, [151.0, 107.0])  # BBB hasn't ticked: last close
    np.testing.assert_allclose(snap.bar_turnover, [12.0, 0.0])
    assert snap.elapsed == 0.5
    assert market.snapshot_live(forming, elapsed=7).elapsed == 1.0


def test_close_clears_live_tick() -> None:
    market = loaded_market()
    market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 150.0), closed=False))
    market.apply(KlineUpdate("AAAUSDT", bar(HISTORY + 3, 140.0), closed=True))
    snap = market.snapshot_live((HISTORY + 4) * INTERVAL_MS, elapsed=0.1)
    assert snap is not None
    assert snap.price[0] == 140.0 and snap.bar_turnover[0] == 0.0
