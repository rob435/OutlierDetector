"""Rolling candle history for the whole universe, stored column-wise.

Each symbol owns one row of fixed-width numpy arrays holding its most recent
closed bars (oldest -> newest), plus the latest tick of the bar currently
forming. Snapshots slice out the symbols that are in sync for a given bar so
the detector can score the whole cross-section in a handful of vector ops.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .models import Candle, KlineUpdate

_NOT_LOADED = -1


class GapError(RuntimeError):
    def __init__(self, symbol: str, expected_ms: int, got_ms: int) -> None:
        super().__init__(f"{symbol}: expected bar {expected_ms}, got {got_ms}")
        self.symbol = symbol


@dataclass(frozen=True, slots=True)
class Snapshot:
    """The cross-section for one bar.

    History arrays are (n_symbols, history_bars) and end at the bar *before*
    `bar_start_ms`; `price`/`bar_turnover` describe the bar under evaluation.
    """

    symbols: tuple[str, ...]
    bar_start_ms: int
    elapsed: float  # fraction of the bar that has elapsed; 1.0 once closed
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    turnover: np.ndarray
    price: np.ndarray
    bar_turnover: np.ndarray

    def __len__(self) -> int:
        return len(self.symbols)


class MarketData:
    def __init__(self, symbols: Sequence[str], history_bars: int, interval_ms: int) -> None:
        if len(set(symbols)) != len(symbols):
            raise ValueError("duplicate symbols")
        self.symbols = tuple(symbols)
        self.history_bars = history_bars
        self.interval_ms = interval_ms
        self._index = {symbol: row for row, symbol in enumerate(self.symbols)}

        # One extra column so a just-closed bar can be evaluated against the
        # `history_bars` that precede it.
        shape = (len(self.symbols), history_bars + 1)
        self._close = np.full(shape, np.nan)
        self._high = np.full(shape, np.nan)
        self._low = np.full(shape, np.nan)
        self._turnover = np.zeros(shape)
        self._last_start = np.full(len(self.symbols), _NOT_LOADED, dtype=np.int64)
        self._live_start = np.full(len(self.symbols), _NOT_LOADED, dtype=np.int64)
        self._live_close = np.full(len(self.symbols), np.nan)
        self._live_turnover = np.zeros(len(self.symbols))

    @property
    def loaded_count(self) -> int:
        return int(np.count_nonzero(self._last_start != _NOT_LOADED))

    @property
    def latest_closed_ms(self) -> int | None:
        """Start time of the newest closed bar across the universe."""
        latest = int(self._last_start.max(initial=_NOT_LOADED))
        return None if latest == _NOT_LOADED else latest

    def is_loaded(self, symbol: str) -> bool:
        return self._last_start[self._index[symbol]] != _NOT_LOADED

    def last_closed_ms(self, symbol: str) -> int | None:
        last = int(self._last_start[self._index[symbol]])
        return None if last == _NOT_LOADED else last

    def closed_count(self, bar_start_ms: int) -> int:
        return int(np.count_nonzero(self._last_start == bar_start_ms))

    def lagging(self, before_ms: int) -> list[str]:
        """Loaded symbols whose newest closed bar is older than `before_ms`."""
        rows = np.flatnonzero((self._last_start != _NOT_LOADED) & (self._last_start < before_ms))
        return [self.symbols[row] for row in rows]

    def load(self, symbol: str, candles: Sequence[Candle]) -> None:
        """Replace a symbol's history with its most recent closed candles."""
        width = self.history_bars + 1
        if len(candles) < width:
            raise ValueError(f"{symbol}: need {width} candles, got {len(candles)}")
        candles = candles[-width:]
        starts = np.fromiter((c.start_ms for c in candles), dtype=np.int64, count=width)
        if np.any(np.diff(starts) != self.interval_ms):
            raise ValueError(f"{symbol}: candles are not contiguous")
        row = self._index[symbol]
        self._close[row] = [c.close for c in candles]
        self._high[row] = [c.high for c in candles]
        self._low[row] = [c.low for c in candles]
        self._turnover[row] = [c.turnover for c in candles]
        self._last_start[row] = starts[-1]
        self._clear_live(row)

    def invalidate(self, symbol: str) -> None:
        """Exclude a symbol from snapshots until it is reloaded."""
        row = self._index[symbol]
        self._last_start[row] = _NOT_LOADED
        self._clear_live(row)

    def apply(self, update: KlineUpdate) -> bool:
        """Fold a stream update into state. Returns True for a new tick or bar.

        Raises GapError when the update skips a bar, i.e. a close was missed.
        """
        row = self._index.get(update.symbol)
        if row is None or self._last_start[row] == _NOT_LOADED:
            return False
        candle = update.candle
        last = int(self._last_start[row])
        if candle.start_ms == last and update.closed:
            # REST can catch a bar a moment before it is final; the stream's
            # close is authoritative.
            self._write_last(row, candle)
            return False
        if candle.start_ms <= last:
            return False  # late update for a bar we already hold
        expected = last + self.interval_ms
        if candle.start_ms != expected:
            raise GapError(update.symbol, expected, candle.start_ms)

        if update.closed:
            for column in (self._close, self._high, self._low, self._turnover):
                column[row, :-1] = column[row, 1:]
            self._write_last(row, candle)
            self._last_start[row] = candle.start_ms
            self._clear_live(row)
        else:
            self._live_start[row] = candle.start_ms
            self._live_close[row] = candle.close
            self._live_turnover[row] = candle.turnover
        return True

    def snapshot_closed(self, bar_start_ms: int) -> Snapshot | None:
        """Symbols whose bar `bar_start_ms` has closed, scored on that bar."""
        rows = np.flatnonzero(self._last_start == bar_start_ms)
        if rows.size == 0:
            return None
        return Snapshot(
            symbols=tuple(self.symbols[row] for row in rows),
            bar_start_ms=bar_start_ms,
            elapsed=1.0,
            close=self._close[rows, :-1],
            high=self._high[rows, :-1],
            low=self._low[rows, :-1],
            turnover=self._turnover[rows, :-1],
            price=self._close[rows, -1],
            bar_turnover=self._turnover[rows, -1],
        )

    def snapshot_live(self, bar_start_ms: int, elapsed: float) -> Snapshot | None:
        """Symbols in sync for the forming bar `bar_start_ms`, scored on their last tick.

        A symbol that has not ticked yet this bar is included at its last close
        with zero turnover: it hasn't moved, which the market median should see.
        """
        rows = np.flatnonzero(self._last_start == bar_start_ms - self.interval_ms)
        if rows.size == 0:
            return None
        ticked = self._live_start[rows] == bar_start_ms
        return Snapshot(
            symbols=tuple(self.symbols[row] for row in rows),
            bar_start_ms=bar_start_ms,
            elapsed=min(max(elapsed, 0.0), 1.0),
            close=self._close[rows, 1:],
            high=self._high[rows, 1:],
            low=self._low[rows, 1:],
            turnover=self._turnover[rows, 1:],
            price=np.where(ticked, self._live_close[rows], self._close[rows, -1]),
            bar_turnover=np.where(ticked, self._live_turnover[rows], 0.0),
        )

    def _write_last(self, row: int, candle: Candle) -> None:
        self._close[row, -1] = candle.close
        self._high[row, -1] = candle.high
        self._low[row, -1] = candle.low
        self._turnover[row, -1] = candle.turnover

    def _clear_live(self, row: int) -> None:
        self._live_start[row] = _NOT_LOADED
        self._live_close[row] = np.nan
        self._live_turnover[row] = 0.0
