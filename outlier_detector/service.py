"""Live service: bootstrap from REST, stream klines, scan, alert.

Two scan paths share one detector:
  * early     - every `scan_interval_seconds` while new ticks arrive, scored
                on the forming bar. This is the fast path.
  * confirmed - once per bar, as soon as every in-sync symbol has closed it
                (or `close_grace_seconds` after the first close, whichever
                comes first), scored on closed bars only.

Everything runs on one event loop; scans are synchronous and never interleave
with state updates, so no locking is needed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import sqlite3
import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from .bybit import BybitClient
from .config import Settings
from .detector import CooldownGate, compute_features, detect, find_breakouts, top_movers
from .market import GapError, MarketData, Snapshot
from .models import Direction, KlineUpdate, Signal, Stage
from .notify import AlertDispatcher, format_duration, format_signal
from .store import SignalStore

LOGGER = logging.getLogger(__name__)

MAX_BACKOFF_SECONDS = 60.0
HEALTHY_SESSION_SECONDS = 300.0


class Service:
    def __init__(
        self,
        settings: Settings,
        client: BybitClient,
        store: SignalStore,
        dispatcher: AlertDispatcher,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self.dispatcher = dispatcher
        self._clock = clock
        cooldown_ms = settings.cooldown_minutes * 60_000
        # Restore cooldowns so a restart doesn't re-fire alerts already sent.
        self.gate = CooldownGate(cooldown_ms, store.last_alerts(since_ms=self._now_ms() - cooldown_ms))
        self._early_bar: dict[tuple[str, Direction], int] = {}
        self._announced = False

        # Per-session state, reset by _start_session().
        self.market: MarketData | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._resyncing: set[str] = set()
        self._dirty = False
        self._last_scanned_close = 0
        self._closing_bar: int | None = None
        self._closing_since = 0.0
        self._bar_closed = asyncio.Event()

    async def run(self) -> None:
        """Run forever, rebuilding state from REST after any stream failure."""
        delay = 1.0
        while True:
            started = time.monotonic()
            try:
                await self.run_session()
            except asyncio.CancelledError:
                raise
            except (ConnectionError, TimeoutError) as exc:
                LOGGER.warning("Stream lost: %s", exc)
            except Exception:
                LOGGER.exception("Session failed")
            if time.monotonic() - started > HEALTHY_SESSION_SECONDS:
                delay = 1.0
            LOGGER.info("Reconnecting in %.0fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, MAX_BACKOFF_SECONDS)

    async def run_session(self) -> None:
        s = self.settings
        symbols = await self.client.resolve_universe(s.symbols, s.universe_size, s.min_turnover_usd)
        market, failed = await self._bootstrap(symbols)
        self._start_session(market)
        if not self._announced:
            self._announced = True
            self.dispatcher.submit(
                f"Outlier detector online · {len(market.symbols)} symbols · "
                f"{format_duration(s.interval_minutes)} bars"
            )
        try:
            for symbol in failed:
                self._start_resync(symbol)
            if s.early_alerts:
                self._spawn(self._scan_loop())
            async for update in self.client.stream(
                market.symbols, s.interval_minutes, stale_seconds=s.stale_stream_seconds
            ):
                self.on_update(update)
            raise ConnectionError("stream ended")
        finally:
            await self._stop_tasks()

    # -- stream handling -------------------------------------------------

    def on_update(self, update: KlineUpdate) -> None:
        market = self._require_market()
        try:
            changed = market.apply(update)
        except GapError as exc:
            LOGGER.warning("%s; resyncing from REST", exc)
            self._start_resync(update.symbol)
            return
        if not changed:
            return
        self._dirty = True
        if update.closed:
            self._on_bar_closed(update.candle.start_ms)

    def _on_bar_closed(self, bar: int) -> None:
        if bar <= self._last_scanned_close:
            return  # straggler for a bar that has already been scanned
        if self._closing_bar is None:
            self._closing_bar = bar
            self._closing_since = time.monotonic()
            self._bar_closed.clear()
            self._spawn(self._close_scan(bar))
        self._check_barrier()

    def _check_barrier(self) -> None:
        """Release the close scan once no symbol can still deliver the closing bar."""
        market = self._require_market()
        if (
            self._closing_bar is not None
            and not self._resyncing
            and market.closed_count(self._closing_bar - market.interval_ms) == 0
        ):
            self._bar_closed.set()

    # -- scans -----------------------------------------------------------

    async def _scan_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.scan_interval_seconds)
            if not self._dirty:
                continue
            self._dirty = False
            try:
                self.scan_live()
            except Exception:
                LOGGER.exception("Live scan failed")

    def scan_live(self) -> None:
        market = self._require_market()
        latest = market.latest_closed_ms
        if latest is None:
            return
        bar = latest + market.interval_ms
        now = self._now_ms()
        snap = market.snapshot_live(bar, (now - bar) / market.interval_ms)
        # Mid close-wave only part of the universe has rolled over; wait for a
        # representative cross-section rather than score a sliver of it.
        if snap is None or not _representative(snap, market):
            return
        for signal in detect(snap, self.settings, Stage.EARLY, now):
            self._emit(signal)

    async def _close_scan(self, bar: int) -> None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._bar_closed.wait(), self.settings.close_grace_seconds)
        self._closing_bar = None
        self._last_scanned_close = bar
        self.scan_closed(bar, waited=time.monotonic() - self._closing_since)

    def scan_closed(self, bar: int, waited: float = 0.0) -> None:
        market = self._require_market()
        snap = market.snapshot_closed(bar)
        if snap is None or not _representative(snap, market):
            LOGGER.warning(
                "Bar %s: only %d/%d symbols closed after %.1fs; skipping confirmed scan",
                _utc(bar),
                0 if snap is None else len(snap),
                market.loaded_count,
                waited,
            )
            return
        features = compute_features(snap, self.settings)
        signals = find_breakouts(snap, features, self.settings, Stage.CONFIRMED, self._now_ms())
        movers = ", ".join(
            f"{sym} {move:+.2f}% ({z:+.1f}σ)" for sym, move, z in top_movers(snap, features, 3)
        )
        LOGGER.info(
            "Bar %s closed: %d/%d symbols in %.1fs | market %+.2f%% | top: %s | %d breakout(s)",
            _utc(bar),
            len(snap),
            market.loaded_count,
            waited,
            math.expm1(features.market_move) * 100,
            movers,
            len(signals),
        )
        for signal in signals:
            self._emit(signal)
        # A symbol that missed an entire close is stuck; reload it.
        for symbol in market.lagging(bar - market.interval_ms):
            LOGGER.warning("%s missed the %s close; resyncing from REST", symbol, _utc(bar))
            self._start_resync(symbol)

    def _emit(self, signal: Signal) -> None:
        if not self.gate.allow(signal):
            return
        key = (signal.symbol, signal.direction)
        held = signal.stage is Stage.CONFIRMED and self._early_bar.get(key) == signal.bar_start_ms
        if signal.stage is Stage.EARLY:
            self._early_bar[key] = signal.bar_start_ms
        s = self.settings
        self.dispatcher.submit(
            format_signal(
                signal,
                impulse_minutes=s.impulse_bars * s.interval_minutes,
                lookback_minutes=s.lookback_bars * s.interval_minutes,
                held=held,
            )
        )
        try:
            self.store.record(signal)
        except sqlite3.Error:
            LOGGER.exception("Failed to record %s %s", signal.symbol, signal.stage)

    # -- REST ------------------------------------------------------------

    async def _bootstrap(self, symbols: list[str]) -> tuple[MarketData, list[str]]:
        """Load every symbol's history; returns the market and the symbols that failed."""
        s = self.settings
        semaphore = asyncio.Semaphore(s.bootstrap_concurrency)

        async def fetch(symbol: str) -> list[Any]:
            async with semaphore:
                return await self.client.fetch_candles(symbol, s.interval_minutes, s.history_bars + 1)

        LOGGER.info("Bootstrapping %d symbols", len(symbols))
        results = await asyncio.gather(*(fetch(symbol) for symbol in symbols), return_exceptions=True)
        history = {}
        failed = []
        for symbol, result in zip(symbols, results, strict=True):
            if isinstance(result, Exception):
                LOGGER.warning("Bootstrap of %s failed (%s); will keep retrying", symbol, result)
                failed.append(symbol)
            elif isinstance(result, BaseException):
                raise result
            else:
                history[symbol] = result
        if len(history) < 2:
            raise RuntimeError(f"bootstrap loaded only {len(history)} symbol(s)")
        # Failed symbols stay in the universe unloaded, and are retried in the background.
        market = MarketData(symbols, s.history_bars, s.interval_ms)
        for symbol, candles in history.items():
            market.load(symbol, candles)
        LOGGER.info("Bootstrap complete: %d/%d symbols", len(history), len(symbols))
        return market, failed

    def _start_resync(self, symbol: str) -> None:
        market = self._require_market()
        market.invalidate(symbol)
        if symbol not in self._resyncing:
            self._resyncing.add(symbol)
            self._spawn(self._resync(market, symbol))

    async def _resync(self, market: MarketData, symbol: str) -> None:
        s = self.settings
        delay = 1.0
        while True:
            try:
                candles = await self.client.fetch_candles(symbol, s.interval_minutes, s.history_bars + 1)
                market.load(symbol, candles)
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                LOGGER.warning("Resync of %s failed (%s); retrying in %.0fs", symbol, exc, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF_SECONDS)
        LOGGER.info("Resynced %s", symbol)
        self._resyncing.discard(symbol)
        # REST may hold a close the stream never delivered (e.g. a gap at a bar
        # boundary): treat it as that close so the bar still gets scanned.
        last_closed = market.last_closed_ms(symbol)
        if last_closed is not None:
            self._on_bar_closed(last_closed)
        self._check_barrier()

    # -- plumbing --------------------------------------------------------

    def _start_session(self, market: MarketData) -> None:
        self.market = market
        self._dirty = False
        self._resyncing = set()
        self._closing_bar = None
        self._bar_closed = asyncio.Event()
        # Never alert on a bar that closed before this session started.
        self._last_scanned_close = market.latest_closed_ms or 0

    def _spawn(self, coro: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            LOGGER.error("Background task failed", exc_info=task.exception())

    async def _stop_tasks(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def _require_market(self) -> MarketData:
        if self.market is None:
            raise RuntimeError("no active session")
        return self.market

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)


def _representative(snap: Snapshot, market: MarketData) -> bool:
    """At least half the loaded universe is in the snapshot."""
    return len(snap) * 2 >= market.loaded_count


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M")
