from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, replace

import pytest
from support import INTERVAL_MS, make_candles, random_walk

from outlier_detector.config import Settings
from outlier_detector.models import Candle, Direction, KlineUpdate, Signal, Stage
from outlier_detector.service import Service
from outlier_detector.store import SignalStore

SYMBOLS = [f"S{i}USDT" for i in range(6)]
SETTINGS = Settings(scan_interval_seconds=0.01, close_grace_seconds=5.0)
BARS = SETTINGS.history_bars + 1
B1 = BARS * INTERVAL_MS  # first bar after bootstrap
B2 = B1 + INTERVAL_MS


class FakeExchange:
    def __init__(self) -> None:
        self.history = {s: make_candles(random_walk(BARS, seed=i).tolist()) for i, s in enumerate(SYMBOLS)}
        self.updates: asyncio.Queue[KlineUpdate | Exception] = asyncio.Queue()
        self.fetches: list[str] = []

    async def resolve_universe(self, symbols: Sequence[str], size: int, min_turnover_usd: float) -> list[str]:
        return list(self.history)

    async def fetch_candles(self, symbol: str, interval_minutes: int, count: int) -> list[Candle]:
        self.fetches.append(symbol)
        return self.history[symbol][-count:]

    async def stream(
        self, symbols: Sequence[str], interval_minutes: int, *, stale_seconds: float
    ) -> AsyncIterator[KlineUpdate]:
        while True:
            item = await self.updates.get()
            if isinstance(item, Exception):
                raise item
            yield item

    def last_close(self, symbol: str) -> float:
        return self.history[symbol][-1].close

    def breakout_price(self, symbol: str) -> float:
        return max(c.high for c in self.history[symbol]) * 1.04

    def push(self, symbol: str, start: int, close: float, *, turnover: float = 1e6, closed: bool) -> None:
        self.updates.put_nowait(KlineUpdate(symbol, Candle(start, close, close, close, turnover), closed))

    def push_bar(self, start: int, *, closed: bool, skip: Sequence[str] = (), **overrides: float) -> None:
        for symbol in SYMBOLS:
            if symbol not in skip:
                self.push(
                    symbol,
                    start,
                    overrides.get(symbol, self.last_close(symbol)),
                    closed=closed,
                    turnover=5e6 if symbol in overrides else 1e6,
                )


class Recorder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def submit(self, text: str) -> None:
        self.texts.append(text)

    def find(self, needle: str) -> list[str]:
        return [text for text in self.texts if needle in text]


class Clock:
    def __init__(self, ms: int) -> None:
        self.ms = ms

    def __call__(self) -> float:
        return self.ms / 1000


async def wait_until(predicate: Callable[[], object], timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.005)


@dataclass
class Harness:
    exchange: FakeExchange
    recorder: Recorder
    clock: Clock
    store: SignalStore
    service: Service


@pytest.fixture
async def harness() -> AsyncIterator[Harness]:
    exchange, recorder, clock = FakeExchange(), Recorder(), Clock(B1 + INTERVAL_MS // 2)
    store = SignalStore(":memory:")
    service = Service(SETTINGS, exchange, store, recorder, clock=clock)  # type: ignore[arg-type]
    task = asyncio.create_task(service.run_session())
    await wait_until(lambda: service.market is not None)
    yield Harness(exchange, recorder, clock, store, service)
    exchange.updates.put_nowait(ConnectionError("test over"))
    with pytest.raises(ConnectionError):
        await task
    store.close()


async def test_early_alert_then_confirmed_on_close(harness: Harness) -> None:
    h = harness
    assert h.recorder.texts == ["Outlier detector online · 6 symbols · 15m bars"]
    price = h.exchange.breakout_price("S0USDT")

    h.exchange.push_bar(B1, closed=False, S0USDT=price)
    await wait_until(lambda: h.recorder.find("S0USDT breakout · early"))

    # Every symbol closes: the barrier fires immediately, well inside the 5s grace.
    h.clock.ms = B2 + 1_000
    h.exchange.push_bar(B1, closed=True, S0USDT=price)
    await wait_until(lambda: h.recorder.find("S0USDT breakout · confirmed (held into close)"), timeout=1.0)

    # Still breaking out next bar, but inside the cooldown: nothing new.
    h.clock.ms = B2 + INTERVAL_MS // 2
    h.exchange.push_bar(B2, closed=False, S0USDT=price * 1.02)
    await asyncio.sleep(0.1)
    assert len(h.recorder.texts) == 3
    assert [(s.symbol, s.stage) for s in h.store.recent()] == [
        ("S0USDT", Stage.CONFIRMED),
        ("S0USDT", Stage.EARLY),
    ]


async def test_gap_triggers_rest_resync(harness: Harness) -> None:
    h = harness
    # REST already has B1 closed; the stream jumps straight to B2.
    h.exchange.history["S1USDT"] = make_candles(random_walk(BARS, seed=1).tolist(), start_ms=INTERVAL_MS)
    h.exchange.push("S1USDT", B2, 100.0, closed=False)  # B1's close was never seen
    await wait_until(lambda: "S1USDT" in h.exchange.fetches[len(SYMBOLS) :])
    await wait_until(lambda: h.service.market.is_loaded("S1USDT"))
    assert h.service.market.closed_count(B1) == 1


async def test_close_scan_runs_after_grace_and_resyncs_laggards(harness: Harness) -> None:
    h = harness
    h.service.settings = Settings(scan_interval_seconds=0.01, close_grace_seconds=0.1)
    price = h.exchange.breakout_price("S0USDT")
    h.clock.ms = B2 + 1_000
    h.exchange.push_bar(B1, closed=True, skip=["S5USDT"], S0USDT=price)
    await wait_until(lambda: h.recorder.find("S0USDT breakout · confirmed"))
    assert "(held into close)" not in h.recorder.find("S0USDT breakout · confirmed")[0]

    h.clock.ms = B2 + INTERVAL_MS + 1_000
    h.exchange.push_bar(B2, closed=True, skip=["S5USDT"], S0USDT=price)
    await wait_until(lambda: "S5USDT" in h.exchange.fetches[len(SYMBOLS) :])


async def test_restart_restores_cooldowns() -> None:
    store = SignalStore(":memory:")
    now = B1 + INTERVAL_MS // 2
    signal = Signal("S0USDT", Stage.EARLY, Direction.UP, B1, now - 60_000, 1.0, 5.0, 0.0, 6.0, 4.0, 0.9)
    store.record(signal)
    service = Service(SETTINGS, FakeExchange(), store, Recorder(), clock=Clock(now))  # type: ignore[arg-type]
    assert not service.gate.allow(replace(signal, detected_at_ms=now))
