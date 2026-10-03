"""Replay recent history bar by bar through the live detector.

Only closed bars exist in history, so this reproduces the `confirmed` path;
early alerts fire within the bar and cannot be replayed from klines.
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from collections.abc import Mapping, Sequence

from .bybit import BybitClient
from .config import Settings
from .detector import CooldownGate, detect
from .market import MarketData
from .models import Candle, KlineUpdate, Signal, Stage

LOGGER = logging.getLogger(__name__)


def replay_history(history: Mapping[str, Sequence[Candle]], settings: Settings) -> list[Signal]:
    """Run every bar after the warm-up window through detection + cooldowns.

    `history` must be aligned: every symbol has candles with identical starts.
    """
    if not history:
        return []
    warmup = settings.history_bars + 1
    reference = next(iter(history.values()))
    starts = [candle.start_ms for candle in reference]
    for symbol, candles in history.items():
        if [candle.start_ms for candle in candles] != starts:
            raise ValueError(f"{symbol}: candles are not aligned with the rest of the universe")
    if len(starts) <= warmup:
        raise ValueError(f"need more than {warmup} candles per symbol, got {len(starts)}")

    market = MarketData(list(history), settings.history_bars, settings.interval_ms)
    for symbol, candles in history.items():
        market.load(symbol, candles[:warmup])
    gate = CooldownGate(settings.cooldown_minutes * 60_000)
    signals: list[Signal] = []
    for index in range(warmup, len(starts)):
        for symbol, candles in history.items():
            market.apply(KlineUpdate(symbol, candles[index], closed=True))
        bar = starts[index]
        snap = market.snapshot_closed(bar)
        if snap is None:
            continue
        # Stamp signals at the moment the bar closed, as the live service would.
        signals.extend(
            signal
            for signal in detect(snap, settings, Stage.CONFIRMED, bar + settings.interval_ms)
            if gate.allow(signal)
        )
    return signals


async def fetch_history(client: BybitClient, settings: Settings, days: float) -> dict[str, list[Candle]]:
    bars = max(1, round(days * 1440 / settings.interval_minutes))
    count = settings.history_bars + 1 + bars
    symbols = await client.resolve_universe(
        settings.symbols, settings.universe_size, settings.min_turnover_usd
    )
    semaphore = asyncio.Semaphore(settings.bootstrap_concurrency)

    async def fetch(symbol: str) -> list[Candle]:
        async with semaphore:
            return await client.fetch_candles(symbol, settings.interval_minutes, count)

    LOGGER.info("Fetching %d bars for %d symbols", count, len(symbols))
    results = await asyncio.gather(*(fetch(symbol) for symbol in symbols), return_exceptions=True)
    history: dict[str, list[Candle]] = {}
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, Exception):
            LOGGER.warning("Skipping %s: %s", symbol, result)
        elif isinstance(result, BaseException):
            raise result
        else:
            history[symbol] = result
    if not history:
        return {}
    # A bar may close mid-fetch; cut everyone back to the common latest bar.
    end = min(candles[-1].start_ms for candles in history.values())
    aligned = {
        symbol: [candle for candle in candles if candle.start_ms <= end][-count:]
        for symbol, candles in history.items()
    }
    return {symbol: candles for symbol, candles in aligned.items() if len(candles) == count}


def summarize(signals: Sequence[Signal], days: float) -> str:
    if not signals:
        return "No breakouts."
    directions = Counter(signal.direction.value for signal in signals)
    symbols = Counter(signal.symbol for signal in signals).most_common(5)
    return (
        f"{len(signals)} breakouts over {days:g} days ({len(signals) / days:.1f}/day): "
        f"{directions.get('up', 0)} up, {directions.get('down', 0)} down. "
        f"Most active: {', '.join(f'{symbol} ({count})' for symbol, count in symbols)}"
    )
