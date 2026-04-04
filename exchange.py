from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

import aiohttp

from config import Settings

LOGGER = logging.getLogger(__name__)


RATE_LIMIT_RETCODE = 10006


class MissingCandlesError(RuntimeError):
    pass


def is_rate_limited_payload(payload: dict) -> bool:
    return payload.get("retCode") == RATE_LIMIT_RETCODE


def interval_to_milliseconds(interval: str) -> int:
    interval = interval.strip()
    if interval == "D":
        return 24 * 60 * 60 * 1000
    if interval == "W":
        return 7 * 24 * 60 * 60 * 1000
    if interval == "M":
        return 30 * 24 * 60 * 60 * 1000
    match = re.fullmatch(r"(?i)(\d+)([mhdw])", interval)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        factors = {
            "m": 60 * 1000,
            "h": 60 * 60 * 1000,
            "d": 24 * 60 * 60 * 1000,
            "w": 7 * 24 * 60 * 60 * 1000,
        }
        return value * factors[unit]
    return int(interval) * 60 * 1000


@dataclass(slots=True)
class BootstrapPayload:
    price_history: dict[str, list[tuple[int, float]]]
    btc_daily_history: list[tuple[int, float]]
    btcdom_history: list[tuple[int, float]]


class BybitMarketDataClient:
    def __init__(self, session: aiohttp.ClientSession, settings: Settings) -> None:
        self.session = session
        self.settings = settings
        self._rest_timeout = aiohttp.ClientTimeout(total=20)

    async def _get_json(self, path: str, params: dict[str, str | int]) -> dict:
        url = f"{self.settings.bybit_rest_base_url.rstrip('/')}{path}"
        for attempt in range(self.settings.rate_limit_retries + 1):
            async with self.session.get(url, params=params, timeout=self._rest_timeout) as response:
                response.raise_for_status()
                payload = await response.json()
            if payload.get("retCode") == 0:
                return payload
            if is_rate_limited_payload(payload) and attempt < self.settings.rate_limit_retries:
                delay = self.settings.rate_limit_backoff_seconds * (2 ** attempt)
                LOGGER.warning(
                    "Bybit rate limit hit for %s with params=%s. Retrying in %.1fs (%s/%s)",
                    path,
                    params,
                    delay,
                    attempt + 1,
                    self.settings.rate_limit_retries,
                )
                await asyncio.sleep(delay)
                continue
            raise RuntimeError(f"Bybit error: {payload}")
        raise RuntimeError(f"Bybit error: exhausted retries for {path} {params}")

    async def _get_binance_json(self, path: str, params: dict[str, str | int]) -> list | dict:
        url = f"{self.settings.binance_futures_base_url.rstrip('/')}{path}"
        retry_statuses = {418, 429}
        for attempt in range(self.settings.rate_limit_retries + 1):
            async with self.session.get(url, params=params, timeout=self._rest_timeout) as response:
                if response.status in retry_statuses and attempt < self.settings.rate_limit_retries:
                    delay = self.settings.rate_limit_backoff_seconds * (2 ** attempt)
                    LOGGER.warning(
                        "Binance rate limit hit for %s with params=%s. Retrying in %.1fs (%s/%s)",
                        path,
                        params,
                        delay,
                        attempt + 1,
                        self.settings.rate_limit_retries,
                    )
                    await asyncio.sleep(delay)
                    continue
                response.raise_for_status()
                payload = await response.json()
            if isinstance(payload, dict) and payload.get("code") not in (None, 0):
                raise RuntimeError(f"Binance error: {payload}")
            return payload
        raise RuntimeError(f"Binance error: exhausted retries for {path} {params}")

    async def fetch_closed_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        category: str | None = None,
    ) -> list[tuple[int, float]]:
        category = category or self.settings.bybit_category
        now_ms = int(time.time() * 1000)
        interval_ms = interval_to_milliseconds(interval)
        attempts = [limit + 1, limit + 4, min(1000, limit + 16)]

        for request_limit in attempts:
            payload = await self._get_json(
                "/v5/market/kline",
                {
                    "category": category,
                    "symbol": symbol,
                    "interval": interval,
                    "limit": request_limit,
                },
            )
            rows = payload["result"]["list"]
            candles = []
            for row in rows:
                start_time = int(row[0])
                close_price = float(row[4])
                if start_time + interval_ms <= now_ms:
                    candles.append((start_time, close_price))
            candles.sort(key=lambda item: item[0])
            if len(candles) >= limit:
                candles = candles[-limit:]
                self._ensure_contiguous(candles, interval_ms, symbol, interval)
                return candles
        raise MissingCandlesError(
            f"Unable to fetch {limit} closed {interval} candles for {symbol}"
        )

    def _ensure_contiguous(
        self,
        candles: list[tuple[int, float]],
        interval_ms: int,
        symbol: str,
        interval: str,
    ) -> None:
        for current, nxt in zip(candles, candles[1:]):
            if nxt[0] - current[0] != interval_ms:
                raise MissingCandlesError(
                    f"Missing {interval} candles for {symbol}: {current[0]} -> {nxt[0]}"
                )

    async def fetch_btcdom_klines(self) -> list[tuple[int, float]]:
        symbol = self.settings.btcdom_symbol.strip().upper()
        if symbol.endswith(".P"):
            symbol = symbol[:-2]
        payload = await self._get_binance_json(
            "/fapi/v1/klines",
            {
                "symbol": symbol,
                "interval": self.settings.btcdom_interval,
                "limit": self.settings.btcdom_history_lookback + 4,
            },
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Binance error: unexpected payload for {symbol}: {payload}")
        interval_ms = interval_to_milliseconds(self.settings.btcdom_interval)
        now_ms = int(time.time() * 1000)
        candles: list[tuple[int, float]] = []
        for row in payload:
            start_time = int(row[0])
            close_price = float(row[4])
            if start_time + interval_ms <= now_ms:
                candles.append((start_time, close_price))
        candles.sort(key=lambda item: item[0])
        if len(candles) < self.settings.btcdom_history_lookback:
            raise MissingCandlesError(
                f"Unable to fetch {self.settings.btcdom_history_lookback} closed "
                f"{self.settings.btcdom_interval} candles for {symbol}"
            )
        candles = candles[-self.settings.btcdom_history_lookback :]
        self._ensure_contiguous(candles, interval_ms, symbol, self.settings.btcdom_interval)
        return candles

    async def bootstrap(self) -> BootstrapPayload:
        semaphore = asyncio.Semaphore(self.settings.bootstrap_concurrency)
        price_history: dict[str, list[tuple[int, float]]] = {}

        async def load_symbol(symbol: str) -> None:
            async with semaphore:
                candles = await self.fetch_closed_klines(
                    symbol=symbol,
                    interval=self.settings.candle_interval,
                    limit=self.settings.state_window,
                )
                price_history[symbol] = candles

        await asyncio.gather(*(load_symbol(symbol) for symbol in self.settings.tracked_symbols))
        btc_daily_history = await self.fetch_closed_klines(
            symbol="BTCUSDT",
            interval="D",
            limit=self.settings.btc_daily_lookback,
        )
        btcdom_history = await self.fetch_btcdom_klines()
        return BootstrapPayload(
            price_history=price_history,
            btc_daily_history=btc_daily_history,
            btcdom_history=btcdom_history,
        )

    async def stream_candles(
        self,
        symbols: list[str],
        on_provisional_candle,
        on_closed_candle,
        on_emerging_event,
        on_confirmed_event,
    ) -> None:
        topics = [f"kline.{self.settings.candle_interval}.{symbol}" for symbol in symbols]
        subscribe_message = {"op": "subscribe", "args": topics}
        ping_message = json.dumps({"op": "ping"})

        async with self.session.ws_connect(
            self.settings.bybit_ws_base_url,
            heartbeat=self.settings.websocket_ping_seconds,
            timeout=30,
        ) as websocket:
            LOGGER.info("Connected to Bybit WebSocket for %s symbols", len(symbols))
            await websocket.send_json(subscribe_message)
            LOGGER.info("Subscribed to kline topics")
            while True:
                try:
                    message = await websocket.receive(timeout=self.settings.websocket_ping_seconds)
                except asyncio.TimeoutError:
                    await websocket.send_str(ping_message)
                    continue

                if message.type == aiohttp.WSMsgType.TEXT:
                    payload = json.loads(message.data)
                    topic = payload.get("topic", "")
                    if not topic.startswith("kline."):
                        continue
                    for candle in payload.get("data", []):
                        symbol = topic.split(".")[-1]
                        close_time_ms = int(candle["start"])
                        close_price = float(candle["close"])
                        if candle.get("confirm"):
                            appended = on_closed_candle(symbol, close_time_ms, close_price)
                            if appended:
                                on_confirmed_event(close_time_ms)
                        else:
                            appended = on_provisional_candle(symbol, close_time_ms, close_price)
                            if appended:
                                on_emerging_event(close_time_ms)
                elif message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}:
                    raise ConnectionError("WebSocket disconnected")
                elif message.type == aiohttp.WSMsgType.CLOSE:
                    raise ConnectionError("WebSocket closed")
