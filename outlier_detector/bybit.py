"""Async client for Bybit V5 public market data (USDT linear perpetuals).

REST bootstraps candle history and picks the universe; the kline WebSocket
drives live updates. All timestamps are epoch milliseconds.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import aiohttp

from .models import Candle, KlineUpdate

LOGGER = logging.getLogger(__name__)

CATEGORY = "linear"
KLINE_PAGE_LIMIT = 1000
INSTRUMENTS_PAGE_LIMIT = 1000
REQUEST_TIMEOUT_SECONDS = 15.0
CONNECT_TIMEOUT_SECONDS = 15.0
MAX_BACKOFF_SECONDS = 30.0
RETRYABLE_RET_CODES = frozenset({10006, 10016})  # rate limited, server error
STABLECOIN_BASES = frozenset({"USDC", "USDE", "FDUSD", "DAI", "TUSD", "USDD", "PYUSD", "USD1"})
SUBSCRIBE_CHUNK_SIZE = 10
ACK_TIMEOUT_SECONDS = 10.0
PING_INTERVAL_SECONDS = 20.0  # Bybit's recommended app-level heartbeat

_CLOSED_TYPES = frozenset(
    {aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR}
)


class BybitError(RuntimeError):
    """Bybit rejected a request, or its data breaks this client's guarantees."""


class _TransientError(Exception):
    """A failure worth retrying: network, timeout, throttling or a server-side error."""


def parse_kline_rows(rows: list[list[str]], server_time_ms: int, interval_ms: int) -> list[Candle]:
    """Closed candles, oldest first, from REST kline rows (newest first, incl. the forming bar)."""
    closed: dict[int, Candle] = {}
    for row in rows:
        start = int(row[0])
        if start + interval_ms <= server_time_ms:
            closed[start] = Candle(
                start_ms=start,
                high=float(row[2]),
                low=float(row[3]),
                close=float(row[4]),
                turnover=float(row[6]),
            )
    return [closed[start] for start in sorted(closed)]


def check_contiguous(candles: Sequence[Candle], interval_ms: int, symbol: str) -> None:
    for previous, current in itertools.pairwise(candles):
        if current.start_ms - previous.start_ms != interval_ms:
            raise BybitError(
                f"{symbol}: candle gap between {previous.start_ms} and {current.start_ms} "
                f"(expected a {interval_ms} ms step)"
            )


def parse_ws_message(payload: dict) -> list[KlineUpdate]:
    """Kline updates in a WebSocket push; [] for pongs, acks and other topics."""
    topic = payload.get("topic")
    if not isinstance(topic, str) or not topic.startswith("kline."):
        return []
    symbol = topic.rsplit(".", 1)[-1]
    return [
        KlineUpdate(symbol=symbol, candle=_ws_candle(item), closed=bool(item.get("confirm")))
        for item in payload.get("data") or ()
    ]


def _ws_candle(item: Mapping[str, Any]) -> Candle:
    return Candle(
        start_ms=int(item["start"]),
        high=float(item["high"]),
        low=float(item["low"]),
        close=float(item["close"]),
        turnover=float(item["turnover"]),
    )


def _backoff_delay(base_seconds: float, attempt: int) -> float:
    return min(base_seconds * 2**attempt, MAX_BACKOFF_SECONDS)


def _is_stablecoin(symbol: str) -> bool:
    return symbol.removesuffix("USDT") in STABLECOIN_BASES


class BybitClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        rest_url: str,
        ws_url: str,
        *,
        retries: int = 5,
        backoff_seconds: float = 1.0,
    ) -> None:
        if retries < 0:
            raise ValueError(f"retries must be >= 0, got {retries}")
        self._session = session
        self._rest_url = rest_url.rstrip("/")
        self._ws_url = ws_url
        self._retries = retries
        self._backoff_seconds = backoff_seconds
        self._timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    # REST -----------------------------------------------------------------

    async def _get(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        """The full payload of a successful (retCode 0) GET, retrying transient failures."""
        attempt = 0
        while True:
            try:
                return await self._request(path, params)
            except _TransientError as exc:
                if attempt >= self._retries:
                    raise BybitError(f"GET {path} failed after {attempt + 1} attempts: {exc}") from exc
                delay = _backoff_delay(self._backoff_seconds, attempt)
                LOGGER.warning(
                    "GET %s failed (%s); retry %d/%d in %.3gs", path, exc, attempt + 1, self._retries, delay
                )
                await asyncio.sleep(delay)
            attempt += 1

    async def _request(self, path: str, params: Mapping[str, str | int]) -> dict[str, Any]:
        try:
            async with self._session.get(
                self._rest_url + path, params=params, timeout=self._timeout
            ) as response:
                if response.status == 429 or response.status >= 500:
                    raise _TransientError(f"HTTP {response.status}")
                if response.status >= 400:
                    body = await response.text()
                    raise BybitError(f"GET {path}: HTTP {response.status}: {body[:200]}")
                payload = await response.json(content_type=None)
        except (aiohttp.ClientError, TimeoutError, json.JSONDecodeError) as exc:
            raise _TransientError(repr(exc)) from exc
        if not isinstance(payload, dict):
            raise BybitError(f"GET {path}: unexpected payload {payload!r:.200}")
        ret_code = payload.get("retCode")
        if ret_code == 0:
            return payload
        message = f"GET {path} {dict(params)}: retCode {ret_code} {payload.get('retMsg')!r}"
        if ret_code in RETRYABLE_RET_CODES:
            raise _TransientError(message)
        raise BybitError(message)

    async def fetch_candles(self, symbol: str, interval_minutes: int, count: int) -> list[Candle]:
        """The last `count` closed candles, oldest first and contiguous."""
        if count < 1:
            raise ValueError(f"count must be >= 1, got {count}")
        interval_ms = interval_minutes * 60_000
        by_start: dict[int, Candle] = {}
        end: int | None = None
        while len(by_start) < count:
            # +1: the newest row of the first page is usually the still-forming bar.
            limit = min(KLINE_PAGE_LIMIT, count - len(by_start) + 1)
            params: dict[str, str | int] = {
                "category": CATEGORY,
                "symbol": symbol,
                "interval": str(interval_minutes),
                "limit": limit,
            }
            if end is not None:
                params["end"] = end
            payload = await self._get("/v5/market/kline", params)
            rows = payload["result"]["list"]
            known = len(by_start)
            for candle in parse_kline_rows(rows, int(payload["time"]), interval_ms):
                by_start[candle.start_ms] = candle
            if len(rows) < limit or len(by_start) == known:
                break  # reached the listing date (or the server ignored `end`)
            end = min(int(row[0]) for row in rows) - 1
        candles = [by_start[start] for start in sorted(by_start)][-count:]
        if len(candles) < count:
            raise BybitError(
                f"{symbol}: only {len(candles)} closed {interval_minutes}m candles available, need {count}"
            )
        check_contiguous(candles, interval_ms, symbol)
        return candles

    async def resolve_universe(self, symbols: Sequence[str], size: int, min_turnover_usd: float) -> list[str]:
        """The symbols to track: the listed subset of `symbols`, or else the most traded perpetuals."""
        listed = await self._listed_perpetuals()
        if symbols:
            requested = list(dict.fromkeys(symbols))
            chosen = [symbol for symbol in requested if symbol in listed]
            dropped = [symbol for symbol in requested if symbol not in listed]
            if dropped:
                LOGGER.warning(
                    "Ignoring symbols that are not trading USDT perpetuals: %s", ", ".join(dropped)
                )
        else:
            chosen = await self._most_traded(listed, size, min_turnover_usd)
        if len(chosen) < 2:
            raise BybitError(f"universe needs at least 2 symbols, got {chosen}")
        return chosen

    async def _listed_perpetuals(self) -> set[str]:
        listed: set[str] = set()
        seen_cursors: set[str] = set()
        cursor = ""
        while True:
            params: dict[str, str | int] = {
                "category": CATEGORY,
                "status": "Trading",
                "limit": INSTRUMENTS_PAGE_LIMIT,
            }
            if cursor:
                params["cursor"] = cursor
            result = (await self._get("/v5/market/instruments-info", params))["result"]
            listed.update(
                item["symbol"]
                for item in result["list"]
                if item.get("contractType") == "LinearPerpetual" and item.get("quoteCoin") == "USDT"
            )
            cursor = result.get("nextPageCursor") or ""
            if not cursor or cursor in seen_cursors:
                return listed
            seen_cursors.add(cursor)

    async def _most_traded(self, listed: set[str], size: int, min_turnover_usd: float) -> list[str]:
        tickers = (await self._get("/v5/market/tickers", {"category": CATEGORY}))["result"]["list"]
        ranked: list[tuple[float, str]] = []
        for ticker in tickers:
            symbol = ticker["symbol"]
            if symbol not in listed or _is_stablecoin(symbol):
                continue
            turnover = float(ticker.get("turnover24h") or 0.0)
            if turnover >= min_turnover_usd:
                ranked.append((turnover, symbol))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [symbol for _, symbol in ranked[:size]]

    # WebSocket ------------------------------------------------------------

    async def stream(
        self, symbols: Sequence[str], interval_minutes: int, *, stale_seconds: float = 60.0
    ) -> AsyncIterator[KlineUpdate]:
        """Live kline updates until the connection fails; the caller reconnects.

        Raises ConnectionError when the socket closes, delivers no kline data for
        `stale_seconds` (pongs don't count: a live socket with a dead feed is still
        dead) or the subscriptions are not acknowledged in time, and BybitError when
        Bybit rejects a subscription.
        """
        if not symbols:
            raise ValueError("stream needs at least one symbol")
        topics = [f"kline.{interval_minutes}.{symbol}" for symbol in symbols]
        loop = asyncio.get_running_loop()
        async with await self._connect() as ws:
            pending_acks = await _subscribe(ws, topics)
            ack_deadline = loop.time() + ACK_TIMEOUT_SECONDS
            data_deadline = loop.time() + stale_seconds
            pinger = asyncio.create_task(_ping_forever(ws), name="bybit-ws-ping")
            try:
                while True:
                    deadline = data_deadline
                    reason = f"stream stalled: no kline data for {stale_seconds:g}s"
                    if pending_acks and ack_deadline < deadline:
                        deadline = ack_deadline
                        reason = f"subscriptions {sorted(pending_acks)} not acknowledged in time"
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        raise ConnectionError(reason)
                    try:
                        payload = await _receive_json(ws, timeout)
                    except TimeoutError:
                        raise ConnectionError(reason) from None
                    if payload.get("op") == "subscribe":
                        _check_ack(payload, pending_acks)
                        if not pending_acks:
                            LOGGER.info("Subscribed to %d kline topics", len(topics))
                        continue
                    updates = parse_ws_message(payload)
                    if updates:
                        data_deadline = loop.time() + stale_seconds
                    for update in updates:
                        yield update
            finally:
                pinger.cancel()
                # gather (unlike suppress(CancelledError)) still propagates our own cancellation.
                await asyncio.gather(pinger, return_exceptions=True)

    async def _connect(self) -> aiohttp.ClientWebSocketResponse:
        try:
            async with asyncio.timeout(CONNECT_TIMEOUT_SECONDS):
                ws = await self._session.ws_connect(self._ws_url, autoping=True)
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise ConnectionError(f"cannot connect to {self._ws_url}: {exc!r}") from exc
        LOGGER.info("Connected to %s", self._ws_url)
        return ws


async def _subscribe(ws: aiohttp.ClientWebSocketResponse, topics: Sequence[str]) -> set[str]:
    """Send the subscriptions in chunks; returns the req_ids awaiting an ack."""
    pending: set[str] = set()
    for offset in range(0, len(topics), SUBSCRIBE_CHUNK_SIZE):
        req_id = f"kline-{offset // SUBSCRIBE_CHUNK_SIZE}"
        args = list(topics[offset : offset + SUBSCRIBE_CHUNK_SIZE])
        await ws.send_json({"op": "subscribe", "req_id": req_id, "args": args})
        pending.add(req_id)
    return pending


def _check_ack(payload: Mapping[str, Any], pending: set[str]) -> None:
    if payload.get("success") is not True:
        raise BybitError(f"subscription rejected: {payload.get('ret_msg') or payload}")
    pending.discard(str(payload.get("req_id", "")))


async def _ping_forever(ws: aiohttp.ClientWebSocketResponse) -> None:
    while True:
        await asyncio.sleep(PING_INTERVAL_SECONDS)
        await ws.send_json({"op": "ping"})


async def _receive_json(ws: aiohttp.ClientWebSocketResponse, timeout: float) -> dict[str, Any]:
    """The next JSON object from the server ({} for frames to ignore); TimeoutError if none arrives."""
    async with asyncio.timeout(timeout):
        message = await ws.receive()
    if message.type in _CLOSED_TYPES:
        raise ConnectionError(f"websocket closed ({message.type.name}): {message.data!r}")
    if message.type is not aiohttp.WSMsgType.TEXT:
        return {}
    try:
        payload = json.loads(message.data)
    except json.JSONDecodeError:
        LOGGER.warning("Ignoring non-JSON websocket message: %.200s", message.data)
        return {}
    return payload if isinstance(payload, dict) else {}
