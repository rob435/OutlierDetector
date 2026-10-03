from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from outlier_detector import bybit
from outlier_detector.bybit import (
    BybitClient,
    BybitError,
    check_contiguous,
    parse_kline_rows,
    parse_ws_message,
)
from outlier_detector.models import Candle, KlineUpdate

BAR = 15 * 60_000  # the 15m interval in ms
T0 = 1_700_000_000_000 // BAR * BAR  # a bar start in 2023, far behind the local clock


# --- fakes -----------------------------------------------------------------


def kline_row(start: int) -> list[str]:
    """A REST kline row whose close (100 + bar index) identifies the bar."""
    close = 100.0 + (start - T0) // BAR
    return [str(start), str(close), str(close + 1), str(close - 1), str(close), "7", str(close * 1000)]


def candle(start: int) -> Candle:
    close = 100.0 + (start - T0) // BAR
    return Candle(start_ms=start, high=close + 1, low=close - 1, close=close, turnover=close * 1000)


def ok(result: dict[str, Any], time_ms: int = T0) -> dict[str, Any]:
    return {"retCode": 0, "retMsg": "OK", "result": result, "retExtInfo": {}, "time": time_ms}


def push(symbol: str, start: int, *, confirm: bool = False) -> dict[str, Any]:
    close = 100.0 + (start - T0) // BAR
    item = {
        "start": start,
        "end": start + BAR - 1,
        "interval": "15",
        "open": "99",
        "close": str(close),
        "high": str(close + 1),
        "low": str(close - 1),
        "volume": "7",
        "turnover": str(close * 1000),
        "confirm": confirm,
        "timestamp": start + 1_000,
    }
    return {"topic": f"kline.15.{symbol}", "type": "snapshot", "ts": start + 1_000, "data": [item]}


@contextlib.asynccontextmanager
async def client_for(app: web.Application, *, retries: int = 3) -> AsyncIterator[BybitClient]:
    async with TestServer(app) as server, aiohttp.ClientSession() as session:
        base = str(server.make_url("/")).rstrip("/")
        yield BybitClient(session, base, f"{base}/ws", retries=retries, backoff_seconds=0.001)


Reply = dict[str, Any] | int  # a JSON body, or a bare HTTP status


def scripted(path: str, *replies: Reply) -> tuple[web.Application, list[dict[str, str]]]:
    """Answers GET `path` with `replies` in turn (the last one repeats); records each query."""
    calls: list[dict[str, str]] = []

    async def handler(request: web.Request) -> web.Response:
        calls.append(dict(request.query))
        reply = replies[min(len(calls), len(replies)) - 1]
        if isinstance(reply, int):
            return web.Response(status=reply, text="nope")
        return web.json_response(reply)

    app = web.Application()
    app.router.add_get(path, handler)
    return app, calls


def kline_app(starts: Sequence[int], server_time_ms: int) -> tuple[web.Application, list[dict[str, str]]]:
    """A kline endpoint honouring `limit` and `end` like Bybit: newest first, forming bar included."""
    calls: list[dict[str, str]] = []

    async def handler(request: web.Request) -> web.Response:
        query = dict(request.query)
        calls.append(query)
        end = int(query.get("end", server_time_ms))
        newest_first = sorted((s for s in starts if s <= end), reverse=True)[: int(query["limit"])]
        result = {
            "category": "linear",
            "symbol": query["symbol"],
            "list": [kline_row(s) for s in newest_first],
        }
        return web.json_response(ok(result, server_time_ms))

    app = web.Application()
    app.router.add_get("/v5/market/kline", handler)
    return app, calls


WsResponder = Callable[["FakeBybitWs", dict[str, Any]], Awaitable[None]]


class FakeBybitWs:
    """A Bybit-like public socket; `respond` scripts the reply to each client message."""

    def __init__(self, respond: WsResponder) -> None:
        self.received: list[dict[str, Any]] = []
        self._respond = respond
        self._ws = web.WebSocketResponse()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/ws", self._handler)
        return app

    async def _handler(self, request: web.Request) -> web.WebSocketResponse:
        await self._ws.prepare(request)
        async for message in self._ws:
            payload = json.loads(message.data)
            self.received.append(payload)
            await self._respond(self, payload)
        return self._ws

    def sent(self, op: str) -> list[dict[str, Any]]:
        return [payload for payload in self.received if payload.get("op") == op]

    async def send(self, payload: dict[str, Any]) -> None:
        await self._ws.send_json(payload)

    async def ack(self, request: dict[str, Any], *, success: bool = True, ret_msg: str = "") -> None:
        reply = {"success": success, "ret_msg": ret_msg, "op": "subscribe", "req_id": request["req_id"]}
        await self.send({**reply, "conn_id": "conn-1"})

    async def close(self) -> None:
        await self._ws.close()


async def ack_subscriptions(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
    if payload["op"] == "subscribe":
        await fake.ack(payload)


async def first_update(client: BybitClient, symbols: Sequence[str], **kwargs: Any) -> KlineUpdate:
    async with contextlib.aclosing(client.stream(symbols, 15, **kwargs)) as updates:
        return await anext(updates)


# --- pure parsing ----------------------------------------------------------


def test_parse_kline_rows_drops_forming_bar_sorts_and_dedupes() -> None:
    rows = [kline_row(T0 + 2 * BAR), kline_row(T0 + BAR), kline_row(T0 + BAR), kline_row(T0)]

    mid_bar = parse_kline_rows(rows, server_time_ms=T0 + 2 * BAR + BAR // 2, interval_ms=BAR)
    at_close = parse_kline_rows(rows, server_time_ms=T0 + 3 * BAR, interval_ms=BAR)

    assert mid_bar == [candle(T0), candle(T0 + BAR)]
    assert mid_bar[0].turnover == 100_000.0  # row[6], not the volume
    assert [c.start_ms for c in at_close] == [T0, T0 + BAR, T0 + 2 * BAR]


def test_check_contiguous_names_the_first_gap() -> None:
    check_contiguous([candle(T0), candle(T0 + BAR)], BAR, "BTCUSDT")
    check_contiguous([], BAR, "BTCUSDT")

    with pytest.raises(BybitError, match=rf"BTCUSDT.*{T0 + BAR}.*{T0 + 3 * BAR}"):
        check_contiguous(
            [candle(T0), candle(T0 + BAR), candle(T0 + 3 * BAR), candle(T0 + 5 * BAR)], BAR, "BTCUSDT"
        )


def test_parse_ws_message_maps_kline_pushes() -> None:
    payload = push("1000PEPEUSDT", T0, confirm=True)
    payload["data"].append(push("1000PEPEUSDT", T0 + BAR)["data"][0])

    assert parse_ws_message(payload) == [
        KlineUpdate(symbol="1000PEPEUSDT", candle=candle(T0), closed=True),
        KlineUpdate(symbol="1000PEPEUSDT", candle=candle(T0 + BAR), closed=False),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "ret_msg": "pong", "conn_id": "c", "op": "ping"},
        {"success": True, "ret_msg": "", "op": "subscribe", "req_id": "kline-0", "conn_id": "c"},
        {"topic": "tickers.BTCUSDT", "type": "snapshot", "data": {"symbol": "BTCUSDT"}},
        {},
    ],
)
def test_parse_ws_message_ignores_everything_else(payload: dict[str, Any]) -> None:
    assert parse_ws_message(payload) == []


def test_backoff_doubles_and_is_capped() -> None:
    assert [bybit._backoff_delay(1.0, attempt) for attempt in range(7)] == [1, 2, 4, 8, 16, 30, 30]


# --- REST: _get ------------------------------------------------------------


@pytest.mark.parametrize(
    "transient",
    [
        {"retCode": 10006, "retMsg": "Too many visits!"},
        {"retCode": 10016, "retMsg": "Server error"},
        429,
        500,
    ],
)
async def test_get_retries_transient_failures(transient: Reply) -> None:
    app, calls = scripted("/v5/x", transient, transient, ok({"value": 1}))

    async with client_for(app) as client:
        payload = await client._get("/v5/x", {"category": "linear", "limit": 5})

    assert payload["result"] == {"value": 1}
    assert calls == [{"category": "linear", "limit": "5"}] * 3


async def test_get_retries_timeouts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bybit, "REQUEST_TIMEOUT_SECONDS", 0.05)
    calls = 0

    async def handler(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            await asyncio.sleep(0.3)
        return web.json_response(ok({"value": 1}))

    app = web.Application()
    app.router.add_get("/v5/x", handler)
    async with client_for(app) as client:
        payload = await client._get("/v5/x", {})

    assert payload["result"] == {"value": 1}
    assert calls == 2


@pytest.mark.parametrize(
    ("fatal", "message"),
    [({"retCode": 10001, "retMsg": "params error"}, "retCode 10001"), (403, "HTTP 403")],
)
async def test_get_raises_immediately_on_other_errors(fatal: Reply, message: str) -> None:
    app, calls = scripted("/v5/x", fatal, ok({}))

    async with client_for(app) as client:
        with pytest.raises(BybitError, match=message):
            await client._get("/v5/x", {})

    assert len(calls) == 1


async def test_get_gives_up_after_retries() -> None:
    app, calls = scripted("/v5/x", {"retCode": 10006, "retMsg": "Too many visits!"})

    async with client_for(app, retries=2) as client:
        with pytest.raises(BybitError, match="after 3 attempts"):
            await client._get("/v5/x", {})

    assert len(calls) == 3


# --- REST: fetch_candles ---------------------------------------------------


async def test_fetch_candles_single_page_uses_server_time() -> None:
    # Bar 49 is still forming by the server's clock, though long closed by the local one.
    app, calls = kline_app([T0 + n * BAR for n in range(50)], server_time_ms=T0 + 49 * BAR + BAR // 3)

    async with client_for(app) as client:
        candles = await client.fetch_candles("BTCUSDT", 15, 10)

    assert candles == [candle(T0 + n * BAR) for n in range(39, 49)]
    assert calls == [{"category": "linear", "symbol": "BTCUSDT", "interval": "15", "limit": "11"}]


async def test_fetch_candles_paginates_backwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bybit, "KLINE_PAGE_LIMIT", 4)
    app, calls = kline_app([T0 + n * BAR for n in range(50)], server_time_ms=T0 + 49 * BAR + BAR // 3)

    async with client_for(app) as client:
        candles = await client.fetch_candles("BTCUSDT", 15, 10)

    assert candles == [candle(T0 + n * BAR) for n in range(39, 49)]
    # page 1: bars 49 (forming), 48..46; page 2: 45..42; page 3: 41..38
    assert [call.get("end") for call in calls] == [None, str(T0 + 46 * BAR - 1), str(T0 + 42 * BAR - 1)]
    assert all(int(call["limit"]) <= 4 for call in calls)


async def test_fetch_candles_raises_when_history_is_too_short() -> None:
    app, calls = kline_app([T0 + n * BAR for n in range(6)], server_time_ms=T0 + 5 * BAR + 1)

    async with client_for(app) as client:
        with pytest.raises(BybitError, match="only 5 closed"):
            await client.fetch_candles("NEWUSDT", 15, 10)

    assert len(calls) == 1


async def test_fetch_candles_raises_on_gap() -> None:
    starts = [T0 + n * BAR for n in range(20) if n != 15]
    app, _ = kline_app(starts, server_time_ms=T0 + 19 * BAR + 1)

    async with client_for(app) as client:
        with pytest.raises(BybitError, match="gap"):
            await client.fetch_candles("BTCUSDT", 15, 10)


# --- REST: resolve_universe ------------------------------------------------


def perp(symbol: str, *, quote: str = "USDT", contract: str = "LinearPerpetual") -> dict[str, str]:
    return {"symbol": symbol, "contractType": contract, "quoteCoin": quote, "status": "Trading"}


INSTRUMENT_PAGES = {
    "": (
        [
            perp("BTCUSDT"),
            perp("ETHUSDT"),
            perp("BTCPERP", quote="USDC"),
            perp("BTCUSDT-26DEC25", contract="LinearFutures"),
        ],
        "page-2",
    ),
    "page-2": (
        [perp("SOLUSDT"), perp("XRPUSDT"), perp("DOGEUSDT"), perp("USDCUSDT"), perp("USDEUSDT")],
        "",
    ),
}
TURNOVER_24H = {
    "BTCPERP": 9e10,  # USDC-quoted
    "BTCUSDT-26DEC25": 8e10,  # dated future
    "USDCUSDT": 7e10,  # stablecoin
    "NEWUSDT": 6e10,  # not trading (e.g. pre-launch)
    "USDEUSDT": 5e10,  # stablecoin
    "ETHUSDT": 3e9,
    "XRPUSDT": 5e8,
    "BTCUSDT": 4e9,
    "DOGEUSDT": 5e6,  # below the turnover floor
    "SOLUSDT": 2e9,
}


def universe_app() -> tuple[web.Application, list[tuple[str, dict[str, str]]]]:
    calls: list[tuple[str, dict[str, str]]] = []

    async def instruments(request: web.Request) -> web.Response:
        calls.append((request.path, dict(request.query)))
        items, next_cursor = INSTRUMENT_PAGES[request.query.get("cursor", "")]
        return web.json_response(ok({"category": "linear", "list": items, "nextPageCursor": next_cursor}))

    async def tickers(request: web.Request) -> web.Response:
        calls.append((request.path, dict(request.query)))
        rows = [{"symbol": s, "lastPrice": "1", "turnover24h": str(t)} for s, t in TURNOVER_24H.items()]
        return web.json_response(ok({"category": "linear", "list": rows}))

    app = web.Application()
    app.router.add_get("/v5/market/instruments-info", instruments)
    app.router.add_get("/v5/market/tickers", tickers)
    return app, calls


async def test_resolve_universe_auto_picks_most_traded_listed_perpetuals() -> None:
    app, calls = universe_app()

    async with client_for(app) as client:
        top3 = await client.resolve_universe((), size=3, min_turnover_usd=1e7)
        everything = await client.resolve_universe((), size=50, min_turnover_usd=1e7)

    assert top3 == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert everything == ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    instruments_query = {"category": "linear", "status": "Trading", "limit": "1000"}
    assert calls[:3] == [
        ("/v5/market/instruments-info", instruments_query),
        ("/v5/market/instruments-info", {**instruments_query, "cursor": "page-2"}),
        ("/v5/market/tickers", {"category": "linear"}),
    ]


async def test_resolve_universe_explicit_keeps_listed_symbols_in_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app, calls = universe_app()

    async with client_for(app) as client:
        with caplog.at_level(logging.WARNING, logger=bybit.__name__):
            symbols = await client.resolve_universe(
                ["SOLUSDT", "NEWUSDT", "BTCPERP", "BTCUSDT"], size=1, min_turnover_usd=1e12
            )

    assert symbols == ["SOLUSDT", "BTCUSDT"]
    assert "NEWUSDT, BTCPERP" in caplog.text
    assert all(path != "/v5/market/tickers" for path, _ in calls)


@pytest.mark.parametrize(
    ("symbols", "min_turnover_usd"), [(["BTCUSDT", "NEWUSDT"], 0.0), ((), 1e12)], ids=["explicit", "auto"]
)
async def test_resolve_universe_needs_two_symbols(symbols: Sequence[str], min_turnover_usd: float) -> None:
    app, _ = universe_app()

    async with client_for(app) as client:
        with pytest.raises(BybitError, match="at least 2"):
            await client.resolve_universe(symbols, size=10, min_turnover_usd=min_turnover_usd)


# --- WebSocket: stream -----------------------------------------------------


async def test_stream_subscribes_in_chunks_and_yields_pushes_interleaved_with_acks() -> None:
    symbols = [f"C{n:02d}USDT" for n in range(23)]

    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        if payload["op"] == "subscribe":
            first_symbol = payload["args"][0].rsplit(".", 1)[-1]
            await fake.send(push(first_symbol, T0, confirm=True))  # lands before this chunk's ack
            await fake.ack(payload)

    fake = FakeBybitWs(respond)
    async with client_for(fake.app()) as client, contextlib.aclosing(client.stream(symbols, 15)) as updates:
        received = [await anext(updates) for _ in range(3)]

    assert received == [KlineUpdate(s, candle(T0), closed=True) for s in ("C00USDT", "C10USDT", "C20USDT")]
    subscriptions = fake.sent("subscribe")
    assert [len(message["args"]) for message in subscriptions] == [10, 10, 3]
    assert [arg for message in subscriptions for arg in message["args"]] == [f"kline.15.{s}" for s in symbols]
    assert len({message["req_id"] for message in subscriptions}) == 3


async def test_stream_raises_on_rejected_subscription() -> None:
    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        if payload["op"] == "subscribe":
            await fake.ack(payload, success=False, ret_msg="error:handler not found,topic:kline.15.NOPEUSDT")

    async with client_for(FakeBybitWs(respond).app()) as client:
        with pytest.raises(BybitError, match="handler not found"):
            await first_update(client, ["NOPEUSDT"])


async def test_stream_raises_when_subscription_is_never_acknowledged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bybit, "ACK_TIMEOUT_SECONDS", 0.05)

    async def ignore(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        pass

    async with client_for(FakeBybitWs(ignore).app()) as client:
        with pytest.raises(ConnectionError, match="not acknowledged"):
            await first_update(client, ["BTCUSDT"], stale_seconds=5)


async def test_stream_raises_when_stalled() -> None:
    async with client_for(FakeBybitWs(ack_subscriptions).app()) as client:
        with pytest.raises(ConnectionError, match="stalled"):
            await first_update(client, ["BTCUSDT"], stale_seconds=0.1)


async def test_stream_stall_watchdog_ignores_pongs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bybit, "PING_INTERVAL_SECONDS", 0.01)

    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        await ack_subscriptions(fake, payload)
        if payload["op"] == "ping":  # socket alive, feed dead
            await fake.send({"success": True, "ret_msg": "pong", "conn_id": "conn-1", "op": "ping"})

    async with client_for(FakeBybitWs(respond).app()) as client:
        with pytest.raises(ConnectionError, match="no kline data"):
            await first_update(client, ["BTCUSDT"], stale_seconds=0.2)


async def test_stream_raises_when_server_closes() -> None:
    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        await ack_subscriptions(fake, payload)
        await fake.close()

    async with client_for(FakeBybitWs(respond).app()) as client:
        with pytest.raises(ConnectionError, match="closed"):
            await first_update(client, ["BTCUSDT"])


async def test_stream_raises_connection_error_when_handshake_fails() -> None:
    async with client_for(web.Application()) as client:  # no /ws route -> HTTP 404
        with pytest.raises(ConnectionError, match="cannot connect"):
            await first_update(client, ["BTCUSDT"])


async def test_stream_pings_and_stops_pinging_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bybit, "PING_INTERVAL_SECONDS", 0.01)

    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        await ack_subscriptions(fake, payload)
        if payload["op"] == "ping":
            await fake.send({"success": True, "ret_msg": "pong", "conn_id": "conn-1", "op": "ping"})
            if len(fake.sent("ping")) == 3:
                await fake.send(push("BTCUSDT", T0))

    fake = FakeBybitWs(respond)
    async with client_for(fake.app()) as client:
        update = await first_update(client, ["BTCUSDT"], stale_seconds=5)

    assert update == KlineUpdate("BTCUSDT", candle(T0), closed=False)  # pongs yield nothing
    assert fake.sent("ping")[:3] == [{"op": "ping"}] * 3
    assert not [task for task in asyncio.all_tasks() if task.get_name() == "bybit-ws-ping"]


async def test_stream_propagates_cancellation() -> None:
    subscribed = asyncio.Event()

    async def respond(fake: FakeBybitWs, payload: dict[str, Any]) -> None:
        await ack_subscriptions(fake, payload)
        subscribed.set()

    async with client_for(FakeBybitWs(respond).app()) as client:
        task = asyncio.create_task(first_update(client, ["BTCUSDT"]))
        await subscribed.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert task.cancelled()
