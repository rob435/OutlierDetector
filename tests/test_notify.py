from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestServer

from outlier_detector import notify
from outlier_detector.models import Direction, Signal, Stage
from outlier_detector.notify import (
    AlertDispatcher,
    TelegramNotifier,
    format_duration,
    format_price,
    format_signal,
)

TOKEN = "123456:SECRET-token"
CHAT_ID = "-100777"

# ---------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("minutes", "expected"),
    [(0, "0m"), (15, "15m"), (60, "1h"), (90, "1h30m"), (240, "4h"), (1440, "24h"), (1445, "24h5m")],
)
def test_format_duration(minutes: int, expected: str) -> None:
    assert format_duration(minutes) == expected


@pytest.mark.parametrize(
    ("price", "expected"),
    [
        (152.34, "152.340"),
        (64123.5, "64123.5"),
        (1.234, "1.23400"),
        (1.0, "1.00000"),
        (0.5, "0.500000"),
        (0.00001234, "0.0000123400"),
        (0.00000098765, "0.000000987650"),
        (123456.0, "123456"),
        (98765432.1, "98765432"),
        (99.99999, "100.000"),
        (0.0, "0"),
    ],
)
def test_format_price(price: float, expected: str) -> None:
    assert format_price(price) == expected


@pytest.mark.parametrize("price", [1e-5, 1.5e-7, 3e-9, 1e12])
def test_format_price_never_uses_scientific_notation(price: float) -> None:
    assert "e" not in format_price(price).lower()


def make_signal(**changes: Any) -> Signal:
    values: dict[str, Any] = {
        "symbol": "SOLUSDT",
        "stage": Stage.EARLY,
        "direction": Direction.UP,
        "bar_start_ms": 1_700_000_000_000,
        "detected_at_ms": 1_700_000_100_000,
        "price": 152.34,
        "move_pct": 3.2,
        "market_move_pct": 0.4,
        "zscore": 4.1,
        "rvol": 5.3,
        "level": 150.1,
    }
    return Signal(**{**values, **changes})


def test_format_signal_breakout() -> None:
    text = format_signal(make_signal(), impulse_minutes=60, lookback_minutes=1440)

    assert text == (
        "▲ SOLUSDT breakout · early\n"
        "152.340  +3.20% in 1h (market +0.40%)\n"
        "4.1σ · 5.3× volume · above 24h high 150.100\n"
        "22:15:00 UTC · https://www.bybit.com/trade/usdt/SOLUSDT"
    )


def test_format_signal_held_breakdown() -> None:
    signal = make_signal(
        symbol="WIFUSDT",
        stage=Stage.CONFIRMED,
        direction=Direction.DOWN,
        price=1.234,
        move_pct=-4.1,
        market_move_pct=-0.2,
        zscore=-3.6,
        rvol=2.1,
        level=1.29,
    )

    text = format_signal(signal, impulse_minutes=60, lookback_minutes=1440, held=True)

    assert text == (
        "▼ WIFUSDT breakdown · confirmed (held into close)\n"
        "1.23400  -4.10% in 1h (market -0.20%)\n"
        "3.6σ · 2.1× volume · below 24h low 1.29000\n"
        "22:15:00 UTC · https://www.bybit.com/trade/usdt/WIFUSDT"
    )


def test_format_signal_confirmed_without_hold_uses_windows() -> None:
    signal = make_signal(stage=Stage.CONFIRMED)

    lines = format_signal(signal, impulse_minutes=90, lookback_minutes=360).splitlines()

    assert lines[0] == "▲ SOLUSDT breakout · confirmed"
    assert lines[1].endswith("in 1h30m (market +0.40%)")
    assert lines[2].endswith("above 6h high 150.100")


# ---------------------------------------------------------------- telegram


class FakeTelegram:
    """Scripted Bot API: replies with `script` in order, repeating the last entry."""

    url: str  # set once the server is listening
    session: aiohttp.ClientSession

    def __init__(self) -> None:
        self.script: list[tuple[int, dict[str, Any] | str]] = [(200, {"ok": True})]
        self.requests: list[dict[str, Any]] = []
        self.app = web.Application()
        self.app.router.add_post(f"/bot{TOKEN}/sendMessage", self.handle)

    async def handle(self, request: web.Request) -> web.Response:
        self.requests.append(await request.json())
        status, body = self.script[min(len(self.requests), len(self.script)) - 1]
        if isinstance(body, str):
            return web.Response(status=status, text=body)
        return web.json_response(body, status=status)

    def notifier(self, attempts: int = 3) -> TelegramNotifier:
        return TelegramNotifier(self.session, TOKEN, CHAT_ID, api_url=self.url, attempts=attempts)


@pytest_asyncio.fixture
async def telegram() -> AsyncIterator[FakeTelegram]:
    fake = FakeTelegram()
    async with TestServer(fake.app) as server, aiohttp.ClientSession() as session:
        fake.url = str(server.make_url("/"))
        fake.session = session
        yield fake


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr(notify, "_sleep", fake_sleep)
    return delays


# Explicit marks keep these working whether pytest-asyncio runs in auto or strict mode.
@pytest.mark.asyncio
class TestTelegramNotifier:
    async def test_success(self, telegram: FakeTelegram, sleeps: list[float]) -> None:
        assert await telegram.notifier().send("hello\nworld") is True

        assert telegram.requests == [
            {"chat_id": CHAT_ID, "text": "hello\nworld", "disable_web_page_preview": True},
        ]
        assert sleeps == []

    async def test_honours_retry_after(self, telegram: FakeTelegram, sleeps: list[float]) -> None:
        telegram.script = [
            (429, {"ok": False, "description": "Too Many Requests", "parameters": {"retry_after": 7}}),
            (200, {"ok": True}),
        ]

        assert await telegram.notifier().send("hi") is True

        assert len(telegram.requests) == 2
        assert sleeps == [7.0]

    async def test_caps_retry_after(self, telegram: FakeTelegram, sleeps: list[float]) -> None:
        telegram.script = [(429, {"ok": False, "parameters": {"retry_after": 600}}), (200, {"ok": True})]

        assert await telegram.notifier().send("hi") is True

        assert sleeps == [30.0]

    async def test_retry_after_zero_needs_no_patching(self, telegram: FakeTelegram) -> None:
        telegram.script = [(429, {"ok": False, "parameters": {"retry_after": 0}}), (200, {"ok": True})]

        assert await telegram.notifier().send("hi") is True
        assert len(telegram.requests) == 2

    async def test_gives_up_on_persistent_server_error(
        self, telegram: FakeTelegram, sleeps: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        telegram.script = [(500, "<html>upstream exploded</html>")]

        assert await telegram.notifier(attempts=3).send("hi") is False

        assert len(telegram.requests) == 3
        assert sleeps == [0.5, 1.0]
        assert "giving up after 3 attempts" in caplog.text
        assert TOKEN not in caplog.text

    async def test_does_not_retry_bad_request(
        self, telegram: FakeTelegram, sleeps: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        telegram.script = [(400, {"ok": False, "description": "Bad Request: chat not found"})]

        assert await telegram.notifier().send("hi") is False

        assert len(telegram.requests) == 1
        assert sleeps == []
        assert "chat not found" in caplog.text
        assert TOKEN not in caplog.text

    async def test_network_error_returns_false(
        self, sleeps: list[float], caplog: pytest.LogCaptureFixture
    ) -> None:
        async with TestServer(web.Application()) as server:
            url = str(server.make_url("/"))  # closed on exit, so connections are refused
        async with aiohttp.ClientSession() as session:
            notifier = TelegramNotifier(session, TOKEN, CHAT_ID, api_url=url, attempts=2)
            assert await notifier.send("hi") is False

        assert sleeps == [0.5]
        assert "ClientConnectorError" in caplog.text
        assert TOKEN not in caplog.text


# ---------------------------------------------------------------- dispatcher


class RecordingNotifier:
    def __init__(self, fail_on: frozenset[str] = frozenset()) -> None:
        self.sent: list[str] = []
        self.fail_on = fail_on

    async def send(self, text: str) -> bool:
        await asyncio.sleep(0)
        if text in self.fail_on:
            raise RuntimeError("boom")
        self.sent.append(text)
        return True


Start = Callable[[AlertDispatcher], AlertDispatcher]


@pytest_asyncio.fixture
async def running() -> AsyncIterator[Start]:
    """Starts a dispatcher's worker; all workers are cancelled at teardown."""
    tasks: list[asyncio.Task[None]] = []

    def start(dispatcher: AlertDispatcher) -> AlertDispatcher:
        tasks.append(asyncio.create_task(dispatcher.run()))
        return dispatcher

    yield start
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
class TestAlertDispatcher:
    async def test_delivers_in_order(self, running: Start, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=notify.__name__)
        notifier = RecordingNotifier()
        dispatcher = running(AlertDispatcher(notifier))

        for text in ["one\nline two", "two", "three"]:
            dispatcher.submit(text)
        await dispatcher.drain(timeout=1.0)

        assert notifier.sent == ["one\nline two", "two", "three"]
        assert "one | line two" in caplog.text

    async def test_survives_notifier_errors(self, running: Start, caplog: pytest.LogCaptureFixture) -> None:
        notifier = RecordingNotifier(fail_on=frozenset({"bad"}))
        dispatcher = running(AlertDispatcher(notifier))

        dispatcher.submit("bad")
        dispatcher.submit("good")
        await dispatcher.drain(timeout=1.0)

        assert notifier.sent == ["good"]
        assert "notifier crashed" in caplog.text

    async def test_without_notifier_only_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.INFO, logger=notify.__name__)
        dispatcher = AlertDispatcher(None)

        dispatcher.submit("▲ SOLUSDT breakout\nsecond line")
        await asyncio.wait_for(dispatcher.drain(), timeout=0.5)

        assert "▲ SOLUSDT breakout | second line" in caplog.text
        worker = asyncio.create_task(dispatcher.run())
        await asyncio.sleep(0)
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    async def test_drops_when_queue_full(self, running: Start, caplog: pytest.LogCaptureFixture) -> None:
        notifier = RecordingNotifier()
        dispatcher = AlertDispatcher(notifier, maxsize=2)

        for text in ["a", "b", "c"]:
            dispatcher.submit(text)
        running(dispatcher)
        await dispatcher.drain(timeout=1.0)

        assert notifier.sent == ["a", "b"]
        assert "dropping alert" in caplog.text

    async def test_drain_is_bounded_without_worker(self, caplog: pytest.LogCaptureFixture) -> None:
        dispatcher = AlertDispatcher(RecordingNotifier())
        dispatcher.submit("stuck")

        loop = asyncio.get_running_loop()
        started = loop.time()
        await dispatcher.drain(timeout=0.05)

        assert loop.time() - started < 0.5
        assert "1 undelivered" in caplog.text
