"""Alert formatting and delivery (Telegram, behind a non-blocking dispatcher)."""

from __future__ import annotations

import asyncio
import logging
import math
from collections.abc import Mapping
from typing import Any, Protocol

import aiohttp

from .models import Direction, Signal

LOGGER = logging.getLogger(__name__)

PRICE_SIGNIFICANT_DIGITS = 6
MAX_RETRY_AFTER_SECONDS = 30.0
BACKOFF_BASE_SECONDS = 0.5
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)
_TRADE_URL = "https://www.bybit.com/trade/usdt/{symbol}"

# Indirection so tests can skip real waits without patching asyncio globally.
_sleep = asyncio.sleep


def format_duration(minutes: int) -> str:
    hours, mins = divmod(minutes, 60)
    if not hours:
        return f"{mins}m"
    return f"{hours}h{mins}m" if mins else f"{hours}h"


def format_price(price: float) -> str:
    """Fixed-point with ~6 significant digits (no scientific notation)."""
    if price == 0 or not math.isfinite(price):
        return f"{price:g}"
    magnitude = math.floor(math.log10(abs(price)))
    # Rounding can carry into the next power of ten (99.99999 -> 100.000).
    if abs(round(price, PRICE_SIGNIFICANT_DIGITS - 1 - magnitude)) >= 10 ** (magnitude + 1):
        magnitude += 1
    decimals = max(0, PRICE_SIGNIFICANT_DIGITS - 1 - magnitude)
    return f"{price:.{decimals}f}"


def format_signal(signal: Signal, *, impulse_minutes: int, lookback_minutes: int, held: bool = False) -> str:
    up = signal.direction is Direction.UP
    arrow, kind = ("▲", "breakout") if up else ("▼", "breakdown")
    level_text = "above {} high" if up else "below {} low"
    header = f"{arrow} {signal.symbol} {kind} · {signal.stage.value}"
    if held:
        header += " (held into close)"
    move = (
        f"{format_price(signal.price)}  {signal.move_pct:+.2f}% in {format_duration(impulse_minutes)}"
        f" (market {signal.market_move_pct:+.2f}%)"
    )
    stats = (
        f"{abs(signal.zscore):.1f}σ · {signal.rvol:.1f}× volume · "
        f"{level_text.format(format_duration(lookback_minutes))} {format_price(signal.level)}"
    )
    return "\n".join([header, move, stats, _TRADE_URL.format(symbol=signal.symbol)])


class Notifier(Protocol):
    async def send(self, text: str) -> bool: ...


class TelegramNotifier:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        token: str,
        chat_id: str,
        *,
        api_url: str = "https://api.telegram.org",
        attempts: int = 3,
    ) -> None:
        self._session = session
        self._token = token
        self._chat_id = chat_id
        self._url = f"{api_url.rstrip('/')}/bot{token}/sendMessage"  # secret: never log
        self._attempts = max(1, attempts)

    async def send(self, text: str) -> bool:
        """Deliver `text`; returns success. Never raises except on cancellation."""
        try:
            return await self._send(text)
        except Exception as exc:
            LOGGER.error("telegram send crashed: %s", self._describe(exc))
            return False

    async def _send(self, text: str) -> bool:
        payload = {"chat_id": self._chat_id, "text": text, "disable_web_page_preview": True}
        for attempt in range(self._attempts):
            tries = f"attempt {attempt + 1}/{self._attempts}"
            try:
                status, body = await self._post(payload)
            except (aiohttp.ClientError, TimeoutError) as exc:
                LOGGER.warning("telegram send failed (%s): %s", tries, self._describe(exc))
                delay = _backoff(attempt)
            else:
                if status == 200:
                    return True
                description = body.get("description", "")
                retry_delay = _retry_delay(status, body, attempt)
                if retry_delay is None:
                    LOGGER.error("telegram rejected message: HTTP %d %s", status, description)
                    return False
                LOGGER.warning("telegram send failed (%s): HTTP %d %s", tries, status, description)
                delay = retry_delay
            if attempt + 1 < self._attempts:
                await _sleep(delay)
        LOGGER.error("telegram: giving up after %d attempts", self._attempts)
        return False

    async def _post(self, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        async with self._session.post(self._url, json=payload, timeout=_REQUEST_TIMEOUT) as response:
            try:
                body = await response.json(content_type=None)
            except ValueError:  # non-JSON error page, e.g. from a proxy
                body = None
            return response.status, body if isinstance(body, dict) else {}

    def _describe(self, exc: BaseException) -> str:
        text = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return text.replace(self._token, "<token>") if self._token else text


def _retry_delay(status: int, body: Mapping[str, Any], attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if the error is permanent."""
    if status == 429:
        parameters = body.get("parameters")
        retry_after = parameters.get("retry_after") if isinstance(parameters, dict) else None
        if isinstance(retry_after, int | float):
            return min(max(float(retry_after), 0.0), MAX_RETRY_AFTER_SECONDS)
        return _backoff(attempt)
    if status >= 500:
        return _backoff(attempt)
    return None


def _backoff(attempt: int) -> float:
    return BACKOFF_BASE_SECONDS * 2**attempt


class AlertDispatcher:
    """Queue + single worker so slow/failed Telegram sends never block scanning."""

    def __init__(self, notifier: Notifier | None, maxsize: int = 1000) -> None:
        self._notifier = notifier
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize)

    def submit(self, text: str) -> None:
        LOGGER.info("ALERT %s", text.replace("\n", " | "))
        if self._notifier is None:
            return
        try:
            self._queue.put_nowait(text)
        except asyncio.QueueFull:
            LOGGER.warning("alert queue full (%d pending); dropping alert", self._queue.qsize())

    async def run(self) -> None:
        while True:
            text = await self._queue.get()
            try:
                await self._deliver(text)
            finally:
                self._queue.task_done()

    async def drain(self, timeout: float = 5.0) -> None:
        try:
            await asyncio.wait_for(self._queue.join(), timeout)
        except TimeoutError:
            LOGGER.warning("gave up draining alerts; %d undelivered", self._queue.qsize())

    async def _deliver(self, text: str) -> None:
        if self._notifier is None:
            return
        summary = text.partition("\n")[0]
        try:
            delivered = await self._notifier.send(text)
        except Exception:
            LOGGER.exception("notifier crashed on alert: %s", summary)
            return
        if not delivered:
            LOGGER.warning("alert not delivered: %s", summary)
