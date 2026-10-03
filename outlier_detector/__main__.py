"""Command line: `outlier-detector [run | replay | signals]`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from collections.abc import Sequence
from contextlib import suppress
from datetime import UTC, datetime

import aiohttp

from .bybit import BybitClient
from .config import ConfigError, Settings, load_settings
from .models import Direction, Signal
from .notify import AlertDispatcher, TelegramNotifier, format_price
from .replay import fetch_history, replay_history, summarize
from .service import Service
from .store import SignalStore

LOGGER = logging.getLogger("outlier_detector")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="outlier-detector", description="Real-time breakout detector for Bybit USDT perpetuals."
    )
    parser.add_argument("--env-file", default=".env", help="optional KEY=VALUE file (default: .env)")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="stream Bybit live and alert on breakouts (default)")
    replay = commands.add_parser("replay", help="run recent history through the detector (closed bars)")
    replay.add_argument("--days", type=float, default=3.0, help="days of history to replay (default: 3)")
    history = commands.add_parser("signals", help="list recent alerts from the database")
    history.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)

    try:
        settings = load_settings(args.env_file)
    except ConfigError as exc:
        print(f"Invalid configuration: {exc}", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if args.command == "replay":
        return asyncio.run(_replay(settings, args.days))
    if args.command == "signals":
        with SignalStore(settings.sqlite_path) as store:
            for row in reversed(store.recent(args.limit)):
                print(format_row(row))
        return 0
    return asyncio.run(_run(settings))


async def _run(settings: Settings) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):  # Windows
            loop.add_signal_handler(sig, stop.set)

    async with aiohttp.ClientSession() as session:
        client = BybitClient(session, settings.bybit_rest_url, settings.bybit_ws_url)
        notifier = (
            TelegramNotifier(session, settings.telegram_bot_token, settings.telegram_chat_id)
            if settings.telegram_enabled
            else None
        )
        if notifier is None:
            LOGGER.warning("Telegram not configured; alerts go to the log only")
        dispatcher = AlertDispatcher(notifier)
        with SignalStore(settings.sqlite_path) as store:
            service = Service(settings, client, store, dispatcher)
            service_task = asyncio.create_task(service.run())
            worker = asyncio.create_task(dispatcher.run())
            stopper = asyncio.create_task(stop.wait())
            done, _ = await asyncio.wait({stopper, service_task, worker}, return_when=asyncio.FIRST_COMPLETED)
            failed = [task for task in done - {stopper} if task.exception() is not None]
            for task in failed:
                LOGGER.error("Exited on error", exc_info=task.exception())
            LOGGER.info("Shutting down")
            service_task.cancel()
            await asyncio.gather(service_task, return_exceptions=True)
            await dispatcher.drain()
            worker.cancel()
            stopper.cancel()
            await asyncio.gather(worker, stopper, return_exceptions=True)
    return 1 if failed else 0


async def _replay(settings: Settings, days: float) -> int:
    async with aiohttp.ClientSession() as session:
        client = BybitClient(session, settings.bybit_rest_url, settings.bybit_ws_url)
        history = await fetch_history(client, settings, days)
    if not history:
        print("No history fetched.", file=sys.stderr)
        return 1
    signals = replay_history(history, settings)
    for row in signals:
        print(format_row(row))
    print(f"\n{len(history)} symbols · {summarize(signals, days)}")
    return 0


def format_row(signal: Signal) -> str:
    arrow = "▲" if signal.direction is Direction.UP else "▼"
    when = datetime.fromtimestamp(signal.detected_at_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"{when}  {signal.stage.value:<9}  {arrow} {signal.symbol:<14} {format_price(signal.price):>14}"
        f"  {signal.move_pct:+6.2f}% (mkt {signal.market_move_pct:+.2f}%)"
        f"  {abs(signal.zscore):4.1f}σ  {signal.rvol:5.1f}× vol"
    )


if __name__ == "__main__":
    sys.exit(main())
