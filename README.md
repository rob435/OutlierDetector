# Outlier Detector

Real-time breakout detection for Bybit USDT perpetuals.

It streams 15-minute klines for the most-traded perps and re-scores the whole universe twice a second. When a coin breaks out of its range on heavy volume while the rest of the market stays put, it sends a Telegram alert while the bar is still forming, instead of waiting for it to close.

```
▲ SOLUSDT breakout · early
152.340  +3.20% in 1h (market +0.40%)
4.1σ · 5.3× volume · above 24h high 150.100
14:32:05 UTC · https://www.bybit.com/trade/usdt/SOLUSDT
```

## What counts as a breakout

A symbol is flagged when **all four** gates pass. Defaults are shown; every gate is configurable.

| Gate | Default | Test |
|---|---|---|
| Range break | 24h | Price is above the highest high (or below the lowest low) of the last 96 bars |
| Impulse | ≥ 1% | Move from the close 4 bars ago to the current price, including the forming bar |
| Outlier | ≥ 3σ | `(move − universe median move) / (σ · √horizon)`, where σ is the symbol's own per-bar volatility |
| Volume | ≥ 2× | Bar turnover vs. the median bar, pro-rated by how much of the bar has elapsed |

The outlier gate subtracts the median move of the universe, so a market-wide pump is not an outlier but a coin leaving the pack is. On 30 days of 15m data for 58 liquid USDT pairs, the defaults flag about **9 breakouts a day**. Without the market adjustment the same gates fire about 25 times a day.

Each breakout can alert twice:

- **early**: scored on the forming bar's latest tick. This is the fast path.
- **confirmed**: scored on closed bars as soon as the whole universe has closed. If an early alert fired in the same bar, the message says it *held into close*.

Each (symbol, direction, stage) has a cooldown (4h by default). Cooldowns are stored in SQLite, so restarting does not re-send alerts.

## How it works

```
Bybit REST ── bootstrap / per-symbol resync ──┐
                                              ▼
Bybit WS kline.15.* ───────────────► MarketData   numpy rows, one per symbol
                                              │   aligned snapshot of the cross-section
                                              ▼
                                       detector   pure, vectorised scoring
                                              │
                                              ▼
                                   CooldownGate ──► Telegram (queued, never blocks) + log
                                              └───► SQLite
```

Robustness built in:

- **Bootstrap**: closed bars are identified with Bybit's server time, never the local clock. A symbol that fails to load (e.g. a new listing) is retried in the background instead of failing startup.
- **Stream**: subscriptions are chunked and each acknowledgement is checked. The client sends Bybit's app-level ping, and a watchdog reconnects if the stream goes quiet.
- **Gaps**: a missed bar re-loads only that symbol from REST, so one hiccup doesn't rebuild the whole universe.
- **Bar close**: the confirmed scan runs as soon as every in-sync symbol has closed, with a grace timeout. It never ranks a half-updated universe.
- **Alignment**: snapshots include only symbols in sync for the bar being scored. A stalled symbol drops out on its own and is re-synced.

## Quick start

Requires Python 3.11+. Bybit blocks some regions (including the US), so run it from a host it serves.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # optional: add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID

outlier-detector            # stream live and alert
outlier-detector replay --days 7   # run recent history through the detector
outlier-detector signals    # list recent alerts from the database
```

`replay` drives the same detector and cooldowns over historical closed bars, which is the quickest way to tune thresholds. History contains no intrabar ticks, so it shows confirmed alerts only.

Without Telegram credentials, alerts go to the log.

## Configuration

All settings are environment variables, optionally read from `.env`. See [`.env.example`](.env.example) for the full list and their defaults. The main ones:

| Variable | Default | |
|---|---|---|
| `SYMBOLS` | *(auto)* | Comma-separated list; empty selects the `UNIVERSE_SIZE` most-traded perps |
| `UNIVERSE_SIZE` / `MIN_TURNOVER_USD` | `100` / `10000000` | Auto-universe size and liquidity floor |
| `LOOKBACK_BARS` / `IMPULSE_BARS` | `96` / `4` | Range window and move window |
| `MIN_ZSCORE` / `MIN_MOVE_PCT` / `MIN_RVOL` | `3.0` / `1.0` / `2.0` | Gate thresholds |
| `EARLY_ALERTS` / `DETECT_BREAKDOWNS` | `true` / `true` | Intrabar alerts; downside breaks |
| `COOLDOWN_MINUTES` | `240` | Per symbol, direction and stage |

Invalid values fail fast at startup with a message naming the variable.

## Layout

```
outlier_detector/
  config.py    settings from env, validated
  models.py    Candle, KlineUpdate, Signal
  bybit.py     REST + WebSocket client
  market.py    columnar rolling state, gap detection, aligned snapshots
  detector.py  breakout scoring and cooldowns (pure)
  service.py   live orchestration: bootstrap, stream, scans, resync
  notify.py    alert formatting, Telegram, non-blocking dispatcher
  store.py     SQLite signal log
  replay.py    historical replay
  __main__.py  CLI
deploy/outlier-detector.service   systemd unit
```

## Development

```bash
pytest          # unit and end-to-end tests; no network needed
ruff check . && ruff format --check .
```

This project is a detector, not a trading strategy. It makes no claim about what happens after a breakout.
