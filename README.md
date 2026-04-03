# Outlier Detector

Real-time crypto breakout detector built around Bybit V5 candle streams, volatility-adjusted momentum, return curvature, BTC regime scoring, and cross-sectional ranking.

## What it does

- Bootstraps 15m history for a manual universe of midcap USDT contracts plus `BTCUSDT`.
- Subscribes to Bybit public `kline.15` WebSocket updates and processes both intrabar updates and confirmed candle closes.
- Recomputes cross-sectional scores on two stages:
  - `emerging`: intrabar ranking off provisional current-candle prices
  - `confirmed`: full 15m close ranking off confirmed history only
- Differentiates intrabar states instead of treating every top-ranked provisional move the same:
  - `WATCHLIST`: ticker enters the broad intrabar leaders
  - `EMERGING`: ticker is strengthening across recent intrabar observations
  - `CONFIRMED`: ticker qualifies on the closed 15m candle
  - `CONFIRMED_STRONG`: ticker qualifies on the closed 15m candle and has also held confirmed leadership across recent confirmed bars
- Uses a short settle delay before each processing cycle so ranking runs on a mostly complete 15m snapshot instead of the first symbol that arrives.
- Throttles `emerging` processing with a minimum interval so intrabar updates stay event-driven without recalculating on every raw WebSocket packet.
- Carries candle timestamps through the queue path for `confirmed` cycles; `emerging` rows use wall-clock detection time because intrabar alerts happen before the candle is final.
- Uses BTC daily regime as a threshold modulator, not a hard block.
- Uses a BTC-vs-alt-basket relative-strength proxy as the dominance rotation signal. This is the honest compromise for a Bybit-only build because Bybit does not publish BTC dominance history.
- Logs every evaluated ticker on each cycle to SQLite and optionally sends Telegram alerts with cooldown control.
- Sends a separate Telegram summary on every confirmed 15m cycle with the top and bottom ranked names, so you can see what the engine is seeing even when no fresh signal transition fires.
- Logs the top-ranked names each cycle so you can see the leaders even when no symbol passes the final alert filters.
- Keeps provisional intrabar prices isolated from the confirmed 15m history so early alerts do not contaminate the close-confirmed signal path.

## Why the implementation differs from the raw spec

- `EMA200` requires more than 30 daily BTC closes. The system stores 220 BTC daily candles so the regime model is valid.
- The spec asks for BTC dominance rotation, but Bybit market data does not provide BTC dominance history. This build uses a BTC relative-strength proxy against the tracked alt basket so the system stays deployable without a second paid data source.

## Layout

- `config.py`: environment-driven settings
- `universe.py`: manual universe list
- `state.py`: in-memory rolling state
- `indicators.py`: pure numeric routines
- `exchange.py`: Bybit REST bootstrap + WebSocket ingestion
- `signal_engine.py`: ranking, signal generation, cooldown, logging, alert dispatch
- `database.py`: SQLite persistence
- `alerting.py`: Telegram output
- `main.py`: process wiring and supervision

## Run

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python main.py
```

## Test

```bash
pytest
```

## Replay

Use the replay harness to drive the production engine over recent Bybit history without waiting days for live validation:

```bash
python replay.py --cycles 96 --db replay-signals.sqlite3
```

This warms state with the first `STATE_WINDOW` candles, then replays the remaining cycles in market-event time.

## Universe Validation

Validate the configured universe against Bybit’s live instruments list before a deployment:

```bash
python universe_validator.py
```

This uses Bybit `GET /v5/market/instruments-info` with pagination because `linear` instruments exceed the default 500-row response.

## Reporting

Summarize a replay or live SQLite log without hand-writing SQL:

```bash
python report.py --db replay-signals.sqlite3 --top 10
```

The report now includes a stage breakdown so you can see how much of the log came from `emerging` versus `confirmed` cycles.

## Smoke Run

Run universe validation plus a short live-data replay in one command:

```bash
python smoke.py --cycles 4 --db smoke-replay.sqlite3 --strict-universe
```

## Benchmark

Measure signal-engine cycle latency against synthetic data:

```bash
python benchmark.py --tickers 100 --cycles 20
```

On this machine, the current engine path measured about `36ms` average for `100` synthetic tickers including SQLite writes, which is inside the `<100ms` target.

## Bounded Live Run

Run the actual service for a fixed time window with Telegram disabled:

```bash
python main.py --run-seconds 300 --disable-telegram
```

This is the honest soak-test path for startup, WebSocket stability, macro refresh, queue handling, and clean shutdown. The process logs a runtime summary on exit.

## Deployment

`deploy/outlier-detector.service` is a minimal `systemd` unit. Adjust paths before installing it.

For the first VPS validation run, use [SOAK_RUN.md](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/deploy/SOAK_RUN.md) and start from [production.env.example](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/deploy/production.env.example).

## Operational notes

- Bybit REST `GET /v5/market/kline` returns candles in reverse chronological order and includes the still-open candle; the bootstrapper normalizes to ascending closed candles only.
- Bootstrap is the heaviest REST phase. Repeated local restarts can hit Bybit `retCode=10006`; the client now backs off and retries automatically.
- The WebSocket connection will drop eventually. The supervisor reconnects, reboots state through REST, and resumes the stream.
- Intrabar cycles will fire repeatedly during an open 15m candle whenever Bybit pushes provisional kline updates. That is intentional; alert sends are transition-gated into `WATCHLIST` and `EMERGING` states instead of firing on every pass.
- `EMERGING` requires strengthening, not just presence. The intrabar state machine looks for repeated watchlist observations with improving rank and rising composite score before promoting a ticker from `WATCHLIST` to `EMERGING`.
- `CONFIRMED_STRONG` is not a hard gate. The first valid confirmed bar can still alert as `CONFIRMED`; the signal upgrades to `CONFIRMED_STRONG` when recent confirmed-bar history also supports it.
- If a candle gap is detected mid-stream, the supervisor falls back to a fresh REST bootstrap instead of pretending state is intact.
- Cooldown only advances after a successful alert send; a failed Telegram request does not silently suppress the next valid signal.
- `emerging` and `confirmed` alerts use separate cooldown state, so an early watchlist alert does not block the later close-confirmed alert for the same ticker.
- Confirmed-cycle summaries are separate from event alerts. They are a routine operator digest, not a signal trigger, and default to the top 5 plus bottom 5 names.
- The runtime uses the `certifi` CA bundle for outbound TLS, which avoids common macOS Python certificate-store breakage.
- If you run from a US-routed host, Bybit mainnet access may be blocked. Use a compliant region or testnet/base URL override.

## Docs used

- Bybit Get Kline: https://bybit-exchange.github.io/docs/v5/market/kline
- Bybit WebSocket Connect: https://bybit-exchange.github.io/docs/v5/ws/connect
