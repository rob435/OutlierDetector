# Decisions

## 2026-04-03

- Chose Bybit V5 as the concrete exchange target because the spec aligns with Bybit topic semantics and closed-candle `confirm` events.
- Kept the repo flat. There is no extra package layering because the project is small and the dataflow is direct.
- Recompute cross-sectional rankings on every closed candle update instead of trying to process only the changed ticker. Ranking is inherently cross-sectional, so partial updates would be wrong.
- Add a short cycle settle delay before each processing pass so the engine ranks against a coherent candle-close wave instead of a half-updated snapshot.
- Pass candle timestamps through the queue into the confirmed signal path so close-confirmed logs and cooldown use market-event time; keep emerging alerts on wall-clock detection time because they fire before the candle is final.
- Treat BTC regime as a threshold modulator only. Regime score `0` removes the composite-score floor because the configured threshold map returns `None`; rank, Hurst, and dominance still apply.
- Use a BTC-vs-alt-basket relative-strength proxy as the dominance rotation signal. Real BTC dominance history is outside Bybit public data.
- Log every evaluated ticker to SQLite on each cycle, including cooldown-suppressed rows with `alerted=0`.
- Batch SQLite inserts per cycle and store `price`, `rank`, and `dom_falling` so the logs are useful for analysis instead of just audit trivia.
- Only update cooldown state after a notifier send actually succeeds.
- Keep a non-trivial minimum volatility floor (`1e-5`) because the raw-price momentum formula otherwise over-rewards ultra-smooth drift series.
- Add a replay harness that drives the production engine over recent historical candles rather than inventing a separate backtest-only code path.
- Add a paginated universe validator against Bybit instruments metadata because the manual symbol list will drift over time and the endpoint now exceeds the default 500-row page.
- Add a lightweight SQLite report CLI instead of expecting operators to inspect signal quality with raw SQL after every replay or live run.
- Add a one-command smoke runner so external validation is repeatable instead of depending on a manual validator -> replay -> report sequence.
- Add a benchmark CLI over the real engine path so the 100-ticker latency target can be measured repeatedly instead of asserted by guesswork.
- Add bounded-runtime support and runtime counters to `main.py` so live soak runs are repeatable and produce a shutdown summary instead of depending on manual observation.
- Add an explicit soak-run guide and production env template because deployment mistakes are more likely than code defects at this stage.
- Keep confirmed 15m history and provisional intrabar prices as separate state paths. Early watchlist logic is useful, but corrupting the confirmed history with partial candles would invalidate the close-confirmed signal.
- Use the same cross-sectional engine for both `emerging` and `confirmed` stages instead of inventing a second ranking model. The difference is the price input and alert gating, not a separate math stack.
- Do not treat every intrabar top-ranked name as a real emerging breakout. Intrabar candidates now progress through `watchlist` first and only promote to `emerging` after repeated observations show improving rank and rising composite score.
- Gate intrabar alerts on state transitions (`neutral -> watchlist`, `watchlist -> emerging`) plus separate cooldowns so the live system stays event-driven without spamming the same ticker on every Bybit kline push.
- Store both `stage` and `signal_kind` in SQLite because once the engine has intrabar state transitions, a single undifferentiated signal log becomes analytically useless.
- Treat confirmed-bar persistence as a confidence upgrade, not a hard prerequisite. A first valid confirmed breakout still matters for momentum capture, so the system upgrades to `confirmed_strong` on repeated confirmed leadership instead of suppressing early confirmed signals.
- Keep routine Telegram summaries separate from event-driven alerts. The confirmed-cycle digest exists to prove liveness and show market context; it does not change signal state or cooldown logic.
- Treat the 1-day Telegram review as evidence that the current ranking stack is too reactive for a 3-7 day momentum objective. The next tuning pass should prioritize reducing curvature dominance, replacing raw-price momentum with percentage/log momentum, and reconsidering the dominance gate as a hard blocker.
- Replace the BTC-vs-alt-basket dominance proxy with Binance futures BTCDOM history on `1h`, normalized into `falling / neutral / rising` using a `+-0.2%` neutral band.
- Modulate thresholds by dominance state instead of hard-blocking signals. Dominance is now a score adjustment, not a binary veto.
- Switch momentum to log-return normalization and reduce curvature weighting to `0.15` so the leaderboard is less sensitive to price scale and curvature noise.
- Suppress watchlist Telegram by config if needed, but keep confirmed-cycle summaries and confirmed event alerts enabled so liveness and signal context remain visible.
- Deduplicate confirmed summaries by cycle time so restarts do not resend an already-logged digest.
- Promote the current runtime behavior into a canonical written specification in `SPEC.md` so the repo no longer depends on an implied or historical spec.
- Treat the second Telegram review as evidence that the momentum/log-normalization and weaker-curvature refactor materially improved short-horizon ranking stability. Do not cut curvature further yet; the next likely tuning target is macro information content and confirmed stability behavior, not another immediate ranking rewrite.
- Add `entry_ready` as the trader-facing midpoint entry tier between `emerging` and `confirmed`, and give it explicit knobs so operators can keep it tighter than broad emerging context without turning it into a close-only filter.
- Keep repo hygiene strict: ignore local runtime artifacts, prefer clean git-based VPS deployment over `scp` of a dirty working tree, and remove dead compatibility shims instead of letting them rot.
