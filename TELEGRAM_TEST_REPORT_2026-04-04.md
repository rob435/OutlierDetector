# Telegram Log Review

Source log: `/Users/jhbvdnsbkvnsd/Downloads/Telegram Desktop/ChatExport_2026-04-04/messages.html`

## Scope

- Reviewed `86` unique confirmed 15-minute summary cycles.
- Coverage window: `2026-04-03 20:45 UTC` through `2026-04-04 18:00 UTC`.
- Confirmed cycle spacing was clean: every unique summary was exactly `900,000 ms` apart.
- There were `5` duplicated confirmed summaries, which indicates restart/resend behavior rather than missing data.

## Executive Summary

The system stayed alive and processed on schedule, but the ranking behavior is too unstable for a 3-7 day momentum capture profile.

The main problem is not that the engine misses data. The main problem is that the current composite ranking is too reactive, especially on curvature, and the dominance gate suppresses a large fraction of otherwise strong names. The result is a leaderboard where the same symbols repeatedly move from the top 5 to the bottom 5 and back again over a small number of cycles. That is not what a stable multi-day momentum engine should look like.

The ugliest evidence:

- `111` immediate consecutive-cycle flips between top-5 and bottom-5 membership.
- `232` top-to-bottom flips within the next `4` confirmed cycles.
- Several names spanned nearly the full observed rank range:
  - `BCHUSDT`: observed from rank `#1` to `#58`
  - `TAOUSDT`: observed from rank `#1` to `#58`
  - `BLURUSDT`: observed from rank `#1` to `#58`
  - `STGUSDT`: observed from rank `#1` to `#58`
  - `JTOUSDT`: observed from rank `#1` to `#58`
  - `RENDERUSDT`: observed from rank `#1` to `#58`

That is not normal leader rotation. That is over-reactive scoring.

## What Worked

- The service itself is operationally fine.
- Confirmed summaries arrived on a strict 15-minute cadence with no missing confirmed cycles in the reviewed window.
- Telegram delivery worked.
- Event-driven alerts worked.
- The summary feature solved the observability problem: silence no longer means the engine might be dead.

## Key Findings

### 1. The ranking is too unstable for the intended holding horizon

This is the core problem.

Examples from the confirmed summaries:

- `JTOUSDT` was top 5 at `2026-04-03 20:00 UTC` and bottom 5 at `2026-04-03 20:15 UTC`.
- `RENDERUSDT` was `#1` at `2026-04-03 20:30 UTC` and bottom 5 by `2026-04-03 21:15 UTC`.
- `STGUSDT` was top 5 repeatedly and later bottom 5 repeatedly.
- `BLURUSDT`, `TAOUSDT`, `BCHUSDT`, and `VETUSDT` all appeared on both extremes many times.

Observed consecutive flip leaders:

- `JTOUSDT`: `11` immediate top-to-bottom flips
- `BLURUSDT`: `9` immediate top-to-bottom flips
- `VETUSDT`: `7` immediate top-to-bottom flips
- `RENDERUSDT`: `6` top-to-bottom and `6` bottom-to-top flips
- `STGUSDT`: `5` top-to-bottom and `5` bottom-to-top flips

For a 3-7 day momentum system, that amount of churn is too high.

### 2. Curvature is driving too much of the leaderboard churn

From the reviewed summaries:

- In the top-5 rows, `72.3%` of entries had `|cur_z| > |mom_z|`.
- In the bottom-5 rows, `71.9%` of entries had `|cur_z| > |mom_z|`.

That means most of the leaderboard is being driven more by the curvature z-score than the momentum z-score.

The current composite is a straight sum:

- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L122)
- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L125)

`composite_score = momentum_z + curvature_z`

That is simple, but it is too unforgiving when curvature snaps from strongly positive to strongly negative. In practice, many of the “top to bottom” moves are really “curvature spike to curvature collapse” moves.

Concrete examples:

- `BLURUSDT`:
  - top 3 with `cur_z` around `+3.08` / `+3.99`
  - later bottom 5 with `cur_z` around `-1.82` / `-2.32`
- `STGUSDT`:
  - top 3 with `cur_z` around `+2.86`
  - later bottom 5 with `cur_z` around `-2.52`
- `JTOUSDT`:
  - top 1 with `cur_z` around `+3.79` / `+6.47`
  - later bottom 5 with `cur_z` around `-3.80` / `-6.62`

That is the statistical signature of a ranking dominated by acceleration reversals, not stable momentum leadership.

### 3. Momentum is also structurally biased by raw price differences

The momentum formula is still raw price delta divided by return volatility:

- [indicators.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/indicators.py#L44)
- [indicators.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/indicators.py#L51)

That means:

- high nominal-price assets can still dominate the raw numerator
- z-scoring later does not fully fix the cross-asset comparability problem

This shows up clearly in names like `TAOUSDT` and `BCHUSDT`, which repeatedly hit the top with huge momentum z-scores and later hit the bottom with huge negative momentum z-scores. That behavior may be directionally “real,” but it is not clean cross-sectional normalization.

Examples:

- `TAOUSDT`:
  - early top ranks with `mom_z` as high as `+6.68` / `+6.72`
  - late bottom ranks with `mom_z` as low as `-6.88` / `-6.84`
- `BCHUSDT`:
  - bottom early with `mom_z` around `-7.11` to `-7.18`
  - later top with `mom_z` around `+4.44` / `+4.99`
  - then bottom again with `mom_z` around `-3.13` / `-3.76`

The rank system is therefore getting hit from both sides:

- curvature whipsaw across most of the universe
- raw-price momentum spikes in high-price symbols

### 4. The dominance gate is acting as a hard on/off switch

This was one of the strongest findings in the entire test.

Across the `86` unique confirmed cycles:

- `33` cycles had `Dominance falling: yes`
- `53` cycles had `Dominance falling: no`
- `33` cycles had at least one qualified signal
- `0` cycles with `Dominance falling: no` had any qualified signal

So during this test window:

- every qualified confirmed cycle happened when dominance was `yes`
- no confirmed cycle qualified anything when dominance was `no`

That is because the core filter is a hard gate:

- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L229)
- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L237)

`item.hurst > hurst_cutoff and dom_falling == 1 and composite_score >= min_score`

This is not threshold modulation. This is a binary filter.

If you are comfortable with that behavior, fine. But the logs show very clearly that dominance is currently deciding when confirmed signals are even allowed to exist.

### 5. The Hurst filter is no longer the main blocker if your cutoff is really `0.40`

Assuming the VPS was actually running `HURST_CUTOFF=0.40` as discussed:

- `346 / 430` top-5 appearances had `hurst > 0.40`
- `256 / 430` top-5 appearances had `hurst > 0.45`
- `155 / 430` top-5 appearances had `hurst > 0.50`
- only `84 / 430` top-5 appearances had `hurst > 0.55`

So:

- at `0.55`, Hurst would still be a severe choke point
- at `0.40`, Hurst is no longer the main reason “good-looking” names fail

In this test window, dominance was the bigger blocker.

### 6. The regime model did not add any discrimination in this sample

Every single confirmed summary had:

- `BTC regime: 1`

So across the reviewed period, the regime model did not change state once.

That means the effective regime floor never changed during the sample.

Operationally, that makes regime mostly irrelevant for this 1-day test. It was not broken, but it was not informative either.

### 7. Watchlist noise is high relative to higher-quality alerts

Telegram candidate alert counts:

- `WATCHLIST`: `228`
- `EMERGING`: `67`
- `CONFIRMED`: `29`
- `CONFIRMED_STRONG`: `9`

So the mix is:

- watchlist-heavy
- emerging much thinner
- confirmed much thinner still

That is not automatically wrong, but it does mean your Telegram feed is currently dominated by low-confidence transitions.

### 8. Confirmed summaries were duplicated 5 times

Duplicate confirmed summaries were observed for these cycle times:

- `1775245500000`
- `1775268900000`
- `1775278800000`
- `1775279700000`
- `1775280600000`

That is probably restart/resubmission behavior rather than ranking logic, but it is still operationally sloppy. The system should not re-send the same confirmed summary for the same cycle time after a restart unless that is explicitly desired.

## Why The “Top Performers Going To The Bottom” Problem Happens

The simplest honest explanation is:

1. The ranking is fully cross-sectional.
2. The composite score is too reactive.
3. Curvature is allowed to swing leadership too hard.
4. Raw-price momentum exaggerates some symbols.
5. The filters then make the final signal set sparse and discontinuous.

So the issue is not just “coins rotate.”

The issue is that the current ranking stack is effectively overfitting to short-horizon acceleration changes while you want a multi-day momentum capture profile.

That mismatch is visible directly in the Telegram history.

## Code-Level Causes

### Equal-weight composite is too sharp

- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L125)

The equal-weight sum of `momentum_z + curvature_z` gives curvature too much power over the ordering.

### Curvature is a second-difference of smoothed returns with no additional clamp

- [indicators.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/indicators.py#L54)
- [indicators.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/indicators.py#L69)

That is mathematically valid, but in this deployment it is producing more churn than is useful for your intended holding period.

### Momentum uses raw price difference

- [indicators.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/indicators.py#L44)

That is the original spec, but it is still a weak cross-sectional design across assets with wildly different price levels.

### Dominance is a hard gate, not just context

- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L235)
- [signal_engine.py](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/signal_engine.py#L238)

This explains the clean split between `dom_falling=yes` cycles having qualified signals and `dom_falling=no` cycles having none.

## Recommendations

### Priority 1: Reduce curvature power

This is the biggest ranking fix.

Recommended changes:

- switch from `composite = mom_z + cur_z` to a weighted form like:
  - `composite = 0.7 * mom_z + 0.3 * cur_z`
- clamp or winsorize curvature z-scores before combining
  - example: clip `cur_z` to `[-2.5, 2.5]`
- consider requiring curvature to confirm momentum rather than dominate it

If you want a 3-7 day momentum engine, curvature should be a timing aid, not the main ranking driver.

### Priority 2: Replace raw-price momentum with percentage/log momentum

Recommended change:

- replace raw price difference with log-price or percentage momentum across the lookback

For example:

- `log(prices[-K] / prices[-(L+K)]) / volatility`

That will make cross-sectional comparisons much less distorted by nominal price level.

### Priority 3: Soften the dominance gate

Right now, dominance acts like a binary kill switch.

Recommended options:

- keep it as a hard block only for confirmed alerts, but not for ranking
- or better, turn it into a score penalty / stricter threshold rather than `dom_falling == 1` as an absolute requirement

If you do not change this, expect large stretches of “good looking leaderboard, no qualified signals.”

### Priority 4: Introduce leader stability into the confirmed ranking

You already have `confirmed_strong`, which is the right idea.

What is still missing is making the displayed “best names” reflect short persistence more explicitly.

Recommended options:

- add a `leader_stability` field to summaries
- show whether a top-5 name was also top-10 on prior confirmed bars
- optionally sort a secondary operator report by persistence-adjusted score

This will help separate genuine multi-bar leaders from one-bar curvature spikes.

### Priority 5: Suppress duplicate confirmed summaries after restart

This is not the main strategy issue, but it is still a quality issue.

The summary sender should remember the last confirmed `cycle_time_ms` it has already emitted and refuse to re-send that same summary after a restart/rebootstrap.

### Priority 6: Reduce watchlist Telegram noise if needed

If the feed starts to feel noisy:

- keep `WATCHLIST` in SQLite
- but send only `EMERGING`, `CONFIRMED`, `CONFIRMED_STRONG`, and the confirmed-cycle summary to Telegram

That would probably improve operator signal-to-noise immediately.

## Recommended Next Iteration

If the goal is to preserve the current architecture but make it behave like a multi-day momentum engine, the next tuning pass should be:

1. Change composite weighting to favor momentum over curvature.
2. Move momentum from raw price delta to log/percentage momentum.
3. Keep dominance in the summary, but stop using it as a hard universal kill switch.
4. Re-run the same 1-day Telegram review and compare:
   - top-to-bottom flip count
   - number of qualified confirmed cycles
   - number of duplicate/noisy alerts

## Bottom Line

The deployment is operationally successful.

The ranking model is not yet behaviorally aligned with the intended strategy.

Right now it behaves more like a short-horizon acceleration detector with a hard dominance gate than a stable 3-7 day momentum capture system.

That is why the same names are repeatedly showing up at both extremes of the leaderboard.
