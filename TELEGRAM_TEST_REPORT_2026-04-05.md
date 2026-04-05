# Telegram Log Review

Source log: `/Users/jhbvdnsbkvnsd/Downloads/Telegram Desktop/ChatExport_2026-04-05/messages.html`

## Scope

- Reviewed messages from `2026-04-04 21:00 UTC` onward.
- Reviewed `93` unique confirmed 15-minute summary cycles.
- Coverage window: `2026-04-04 21:00 UTC` through `2026-04-05 20:00 UTC`.
- Confirmed cycle spacing was clean: every unique summary `cycle_time_ms` was exactly `900,000 ms` apart.
- There were `0` duplicated confirmed summaries in the reviewed window.

## Executive Summary

This run is materially better than the previous one.

The most important improvement is that the ugly top-to-bottom whipsaw behavior has largely disappeared at the short horizon that mattered most in the first review.

Compared with the prior report in [TELEGRAM_TEST_REPORT_2026-04-04.md](/Users/jhbvdnsbkvnsd/Desktop/OutlierDetector/TELEGRAM_TEST_REPORT_2026-04-04.md):

- immediate top-5 to bottom-5 or bottom-5 to top-5 flips fell from `111` to `0`
- top-to-bottom flips within the next `4` confirmed cycles fell from `232` to `0`
- curvature dominance in top/bottom leaderboard rows fell from roughly `72%` to `17.4%`

That is a real improvement, not noise.

The system now looks much more like a momentum engine and much less like a curvature-reversal detector.

But it is not clean yet.

The remaining problems are different now:

- BTC regime is still flat and contributes almost no information in this window
- BTCDOM state is better than before, but this sample only exercised `neutral` and `rising`, not `falling`
- some names still span both top and bottom extremes over the full day, even though they are no longer teleporting between them on consecutive cycles

So the main failure mode changed from:

- violent short-horizon leaderboard instability

to:

- longer-horizon ranking drift and low-information macro state

That is progress.

## What Worked

### 1. Confirmed summaries stayed clean and regular

- `93` unique confirmed summaries arrived over `23.0` hours
- `0` duplicate summaries were observed
- confirmed cycle spacing stayed exact at `900,000 ms`

This means:

- the summary dedupe fix worked
- restart-resend noise appears gone in this window
- the system still looks operationally stable

### 2. The top-to-bottom flip problem is dramatically improved

This was the biggest weakness in the prior run. It is the biggest improvement in this run.

Observed in this window:

- `0` immediate extreme flips between consecutive confirmed cycles
- `0` top/bottom extreme flips within the next `4` confirmed cycles

That is a complete removal of the specific failure mode that dominated the first report.

### 3. Momentum now dominates the ranking more than curvature

Across the `93` confirmed summaries:

- `82.6%` of top/bottom leaderboard rows had `|mom_z| > |cur_z|`
- only `17.4%` had `|cur_z| > |mom_z|`

This is exactly what the refactor was supposed to do.

The current leaderboard is now primarily driven by momentum, with curvature acting more like a timing modifier instead of the main rank driver.

### 4. Qualified confirmed signals are no longer starved

Across the window:

- average qualified confirmed signals per cycle: `2.53`
- minimum: `0`
- maximum: `3`
- only `2` confirmed cycles had zero qualified signals

That is a healthier profile than the earlier “dead board unless dominance agrees” behavior.

### 5. Dominance is no longer acting like a hard kill switch

Confirmed summaries in this window had:

- `65` cycles with `Dominance state: neutral`
- `28` cycles with `Dominance state: rising`
- `0` cycles with `Dominance state: falling`

Despite that, confirmed qualifications still occurred consistently:

- average qualified signals in `neutral` cycles: `2.46`
- average qualified signals in `rising` cycles: `2.68`

That means the dominance refactor worked in the specific sense that:

- the system still produces qualified signals outside a single preferred dominance state
- dominance is no longer behaving like a binary veto

### 6. Emerging alerts are mostly useful

The window contained `10` `EMERGING` Telegram alerts.

Of those:

- `8` converted into a qualified confirmed signal within `60` minutes

Examples:

- `ADAUSDT`: `00:30:56` emerging -> `01:15:02` confirmed
- `ETCUSDT`: `00:31:18` emerging -> `00:45:02` confirmed
- `TONUSDT`: `05:15:25` emerging -> `06:00:02` confirmed
- `RENDERUSDT`: `06:01:30` emerging -> `06:30:02` confirmed
- `TRXUSDT`: `06:56:14` emerging -> `07:00:02` confirmed
- `STGUSDT`: `09:46:22` emerging -> `10:00:02` confirmed
- `VETUSDT`: `10:19:08` emerging -> `10:30:02` confirmed

The two emerging alerts that did not convert within `60` minutes were:

- `ATOMUSDT`
- `HBARUSDT`

That is not perfect, but `8/10` is a good sign for the emerging-stage transition logic.

### 7. Watchlist Telegram suppression worked

There were `0` Telegram messages containing `WATCHLIST` in the reviewed window.

That is good. It means the feed is no longer wasting operator attention on broad provisional watchlist noise.

## Key Findings

### 1. Leaderboard stability is much better, but not fully solved

Short-horizon instability is dramatically better, but some symbols still span both extremes over the full day.

Names that appeared in both top-5 and bottom-5 sets during the window include:

- `STGUSDT`: top-5 `38` times, bottom-5 `23` times
- `RUNEUSDT`: top-5 `29` times, bottom-5 `23` times
- `SHIB1000USDT`: top-5 `22` times, bottom-5 `24` times
- `ETCUSDT`: top-5 `35` times, bottom-5 `2` times
- `RENDERUSDT`: top-5 `16` times, bottom-5 `20` times
- `TAOUSDT`: top-5 `4` times, bottom-5 `22` times
- `ORDIUSDT`: top-5 `9` times, bottom-5 `33` times
- `PENDLEUSDT`: top-5 `3` times, bottom-5 `39` times

This is not the same failure as the prior run. These are not immediate whipsaws.

But it still means some names are not expressing clean multi-day momentum leadership. They are rotating from strength to weakness over the day strongly enough to hit both extremes.

For a 3-7 day momentum system, that is still a real issue.

### 2. BTC regime is still basically dead information

Every confirmed summary in the window had:

- `BTC regime: 1`

So the regime model currently contributed zero real state variation over the full sample.

That does not mean the regime code is broken.
It means that in this market window, the regime model was not informative.

Operationally, this is weak because:

- regime cannot help separate cleaner from dirtier conditions if it never changes
- the threshold modulation is effectively constant

### 3. BTCDOM improved behavior, but this sample did not exercise the full state machine

The window only saw:

- `neutral`
- `rising`

It never saw:

- `falling`

So the dominance redesign looks better, but this run did not test the full intended tri-state behavior.

Also:

- dominance change ranged only from `-0.19%` to `+0.30%`

That is not much spread.

So the system is using the new BTCDOM input, but this sample is not yet enough to conclude that the full dominance-state design is well tuned.

### 4. Confirmed leadership is now more persistent, which is good

Top-1 leadership was concentrated instead of flapping every bar:

- `TONUSDT` was top-ranked in `30` confirmed cycles
- `TRXUSDT` was top-ranked in `27` confirmed cycles
- `ETCUSDT` was top-ranked in `14` confirmed cycles

Longest top-1 streaks:

- `TONUSDT`: `29` consecutive cycles
- `TRXUSDT`: `10` consecutive cycles
- `TRXUSDT`: `9` consecutive cycles
- `ETCUSDT`: `8` consecutive cycles

This is a good change.

The earlier system looked like it could barely decide who the leader was from one cycle to the next. This one is much more willing to let a leader remain a leader.

### 5. Top-5 churn still exists, but it is now moderate instead of absurd

Across consecutive confirmed cycles:

- average top-5 overlap: `3.78`
- minimum overlap: `1`
- maximum overlap: `5`
- average symmetric difference in top-5 membership: `2.43`

That means the top 5 usually retains about `3 to 4` names from one cycle to the next.

That is not perfectly stable, but it is dramatically less chaotic than the prior sample.

### 6. Confirmed signal distribution is now led by a smaller set of names

Most frequent qualified confirmed tickers:

- `TRXUSDT`: `54`
- `TONUSDT`: `45`
- `VETUSDT`: `26`
- `ETCUSDT`: `23`
- `STGUSDT`: `22`
- `SHIB1000USDT`: `14`
- `ENSUSDT`: `13`
- `NEARUSDT`: `12`

That concentration can be interpreted two ways:

- good: the system is identifying persistent leadership instead of constantly rotating noise
- bad: it may be over-concentrating on a handful of names

Right now I would read it as mostly good, but it is something to monitor.

## Issues

### 1. Macro regime still needs work

If regime stays at `1` for an entire day again, then it is not helping enough.

Possible next questions:

- is the regime definition too coarse?
- is the daily timeframe too slow for what you want it to do?
- or is the current market genuinely that flat by this regime model?

This is now more of an information-content problem than a code problem.

### 2. Longer-horizon top/bottom span still exists

The short-horizon flip problem is fixed enough to stop being the headline issue.

But if symbols like `STGUSDT`, `RUNEUSDT`, `SHIB1000USDT`, `ORDIUSDT`, and `PENDLEUSDT` still appear on both extremes in the same day, then the ranking stack is still not fully aligned with a stable 3-7 day momentum objective.

It is better, but not clean.

### 3. The sample did not prove the `falling` dominance state

No `falling` BTCDOM states were observed in the reviewed window.

That means:

- the new logic is running
- but one-third of the intended macro-state machine still did not get exercised

So the design is more complete than before, but the live evidence is still incomplete.

### 4. Two emerging alerts failed to confirm quickly

The non-converting emerging alerts were:

- `ATOMUSDT`
- `HBARUSDT`

That is acceptable, but it is still a reminder that the intrabar path is not equivalent to the confirmed path.

## Good Things

- The summary system is now operationally clean: no duplicates, no cadence gaps.
- The top-to-bottom teleport problem has been removed in this sample.
- Momentum is now clearly the dominant ranking component.
- Emerging alerts are often useful precursors rather than pure noise.
- Watchlist Telegram noise is gone.
- Dominance no longer appears to be suffocating the board.
- Confirmed leaders are now allowed to persist instead of flipping constantly.

## Bottom Line

This run is a clear improvement over the prior one.

The main original problem, violent short-horizon leaderboard instability, is no longer the dominant issue.

The system now looks much closer to a usable momentum detector.

But it is not fully where it should be for a 3-7 day momentum engine.

The remaining weakness is no longer “curvature is breaking everything.”
The remaining weakness is:

- the macro layer is still low-information
- some names still traverse the full rank range over longer horizons

That is a much better class of problem to have.

## Recommended Next Focus

1. Keep the current momentum / curvature weighting as-is for another live window.
2. Observe whether BTC regime remains stuck at `1` across another day.
3. Watch whether BTCDOM ever enters a meaningful `falling` state.
4. If long-horizon extreme-span names remain common, the next likely tuning target is confirmed persistence / stability weighting rather than more curvature cuts.
