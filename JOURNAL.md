# DEVELOPMENT JOURNAL

Personal notes, thoughts, learnings, and decisions during development.

Format: Date-stamped entries with context, questions, insights, and next steps.

---

## 2025-10-17 - Strategy Pivot: HFT → Statistical Arbitrage

### Context
Started questioning whether the 24-hour z-score window was too short. Realized I was trying to mix HFT patterns (ultra-short windows) with stat-arb logic (mean reversion).

### Key Realizations

**"Would Jim Simons use EWMA smoothing?"**
- No - smoothing introduces lag, kills edge in stat-arb
- If signals are jumpy, that's information (volatility clustering)
- Better approach: position sizing based on confidence, not smoothing signals

**"What if it doesn't revert?"**
- This is the risk that kills stat-arb traders (LTCM example)
- Need mandatory stop losses (z-score >5 = regime change, admit wrong)
- Position sizing critical (2% max per trade)
- Portfolio heat limits (20% max capital deployed)
- Expected to be wrong 40-45% of the time - edge comes from asymmetry + sample size

**"How is this different from CoinMarketCap top movers?"**
- CMC: "What moved the most?" (reactive, no edge)
- This system: "What's statistically overextended?" (predictive mean reversion)
- We're betting on reversion of extremes, not chasing what already moved

**"Can't backtest yet"**
- Survivorship bias (coin list = today's winners, missing dead coins)
- Look-ahead bias (using current market cap on historical data)
- No transaction costs modeled
- No execution simulation
- Jim Simons would say: "Come back with clean data"
- Better path: Paper trade forward for 2-3 months

### Technical Decisions Made

**Configuration changes:**
- Z-score window: 24h → 14 days (4032 bars)
- Velocity window: 2h → 24h (288 bars)
- Acceleration: DISABLED (third derivative = noise)
- Data requirement: 3 days → 30 days

**Strategy confirmed:**
- Mean reversion stat-arb (NOT momentum)
- Primary edge: BTC-relative z-score
- Entry: |Z| > 2.0
- Exit: Z crosses back to 0
- Hold time: Days/weeks

**Risk framework:**
- 2% position size max
- Z-score >5 stop loss
- 20% portfolio heat limit
- Correlation checks (don't short 5 correlated memecoins)
- Drawdown circuit breaker (>10% = cut size 50%, >20% = pause)

### Data Sources Decision

**Derivatives data:**
- Binance API (free) is sufficient
- Coinglass premium NOT needed (too expensive, diminishing returns)
- Binance alone = 40%+ of derivatives market
- Renaissance didn't have derivatives data in the 90s - made billions on price/volume alone

### Questions Still Open

1. **Momentum strategy later?**
   - Could add 90-day momentum on top of 14-day mean reversion
   - Different timeframes = complementary, not contradictory
   - AQR, Winton, Man Group do this (trend following on months/years)

2. **When to actually trade?**
   - Need to paper trade forward first (2-3 months minimum)
   - Collect real execution metrics
   - Validate that mean reversion edge actually exists in crypto

3. **How to handle regime changes?**
   - MTF confirmation helps (don't fade strong aligned trends)
   - But still vulnerable to structural shifts
   - Need monitoring for correlation regime changes

### Insights & Learnings

**Stat-arb is HARD:**
- Not just "find extremes and fade them"
- Risk management is 80% of the edge
- LTCM had Nobel Prize winners and still blew up
- "It keeps going" scenarios need explicit defenses

**Data quality matters more than data quantity:**
- Binance free API > expensive Coinglass premium with bad methodology
- Clean OHLCV + basic derivatives > fancy features with lookahead bias

**Focus hierarchy:**
1. Get 30 days of clean data
2. Verify signals are stable with 14-day windows
3. Paper trade forward (NOT backtest)
4. Add risk management layer
5. Then consider live trading

### Personal Notes

Felt overwhelmed today thinking about how hard this is. Derivatives data is expensive. School internet blocks Binance API.

But realized: I'm asking the right questions. Most people don't question their assumptions. I'm thinking about risk management BEFORE blowing up. That's the important part.

The code is easy. The framework is hard. But I'm building the framework by asking questions like:
- "What if it doesn't revert?"
- "Would Jim Simons do this?"
- "How is this different from CMC?"

That's the real work.

### Next Steps

**When I get home (internet access):**
1. Enable data download (ENABLE_DATA_DOWNLOAD = True)
2. Download 30 days of OHLCV data for all coins
3. Run system and observe signal stability
4. Look at how often z-score >2 happens
5. Check half-life distribution (are they realistic?)

**Before live trading:**
1. Paper trade for 2-3 months
2. Track every signal, every entry, every exit
3. Measure actual win rate, average win/loss, max drawdown
4. See if the edge actually exists

**Not doing yet:**
- Backtesting (too many biases, not ready)
- Live trading (need paper trading validation first)
- Coinglass premium (don't need it)

---






## Template for Future Entries

```markdown
## YYYY-MM-DD - [Title]

### Context
What was I working on? What triggered this session?

### Key Realizations
What did I learn? What clicked?

### Technical Decisions Made
What did I change and why?

### Questions Still Open
What don't I know yet?

### Insights & Learnings
Broader lessons beyond just technical details

### Personal Notes
How am I feeling? Challenges? Wins?

### Next Steps
What to do next session
```
