# What to Learn While Paper Trading (2-3 Months)

Your stat-arb system is running. Here's what to do while collecting 50 trades.

---

## Week 1-4: Foundation Building

### 1. **Statistical Arbitrage Deep Dive**

**Books** (Read in order):
1. **"Dynamic Hedging" by Nassim Taleb** (Chapters 1-5)
   - Understand options, hedging, risk management
   - Foundation for pairs trading mechanics
   - Why: You're trading spreads, not directions

2. **"Quantitative Trading" by Ernest Chan** (Chapters 3-4)
   - Mean reversion strategies
   - Cointegration vs correlation
   - Why: Chan built stat-arb systems at hedge funds

3. **"Algorithmic Trading" by Ernest Chan** (Chapter 7)
   - Pairs trading in practice
   - Transaction costs modeling
   - Why: Real-world implementation details

**Papers** (Free on SSRN):
- "Pairs Trading: Performance of a Relative-Value Arbitrage Rule" (Gatev et al., 2006)
- "Statistical Arbitrage in the U.S. Equities Market" (Avellaneda & Lee, 2010)

**YouTube Courses**:
- QuantInsti: "Pairs Trading Strategy" (free playlist)
- Khan Academy: "Statistics & Probability" (refresh z-scores, distributions)

### 2. **Python for Quant Finance**

**Practice Projects** (Build these while waiting):
1. **Correlation Matrix Calculator**
   - Build heatmap of all 62 coins
   - Identify correlated pairs (avoid opening 5 memecoins at once)
   - Tools: pandas, seaborn

2. **Performance Dashboard**
   - Read `paper_trades/positions.parquet`
   - Plot: Cumulative P&L, drawdown curve, win rate over time
   - Tools: plotly, dash (interactive)

3. **Signal Quality Analyzer**
   - Analyze: Which coins have best win rates?
   - Analyze: Which half-life ranges perform best?
   - Analyze: Volume surge correlation with success?

**Courses**:
- "Python for Finance" (freeCodeCamp YouTube - 4 hours)
- "Pandas for Financial Analysis" (Coursera - free audit)

### 3. **Risk Management & Position Sizing**

**Key Concepts to Study**:
- **Kelly Criterion**: Optimal bet sizing based on edge
- **Sharpe Ratio**: Risk-adjusted returns (your target: >1.0)
- **Sortino Ratio**: Downside risk only (better than Sharpe)
- **Maximum Drawdown**: How much you can lose before stopping
- **Value at Risk (VaR)**: Probability of losses

**Exercise**: Calculate Kelly position size for your strategy
```python
# After 20 trades, calculate:
win_rate = 0.60  # Example
avg_win = 0.05   # 5% per trade
avg_loss = 0.03  # 3% per trade

# Kelly formula: f = (p*b - q) / b
# f = optimal position size
# p = win rate, q = loss rate, b = win/loss ratio
```

---

## Month 2: Advanced Topics

### 1. **Cointegration vs Correlation**

**Your current system**: Uses correlation (BTC-relative z-score)

**Upgrade path**: Study cointegration
- Pairs with cointegration mean-revert MORE reliably
- Can find coin-coin pairs (not just coin-BTC)
- Tools: `statsmodels.tsa.stattools.coint()`

**Reading**:
- "Cointegration and Pairs Trading" (Vidyamurthy, 2004)
- "Quantitative Momentum" by Wesley Gray (Chapter 6)

**Project**: Test if AAVE-ETH is cointegrated (better than AAVE-BTC?)

### 2. **Transaction Cost Analysis**

**Study these costs** (you're modeling 0.3%, but is it accurate?):
- Binance futures fees: 0.02% maker, 0.05% taker
- Slippage: Varies by coin liquidity (PEPE > BTC)
- Funding rate: 0.01% per 8h (but varies by market sentiment)
- Opportunity cost: Capital tied up = missed trades

**Project**: Build transaction cost breakdown per coin
```python
# Calculate actual costs from closed trades
actual_costs = (entry_fee + exit_fee + slippage + funding)
```

### 3. **Regime Detection**

**Problem**: Mean reversion works in ranging markets, fails in trends

**Study**:
- Hidden Markov Models (HMM) for regime detection
- Volatility clustering (GARCH models)
- Market microstructure changes

**Project**: Add regime filter
- Don't open positions during trending markets
- Use 50-day moving average as simple filter
- If BTC trending >5% per day → pause new trades

---

## Month 3: Optimization & Live Prep

### 1. **Performance Analysis**

**By Month 3, you should have 30-50 closed trades**

**Analysis Checklist**:
- [ ] Win rate: Is it 55-60%? (If <50%, stop trading)
- [ ] Profit factor: >1.5? (Total wins / total losses)
- [ ] Sharpe ratio: >1.0? (Risk-adjusted returns)
- [ ] Max drawdown: <15%? (Capital preservation)
- [ ] Stop-out rate: <15%? (If >15%, signals breaking)

**Breakdown by Variable**:
- Win rate by coin (which coins work best?)
- Win rate by half-life (<24h vs >48h)
- Win rate by z-score entry (z=2.0 vs z=2.5)
- Win rate by volume surge (with vs without)

### 2. **Strategy Improvements** (Evidence-Based Only)

**IF win rate <55%**:
- Entry threshold: Test z=2.5 instead of z=2.0
- Half-life filter: Skip trades with half-life >48h
- Coin filter: Remove low-liquidity coins (market cap <$500M)

**IF Sharpe <1.0**:
- Position sizing: Reduce to 1.5% per leg
- Portfolio heat: Reduce max to 30% (from 40%)
- Risk per trade: Tighten stops to z=4.0 (from 5.0)

**IF stop-out rate >15%**:
- Don't tighten stops → tighten ENTRY (z=2.5)
- Add MTF filter: Require 3+ aligned timeframes (from 2+)
- Add trend filter: Skip if BTC trending >5% per day

### 3. **Live Trading Preparation**

**Before going live** (after 50+ paper trades):
1. **Capital Plan**: How much to risk? ($10k minimum for stat-arb)
2. **Exchange Setup**: Binance Futures account, API keys, 2FA
3. **Execution System**: Build order management (limit orders, not market)
4. **Monitoring**: Alerts for stop losses, position limits, API failures
5. **Tax Planning**: Track cost basis, P&L for taxes

**Red Flags** (Don't go live if):
- ❌ Win rate <50% (no edge)
- ❌ Sharpe <0.5 (risk not worth reward)
- ❌ Max drawdown >20% (too risky)
- ❌ <50 closed trades (not statistically significant)

---

## Side Projects to Build

### 1. **Real-Time Dashboard**
- Live positions, P&L, current z-scores
- Alerts when positions approach stop loss
- Tools: Streamlit, Dash, or Flask

### 2. **Backtesting Engine**
- Download 1 year of historical data
- Test different entry thresholds (z=2.0 vs 2.5 vs 3.0)
- Test different stop losses (z=4.0 vs 5.0 vs 6.0)
- Walk-forward validation (train on 6mo, test on 3mo)

### 3. **Correlation Monitor**
- Calculate 62x62 correlation matrix daily
- Alert if opening 3+ correlated positions
- Visualize: Which coins move together?

### 4. **Alternative Strategies** (Research only)
- Momentum: Ride winners instead of fading extremes
- Breakout: Trade z-score >3 in direction of trend
- Multi-strategy: Combine mean reversion + momentum

---

## Learning Resources

### Free Courses
- **QuantInsti**: Statistical Arbitrage (free webinar)
- **Coursera**: "Machine Learning for Trading" (Georgia Tech)
- **Khan Academy**: Statistics & Probability (refresh fundamentals)

### Books (Must-Read)
1. "The Man Who Solved the Market" by Gregory Zuckerman (Simons biography)
2. "Quantitative Trading" by Ernest Chan
3. "Algorithmic Trading" by Ernest Chan
4. "Dynamic Hedging" by Nassim Taleb (advanced)

### Podcasts
- "Chat With Traders" (episodes on stat-arb)
- "Flirting With Models" (quant interviews)
- "Top Traders Unplugged" (macro + quant)

### Communities
- r/algotrading (Reddit)
- Quantocracy.com (quant blog aggregator)
- QuantConnect forums (quant developers)

---

## Weekly Checklist (During Paper Trading)

**Monday**:
- [ ] Check automation logs (did system run over weekend?)
- [ ] Review open positions (`python3 src/view_positions.py`)
- [ ] Check for new closed trades

**Wednesday**:
- [ ] Spot-check z-scores (are signals making sense?)
- [ ] Review any stop losses (why did they trigger?)

**Friday**:
- [ ] Weekly P&L review
- [ ] Update learning journal (what did I learn this week?)
- [ ] Read 1 paper or 1 book chapter

**Monthly**:
- [ ] Performance analysis (win rate, Sharpe, drawdown)
- [ ] Update JOURNAL.md with learnings
- [ ] Adjust strategy IF evidence supports (not based on 1 bad week)

---

## The Patience Game

**Jim Simons took 10 years** to build Medallion Fund. You're in Week 1.

**Your timeline**:
- **Month 1**: System running, learning basics
- **Month 2**: 20-30 trades, pattern recognition
- **Month 3**: 50+ trades, statistical significance
- **Month 4**: Decision point (go live, adjust, or stop)

**What NOT to do**:
- ❌ Change thresholds after 5 trades
- ❌ Add complexity because "it's not working yet"
- ❌ Panic after first losing week
- ❌ Go live before 50 trades

**The Simons Way**:
- Gather data
- Analyze systematically
- Let statistics decide
- Be patient

You're on the right path. Now let the system work.
