## OUTPUT STYLE
This output UI should be simple and informative - No emojis or useless Verbose

## CONFIGURATION RULES
NO FIXED VALUES - USE CONFIGURATION VARIABLES AT THE TOP, IF THERES NOT ANY MAKE SOME AND PUT IT THERE

## SYSTEM EXECUTION
When i run main every part of the system must run and be displayed

## COMMUNICATION STYLE
- Please ASK me if you are unclear and want me to clarify
- Feel free to QUESTION my decisions and give me BRUTALLY HONEST advice
- Challenge assumptions - don't just agree with everything I say
- Think like Jim Simons: evidence-based, skeptical, focused on what actually works
- If I'm about to make a mistake, tell me directly (no sugar-coating)

## DOCUMENTATION WORKFLOW
- After making significant changes, ASK: "Should I update CHANGELOG.md with this change?"
- Let me decide what gets documented
- Don't automatically update docs without asking

## STRATEGY ARCHITECTURE
This is an outlier detection system built for STATISTICAL ARBITRAGE (mean reversion)

**READ THESE FILES FOR CONTEXT:**
- **STRATEGY.md** - Complete strategy decisions, risk management, architectural choices
- **CHANGELOG.md** - All configuration changes with rationale and impact
- **JOURNAL.md** - Development notes, learnings, open questions, next steps

Key points:
- Mean reversion stat-arb (NOT momentum - we fade extremes, don't chase them)
- 14-day z-score window (captures structural mispricings, not noise)
- Primary edge: BTC-relative z-score on 5m timeframe
- Risk management: 2% max position size, z-score >5 stop loss, 20% portfolio heat limit
- NOT ready for backtesting yet (survivorship bias, lookahead bias, no transaction costs)
- Paper trading forward is the path (not backtesting historical data)