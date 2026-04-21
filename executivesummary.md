# TradingBot — Executive Summary

## What It Is

An autonomous paper-trading system that runs on a ~$10,000 Alpaca paper account. It combines quantitative technical signals with LLM-driven news sentiment to trade a 32-ticker watchlist across 10 sectors. The system is paper-only by hard constraint — it will not start if a live API key is detected.

## How It Makes Decisions

**The LLM does not find trades. It filters out bad ones.**

Each trading day follows a fixed schedule:

- **3× daily (9AM, 10AM, 1PM ET)** — an LLM pipeline fetches financial headlines from Finnhub, runs them through a cheap keyword filter and then Claude Haiku/Sonnet scoring, and writes a BULLISH / NEUTRAL / BEARISH bias per ticker to SQLite. Cost: ~$2–3.50/month.
- **6× daily (10:30–15:30 ET, every hour)** — a pure-quant scanner reads those biases for free, finds technical setups, and submits bracket orders. Zero LLM API calls.
- **4:30 PM ET** — Claude Haiku writes a one-paragraph daily recap with a performance grade, sent as an HTML email.

## Entry Criteria

A ticker only qualifies for entry when **all** of the following are true:

1. Price above EMA(20) — uptrend confirmed
2. RSI(14) between 50–80 — momentum without being overbought
3. Volume ≥ 1.2× 20-day average (2.0× at the 10:30 opening scan)
4. Sentiment bias is not contradicting the direction (BEARISH blocks longs; BULLISH blocks shorts; NEUTRAL passes)
5. Ticker's 20-day return exceeds SPY's (relative strength — longs only)
6. Price within 10% of its 63-day high (near-high filter — longs only)

## Risk Management

Every guardrail is hard-coded. None can be overridden at runtime.

| What | How |
|---|---|
| Stop-loss | ATR × 1.5 from entry (minimum 1%) |
| Take-profit | +6% from entry |
| Partial exit | Scale out 50% at +3% unrealized gain — locks in profit, keeps a runner |
| Trailing stop | Activates at +1.5% gain, trails at 1% |
| Position sizing | ATR-based: each trade risks exactly 0.5% of equity ($50 on a $10k account) |
| Sentiment boost | When bias matches direction, position size scales up 1.25× |
| Max open positions | 5 (reduced to 2 when market volatility is elevated) |
| Max per position | 10% of equity |
| Max per sector | 30% of equity |
| Portfolio heat | Total open dollar-risk capped at 4% of equity |
| Daily loss limit | If down 3% intraday, liquidate everything and send an immediate alert email |
| Regime filter | SPY EMA(50/200) → CAUTION blocks new longs; BEAR halts all entries |
| Overnight exposure | Flatten all positions at 15:30 ET (hold exception if up >1% + matching bias) |
| Earnings / macro | No new entries within 2 days of earnings or on FOMC/CPI dates |

## Backtest Gate

`main.py` refuses to start without a passing backtest report in `reports/`. The harness replays the exact same signal and exit logic as the live system on historical daily bars, with 10bps slippage applied to every fill. The report is fingerprinted against the current config — any strategy-relevant setting change invalidates it and requires a re-run.

## Tech Stack (one-line)

Python 3.12 · Alpaca paper API · Anthropic Haiku + Sonnet · Finnhub news · pandas-ta · APScheduler · SQLite (WAL) · Streamlit dashboard · 374 pytest tests

## Key Numbers

| | |
|---|---|
| Account size | ~$10,000 (paper) |
| Watchlist | 32 tickers, 10 sectors |
| LLM budget | Hard-stop at $10/month (~$2–4 typical) |
| Risk per trade | $50 (0.5% of equity) |
| Max daily loss | $300 (3% circuit breaker) |
| Scheduled runs/day | 10 (3 LLM + 6 quant + 1 email) |
