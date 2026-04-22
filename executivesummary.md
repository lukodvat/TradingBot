# TradingBot — Executive Summary

## What It Is

A fully autonomous paper-trading bot running on a $10,000 simulated Alpaca account. It trades a curated watchlist of 32 large-cap stocks across 10 sectors. No real money is ever at risk — the system refuses to start if live trading credentials are detected.

## How It Makes Decisions

The bot combines two independent signals:

- **Market signal** — technical indicators identify tickers in confirmed uptrends with strong momentum and above-average volume.
- **News signal** — an AI pipeline reads financial headlines twice a day (morning + midday) and assigns each ticker a bullish, neutral, or bearish lean.

A trade is only taken when both signals agree. The AI never picks trades — it only vetoes them when the news looks bad.

## Risk Posture

Every risk rule is hard-coded and tested — nothing can be overridden at runtime.

- Every trade risks a fixed **$50** (0.5% of the account), regardless of stock volatility.
- **Stop-losses, take-profits, and trailing stops** are attached to every order at entry.
- A **3% daily loss limit** liquidates all positions and emails an alert immediately.
- **No overnight or weekend exposure** — positions flatten by market close.
- **No new trades around earnings reports** or on Fed / inflation-data days.
- **Sector and portfolio caps** prevent concentration in any single theme.

## Operational Cadence

- **Morning + midday** — AI reads the news, assigns biases
- **Every hour from 10:30–15:30 ET** — scanner looks for setups and places trades
- **4:30 PM ET** — daily recap email with a performance grade and a summary written by the AI

## Key Numbers

| | |
|---|---|
| Account size | $10,000 (paper) |
| Risk per trade | $50 (0.5% of equity) |
| Daily loss limit | $300 (3% — triggers full liquidation) |
| Watchlist | 32 tickers across 10 sectors |
| Monthly AI spend | ~$3–4 (hard cap at $10) |
| Infrastructure | One always-on server, deploys automatically on code push |
