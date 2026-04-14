# TradingBot — Hybrid Quant + LLM Paper Trading Bot

## Project Overview

Autonomous paper-trading system combining technical indicators with LLM-driven news
sentiment confirmation and hard-coded risk guardrails. Runs on Alpaca paper trading
with ~$10k account. The LLM does NOT find trades — it vetoes bad ones.

**Core thesis:** Technical-momentum entries confirmed by news sentiment, with
mechanical risk management. The primary signal is quantitative (trend + momentum +
volume). Sentiment acts as a daily GO/NO-GO gate — it sets a per-ticker bias
(BULLISH / NEUTRAL / BEARISH) twice per day that the quant scanner reads for free
from SQLite on every hourly pass. The LLM does not fire on every quant run.

## Hard Constraints (non-negotiable)

- **Paper trading ONLY.** Alpaca paper endpoint. Fails loudly if live key detected.
- **Account size:** ~$10,000
- **LLM budget:** <$20/month. SQLite ledger tracks per-call spend. Hard-stop at $18.
- **LLM runs 2x per trading day:** 10:00 ET (morning) and 13:00 ET (midday).
- **Quant scanner runs every hour:** 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
  (6 windows). Reads sentiment bias from SQLite — no LLM API call.
- **Daily email at 16:30 ET:** LLM-written summary with grade 1–10, P&L, trades,
  period performance. Sent via Gmail SMTP.
- **Same-ticker cooldown:** once a position is entered, that ticker is skipped for
  the remainder of the trading day.
- **No Docker for v1.** Python venv + APScheduler in long-running process.
- **Backtest gate:** `main.py` refuses to start without a passing backtest report
  in `reports/`. Run `python run_backtest.py` first.

## Directory Structure

```
TradingBot/
├── config/
│   ├── settings.py          # Pydantic BaseSettings — all tunables, paper-only guard
│   └── watchlist.yaml       # 20 tickers + static sector map
├── core/
│   ├── broker.py            # Alpaca paper client; raises LiveKeyError on live URL
│   ├── risk.py              # All guardrails: sizing, stops, sector cap, circuit breaker
│   └── portfolio.py         # Trailing stop activation, flatten logic, snapshot recording
├── data/
│   ├── market.py            # OHLCV fetching via alpaca-py historical bars
│   └── news.py              # NewsProvider ABC + FinnhubProvider; headline de-dupe
├── analysis/
│   ├── volatility.py        # Quant gate: ATR/price + realized vol (pandas-ta)
│   ├── sentiment.py         # Tiered LLM triage orchestrator (T1 → T2 → T3)
│   └── signals.py           # Quant signal scan; reads sentiment_bias from SQLite
├── llm/
│   ├── client.py            # Anthropic SDK wrapper; per-call cost tracking
│   ├── prompts.py           # Versioned prompt templates (Haiku vs Sonnet)
│   └── budget.py            # Monthly spend ledger; hard-stop at $18
├── backtest/
│   ├── harness.py           # Timestamp-gated replay engine + check_backtest_gate()
│   ├── metrics.py           # Sharpe, max drawdown, win rate, expectancy
│   └── loader.py            # Historical bar loading with min-days validation
├── db/
│   ├── schema.py            # 9 SQLite tables; init_db()
│   └── store.py             # Thin parameterized read/write helpers (no ORM)
├── notifications/
│   └── email.py             # Daily HTML email: LLM summary, P&L, trades, performance
├── tests/
│   ├── test_risk.py
│   ├── test_volatility.py
│   ├── test_backtest.py
│   ├── test_sentiment.py
│   ├── test_signals.py
│   ├── test_portfolio.py
│   ├── test_main.py
│   └── test_email.py
├── main.py                  # Entrypoint: pre-flight → APScheduler (Jobs A, B, C)
├── run_backtest.py          # Standalone CLI: run backtest, save report to reports/
├── dashboard.py             # Streamlit dashboard (read-only, port 8501)
├── .env                     # Your keys — never commit this
├── .env.example
├── requirements.txt
└── README.md
```

## Three Scheduled Jobs

All share one SQLite database. No direct coupling between jobs.

---

### Job A — LLM Sentiment (10:00 ET, 13:00 ET)

**Purpose:** Set per-ticker daily sentiment bias that Job B reads for free.

1. Calendar gate — weekday only.
2. LLM budget check — abort if MTD spend >= $18.
3. Fetch Finnhub headlines (4h lookback at 10:00, 2h at 13:00).
4. De-duplicate against `headlines_seen`. Persist new ones.
5. **Tier 1** — regex/keyword rejection (free). Drops SEC filings, routine dividends, etc.
6. **Tier 2** — Haiku: `{sentiment: -1..+1, confidence: 0..1}`. Logs cost to `llm_calls`.
7. **Tier 3** — Sonnet: only if `|sentiment| > 0.6` AND `confidence > 0.75`. Cap 5/run.
8. Aggregate per-ticker → BULLISH / NEUTRAL / BEARISH. Upsert `sentiment_bias`.

**Cost estimate:** ~$0.06/day → ~$1.25/month.

---

### Job B — Quant Scanner (10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET)

**Purpose:** Find and execute technical setups. Zero LLM cost.

1. Calendar + time gate.
2. Circuit breaker — if daily P&L < -3% equity, liquidate all and halt.
3. Trailing stop activation — upgrade fixed stops on positions up ≥ 1.5%.
4. Flatten check — at 15:30 ET, close positions unless up >1% AND bias matches direction.
   Friday: always flatten all regardless.
5. Fetch OHLCV (last 25 trading days). Run volatility filter (ATR/price, realized vol).
6. Signal scan — EMA(20) trend + RSI(14) in [40,70] + volume > 1.5x avg + matching bias.
7. Risk pre-checks per candidate: position limit (5), sector cap (25%), buying power.
8. Submit bracket orders: limit entry + 2% stop-loss.
9. Record equity snapshot + upsert daily_pnl.

---

### Job C — Daily Email (16:30 ET)

1. Collect day's trades, P&L, open positions, sentiment biases from SQLite.
2. Call Haiku to write a 2–3 sentence summary and assign a grade 1–10.
3. Build HTML email: summary, grade, performance table (Today/Week/Month/6M/YTD/1Y),
   trades table, open positions, LLM spend bar.
4. Send via Gmail SMTP.

---

## SQLite Tables (9)

| Table | Purpose |
|---|---|
| `trades` | Filled trades: ticker, side, qty, fill price, realized P&L, slippage |
| `orders` | All orders (pending/filled/cancelled) with limit vs fill price |
| `llm_calls` | Every LLM call: model, tier, tokens, cost, sentiment, confidence |
| `daily_pnl` | Per-day: open equity, close equity, realized, unrealized |
| `equity_snapshots` | Per-run equity snapshot — powers the equity curve chart |
| `headlines_seen` | Headline de-dupe store + full T1/T2/T3 triage audit trail |
| `sentiment_bias` | Ticker + date + bias. **Key decoupling table between Job A and B.** |
| `session_log` | One row per run: tickers evaluated, tier counts, circuit breaker state |
| `volatility_filter_log` | Per-ticker ATR/vol values and pass/fail reason |

## Risk Guardrails

| Guardrail | Value | Where enforced |
|---|---|---|
| Stop-loss | 2% from entry | bracket order at submission |
| Trailing stop activation | +1.5% unrealized | Job B position management |
| Trailing stop distance | 3% | `broker.replace_stop_with_trailing()` |
| Max positions | 5 | `risk.size_position()` |
| Max position size | 10% of equity | `risk.size_position()` |
| Max sector concentration | 25% of equity | `risk.size_position()` |
| Daily circuit breaker | -3% equity | Job B pre-flight |
| Overnight flatten | 15:30 ET (with exception) | `portfolio.manage_flattens()` |
| Weekend flatten | Friday 15:30 ET, always | `portfolio.manage_flattens()` |
| LLM hard-stop | $18/month | `llm/budget.py` |
| Paper-only guard | URL must contain "paper" | `Settings` model_validator |
| Backtest gate | Must have passing report | `main.py` startup |

## Backtest

Two-phase strategy:

- **Phase 1 (now):** Quant-only. Establishes baseline. Must show positive expectancy
  (`expectancy > 0`, `trades >= 10`, `sharpe > 0`) before paper trading starts.
- **Phase 2 (after 2–3 weeks live):** Add sentiment filter. Refine thresholds.

Slippage model: configurable `--slippage-bps` (default 10bps) on every fill.
Lookahead prevention: signal on day N → entry fills at day N+1 open price.

## Dashboard (Streamlit)

`dashboard.py` — read-only, connects to the same SQLite database.

Sections: live stats header, equity curve, drawdown chart, performance table,
trades (today / all / win-loss histogram), open positions, sentiment bias heatmap,
LLM cost tracker, volatility filter log, session log, headlines + triage scores.

Run: `streamlit run dashboard.py --server.port 8501`
Access: SSH tunnel → `ssh -L 8501:localhost:8501 user@your-server`

## Tech Stack

- Python 3.12
- `alpaca-py` — broker + market data
- `anthropic` SDK — Haiku (Tier 2 + email) and Sonnet (Tier 3)
- `finnhub-python` — news headlines (free tier, rate-limited)
- `pandas` + `pandas-ta` — indicators (functional API)
- `APScheduler` — three scheduled jobs in one long-running process
- `pydantic-settings` — typed config with validation
- `plotly` + `streamlit` — interactive dashboard
- SQLite — single-file database, WAL mode, no ORM
- `pytest` — 291 tests across all modules

## Default Watchlist (20 stocks, 6 sectors)

| Sector | Tickers |
|---|---|
| Tech | AAPL, MSFT, GOOGL, AMZN, NVDA, META, AMD, CRM |
| Finance | JPM, BAC, GS |
| Healthcare | JNJ, PFE, UNH |
| Energy | XOM, CVX |
| Consumer | HD, WMT, DIS |

## Known Risks and Mitigations

1. **Stale news** — sentiment is a veto gate, not a signal source. Tight lookback window.
2. **Lookahead bias** — timestamp-gated joins, next-bar fills, slippage model, unit tests.
3. **Paper-fill optimism** — Alpaca fills at limit even when real markets wouldn't.
   Slippage model in backtest partially compensates.
4. **Headline recycling** — SQLite de-dupe by headline ID before any triage.
5. **Trailing stop management** — not fire-and-forget. Position management runs at the
   top of every Job B run to update active stops.
6. **SQLite concurrency** — WAL mode allows concurrent reads (dashboard + bot).
   Dashboard is read-only; no write contention.
