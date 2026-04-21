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
- **LLM budget:** <$10/month. SQLite ledger tracks per-call spend. Hard-stop at $10.
- **LLM runs 3x per trading day:** 09:00 ET (pre-market, 8h overnight lookback), 10:00 ET (morning, 24h lookback), 13:00 ET (midday, 4h lookback).
- **Quant scanner runs every hour:** 10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET
  (6 windows). Reads sentiment bias from SQLite — no LLM API call.
- **Daily email at 16:30 ET:** LLM-written summary with grade 1–10, P&L, trades,
  period performance. Sent via Gmail SMTP.
- **Immediate circuit breaker alert:** Separate email fires at the moment of
  liquidation — does not wait for 16:30 end-of-day summary.
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
│   ├── watchlist.yaml       # 32 tickers + static sector map (10 sectors)
│   └── macro_events.yaml    # FOMC + CPI blackout dates (no new entries on these days)
├── core/
│   ├── broker.py            # Alpaca paper client; raises LiveKeyError on live URL
│   ├── risk.py              # All guardrails: sizing, stops, sector cap, circuit breaker
│   └── portfolio.py         # Partial exits, trailing stops, flatten logic, snapshot recording
├── data/
│   ├── market.py            # OHLCV fetching via alpaca-py historical bars
│   └── news.py              # NewsProvider ABC + FinnhubProvider; headline de-dupe + earnings calendar
├── analysis/
│   ├── regime.py            # Market regime filter: BULL/CAUTION/BEAR via SPY EMA50/200 + vol
│   ├── volatility.py        # Quant gate: ATR/price + realized vol (pandas-ta)
│   ├── sentiment.py         # Tiered LLM triage orchestrator (T1 → T2 → T3)
│   └── signals.py           # Quant signal scan; reads sentiment_bias from SQLite
├── llm/
│   ├── client.py            # Anthropic SDK wrapper; per-call cost tracking
│   ├── prompts.py           # Versioned prompt templates (Haiku vs Sonnet)
│   └── budget.py            # Monthly spend ledger; hard-stop at $10
├── backtest/
│   ├── harness.py           # Timestamp-gated replay engine + check_backtest_gate()
│   ├── metrics.py           # Sharpe, max drawdown, win rate, expectancy
│   └── loader.py            # Historical bar loading with min-days validation
├── db/
│   ├── schema.py            # 11 SQLite tables (incl. partial_exits, email_log); init_db()
│   └── store.py             # Thin parameterized read/write helpers (no ORM)
├── notifications/
│   └── email.py             # Daily HTML email + immediate circuit breaker alert
├── tests/
│   ├── test_risk.py
│   ├── test_volatility.py
│   ├── test_backtest.py
│   ├── test_sentiment.py
│   ├── test_signals.py
│   ├── test_portfolio.py
│   ├── test_main.py
│   ├── test_regime.py
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

### Job A — LLM Sentiment (09:00, 10:00, 13:00 ET)

**Purpose:** Set per-ticker daily sentiment bias that Job B reads for free.

Three runs per trading day:
- **09:00 ET pre-market** — 8h lookback (captures overnight news 1AM–9AM). Biases ready before first quant scan.
- **10:00 ET morning** — 24h lookback (full prior-day narrative + morning). SQLite de-dup prevents re-processing.
- **13:00 ET midday** — 4h lookback (intraday update; overwrites morning bias if stronger signal).

Bias priority when multiple runs exist for the same ticker+date: midday > morning > premarket.

1. Calendar gate — weekday + NYSE holiday check.
2. LLM budget check — abort if MTD spend >= $10.
3. Fetch Finnhub headlines per lookback window.
4. De-duplicate against `headlines_seen`. Persist new ones.
5. **Tier 1** — regex/keyword rejection (free). Drops SEC filings, routine dividends, etc.
6. **Tier 2** — Haiku: `{sentiment: -1..+1, confidence: 0..1}`. Logs cost to `llm_calls`.
7. **Tier 3** — Sonnet: only if `|sentiment| > 0.6` AND `confidence > 0.75`. Cap 12/run.
8. Aggregate per-ticker → BULLISH / NEUTRAL / BEARISH. Upsert `sentiment_bias`.

**Cost estimate:** ~$0.10–0.16/day → ~$2.20–3.50/month (well within $10 cap).

---

### Job B — Quant Scanner (10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET)

**Purpose:** Find and execute technical setups. Zero LLM cost.

1. Calendar + time gate (NYSE holiday-aware).
2. **Fill reconciliation** — poll Alpaca for filled orders in last 24h, upsert into trades table.
3. Account snapshot.
4. **Circuit breaker** — if daily P&L < -3% equity, liquidate all, send immediate alert email, halt.
5. **Partial exits** — scale out 50% of position at +3% unrealized gain (single-shot per entry, tracked in `partial_exits` table).
6. **Trailing stop activation** — upgrade fixed stops on positions up ≥ 1.5%.
7. **Time-based exit** — close positions held ≥ 10 calendar days with < +1% unrealized gain.
8. Flatten check — at 15:30 ET (management-only run, no new entries), close positions unless up >1% AND bias matches direction. Friday: always flatten all regardless.
9. Same-ticker cooldown check — skip tickers already entered today.
10. Fetch OHLCV for watchlist + SPY (260 days for regime/near-high/rel-strength).
11. **Macro blackout** — if today is a FOMC/CPI date, skip new entries (still manages positions).
12. **Market regime filter** — SPY EMA(50/200) + realized vol → BULL/CAUTION/BEAR/UNKNOWN.
    - CAUTION: blocks long entries. BEAR: halts all new entries.
    - High vol (>30% annualized): reduces max positions to 2.
13. **Earnings blackout** — skip tickers with earnings within 2 calendar days.
14. Volatility filter (ATR/price, realized vol).
15. Signal scan — EMA(20) trend + RSI(14) in [50,80] + volume > 1.2x avg (2.0x at 10:30 open)
    + matching bias + **relative strength** (ticker 20d return > SPY 20d return, longs only)
    + **near-high filter** (price within 10% of 63-day high, longs only).
16. Apply regime direction filter to candidates.
17. **ATR-based sizing** — stop distance = ATR × 1.5; qty set so dollar-risk = 0.5% equity. Caps at max_position_pct and sector_headroom.
18. **Sentiment-as-sizer** — BULLISH bias on a LONG (or BEARISH on SHORT) scales target notional ×1.25.
19. Risk pre-checks per candidate: position limit (regime-adjusted), sector cap (30%), portfolio heat cap (4%), buying power.
20. Submit bracket orders: limit entry + ATR-derived stop-loss + **6% take-profit leg**.
21. Record equity snapshot + upsert daily_pnl.

---

### Job C — Daily Email (16:30 ET)

1. Collect day's trades, P&L, open positions, sentiment biases from SQLite.
2. Call Haiku to write a 2–3 sentence summary and assign a grade 1–10.
3. Build HTML email: summary, grade, performance table (Today/Week/Month/6M/YTD/1Y),
   trades table, open positions, LLM spend bar.
4. Send via Gmail SMTP.

---

## SQLite Tables (11)

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
| `partial_exits` | Scale-out records: symbol, entry_run_ts, qty_sold, fill_price — one row per partial exit |
| `email_log` | Every email send attempt: kind, recipient, subject, status (sent/failed), error |

**Trades table population:** `main.py` only writes to `orders` at submission time.
The `trades` table is populated by `reconcile_fills()` which polls Alpaca's `status=closed`
endpoint each Job B run and upserts actual fill data. This is the authoritative mechanism.

## Risk Guardrails

| Guardrail | Value | Where enforced |
|---|---|---|
| ATR-based stop-loss | ATR × 1.5 from entry (floor: 1%) | bracket order at submission |
| Take-profit | +6% from entry | bracket order at submission |
| Partial exit | Scale out 50% at +3% unrealized gain | `portfolio.manage_partial_exits()` |
| Trailing stop activation | +1.5% unrealized | Job B position management |
| Trailing stop distance | 1% | `broker.replace_stop_with_trailing()` |
| Time-based exit | 10 days held + <1% gain | `portfolio.manage_time_based_exits()` |
| Risk per trade | 0.5% of equity (ATR-scaled) | `risk.size_position()` |
| Max positions | 5 (2 in high-vol regime) | `risk.size_position()` + regime cap |
| Max position size | 10% of equity | `risk.size_position()` |
| Max sector concentration | 30% of equity | `risk.size_position()` |
| Portfolio heat cap | 4% aggregate open dollar-risk | `risk.size_position()` |
| Sentiment sizer | ×1.25 notional when bias matches direction | `risk.size_position()` |
| Daily circuit breaker | -3% equity | Job B pre-flight + immediate alert email |
| Overnight flatten | 15:30 ET (with exception) | `portfolio.manage_flattens()` |
| Weekend flatten | Friday 15:30 ET, always | `portfolio.manage_flattens()` |
| 15:30 entries gate | No new entries at final scan | `main.run_quant_job()` |
| Open session vol | 2.0× volume required at 10:30 ET | `signals.scan(volume_multiplier_override)` |
| Earnings blackout | 2 days before earnings | `news.get_upcoming_earnings()` |
| Macro blackout | FOMC + CPI dates | `config/macro_events.yaml` + `main._is_macro_blackout()` |
| Market regime | SPY EMA50/200 gates | `analysis/regime.py` → BULL/CAUTION/BEAR |
| Relative strength | Ticker 20d > SPY 20d | `analysis/signals.py` (longs only) |
| Near-high filter | Price within 10% of 63d high | `analysis/signals.py` (longs only) |
| LLM hard-stop | $10/month | `llm/budget.py` |
| Paper-only guard | URL must contain "paper" | `Settings` model_validator |
| Backtest gate | Must have passing report | `main.py` startup |
| NYSE holidays | 2025–2026 holiday calendar | `main._NYSE_HOLIDAYS` frozenset |

## Backtest

Two-phase strategy:

- **Phase 1 (now):** Quant-only. Establishes baseline. Must show positive expectancy
  (`expectancy > 0`, `trades >= 10`, `sharpe > 0`) before paper trading starts.
- **Phase 2 (after 2–3 weeks live):** Add sentiment filter. Refine thresholds.

The harness certifies the **same signal and exit criteria as the live system**:

Signal (`_compute_signal`, warmup = 65 bars):
- EMA(20) trend + RSI(14) in [rsi_min, rsi_max] + volume > volume_multiplier × average
- **Near-high proximity**: price within `near_high_max_drawdown` of the 63-bar rolling high
- **Relative strength vs SPY**: ticker 20d return > SPY 20d return (longs only)
- SPY bars are always fetched by the loader to compute the RS filter each day

Exits (in priority order):
- **Take-profit**: `_check_exits` exits at `entry × (1 + take_profit_pct)` — checked first
- **Partial exit**: scale out `partial_exit_fraction` at `partial_exit_trigger_pct` unrealized gain (single-shot; residual re-evaluated same bar)
- **Trailing stop**: activates at `trail_activation_pct` unrealized gain, then trails at `trail_pct`
- **ATR-derived stop-loss**: stop distance = max(atr_stop_multiplier × ATR/price, min_stop_pct)
- **Time-based exit**: `_check_time_exit()` — close positions held ≥ `max_hold_days` with < 1% gain

Sizing:
- `_compute_qty`: risk-based — qty × stop_distance ≈ `risk_per_trade_pct` × equity
- Portfolio heat cap: aggregate open risk checked before each new entry

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
- `finnhub-python` — news headlines (free tier, rate-limited) + earnings calendar
- `pandas` + `pandas-ta` — indicators (functional API)
- `APScheduler` — three scheduled jobs in one long-running process
- `pydantic-settings` — typed config with validation
- `plotly` + `streamlit` — interactive dashboard
- SQLite — single-file database, WAL mode, no ORM
- `pytest` — 374 tests across all modules (all passing)

## Deployment

Hosted on a DigitalOcean VPS. A CD pipeline via GitHub Actions automatically deploys
on every push to `main`: pulls the latest code, restarts the PM2 process (`trading-bot`),
and the dashboard process (`trading-dashboard`). No manual deploy steps needed.

To check bot status: `pm2 list` and `pm2 logs trading-bot` (requires NVM: `source ~/.nvm/nvm.sh`).

## Default Watchlist (32 stocks, 10 sectors)

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, AMZN, NVDA, META, AMD, CRM |
| Semiconductors | AVGO, MU, TSM, QCOM |
| Finance | JPM, BAC, GS |
| Fintech | V, MA, PYPL |
| Healthcare | JNJ, AMGN, UNH |
| Energy | XOM, CVX |
| Consumer | HD, WMT, TSLA, MCD |
| Industrials | CAT, DE, BA |
| Biotech | MRNA, REGN |

## Known Risks and Mitigations

1. **Stale news** — sentiment is a veto gate, not a signal source. Tight lookback window.
   Job B logs a WARNING when no sentiment biases exist for the day.
2. **Lookahead bias** — timestamp-gated joins, next-bar fills, slippage model, unit tests.
3. **Paper-fill optimism** — Alpaca fills at limit even when real markets wouldn't.
   Slippage model in backtest partially compensates.
4. **Headline recycling** — SQLite de-dupe by headline ID before any triage.
5. **Trailing stop management** — not fire-and-forget. Position management runs at the
   top of every Job B run with all required parameters (symbol, side, qty, trail_pct).
6. **SQLite concurrency** — WAL mode allows concurrent reads (dashboard + bot).
   Dashboard is read-only; no write contention.
7. **Fill reconciliation** — trades table is populated by polling Alpaca after each run,
   not at submission time. `trade_exists()` prevents duplicate rows.
8. **Macro event risk** — FOMC and CPI dates are loaded from `config/macro_events.yaml`.
   No new entries on those days; existing positions are still managed.
9. **Regime misclassification** — `UNKNOWN` regime (insufficient SPY data) is a pass-through
   that does not block entries. Treated as BULL for entry purposes.
