# TradingBot

An autonomous algorithmic trading system that combines technical analysis with LLM-driven news sentiment to trade a ~$10,000 paper account on Alpaca. The LLM does not find trades — it filters out bad ones.

---

## How It Works

The system runs three scheduled jobs on a server, all sharing a single SQLite database.

```
10:00 ET  ─── Job A: LLM fetches headlines, scores sentiment, writes bias to DB
10:30 ET  ─┐
11:30 ET  ─┤
12:30 ET  ─┤─ Job B: Quant scanner reads bias, finds setups, executes orders
13:00 ET  ─── Job A: Midday sentiment refresh
13:30 ET  ─┤
14:30 ET  ─┤─ Job B (continued)
15:30 ET  ─┘
16:30 ET  ─── Job C: Daily summary email with LLM-written recap and grade
```

**Job A — LLM Sentiment Pipeline**

Fetches financial headlines from Finnhub for 20 watchlist tickers and runs them through a three-tier triage:

- **Tier 1** — Free regex filter. Instantly drops SEC filings, routine dividends, index rebalances.
- **Tier 2** — Claude Haiku scores each headline: `{sentiment: -1..+1, confidence: 0..1}`.
- **Tier 3** — Claude Sonnet deep assessment. Only fires when sentiment is strong and confident (capped at 5 calls per run to control cost).

The result is a per-ticker bias — BULLISH, NEUTRAL, or BEARISH — written to SQLite. Total LLM cost: ~$1.25/month.

**Job B — Quant Scanner**

Runs six times per day. Makes zero LLM API calls — it reads the bias from SQLite for free.

For each run:
1. Circuit breaker check — if daily P&L is below -3%, liquidate everything and stop.
2. Position management — upgrade fixed stops to trailing stops on winning positions (+1.5% threshold).
3. Flatten check — at 15:30 ET, close positions unless they are up >1% with matching sentiment. Always flatten on Fridays.
4. Fetch OHLCV bars and run the volatility filter (ATR/price ratio + realized vol).
5. Signal scan — a ticker only qualifies if ALL are true: price above EMA(20), RSI(14) between 40–70, volume above 1.5× average, and today's sentiment bias matches the direction.
6. Risk checks per candidate — position limit (5), sector cap (25%), buying power.
7. Submit bracket orders: limit entry + 2% stop-loss.

**Job C — Daily Email**

At 4:30 PM ET, sends an HTML email containing:
- A 2–3 sentence LLM-written summary of the day and a grade out of 10
- Performance table: today, this week, month, 6 months, YTD, 1 year
- Table of today's trades with P&L
- Open positions
- LLM spend vs. monthly budget

---

## Tech Stack

| Component | Technology |
|---|---|
| Broker + market data | `alpaca-py` (paper trading only) |
| LLM | Anthropic API — Haiku (fast/cheap) and Sonnet (deep assessment) |
| News | Finnhub free tier |
| Indicators | `pandas` + `pandas-ta` |
| Scheduler | APScheduler — three jobs in one process |
| Database | SQLite (WAL mode, no ORM) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Config | Pydantic Settings |
| Tests | pytest — 291 tests |
| Language | Python 3.12 |

---

## Risk Guardrails

Every guardrail is hard-coded and tested. None are configurable at runtime.

| Guardrail | Value |
|---|---|
| Stop-loss | 2% from entry (bracket order) |
| Trailing stop activates at | +1.5% unrealized gain |
| Trailing stop distance | 3% |
| Max open positions | 5 |
| Max position size | 10% of equity |
| Max sector concentration | 25% of equity |
| Daily circuit breaker | -3% equity → liquidate all |
| Overnight flatten | 15:30 ET (hold exception if up >1% + matching bias) |
| Weekend flatten | Friday 15:30 ET, always, no exceptions |
| LLM monthly hard-stop | $18 USD |
| Live key guard | Process exits immediately if a non-paper Alpaca URL is detected |
| Backtest gate | `main.py` refuses to start without a passing backtest report |

---

## Dashboard

A Streamlit dashboard connects to the same SQLite database and provides a full read-only view of everything the bot produces.

**Sections:** live account stats, equity curve, drawdown chart, performance by period, trades table, win/loss distribution, open positions, sentiment bias heatmap, LLM cost tracker, volatility filter log, session log, headlines and triage scores.

---

## Getting Started

### 1. Prerequisites

- Python 3.12
- An [Alpaca](https://alpaca.markets) paper trading account
- An [Anthropic](https://console.anthropic.com) API key
- A [Finnhub](https://finnhub.io) free API key
- A Gmail account with an [App Password](https://support.google.com/accounts/answer/185833) (for email notifications)

### 2. Install

```bash
git clone https://github.com/yourname/TradingBot.git
cd TradingBot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
# Alpaca (paper only)
ALPACA_API_KEY=your_paper_key
ALPACA_SECRET_KEY=your_paper_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...

# Finnhub
FINNHUB_API_KEY=your_finnhub_key

# Email notifications (optional)
EMAIL_ENABLED=true
EMAIL_RECIPIENT=you@example.com
EMAIL_SENDER=yourbot@gmail.com
EMAIL_APP_PASSWORD=your_gmail_app_password
```

### 4. Run the backtest

The bot will not start without a passing backtest report. Run this first:

```bash
python run_backtest.py
```

Options:
```bash
python run_backtest.py --slippage-bps 15
python run_backtest.py --start 2024-01-01 --end 2024-12-31
python run_backtest.py --tickers AAPL MSFT NVDA
```

The report is saved to `reports/`. The bot checks for it at startup.

### 5. Run the bot

```bash
python main.py
```

The scheduler starts immediately and waits for the next scheduled window. It runs indefinitely — use a process manager (systemd, screen, tmux) to keep it alive on a server.

### 6. Run the dashboard

```bash
streamlit run dashboard.py --server.port 8501
```

**Accessing from a remote server (recommended):**

```bash
# On your laptop — creates a secure tunnel
ssh -L 8501:localhost:8501 user@your-server-ip

# Then open in your browser
http://localhost:8501
```

### 7. Run the tests

```bash
pytest
```

---

## Project Structure

```
TradingBot/
├── config/
│   ├── settings.py          # All configuration — paper-only guard enforced here
│   └── watchlist.yaml       # 20 tickers across 6 sectors
├── core/
│   ├── broker.py            # Alpaca wrapper — raises on live keys
│   ├── risk.py              # All guardrails: sizing, stops, sector cap, circuit breaker
│   └── portfolio.py         # Trailing stops, flatten logic, equity snapshots
├── data/
│   ├── market.py            # OHLCV bar fetching
│   └── news.py              # Finnhub headlines with rate limiting and de-dupe
├── analysis/
│   ├── volatility.py        # ATR and realized vol filter
│   ├── sentiment.py         # Three-tier LLM triage orchestrator
│   └── signals.py           # Technical signal scanner
├── llm/
│   ├── client.py            # Anthropic SDK wrapper with cost tracking
│   ├── prompts.py           # Prompt templates for Haiku and Sonnet
│   └── budget.py            # Monthly spend enforcement
├── backtest/
│   ├── harness.py           # Day-by-day replay engine
│   ├── metrics.py           # Sharpe, drawdown, win rate, expectancy
│   └── loader.py            # Historical bar loading
├── db/
│   ├── schema.py            # 9 SQLite tables
│   └── store.py             # Read/write helpers
├── notifications/
│   └── email.py             # Daily HTML email with LLM summary
├── tests/                   # 291 tests
├── main.py                  # Entrypoint — APScheduler with Jobs A, B, C
├── run_backtest.py          # Standalone backtest CLI
└── dashboard.py             # Streamlit dashboard
```

---

## Watchlist

20 stocks across 6 sectors. Edit `config/watchlist.yaml` to change.

| Sector | Tickers |
|---|---|
| Technology | AAPL, MSFT, GOOGL, AMZN, NVDA, META, AMD, CRM |
| Finance | JPM, BAC, GS |
| Healthcare | JNJ, PFE, UNH |
| Energy | XOM, CVX |
| Consumer | HD, WMT, DIS |

---

## Deploying to a Server (Digital Ocean)

```bash
# 1. SSH into your droplet
ssh root@your-server-ip

# 2. Clone and install
git clone https://github.com/yourname/TradingBot.git
cd TradingBot
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Add your .env file
nano .env

# 4. Run the backtest
python run_backtest.py

# 5. Run the bot in a persistent session
screen -S tradingbot
python main.py
# Ctrl+A, D to detach

# 6. Run the dashboard in another session
screen -S dashboard
streamlit run dashboard.py --server.port 8501 --server.address 127.0.0.1
# Ctrl+A, D to detach

# 7. Access the dashboard from your laptop
ssh -L 8501:localhost:8501 root@your-server-ip
# Open http://localhost:8501
```

---

## Disclaimer

This is a paper trading system. It uses simulated money only. Nothing here constitutes financial advice. Past backtest performance does not guarantee future results.
