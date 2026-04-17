"""
SQLite schema definitions.

All CREATE TABLE statements live here. Call init_db() once at startup.
No migration system for v1 — if you change a table, drop and recreate the DB.
"""

import sqlite3

SCHEMA_SQL = """
-- -----------------------------------------------------------------------
-- trades: one row per filled entry or exit
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT    UNIQUE NOT NULL,
    symbol          TEXT    NOT NULL,
    side            TEXT    NOT NULL,          -- 'buy' | 'sell'
    qty             REAL    NOT NULL,
    fill_price      REAL    NOT NULL,
    limit_price     REAL,                      -- submitted limit
    stop_price      REAL,                      -- initial stop set at entry
    notional        REAL    NOT NULL,          -- qty * fill_price
    session         TEXT    NOT NULL,          -- 'morning' | 'afternoon'
    run_timestamp   TEXT    NOT NULL,          -- ISO8601 of the run that placed it
    filled_at       TEXT,                      -- ISO8601 Alpaca fill time
    realized_pnl    REAL,                      -- populated when position closes
    slippage_bps    REAL,                      -- (fill - limit) / limit * 10000
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol       ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_run          ON trades(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_filled_at    ON trades(filled_at);

-- -----------------------------------------------------------------------
-- orders: every order submitted, regardless of fill status
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT    UNIQUE NOT NULL,
    symbol          TEXT    NOT NULL,
    order_type      TEXT    NOT NULL,          -- 'bracket' | 'market' | 'trailing_stop'
    side            TEXT    NOT NULL,
    qty             REAL    NOT NULL,
    limit_price     REAL,
    stop_price      REAL,
    status          TEXT    NOT NULL,          -- 'pending' | 'filled' | 'cancelled' | 'expired'
    submitted_at    TEXT    NOT NULL,
    updated_at      TEXT,
    fill_price      REAL,
    filled_at       TEXT,
    cancel_reason   TEXT,
    run_timestamp   TEXT    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol    ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status    ON orders(status);

-- -----------------------------------------------------------------------
-- llm_calls: every Anthropic API call with cost tracking
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    model            TEXT    NOT NULL,         -- full model ID
    tier             INTEGER NOT NULL,         -- 2 (Haiku) or 3 (Sonnet)
    symbol           TEXT,
    headline_id      TEXT,
    prompt_tokens    INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens     INTEGER NOT NULL,
    cost_usd         REAL    NOT NULL,
    sentiment        REAL,                     -- -1..+1, NULL for Tier 3
    confidence       REAL,                     -- 0..1, NULL for Tier 3
    response_json    TEXT,                     -- full raw response for audit
    run_timestamp    TEXT    NOT NULL,
    called_at        TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_run        ON llm_calls(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_llm_calls_called_at  ON llm_calls(called_at);

-- -----------------------------------------------------------------------
-- daily_pnl: one row per trading day, upserted at end of each run
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS daily_pnl (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    UNIQUE NOT NULL,   -- YYYY-MM-DD
    open_equity     REAL,
    close_equity    REAL,
    realized_pnl    REAL,
    unrealized_pnl  REAL,
    total_pnl       REAL,
    trade_count     INTEGER DEFAULT 0,
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- equity_snapshots: per-run equity snapshot for equity curve
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp   TEXT    NOT NULL,
    session         TEXT    NOT NULL,          -- 'morning' | 'afternoon'
    equity          REAL    NOT NULL,
    cash            REAL    NOT NULL,
    portfolio_value REAL    NOT NULL,
    open_positions  INTEGER NOT NULL,
    daily_pnl       REAL    NOT NULL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_equity_snapshots_run ON equity_snapshots(run_timestamp);

-- -----------------------------------------------------------------------
-- headlines_seen: de-dupe store + triage audit trail
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS headlines_seen (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    headline_id      TEXT    UNIQUE NOT NULL,  -- Finnhub article ID
    symbol           TEXT    NOT NULL,
    headline         TEXT    NOT NULL,
    source           TEXT,
    published_at     TEXT    NOT NULL,         -- ISO8601
    fetched_at       TEXT    NOT NULL,         -- ISO8601
    tier1_pass       INTEGER,                  -- 1=pass, 0=fail, NULL=not evaluated
    tier1_reason     TEXT,
    tier2_sentiment  REAL,                     -- NULL if T1 rejected
    tier2_confidence REAL,
    tier3_assessment TEXT,                     -- NULL if not escalated
    run_timestamp    TEXT    NOT NULL,
    created_at       TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_headlines_symbol      ON headlines_seen(symbol);
CREATE INDEX IF NOT EXISTS idx_headlines_published   ON headlines_seen(published_at);
CREATE INDEX IF NOT EXISTS idx_headlines_run         ON headlines_seen(run_timestamp);

-- -----------------------------------------------------------------------
-- session_log: one row per scheduled run for operational monitoring
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS session_log (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp               TEXT    UNIQUE NOT NULL,
    session                     TEXT    NOT NULL,          -- 'morning' | 'afternoon'
    tickers_evaluated           INTEGER NOT NULL DEFAULT 0,
    tickers_vol_filtered        INTEGER NOT NULL DEFAULT 0,
    headlines_fetched           INTEGER NOT NULL DEFAULT 0,
    headlines_deduped           INTEGER NOT NULL DEFAULT 0,
    tier1_passes                INTEGER NOT NULL DEFAULT 0,
    tier2_calls                 INTEGER NOT NULL DEFAULT 0,
    tier3_calls                 INTEGER NOT NULL DEFAULT 0,
    orders_submitted            INTEGER NOT NULL DEFAULT 0,
    circuit_breaker_triggered   INTEGER NOT NULL DEFAULT 0,
    llm_budget_mode             TEXT    NOT NULL DEFAULT 'normal', -- 'normal' | 'quant_only'
    error                       TEXT,
    duration_seconds            REAL,
    created_at                  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- -----------------------------------------------------------------------
-- sentiment_bias: daily per-ticker bias written by Job A, read by Job B
-- Key decoupling table — Job B never calls the LLM directly.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sentiment_bias (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker          TEXT    NOT NULL,
    date            TEXT    NOT NULL,          -- YYYY-MM-DD
    bias            TEXT    NOT NULL,          -- 'BULLISH' | 'NEUTRAL' | 'BEARISH'
    aggregated_score REAL   NOT NULL,          -- mean T2 sentiment across headlines
    headline_count  INTEGER NOT NULL DEFAULT 0,
    llm_run         TEXT    NOT NULL,          -- 'morning' | 'midday'
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(ticker, date, llm_run)
);

CREATE INDEX IF NOT EXISTS idx_sentiment_bias_ticker_date ON sentiment_bias(ticker, date);

-- -----------------------------------------------------------------------
-- volatility_filter_log: per-ticker vol gate decisions
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS volatility_filter_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp   TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    atr_price_ratio REAL,
    realized_vol    REAL,
    atr_threshold   REAL    NOT NULL,
    vol_threshold   REAL    NOT NULL,
    passed          INTEGER NOT NULL,          -- 1=passed, 0=filtered out
    fail_reason     TEXT,                      -- 'atr_too_high' | 'vol_too_high' | NULL
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_vol_filter_run    ON volatility_filter_log(run_timestamp);
CREATE INDEX IF NOT EXISTS idx_vol_filter_symbol ON volatility_filter_log(symbol);

-- -----------------------------------------------------------------------
-- partial_exits: one row per partial profit-taking action
-- Used to ensure each position is partially exited at most once.
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS partial_exits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT    NOT NULL,
    entry_run_ts    TEXT    NOT NULL,        -- run_timestamp of the entry trade
    qty_sold        REAL    NOT NULL,
    fill_price      REAL,
    order_id        TEXT,
    exit_at         TEXT    NOT NULL,        -- ISO8601 of when the partial fired
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_partial_exits_symbol ON partial_exits(symbol);
"""


def init_db(conn_or_path: "sqlite3.Connection | str") -> None:
    """
    Create all tables and indexes if they don't exist.
    Safe to call on every startup — all statements are idempotent.

    Accepts either an open sqlite3.Connection or a file path string.
    """
    if isinstance(conn_or_path, str):
        with sqlite3.connect(conn_or_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.commit()
    else:
        conn = conn_or_path
        conn.executescript(SCHEMA_SQL)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.commit()


EXPECTED_TABLES = {
    "trades",
    "orders",
    "llm_calls",
    "daily_pnl",
    "equity_snapshots",
    "headlines_seen",
    "sentiment_bias",
    "session_log",
    "volatility_filter_log",
    "partial_exits",
}
