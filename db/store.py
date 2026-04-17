"""
Thin read/write helpers for every SQLite table.

Rules:
- All queries are parameterized — no string interpolation ever.
- Functions accept an open sqlite3.Connection (caller owns the connection lifecycle).
- Returns are plain dicts or primitives — no custom types.
- No ORM. No abstractions beyond what the run loop actually uses.
"""

import sqlite3
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Open a database connection with row_factory set so rows behave like dicts.
    Caller is responsible for closing (use as a context manager).
    """
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# LLM budget
# ---------------------------------------------------------------------------

def get_monthly_llm_spend(conn: sqlite3.Connection, year_month: str) -> float:
    """
    Return total USD spent on LLM calls in the given month.

    year_month: 'YYYY-MM' format (e.g. '2025-04')
    """
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0.0) AS total "
        "FROM llm_calls "
        "WHERE strftime('%Y-%m', called_at) = ?",
        (year_month,),
    ).fetchone()
    return float(row["total"])


def record_llm_call(
    conn: sqlite3.Connection,
    *,
    model: str,
    tier: int,
    symbol: Optional[str],
    headline_id: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    sentiment: Optional[float],
    confidence: Optional[float],
    response_json: Optional[str],
    run_timestamp: str,
) -> int:
    """Insert an LLM call record. Returns the new row id."""
    cursor = conn.execute(
        """
        INSERT INTO llm_calls (
            model, tier, symbol, headline_id,
            prompt_tokens, completion_tokens, total_tokens,
            cost_usd, sentiment, confidence, response_json, run_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model, tier, symbol, headline_id,
            prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
            cost_usd, sentiment, confidence, response_json, run_timestamp,
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------

def is_headline_seen(conn: sqlite3.Connection, headline_id: str) -> bool:
    """Return True if this headline ID has already been processed."""
    row = conn.execute(
        "SELECT 1 FROM headlines_seen WHERE headline_id = ? LIMIT 1",
        (headline_id,),
    ).fetchone()
    return row is not None


def insert_headline(
    conn: sqlite3.Connection,
    *,
    headline_id: str,
    symbol: str,
    headline: str,
    source: Optional[str],
    published_at: str,
    fetched_at: str,
    run_timestamp: str,
) -> None:
    """
    Persist a new headline before triage begins.
    Call this immediately after de-duplication — before any LLM calls.
    Use update_headline_triage() later to fill in tier scores.
    """
    conn.execute(
        """
        INSERT OR IGNORE INTO headlines_seen (
            headline_id, symbol, headline, source,
            published_at, fetched_at, run_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (headline_id, symbol, headline, source, published_at, fetched_at, run_timestamp),
    )
    conn.commit()


def update_headline_triage(
    conn: sqlite3.Connection,
    *,
    headline_id: str,
    tier1_pass: Optional[int] = None,
    tier1_reason: Optional[str] = None,
    tier2_sentiment: Optional[float] = None,
    tier2_confidence: Optional[float] = None,
    tier3_assessment: Optional[str] = None,
) -> None:
    """Update triage results for a previously inserted headline."""
    conn.execute(
        """
        UPDATE headlines_seen
        SET tier1_pass       = COALESCE(?, tier1_pass),
            tier1_reason     = COALESCE(?, tier1_reason),
            tier2_sentiment  = COALESCE(?, tier2_sentiment),
            tier2_confidence = COALESCE(?, tier2_confidence),
            tier3_assessment = COALESCE(?, tier3_assessment)
        WHERE headline_id = ?
        """,
        (
            tier1_pass, tier1_reason,
            tier2_sentiment, tier2_confidence,
            tier3_assessment,
            headline_id,
        ),
    )
    conn.commit()


def get_latest_headline_sentiment(
    conn: sqlite3.Connection, symbol: str
) -> Optional[dict[str, Any]]:
    """
    Return the most recent T2 sentiment score for a symbol.
    Used by the overnight/flatten policy to check if sentiment is still positive.
    """
    row = conn.execute(
        """
        SELECT tier2_sentiment, tier2_confidence, published_at
        FROM headlines_seen
        WHERE symbol = ? AND tier2_sentiment IS NOT NULL
        ORDER BY published_at DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

def record_order(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    symbol: str,
    order_type: str,
    side: str,
    qty: float,
    limit_price: Optional[float],
    stop_price: Optional[float],
    status: str,
    submitted_at: str,
    run_timestamp: str,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO orders (
            order_id, symbol, order_type, side, qty,
            limit_price, stop_price, status, submitted_at, run_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, symbol, order_type, side, qty,
            limit_price, stop_price, status, submitted_at, run_timestamp,
        ),
    )
    conn.commit()


def update_order_status(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    status: str,
    fill_price: Optional[float] = None,
    filled_at: Optional[str] = None,
    cancel_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """
        UPDATE orders
        SET status       = ?,
            fill_price   = COALESCE(?, fill_price),
            filled_at    = COALESCE(?, filled_at),
            cancel_reason = COALESCE(?, cancel_reason),
            updated_at   = ?
        WHERE order_id = ?
        """,
        (status, fill_price, filled_at, cancel_reason, _now(), order_id),
    )
    conn.commit()


def get_pending_orders(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all orders with status='pending' for stale-order sweep."""
    rows = conn.execute(
        "SELECT * FROM orders WHERE status = 'pending' ORDER BY submitted_at",
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

def trade_exists(conn: sqlite3.Connection, order_id: str) -> bool:
    """Return True if a trade with this order_id has already been recorded."""
    row = conn.execute(
        "SELECT 1 FROM trades WHERE order_id = ? LIMIT 1",
        (order_id,),
    ).fetchone()
    return row is not None


def record_trade(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    symbol: str,
    side: str,
    qty: float,
    fill_price: float,
    limit_price: Optional[float],
    stop_price: Optional[float],
    notional: float,
    session: str,
    run_timestamp: str,
    filled_at: Optional[str] = None,
    realized_pnl: Optional[float] = None,
) -> None:
    slippage_bps: Optional[float] = None
    if limit_price and limit_price > 0:
        slippage_bps = (fill_price - limit_price) / limit_price * 10_000

    conn.execute(
        """
        INSERT OR IGNORE INTO trades (
            order_id, symbol, side, qty, fill_price, limit_price,
            stop_price, notional, session, run_timestamp, filled_at,
            realized_pnl, slippage_bps
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            order_id, symbol, side, qty, fill_price, limit_price,
            stop_price, notional, session, run_timestamp, filled_at,
            realized_pnl, slippage_bps,
        ),
    )
    conn.commit()


def update_trade_pnl(
    conn: sqlite3.Connection, *, order_id: str, realized_pnl: float
) -> None:
    conn.execute(
        "UPDATE trades SET realized_pnl = ? WHERE order_id = ?",
        (realized_pnl, order_id),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Equity snapshots
# ---------------------------------------------------------------------------

def record_equity_snapshot(
    conn: sqlite3.Connection,
    *,
    run_timestamp: str,
    session: str,
    equity: float,
    cash: float,
    portfolio_value: float,
    open_positions: int,
    daily_pnl: float,
) -> None:
    conn.execute(
        """
        INSERT INTO equity_snapshots (
            run_timestamp, session, equity, cash,
            portfolio_value, open_positions, daily_pnl
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (run_timestamp, session, equity, cash, portfolio_value, open_positions, daily_pnl),
    )
    conn.commit()


def get_equity_curve(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all equity snapshots ordered by time — used for charting."""
    rows = conn.execute(
        "SELECT run_timestamp, session, equity, daily_pnl "
        "FROM equity_snapshots ORDER BY run_timestamp",
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

def get_open_equity_for_date(
    conn: sqlite3.Connection, date: str
) -> Optional[float]:
    """Return stored open_equity for `date` or None if not yet recorded."""
    row = conn.execute(
        "SELECT open_equity FROM daily_pnl WHERE date = ?",
        (date,),
    ).fetchone()
    if row is None or row["open_equity"] is None:
        return None
    return float(row["open_equity"])


def upsert_daily_pnl(
    conn: sqlite3.Connection,
    *,
    date: str,                  # YYYY-MM-DD
    open_equity: Optional[float] = None,
    close_equity: Optional[float] = None,
    realized_pnl: Optional[float] = None,
    unrealized_pnl: Optional[float] = None,
    total_pnl: Optional[float] = None,
    trade_count: Optional[int] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO daily_pnl (date, open_equity, close_equity, realized_pnl,
                                unrealized_pnl, total_pnl, trade_count, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            open_equity    = COALESCE(excluded.open_equity, open_equity),
            close_equity   = COALESCE(excluded.close_equity, close_equity),
            realized_pnl   = COALESCE(excluded.realized_pnl, realized_pnl),
            unrealized_pnl = COALESCE(excluded.unrealized_pnl, unrealized_pnl),
            total_pnl      = COALESCE(excluded.total_pnl, total_pnl),
            trade_count    = COALESCE(excluded.trade_count, trade_count),
            updated_at     = excluded.updated_at
        """,
        (date, open_equity, close_equity, realized_pnl,
         unrealized_pnl, total_pnl, trade_count, _now()),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Session log
# ---------------------------------------------------------------------------

def start_session_log(
    conn: sqlite3.Connection, *, run_timestamp: str, session: str
) -> None:
    """Insert a session row with all counters at zero. Update as the run progresses."""
    conn.execute(
        """
        INSERT OR IGNORE INTO session_log (run_timestamp, session)
        VALUES (?, ?)
        """,
        (run_timestamp, session),
    )
    conn.commit()


def update_session_log(
    conn: sqlite3.Connection,
    *,
    run_timestamp: str,
    **fields: Any,
) -> None:
    """
    Update any subset of session_log columns by name.

    Usage:
        update_session_log(conn, run_timestamp=ts, tier2_calls=3, orders_submitted=1)
    """
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [run_timestamp]
    conn.execute(
        f"UPDATE session_log SET {set_clause} WHERE run_timestamp = ?",  # noqa: S608
        values,
    )
    conn.commit()


def get_session_log(conn: sqlite3.Connection, limit: int = 20) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM session_log ORDER BY run_timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Sentiment bias (written by Job A, read by Job B)
# ---------------------------------------------------------------------------

def upsert_sentiment_bias(
    conn: sqlite3.Connection,
    *,
    ticker: str,
    date: str,                   # YYYY-MM-DD
    bias: str,                   # 'BULLISH' | 'NEUTRAL' | 'BEARISH'
    aggregated_score: float,
    headline_count: int,
    llm_run: str,                # 'morning' | 'midday'
) -> None:
    conn.execute(
        """
        INSERT INTO sentiment_bias
            (ticker, date, bias, aggregated_score, headline_count, llm_run, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date, llm_run) DO UPDATE SET
            bias             = excluded.bias,
            aggregated_score = excluded.aggregated_score,
            headline_count   = excluded.headline_count,
            updated_at       = excluded.updated_at
        """,
        (ticker, date, bias, aggregated_score, headline_count, llm_run, _now()),
    )
    conn.commit()


def get_sentiment_bias(
    conn: sqlite3.Connection, ticker: str, date: str
) -> Optional[dict[str, Any]]:
    """
    Return the most recent sentiment bias for a ticker on a given date.
    Prefers 'midday' over 'morning' if both exist (midday is more current).
    Returns None if no bias has been set — caller should treat as NEUTRAL.
    """
    row = conn.execute(
        """
        SELECT bias, aggregated_score, headline_count, llm_run, updated_at
        FROM sentiment_bias
        WHERE ticker = ? AND date = ?
        ORDER BY CASE llm_run
            WHEN 'midday'    THEN 0
            WHEN 'morning'   THEN 1
            WHEN 'premarket' THEN 2
            ELSE 3
        END
        LIMIT 1
        """,
        (ticker, date),
    ).fetchone()
    return dict(row) if row else None


def get_all_sentiment_biases_for_date(
    conn: sqlite3.Connection, date: str
) -> dict[str, str]:
    """
    Return {ticker: bias} for all tickers that have a bias set on the given date.
    Uses most recent run (midday preferred over morning).
    Job B calls this once at the start of each quant scan.
    """
    rows = conn.execute(
        """
        SELECT ticker, bias
        FROM (
            SELECT ticker, bias,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker
                       ORDER BY CASE llm_run
                           WHEN 'midday'    THEN 0
                           WHEN 'morning'   THEN 1
                           WHEN 'premarket' THEN 2
                           ELSE 3
                       END
                   ) AS rn
            FROM sentiment_bias
            WHERE date = ?
        )
        WHERE rn = 1
        """,
        (date,),
    ).fetchall()
    return {r["ticker"]: r["bias"] for r in rows}


# ---------------------------------------------------------------------------
# Volatility filter log
# ---------------------------------------------------------------------------

def record_vol_filter(
    conn: sqlite3.Connection,
    *,
    run_timestamp: str,
    symbol: str,
    atr_price_ratio: Optional[float],
    realized_vol: Optional[float],
    atr_threshold: float,
    vol_threshold: float,
    passed: bool,
    fail_reason: Optional[str] = None,
) -> None:
    conn.execute(
        """
        INSERT INTO volatility_filter_log (
            run_timestamp, symbol, atr_price_ratio, realized_vol,
            atr_threshold, vol_threshold, passed, fail_reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_timestamp, symbol, atr_price_ratio, realized_vol,
            atr_threshold, vol_threshold, int(passed), fail_reason,
        ),
    )
    conn.commit()
