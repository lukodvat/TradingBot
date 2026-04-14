"""
Tests for db/schema.py and db/store.py.

All tests use an in-memory SQLite database — no files written to disk.
"""

import sqlite3
import pytest
from datetime import datetime, timezone

from db.schema import init_db, EXPECTED_TABLES
from db import store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    """In-memory DB, fully initialized, torn down after each test."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    # Replay init_db logic against in-memory conn
    from db.schema import SCHEMA_SQL
    c.executescript(SCHEMA_SQL)
    c.execute("PRAGMA foreign_keys=ON")
    yield c
    c.close()


RUN_TS = "2025-04-13T10:30:00+00:00"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:

    def test_init_db_creates_all_tables(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        with sqlite3.connect(db_path) as c:
            rows = c.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {r[0] for r in rows}
        assert EXPECTED_TABLES.issubset(names)

    def test_init_db_is_idempotent(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        init_db(db_path)
        init_db(db_path)  # second call must not raise


# ---------------------------------------------------------------------------
# LLM budget
# ---------------------------------------------------------------------------

class TestLlmBudget:

    def _insert_call(self, conn, cost: float, called_at: str = "2025-04-13T10:30:00"):
        store.record_llm_call(
            conn,
            model="claude-haiku-4-5",
            tier=2,
            symbol="AAPL",
            headline_id="h1",
            prompt_tokens=100,
            completion_tokens=50,
            cost_usd=cost,
            sentiment=0.5,
            confidence=0.8,
            response_json='{"sentiment": 0.5}',
            run_timestamp=RUN_TS,
        )
        # Override called_at for deterministic month filtering
        conn.execute(
            "UPDATE llm_calls SET called_at = ? WHERE id = (SELECT MAX(id) FROM llm_calls)",
            (called_at,),
        )
        conn.commit()

    def test_zero_spend_when_no_calls(self, conn):
        total = store.get_monthly_llm_spend(conn, "2025-04")
        assert total == 0.0

    def test_spend_accumulates(self, conn):
        self._insert_call(conn, 0.50)
        self._insert_call(conn, 0.75)
        total = store.get_monthly_llm_spend(conn, "2025-04")
        assert abs(total - 1.25) < 1e-9

    def test_spend_is_month_scoped(self, conn):
        self._insert_call(conn, 5.00, called_at="2025-03-15T10:00:00")
        self._insert_call(conn, 1.00, called_at="2025-04-01T10:00:00")
        assert abs(store.get_monthly_llm_spend(conn, "2025-04") - 1.00) < 1e-9
        assert abs(store.get_monthly_llm_spend(conn, "2025-03") - 5.00) < 1e-9

    def test_record_llm_call_returns_row_id(self, conn):
        row_id = store.record_llm_call(
            conn,
            model="claude-haiku-4-5",
            tier=2,
            symbol="MSFT",
            headline_id="h999",
            prompt_tokens=200,
            completion_tokens=100,
            cost_usd=0.01,
            sentiment=-0.3,
            confidence=0.9,
            response_json=None,
            run_timestamp=RUN_TS,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_total_tokens_computed_correctly(self, conn):
        store.record_llm_call(
            conn,
            model="claude-haiku-4-5",
            tier=2,
            symbol="X",
            headline_id="h2",
            prompt_tokens=300,
            completion_tokens=100,
            cost_usd=0.01,
            sentiment=0.0,
            confidence=0.5,
            response_json=None,
            run_timestamp=RUN_TS,
        )
        row = conn.execute("SELECT total_tokens FROM llm_calls ORDER BY id DESC LIMIT 1").fetchone()
        assert row["total_tokens"] == 400


# ---------------------------------------------------------------------------
# Headlines
# ---------------------------------------------------------------------------

class TestHeadlines:

    def _insert(self, conn, headline_id="h1", symbol="AAPL"):
        store.insert_headline(
            conn,
            headline_id=headline_id,
            symbol=symbol,
            headline="Apple announces record earnings",
            source="reuters",
            published_at="2025-04-13T09:00:00+00:00",
            fetched_at="2025-04-13T10:00:00+00:00",
            run_timestamp=RUN_TS,
        )

    def test_new_headline_not_seen(self, conn):
        assert not store.is_headline_seen(conn, "h_unknown")

    def test_headline_seen_after_insert(self, conn):
        self._insert(conn)
        assert store.is_headline_seen(conn, "h1")

    def test_insert_is_idempotent(self, conn):
        self._insert(conn)
        self._insert(conn)  # INSERT OR IGNORE — should not raise
        count = conn.execute("SELECT COUNT(*) FROM headlines_seen WHERE headline_id='h1'").fetchone()[0]
        assert count == 1

    def test_update_triage_tier1(self, conn):
        self._insert(conn)
        store.update_headline_triage(conn, headline_id="h1", tier1_pass=1, tier1_reason="keyword_match")
        row = conn.execute("SELECT tier1_pass, tier1_reason FROM headlines_seen WHERE headline_id='h1'").fetchone()
        assert row["tier1_pass"] == 1
        assert row["tier1_reason"] == "keyword_match"

    def test_update_triage_tier2(self, conn):
        self._insert(conn)
        store.update_headline_triage(conn, headline_id="h1", tier1_pass=1, tier2_sentiment=0.7, tier2_confidence=0.85)
        row = conn.execute("SELECT tier2_sentiment, tier2_confidence FROM headlines_seen WHERE headline_id='h1'").fetchone()
        assert abs(row["tier2_sentiment"] - 0.7) < 1e-9
        assert abs(row["tier2_confidence"] - 0.85) < 1e-9

    def test_get_latest_sentiment_returns_none_when_empty(self, conn):
        result = store.get_latest_headline_sentiment(conn, "AAPL")
        assert result is None

    def test_get_latest_sentiment_returns_most_recent(self, conn):
        store.insert_headline(conn, headline_id="h1", symbol="AAPL", headline="Old news",
                               source=None, published_at="2025-04-12T09:00:00+00:00",
                               fetched_at="2025-04-12T10:00:00+00:00", run_timestamp=RUN_TS)
        store.insert_headline(conn, headline_id="h2", symbol="AAPL", headline="New news",
                               source=None, published_at="2025-04-13T09:00:00+00:00",
                               fetched_at="2025-04-13T10:00:00+00:00", run_timestamp=RUN_TS)
        store.update_headline_triage(conn, headline_id="h1", tier1_pass=1, tier2_sentiment=0.3, tier2_confidence=0.6)
        store.update_headline_triage(conn, headline_id="h2", tier1_pass=1, tier2_sentiment=0.8, tier2_confidence=0.9)
        result = store.get_latest_headline_sentiment(conn, "AAPL")
        assert result is not None
        assert abs(result["tier2_sentiment"] - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# Orders
# ---------------------------------------------------------------------------

class TestOrders:

    def test_record_and_retrieve_pending_order(self, conn):
        store.record_order(
            conn,
            order_id="ord-001",
            symbol="NVDA",
            order_type="bracket",
            side="buy",
            qty=5.0,
            limit_price=450.00,
            stop_price=441.00,
            status="pending",
            submitted_at=RUN_TS,
            run_timestamp=RUN_TS,
        )
        pending = store.get_pending_orders(conn)
        assert len(pending) == 1
        assert pending[0]["order_id"] == "ord-001"

    def test_update_order_status_to_filled(self, conn):
        store.record_order(conn, order_id="ord-002", symbol="MSFT", order_type="bracket",
                            side="buy", qty=3.0, limit_price=400.0, stop_price=392.0,
                            status="pending", submitted_at=RUN_TS, run_timestamp=RUN_TS)
        store.update_order_status(conn, order_id="ord-002", status="filled",
                                   fill_price=399.50, filled_at=RUN_TS)
        pending = store.get_pending_orders(conn)
        assert len(pending) == 0

    def test_record_order_is_idempotent(self, conn):
        for _ in range(2):
            store.record_order(conn, order_id="ord-003", symbol="AAPL", order_type="market",
                                side="sell", qty=10.0, limit_price=None, stop_price=None,
                                status="pending", submitted_at=RUN_TS, run_timestamp=RUN_TS)
        count = conn.execute("SELECT COUNT(*) FROM orders WHERE order_id='ord-003'").fetchone()[0]
        assert count == 1


# ---------------------------------------------------------------------------
# Trades
# ---------------------------------------------------------------------------

class TestTrades:

    def test_record_trade_and_slippage_computed(self, conn):
        store.record_trade(
            conn,
            order_id="ord-010",
            symbol="AAPL",
            side="buy",
            qty=10.0,
            fill_price=150.15,
            limit_price=150.00,
            stop_price=147.00,
            notional=1501.50,
            session="morning",
            run_timestamp=RUN_TS,
        )
        row = conn.execute("SELECT slippage_bps FROM trades WHERE order_id='ord-010'").fetchone()
        # slippage = (150.15 - 150.00) / 150.00 * 10000 = 10 bps
        assert abs(row["slippage_bps"] - 10.0) < 0.01

    def test_update_trade_pnl(self, conn):
        store.record_trade(conn, order_id="ord-011", symbol="MSFT", side="sell",
                            qty=5.0, fill_price=400.0, limit_price=400.0,
                            stop_price=None, notional=2000.0, session="afternoon",
                            run_timestamp=RUN_TS)
        store.update_trade_pnl(conn, order_id="ord-011", realized_pnl=85.50)
        row = conn.execute("SELECT realized_pnl FROM trades WHERE order_id='ord-011'").fetchone()
        assert abs(row["realized_pnl"] - 85.50) < 1e-9


# ---------------------------------------------------------------------------
# Equity snapshots
# ---------------------------------------------------------------------------

class TestEquitySnapshots:

    def test_record_and_retrieve_curve(self, conn):
        store.record_equity_snapshot(conn, run_timestamp=RUN_TS, session="morning",
                                      equity=10_200.0, cash=5_100.0,
                                      portfolio_value=5_100.0, open_positions=2,
                                      daily_pnl=200.0)
        curve = store.get_equity_curve(conn)
        assert len(curve) == 1
        assert abs(curve[0]["equity"] - 10_200.0) < 1e-9


# ---------------------------------------------------------------------------
# Daily P&L
# ---------------------------------------------------------------------------

class TestDailyPnl:

    def test_upsert_inserts_new_row(self, conn):
        store.upsert_daily_pnl(conn, date="2025-04-13", open_equity=10_000.0,
                                close_equity=10_200.0, total_pnl=200.0, trade_count=2)
        row = conn.execute("SELECT total_pnl FROM daily_pnl WHERE date='2025-04-13'").fetchone()
        assert abs(row["total_pnl"] - 200.0) < 1e-9

    def test_upsert_updates_existing_row(self, conn):
        store.upsert_daily_pnl(conn, date="2025-04-13", open_equity=10_000.0)
        store.upsert_daily_pnl(conn, date="2025-04-13", close_equity=10_150.0, total_pnl=150.0)
        row = conn.execute("SELECT open_equity, close_equity FROM daily_pnl WHERE date='2025-04-13'").fetchone()
        assert abs(row["open_equity"] - 10_000.0) < 1e-9   # preserved from first upsert
        assert abs(row["close_equity"] - 10_150.0) < 1e-9  # updated by second upsert


# ---------------------------------------------------------------------------
# Session log
# ---------------------------------------------------------------------------

class TestSessionLog:

    def test_start_and_update_session(self, conn):
        store.start_session_log(conn, run_timestamp=RUN_TS, session="morning")
        store.update_session_log(conn, run_timestamp=RUN_TS,
                                  tickers_evaluated=20, tier2_calls=5, orders_submitted=2)
        rows = store.get_session_log(conn)
        assert len(rows) == 1
        assert rows[0]["tickers_evaluated"] == 20
        assert rows[0]["orders_submitted"] == 2

    def test_start_session_is_idempotent(self, conn):
        store.start_session_log(conn, run_timestamp=RUN_TS, session="morning")
        store.start_session_log(conn, run_timestamp=RUN_TS, session="morning")
        count = conn.execute("SELECT COUNT(*) FROM session_log").fetchone()[0]
        assert count == 1

    def test_get_session_log_is_ordered_descending(self, conn):
        store.start_session_log(conn, run_timestamp="2025-04-13T10:30:00+00:00", session="morning")
        store.start_session_log(conn, run_timestamp="2025-04-13T15:30:00+00:00", session="afternoon")
        rows = store.get_session_log(conn)
        assert rows[0]["session"] == "afternoon"
        assert rows[1]["session"] == "morning"


# ---------------------------------------------------------------------------
# Volatility filter log
# ---------------------------------------------------------------------------

class TestVolFilterLog:

    def test_record_pass(self, conn):
        store.record_vol_filter(conn, run_timestamp=RUN_TS, symbol="AAPL",
                                 atr_price_ratio=0.02, realized_vol=0.25,
                                 atr_threshold=0.04, vol_threshold=0.50, passed=True)
        row = conn.execute("SELECT passed, fail_reason FROM volatility_filter_log WHERE symbol='AAPL'").fetchone()
        assert row["passed"] == 1
        assert row["fail_reason"] is None

    def test_record_fail(self, conn):
        store.record_vol_filter(conn, run_timestamp=RUN_TS, symbol="TSLA",
                                 atr_price_ratio=0.07, realized_vol=0.80,
                                 atr_threshold=0.04, vol_threshold=0.50,
                                 passed=False, fail_reason="atr_too_high")
        row = conn.execute("SELECT passed, fail_reason FROM volatility_filter_log WHERE symbol='TSLA'").fetchone()
        assert row["passed"] == 0
        assert row["fail_reason"] == "atr_too_high"
