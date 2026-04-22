"""
Tests for notifications/email.py.

SMTP and Anthropic API calls are mocked.
All DB operations use in-memory SQLite.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from db.schema import SCHEMA_SQL
from notifications.email import (
    collect_daily_data,
    compute_period_returns,
    generate_llm_summary,
    build_html_email,
    send_daily_email,
)

TODAY = "2025-04-14"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    defaults = dict(
        alpaca_api_key="test", alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test", finnhub_api_key="test",
        email_enabled=True,
        email_recipient="trader@example.com",
        email_sender="onboarding@resend.dev",
        resend_api_key="re_test_key",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def seed_trade(conn, symbol="AAPL", side="buy", realized_pnl=None,
               date=TODAY, notional=1000.0) -> None:
    conn.execute(
        """
        INSERT INTO trades (order_id, symbol, side, qty, fill_price, notional,
                            session, run_timestamp, realized_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (f"ord_{symbol}_{side}_{date}", symbol, side, 10, 100.0, notional,
         "quant_1030", f"{date}T10:30:00+00:00", realized_pnl),
    )
    conn.commit()


def seed_equity(conn, equity: float, ts: str) -> None:
    conn.execute(
        """
        INSERT INTO equity_snapshots
            (run_timestamp, session, equity, cash, portfolio_value, open_positions, daily_pnl)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, "quant_1030", equity, equity * 0.5, equity, 0, 0.0),
    )
    conn.commit()


def seed_daily_pnl(conn, date=TODAY, realized=100.0, total=100.0, open_eq=10000.0):
    conn.execute(
        """
        INSERT OR REPLACE INTO daily_pnl
            (date, realized_pnl, unrealized_pnl, total_pnl, open_equity, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        """,
        (date, realized, 0.0, total, open_eq),
    )
    conn.commit()


def seed_bias(conn, ticker="AAPL", bias="BULLISH", score=0.8, date=TODAY):
    conn.execute(
        """
        INSERT OR REPLACE INTO sentiment_bias
            (ticker, date, bias, aggregated_score, headline_count, llm_run, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        """,
        (ticker, date, bias, score, 3, "morning"),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# collect_daily_data
# ---------------------------------------------------------------------------

class TestCollectDailyData:
    def test_trades_today_populated(self):
        conn = make_db()
        seed_trade(conn, "AAPL", "buy", realized_pnl=150.0)
        data = collect_daily_data(conn, TODAY)
        assert len(data["trades_today"]) == 1
        assert data["trades_today"][0]["symbol"] == "AAPL"

    def test_open_positions_excludes_closed(self):
        conn = make_db()
        seed_trade(conn, "AAPL", "buy")
        seed_trade(conn, "AAPL", "sell")   # closes it
        data = collect_daily_data(conn, TODAY)
        assert len(data["open_positions"]) == 0

    def test_open_positions_includes_unclosed(self):
        conn = make_db()
        seed_trade(conn, "AAPL", "buy")
        seed_trade(conn, "MSFT", "buy")
        data = collect_daily_data(conn, TODAY)
        syms = {p["symbol"] for p in data["open_positions"]}
        assert "AAPL" in syms and "MSFT" in syms

    def test_biases_populated(self):
        conn = make_db()
        seed_bias(conn, "AAPL", "BULLISH")
        data = collect_daily_data(conn, TODAY)
        assert any(b["ticker"] == "AAPL" for b in data["biases"])

    def test_empty_db_returns_safe_defaults(self):
        conn = make_db()
        data = collect_daily_data(conn, TODAY)
        assert data["trades_today"] == []
        assert data["open_positions"] == []
        assert data["daily_pnl"] == {}
        assert data["snapshots"] == []


# ---------------------------------------------------------------------------
# compute_period_returns
# ---------------------------------------------------------------------------

class TestComputePeriodReturns:
    def _make_snapshots(self, days_back: int, start_eq=9000.0, end_eq=10000.0) -> list[dict]:
        snapshots = []
        now = datetime(2025, 4, 14, 16, 0, tzinfo=timezone.utc)
        start = now - timedelta(days=days_back)
        snapshots.append({"run_timestamp": start.isoformat(), "equity": start_eq})
        snapshots.append({"run_timestamp": now.isoformat(), "equity": end_eq})
        return snapshots

    def test_today_return_computed(self):
        # days_back=2 puts the start snapshot before the "today" cutoff (yesterday midnight UTC)
        snaps = self._make_snapshots(2, start_eq=9000.0, end_eq=10000.0)
        returns = compute_period_returns(snaps, TODAY)
        assert returns.get("Today") == pytest.approx(1000 / 9000, rel=0.01)

    def test_insufficient_history_returns_none(self):
        # Only 2 days of data — 1-year period should be None
        snaps = self._make_snapshots(2, 9000.0, 10000.0)
        returns = compute_period_returns(snaps, TODAY)
        assert returns.get("1 Year") is None

    def test_empty_snapshots_returns_empty(self):
        returns = compute_period_returns([], TODAY)
        assert returns == {}


# ---------------------------------------------------------------------------
# generate_llm_summary
# ---------------------------------------------------------------------------

class TestGenerateLlmSummary:
    def _mock_anthropic(self, text: str):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=text)]
        mock_client.messages.create.return_value = mock_resp
        return mock_client

    @patch("notifications.email.anthropic.Anthropic")
    def test_extracts_grade_correctly(self, mock_cls):
        raw = "Strong day with 3 winning trades and good momentum.\nGRADE: 8/10"
        mock_cls.return_value = self._mock_anthropic(raw)
        data = {"date": TODAY, "trades_today": [], "daily_pnl": {"total_pnl": 200.0, "open_equity": 10000.0}, "biases": []}
        summary, grade = generate_llm_summary(data, make_settings())
        assert grade == 8
        assert "GRADE" not in summary

    @patch("notifications.email.anthropic.Anthropic")
    def test_grade_clamped_to_1_10(self, mock_cls):
        raw = "Bad day.\nGRADE: 15/10"
        mock_cls.return_value = self._mock_anthropic(raw)
        data = {"date": TODAY, "trades_today": [], "daily_pnl": {"total_pnl": -100.0, "open_equity": 10000.0}, "biases": []}
        _, grade = generate_llm_summary(data, make_settings())
        assert 1 <= grade <= 10

    @patch("notifications.email.anthropic.Anthropic")
    def test_api_failure_returns_fallback(self, mock_cls):
        mock_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        data = {"date": TODAY, "trades_today": [], "daily_pnl": {"total_pnl": 50.0, "open_equity": 10000.0}, "biases": []}
        summary, grade = generate_llm_summary(data, make_settings())
        assert isinstance(summary, str) and len(summary) > 0
        assert grade == 5  # default fallback


# ---------------------------------------------------------------------------
# build_html_email
# ---------------------------------------------------------------------------

class TestBuildHtmlEmail:
    def _make_data(self):
        return {
            "date": TODAY,
            "trades_today": [
                {"symbol": "AAPL", "side": "buy", "qty": 10, "fill_price": 150.0,
                 "realized_pnl": 75.0, "notional": 1500.0},
            ],
            "open_positions": [],
            "daily_pnl": {"total_pnl": 75.0, "realized_pnl": 75.0, "open_equity": 10000.0},
            "snapshots": [],
            "biases": [],
        }

    def test_returns_subject_and_html(self):
        data = self._make_data()
        subject, html = build_html_email(
            data=data,
            period_returns={"Today": 0.0075, "1 Year": None},
            summary="Good day.",
            grade=7,
            mtd_spend=1.5,
            budget=18.0,
        )
        assert isinstance(subject, str) and len(subject) > 0
        assert "<html>" in html

    def test_subject_contains_date_and_grade(self):
        data = self._make_data()
        subject, _ = build_html_email(
            data=data,
            period_returns={},
            summary="OK day.",
            grade=6,
            mtd_spend=0.5,
            budget=18.0,
        )
        assert TODAY in subject
        assert "6/10" in subject

    def test_positive_pnl_shows_in_html(self):
        data = self._make_data()
        _, html = build_html_email(
            data=data,
            period_returns={},
            summary="Good day.",
            grade=8,
            mtd_spend=1.0,
            budget=18.0,
        )
        assert "AAPL" in html
        assert "$+75.00" in html

    def test_no_trades_shows_placeholder(self):
        data = self._make_data()
        data["trades_today"] = []
        _, html = build_html_email(
            data=data, period_returns={}, summary="Quiet day.", grade=5,
            mtd_spend=0.0, budget=18.0,
        )
        assert "No closed trades today" in html


# ---------------------------------------------------------------------------
# send_daily_email (integration of all steps)
# ---------------------------------------------------------------------------

class TestSendDailyEmail:
    def test_skips_when_disabled(self):
        conn = make_db()
        s = make_settings(email_enabled=False)
        result = send_daily_email(conn, s, date=TODAY)
        assert result is False

    def test_skips_when_missing_settings(self):
        conn = make_db()
        s = make_settings(email_recipient="", email_enabled=True)
        result = send_daily_email(conn, s, date=TODAY)
        assert result is False

    @patch("notifications.email._send_via_resend")
    @patch("notifications.email.generate_llm_summary")
    def test_calls_sender_when_enabled(self, mock_summary, mock_send):
        mock_summary.return_value = ("Good day.", 7)
        mock_send.return_value = True
        conn = make_db()
        seed_daily_pnl(conn)
        result = send_daily_email(conn, make_settings(), date=TODAY)
        assert result is True
        mock_send.assert_called_once()

    @patch("notifications.email._send_via_resend")
    @patch("notifications.email.generate_llm_summary")
    def test_send_failure_returns_false(self, mock_summary, mock_send):
        mock_summary.return_value = ("OK.", 5)
        mock_send.return_value = False
        conn = make_db()
        result = send_daily_email(conn, make_settings(), date=TODAY)
        assert result is False
