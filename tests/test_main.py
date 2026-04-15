"""
Tests for main.py helpers and job orchestration.

The calendar/time helpers are pure functions — tested directly.
The job functions (run_llm_job, run_quant_job) are tested with heavy mocking
to verify control flow: which steps run, which are skipped, and what is
written to SQLite.
"""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, call
from zoneinfo import ZoneInfo

import pytest

from config.settings import Settings
from db.schema import SCHEMA_SQL
from main import (
    _is_market_day,
    _is_within_trading_hours,
    _is_friday,
    reconcile_fills,
    run_llm_job,
    run_quant_job,
)

_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    defaults = dict(
        alpaca_api_key="test", alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test", finnhub_api_key="test",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------

class TestIsMarketDay:
    def test_monday_is_market_day(self):
        dt = datetime(2025, 4, 14, 10, 0, tzinfo=_ET)  # Monday
        assert _is_market_day(dt) is True

    def test_friday_is_market_day(self):
        dt = datetime(2025, 4, 11, 10, 0, tzinfo=_ET)  # Friday
        assert _is_market_day(dt) is True

    def test_saturday_is_not_market_day(self):
        dt = datetime(2025, 4, 12, 10, 0, tzinfo=_ET)  # Saturday
        assert _is_market_day(dt) is False

    def test_sunday_is_not_market_day(self):
        dt = datetime(2025, 4, 13, 10, 0, tzinfo=_ET)  # Sunday
        assert _is_market_day(dt) is False

    def test_utc_datetime_converted_correctly(self):
        # Monday 14:30 UTC = Monday 10:30 ET
        dt = datetime(2025, 4, 14, 14, 30, tzinfo=timezone.utc)
        assert _is_market_day(dt) is True

    def test_nyse_holiday_is_not_market_day(self):
        # Good Friday 2025 — NYSE closed, but falls on a Friday (weekday)
        dt = datetime(2025, 4, 18, 10, 0, tzinfo=_ET)
        assert _is_market_day(dt) is False

    def test_christmas_is_not_market_day(self):
        dt = datetime(2025, 12, 25, 10, 0, tzinfo=_ET)
        assert _is_market_day(dt) is False

    def test_day_after_holiday_is_market_day(self):
        # April 19, 2025 (Saturday after Good Friday) is already a weekend,
        # use April 21 (Monday) which is a normal trading day
        dt = datetime(2025, 4, 21, 10, 0, tzinfo=_ET)
        assert _is_market_day(dt) is True


class TestIsWithinTradingHours:
    def _s(self, after_open=30, before_close=10):
        return make_settings(
            no_trade_after_open_minutes=after_open,
            no_trade_before_close_minutes=before_close,
        )

    def test_10_30_is_within_hours(self):
        dt = datetime(2025, 4, 14, 10, 30, tzinfo=_ET)
        assert _is_within_trading_hours(dt, self._s()) is True

    def test_9_30_before_cutoff_is_outside(self):
        # 9:30 ET is exactly at open — open cutoff is 9:30+30 = 10:00
        dt = datetime(2025, 4, 14, 9, 30, tzinfo=_ET)
        assert _is_within_trading_hours(dt, self._s()) is False

    def test_15_50_after_close_cutoff_is_outside(self):
        # close cutoff = 16:00 - 10min = 15:50, so 15:50 is NOT within
        dt = datetime(2025, 4, 14, 15, 50, tzinfo=_ET)
        assert _is_within_trading_hours(dt, self._s()) is False

    def test_15_30_is_within_hours(self):
        dt = datetime(2025, 4, 14, 15, 30, tzinfo=_ET)
        assert _is_within_trading_hours(dt, self._s()) is True

    def test_after_market_close_is_outside(self):
        dt = datetime(2025, 4, 14, 16, 0, tzinfo=_ET)
        assert _is_within_trading_hours(dt, self._s()) is False


class TestIsFriday:
    def test_friday_returns_true(self):
        dt = datetime(2025, 4, 11, 15, 30, tzinfo=_ET)
        assert _is_friday(dt) is True

    def test_monday_returns_false(self):
        dt = datetime(2025, 4, 14, 15, 30, tzinfo=_ET)
        assert _is_friday(dt) is False


# ---------------------------------------------------------------------------
# run_llm_job
# ---------------------------------------------------------------------------

class TestRunLlmJob:
    def _make_mocks(self, conn=None):
        s = make_settings()
        c = conn or make_db()
        llm = MagicMock()
        news = MagicMock()
        return s, c, llm, news

    @patch("main.datetime")
    def test_skips_on_weekend(self, mock_dt):
        # Sunday
        mock_dt.now.return_value = datetime(2025, 4, 13, 15, 0, tzinfo=timezone.utc)
        s, conn, llm, news = self._make_mocks()
        run_llm_job("morning", s, llm, news, conn)
        news.get_headlines.assert_not_called()

    @patch("main.assert_budget_ok")
    @patch("main.datetime")
    def test_skips_when_budget_exceeded(self, mock_dt, mock_budget):
        # Monday
        mock_dt.now.return_value = datetime(2025, 4, 14, 15, 0, tzinfo=timezone.utc)
        mock_budget.side_effect = RuntimeError("budget exceeded")
        s, conn, llm, news = self._make_mocks()
        run_llm_job("morning", s, llm, news, conn)
        news.get_headlines.assert_not_called()

    @patch("main.fetch_all_headlines")
    @patch("main.load_watchlist")
    @patch("main.assert_budget_ok")
    @patch("main.datetime")
    def test_no_new_headlines_exits_early(
        self, mock_dt, mock_budget, mock_wl, mock_fetch
    ):
        mock_dt.now.return_value = datetime(2025, 4, 14, 15, 0, tzinfo=timezone.utc)
        mock_wl.return_value = ["AAPL"]
        mock_fetch.return_value = []  # no headlines
        s, conn, llm, news = self._make_mocks()
        with patch("main.SentimentAnalyzer") as mock_analyzer:
            run_llm_job("morning", s, llm, news, conn)
            mock_analyzer.assert_not_called()

    @patch("main.fetch_all_headlines")
    @patch("main.load_watchlist")
    @patch("main.assert_budget_ok")
    @patch("main.SentimentAnalyzer")
    @patch("main.datetime")
    def test_new_headlines_trigger_analysis(
        self, mock_dt, mock_analyzer_cls, mock_budget, mock_wl, mock_fetch
    ):
        mock_dt.now.return_value = datetime(2025, 4, 14, 15, 0, tzinfo=timezone.utc)
        mock_wl.return_value = ["AAPL"]

        # Build a fake headline
        h = MagicMock()
        h.id = "h1"
        h.symbol = "AAPL"
        h.headline = "Apple earnings beat"
        h.source = "Reuters"
        h.published_at = None
        mock_fetch.return_value = [h]

        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = {}
        mock_analyzer_cls.return_value = mock_analyzer

        s, conn, llm, news = self._make_mocks()
        run_llm_job("morning", s, llm, news, conn)
        mock_analyzer.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# run_quant_job
# ---------------------------------------------------------------------------

class TestRunQuantJob:
    def _make_mocks(self, conn=None):
        s = make_settings()
        c = conn or make_db()
        broker = MagicMock()
        risk = MagicMock()
        market_data = MagicMock()
        news = MagicMock()
        news.get_upcoming_earnings.return_value = set()
        return s, c, broker, risk, market_data, news

    @patch("main.datetime")
    def test_skips_on_weekend(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 4, 13, 15, 30, tzinfo=timezone.utc)
        s, conn, broker, risk, md, news = self._make_mocks()
        run_quant_job("quant_1530", s, broker, risk, md, news, conn)
        broker.snapshot.assert_not_called()

    @patch("main.datetime")
    def test_skips_outside_trading_hours(self, mock_dt):
        # 21:30 UTC = 17:30 ET — after close
        mock_dt.now.return_value = datetime(2025, 4, 14, 21, 30, tzinfo=timezone.utc)
        s, conn, broker, risk, md, news = self._make_mocks()
        run_quant_job("quant_1530", s, broker, risk, md, news, conn)
        broker.snapshot.assert_not_called()

    @patch("main.load_watchlist")
    @patch("main.load_sector_map")
    @patch("main.filter_watchlist")
    @patch("main.passing_tickers")
    @patch("main.SignalScanner")
    @patch("main.send_circuit_breaker_alert")
    @patch("main.datetime")
    def test_circuit_breaker_liquidates_and_returns(
        self, mock_dt, mock_alert, mock_scanner_cls, mock_passing,
        mock_filter, mock_sector, mock_wl,
    ):
        # Monday 15:30 ET = 19:30 UTC
        mock_dt.now.return_value = datetime(2025, 4, 14, 19, 30, tzinfo=timezone.utc)
        mock_wl.return_value = ["AAPL"]
        mock_sector.return_value = {}
        s, conn, broker, risk, md, news = self._make_mocks()

        # Snapshot: big loss triggers circuit breaker
        snap = MagicMock()
        snap.daily_pnl = -400.0
        snap.equity = 10_000.0
        snap.positions = []
        snap.buying_power = 5000.0
        broker.snapshot.return_value = snap

        # Risk says breaker triggered
        cb = MagicMock()
        cb.triggered = True
        cb.daily_pnl_pct = -0.04
        risk.check_circuit_breaker.return_value = cb

        run_quant_job("quant_1530", s, broker, risk, md, news, conn)

        broker.close_all_positions.assert_called_once()
        mock_alert.assert_called_once_with(s, cb.daily_pnl_pct, snap.equity)
        mock_scanner_cls.assert_not_called()

    @patch("main._is_macro_blackout", return_value=False)
    @patch("main.MarketRegimeFilter")
    @patch("main.load_watchlist")
    @patch("main.load_sector_map")
    @patch("main.filter_watchlist")
    @patch("main.passing_tickers")
    @patch("main.SignalScanner")
    @patch("main.PortfolioManager")
    @patch("main.datetime")
    def test_no_candidates_no_orders(
        self, mock_dt, mock_pm_cls, mock_scanner_cls,
        mock_passing, mock_filter, mock_sector, mock_wl,
        mock_regime_cls, mock_blackout,
    ):
        mock_dt.now.return_value = datetime(2025, 4, 14, 19, 30, tzinfo=timezone.utc)
        mock_wl.return_value = ["AAPL"]
        mock_sector.return_value = {}
        mock_filter.return_value = {}
        mock_passing.return_value = []

        # Regime: full BULL, no cap reduction
        regime = MagicMock()
        regime.allow_any_entries = True
        regime.allow_long_entries = True
        regime.allow_short_entries = True
        regime.max_positions_override = 5
        regime.label = "BULL"
        mock_regime_cls.return_value.evaluate.return_value = regime

        s, conn, broker, risk, md, news = self._make_mocks()

        snap = MagicMock()
        snap.daily_pnl = 0.0
        snap.equity = 10_000.0
        snap.positions = []
        snap.buying_power = 10_000.0
        broker.snapshot.return_value = snap

        cb = MagicMock(); cb.triggered = False
        risk.check_circuit_breaker.return_value = cb

        scanner = MagicMock()
        scanner.scan.return_value = []  # no candidates
        mock_scanner_cls.return_value = scanner

        pm = MagicMock()
        pm.get_held_today.return_value = set()
        mock_pm_cls.return_value = pm

        md.get_daily_bars.return_value = {}

        run_quant_job("quant_1530", s, broker, risk, md, news, conn)

        broker.submit_bracket_order.assert_not_called()


# ---------------------------------------------------------------------------
# reconcile_fills
# ---------------------------------------------------------------------------

class TestReconcileFills:
    def _make_order(self, order_id="ord1", symbol="AAPL", side="buy",
                    filled_qty=10.0, filled_avg_price=150.0,
                    limit_price=None, stop_price=None, filled_at=None):
        o = MagicMock()
        o.id = order_id
        o.symbol = symbol
        o.side = side
        o.filled_qty = filled_qty
        o.filled_avg_price = filled_avg_price
        o.limit_price = limit_price
        o.stop_price = stop_price
        o.filled_at = filled_at
        return o

    def test_inserts_new_fill(self):
        conn = make_db()
        broker = MagicMock()
        order = self._make_order()
        broker.get_filled_orders_since.return_value = [order]

        inserted = reconcile_fills(broker, conn, "2025-04-14T19:30:00+00:00")
        assert inserted == 1

        row = conn.execute("SELECT * FROM trades WHERE order_id = 'ord1'").fetchone()
        assert row is not None
        assert row["symbol"] == "AAPL"
        assert row["fill_price"] == 150.0

    def test_skips_already_seen_fill(self):
        conn = make_db()
        broker = MagicMock()
        order = self._make_order()
        broker.get_filled_orders_since.return_value = [order]

        # Insert once
        reconcile_fills(broker, conn, "2025-04-14T19:30:00+00:00")
        # Second call should skip
        inserted = reconcile_fills(broker, conn, "2025-04-14T19:30:00+00:00")
        assert inserted == 0

        rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        assert rows == 1

    def test_no_filled_orders_returns_zero(self):
        conn = make_db()
        broker = MagicMock()
        broker.get_filled_orders_since.return_value = []

        inserted = reconcile_fills(broker, conn, "2025-04-14T19:30:00+00:00")
        assert inserted == 0

    def test_multiple_fills_inserted(self):
        conn = make_db()
        broker = MagicMock()
        orders = [
            self._make_order("o1", "AAPL", "buy", 10, 150.0),
            self._make_order("o2", "MSFT", "sell", 5, 380.0),
        ]
        broker.get_filled_orders_since.return_value = orders

        inserted = reconcile_fills(broker, conn, "2025-04-14T19:30:00+00:00")
        assert inserted == 2
