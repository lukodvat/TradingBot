"""
Tests for core/portfolio.py — PortfolioManager.

All broker calls are mocked. DB uses in-memory SQLite.
"""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch
from zoneinfo import ZoneInfo

import pytest

from config.settings import Settings
from core.broker import AccountSnapshot, BrokerClient
from core.portfolio import PortfolioManager, _minutes_to_market_close, _unrealized_pct
from core.risk import RiskManager, TrailingStopCheck
from db.schema import SCHEMA_SQL

_ET = ZoneInfo("America/New_York")
TODAY = "2025-04-13"


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


def make_position(
    symbol: str = "AAPL",
    qty: float = 10.0,
    unrealized_plpc: float = 0.02,
    cost_basis: float = 10_000.0,
) -> MagicMock:
    """
    Build a mock Alpaca Position. market_value is derived from cost_basis and
    unrealized_plpc so that risk.check_trailing_stop_activation sees consistent values.
    """
    market_value = cost_basis * (1 + unrealized_plpc)
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    pos.unrealized_plpc = str(unrealized_plpc)
    pos.unrealized_pl = str(market_value - cost_basis)
    pos.cost_basis = str(cost_basis)
    pos.market_value = str(market_value)
    return pos


def make_snapshot(
    equity: float = 10_000.0,
    cash: float = 5_000.0,
    portfolio_value: float = 10_200.0,
    buying_power: float = 5_000.0,
    daily_pnl: float = 100.0,
    positions: list = None,
) -> AccountSnapshot:
    return AccountSnapshot(
        equity=equity,
        cash=cash,
        portfolio_value=portfolio_value,
        buying_power=buying_power,
        daily_pnl=daily_pnl,
        positions=positions or [],
        open_order_count=0,
    )


def make_manager(settings=None, conn=None):
    s = settings or make_settings()
    c = conn or make_db()
    broker = MagicMock(spec=BrokerClient)
    risk = RiskManager(s)
    pm = PortfolioManager(s, broker, risk, c)
    return pm, broker, c


# ---------------------------------------------------------------------------
# _minutes_to_market_close tests
# ---------------------------------------------------------------------------

class TestMinutesToClose:
    def test_returns_30_at_1530_et(self):
        dt = datetime(2025, 4, 13, 15, 30, 0, tzinfo=_ET)
        assert _minutes_to_market_close(dt) == 30

    def test_returns_0_at_or_after_close(self):
        dt = datetime(2025, 4, 13, 16, 0, 0, tzinfo=_ET)
        assert _minutes_to_market_close(dt) == 0

    def test_returns_0_after_close(self):
        dt = datetime(2025, 4, 13, 17, 0, 0, tzinfo=_ET)
        assert _minutes_to_market_close(dt) == 0

    def test_returns_90_at_1430_et(self):
        dt = datetime(2025, 4, 13, 14, 30, 0, tzinfo=_ET)
        assert _minutes_to_market_close(dt) == 90

    def test_works_with_utc_input(self):
        # 19:30 UTC = 15:30 ET
        dt = datetime(2025, 4, 13, 19, 30, 0, tzinfo=timezone.utc)
        assert _minutes_to_market_close(dt) == 30


# ---------------------------------------------------------------------------
# _unrealized_pct tests
# ---------------------------------------------------------------------------

class TestUnrealizedPct:
    def test_uses_plpc_when_available(self):
        pos = make_position(unrealized_plpc=0.015)
        assert _unrealized_pct(pos) == pytest.approx(0.015)

    def test_negative_plpc(self):
        pos = make_position(unrealized_plpc=-0.03)
        assert _unrealized_pct(pos) == pytest.approx(-0.03)

    def test_fallback_when_plpc_none(self):
        pos = MagicMock()
        pos.unrealized_plpc = None
        pos.cost_basis = "10000"
        pos.market_value = "10150"
        assert _unrealized_pct(pos) == pytest.approx(0.015)

    def test_zero_cost_basis_returns_zero(self):
        pos = MagicMock()
        pos.unrealized_plpc = None
        pos.cost_basis = "0"
        pos.market_value = "1000"
        assert _unrealized_pct(pos) == 0.0


# ---------------------------------------------------------------------------
# manage_partial_exits tests
# ---------------------------------------------------------------------------

class TestManagePartialExits:
    def _seed_entry(self, conn, symbol, run_ts="2025-04-10T14:30:00+00:00"):
        conn.execute(
            """INSERT INTO trades(order_id, symbol, side, qty, fill_price,
               notional, session, run_timestamp)
               VALUES (?, ?, 'buy', 10, 100, 1000, 'morning', ?)""",
            (f"oid-{symbol}", symbol, run_ts),
        )
        conn.commit()
        return run_ts

    def test_no_op_when_disabled(self):
        s = make_settings(partial_exit_enabled=False)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position(unrealized_plpc=0.05)
        assert pm.manage_partial_exits([pos], "2025-04-13T15:30:00+00:00") == 0
        broker.submit_market_order.assert_not_called()

    def test_skips_when_below_trigger(self):
        pm, broker, conn = make_manager()
        self._seed_entry(conn, "AAPL")
        pos = make_position(symbol="AAPL", unrealized_plpc=0.02)  # below 3% default
        assert pm.manage_partial_exits([pos], "2025-04-13T15:30:00+00:00") == 0
        broker.submit_market_order.assert_not_called()

    def test_fires_at_or_above_trigger(self):
        from alpaca.trading.enums import OrderSide
        pm, broker, conn = make_manager()
        self._seed_entry(conn, "AAPL")
        broker.submit_market_order.return_value = MagicMock(id="ord-1")
        pos = make_position(symbol="AAPL", qty=10.0, unrealized_plpc=0.04)
        count = pm.manage_partial_exits([pos], "2025-04-13T15:30:00+00:00")
        assert count == 1
        broker.submit_market_order.assert_called_once_with(
            symbol="AAPL", side=OrderSide.SELL, qty=5.0,
        )

    def test_fires_only_once_per_entry(self):
        pm, broker, conn = make_manager()
        entry_ts = self._seed_entry(conn, "AAPL")
        broker.submit_market_order.return_value = MagicMock(id="ord-1")
        pos = make_position(symbol="AAPL", qty=10.0, unrealized_plpc=0.04)
        pm.manage_partial_exits([pos], "2025-04-13T15:30:00+00:00")
        # Re-run with same position — should not fire again
        broker.submit_market_order.reset_mock()
        count = pm.manage_partial_exits([pos], "2025-04-13T15:35:00+00:00")
        assert count == 0
        broker.submit_market_order.assert_not_called()

    def test_skips_when_no_entry_record(self):
        pm, broker, _ = make_manager()
        pos = make_position(symbol="MSFT", unrealized_plpc=0.05)
        assert pm.manage_partial_exits([pos], "2025-04-13T15:30:00+00:00") == 0
        broker.submit_market_order.assert_not_called()


# ---------------------------------------------------------------------------
# manage_trailing_stops tests
# ---------------------------------------------------------------------------

class TestManageTrailingStops:
    def test_activates_when_threshold_met(self):
        from alpaca.trading.enums import OrderSide
        s = make_settings(trail_activation_pct=0.015, trail_pct=0.03)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position(symbol="AAPL", unrealized_plpc=0.02)  # 2% > 1.5%
        count = pm.manage_trailing_stops([pos], run_timestamp="2025-04-13T15:30:00+00:00")
        broker.replace_stop_with_trailing.assert_called_once_with(
            symbol="AAPL", side=OrderSide.SELL, qty=10.0, trail_pct=0.03
        )
        assert count == 1

    def test_no_activation_below_threshold(self):
        s = make_settings(trail_activation_pct=0.015)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position(symbol="AAPL", unrealized_plpc=0.01)  # 1% < 1.5%
        count = pm.manage_trailing_stops([pos], run_timestamp="ts")
        broker.replace_stop_with_trailing.assert_not_called()
        assert count == 0

    def test_multiple_positions_partial_activation(self):
        from alpaca.trading.enums import OrderSide
        s = make_settings(trail_activation_pct=0.015, trail_pct=0.03)
        pm, broker, _ = make_manager(settings=s)
        pos_above = make_position(symbol="AAPL", unrealized_plpc=0.02)
        pos_below = make_position(symbol="MSFT", unrealized_plpc=0.005)
        count = pm.manage_trailing_stops([pos_above, pos_below], run_timestamp="ts")
        broker.replace_stop_with_trailing.assert_called_once_with(
            symbol="AAPL", side=OrderSide.SELL, qty=10.0, trail_pct=0.03
        )
        assert count == 1

    def test_broker_error_does_not_raise(self):
        pm, broker, _ = make_manager()
        broker.replace_stop_with_trailing.side_effect = RuntimeError("API error")
        pos = make_position(unrealized_plpc=0.05)
        # Should not raise — just log the error
        count = pm.manage_trailing_stops([pos], run_timestamp="ts")
        assert count == 0

    def test_empty_positions_returns_zero(self):
        pm, broker, _ = make_manager()
        count = pm.manage_trailing_stops([], run_timestamp="ts")
        assert count == 0


# ---------------------------------------------------------------------------
# manage_flattens tests
# ---------------------------------------------------------------------------

class TestManageFlattens:
    def _run_dt(self, hour: int = 15, minute: int = 30) -> datetime:
        return datetime(2025, 4, 13, hour, minute, tzinfo=_ET)

    def test_friday_flattens_all(self):
        pm, broker, _ = make_manager()
        positions = [make_position("AAPL"), make_position("MSFT")]
        count = pm.manage_flattens(
            positions, biases={}, run_dt=self._run_dt(),
            run_timestamp="ts", is_friday=True,
        )
        assert count == 2
        assert broker.close_position.call_count == 2

    def test_within_flatten_window_closes_position(self):
        # 15:30 ET → 30 min to close, flatten_before_close_minutes=30 by default
        s = make_settings(flatten_before_close_minutes=30)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position("AAPL", unrealized_plpc=0.005)  # only 0.5% — no hold exception
        count = pm.manage_flattens(
            [pos], biases={"AAPL": "BULLISH"},
            run_dt=self._run_dt(15, 30), run_timestamp="ts",
        )
        assert count == 1
        broker.close_position.assert_called_once_with("AAPL")

    def test_hold_exception_bullish_long(self):
        # Position is up >1% AND bias is BULLISH → hold
        s = make_settings(flatten_before_close_minutes=30)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position("AAPL", qty=10.0, unrealized_plpc=0.015)  # 1.5% gain, long
        count = pm.manage_flattens(
            [pos], biases={"AAPL": "BULLISH"},
            run_dt=self._run_dt(15, 30), run_timestamp="ts",
        )
        assert count == 0
        broker.close_position.assert_not_called()

    def test_hold_exception_bearish_short(self):
        # Short position up >1% AND bias is BEARISH → hold
        s = make_settings(flatten_before_close_minutes=30)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position("AAPL", qty=-10.0, unrealized_plpc=0.015)  # short, 1.5% gain
        count = pm.manage_flattens(
            [pos], biases={"AAPL": "BEARISH"},
            run_dt=self._run_dt(15, 30), run_timestamp="ts",
        )
        assert count == 0
        broker.close_position.assert_not_called()

    def test_early_run_does_not_flatten(self):
        # 13:30 ET → 150 min to close, outside flatten window
        s = make_settings(flatten_before_close_minutes=30)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position("AAPL", unrealized_plpc=-0.01)
        count = pm.manage_flattens(
            [pos], biases={},
            run_dt=self._run_dt(13, 30), run_timestamp="ts",
        )
        assert count == 0
        broker.close_position.assert_not_called()

    def test_mismatched_bias_no_hold_exception(self):
        # Long position but bias is BEARISH → no hold exception, flatten
        s = make_settings(flatten_before_close_minutes=30)
        pm, broker, _ = make_manager(settings=s)
        pos = make_position("AAPL", qty=10.0, unrealized_plpc=0.02)  # up 2%, but bias mismatch
        count = pm.manage_flattens(
            [pos], biases={"AAPL": "BEARISH"},
            run_dt=self._run_dt(15, 30), run_timestamp="ts",
        )
        assert count == 1
        broker.close_position.assert_called_once_with("AAPL")

    def test_broker_close_error_does_not_raise(self):
        pm, broker, _ = make_manager(settings=make_settings(flatten_before_close_minutes=30))
        broker.close_position.side_effect = RuntimeError("API error")
        pos = make_position("AAPL", unrealized_plpc=0.0)
        # Should not raise
        pm.manage_flattens([pos], biases={}, run_dt=self._run_dt(15, 30), run_timestamp="ts")


# ---------------------------------------------------------------------------
# record_snapshot tests
# ---------------------------------------------------------------------------

class TestRecordSnapshot:
    def test_snapshot_written_to_db(self):
        pm, _, conn = make_manager()
        snap = make_snapshot(equity=10_500.0, cash=5_000.0, daily_pnl=500.0)
        pm.record_snapshot(snap, "2025-04-13T15:30:00+00:00", "quant_1530", date=TODAY)
        rows = conn.execute("SELECT * FROM equity_snapshots").fetchall()
        assert len(rows) == 1
        assert rows[0]["equity"] == pytest.approx(10_500.0)

    def test_daily_pnl_upserted(self):
        pm, _, conn = make_manager()
        snap = make_snapshot(daily_pnl=300.0, positions=[make_position(unrealized_plpc=0.03)])
        pm.record_snapshot(snap, "2025-04-13T15:30:00+00:00", "quant_1530", date=TODAY)
        rows = conn.execute("SELECT * FROM daily_pnl WHERE date = ?", (TODAY,)).fetchall()
        assert len(rows) == 1
        assert rows[0]["total_pnl"] == pytest.approx(300.0)

    def test_multiple_snapshots_accumulate(self):
        pm, _, conn = make_manager()
        pm.record_snapshot(make_snapshot(equity=10_000.0), "ts1", "quant_1030", date=TODAY)
        pm.record_snapshot(make_snapshot(equity=10_100.0), "ts2", "quant_1130", date=TODAY)
        count = conn.execute("SELECT COUNT(*) FROM equity_snapshots").fetchone()[0]
        assert count == 2


# ---------------------------------------------------------------------------
# get_held_today tests
# ---------------------------------------------------------------------------

class TestGetHeldToday:
    def _seed_trade(self, conn: sqlite3.Connection, symbol: str, side: str = "buy",
                    date: str = TODAY) -> None:
        conn.execute(
            """
            INSERT INTO trades (order_id, symbol, side, qty, fill_price, notional,
                                session, run_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"ord_{symbol}_{side}", symbol, side, 10, 100.0, 1000.0,
             "quant_1030", f"{date}T10:30:00+00:00"),
        )
        conn.commit()

    def test_returns_bought_tickers(self):
        pm, _, conn = make_manager()
        self._seed_trade(conn, "AAPL", side="buy")
        held = pm.get_held_today(TODAY)
        assert "AAPL" in held

    def test_returns_short_tickers(self):
        pm, _, conn = make_manager()
        self._seed_trade(conn, "MSFT", side="sell_short")
        held = pm.get_held_today(TODAY)
        assert "MSFT" in held

    def test_excludes_close_trades(self):
        pm, _, conn = make_manager()
        self._seed_trade(conn, "AAPL", side="sell")   # closing trade, not entry
        held = pm.get_held_today(TODAY)
        assert "AAPL" not in held

    def test_excludes_other_dates(self):
        pm, _, conn = make_manager()
        self._seed_trade(conn, "AAPL", side="buy", date="2025-04-12")  # yesterday
        held = pm.get_held_today(TODAY)
        assert "AAPL" not in held

    def test_empty_when_no_trades(self):
        pm, _, conn = make_manager()
        assert pm.get_held_today(TODAY) == set()

    def test_deduplicates_symbol(self):
        pm, _, conn = make_manager()
        # Two trades for same ticker (shouldn't happen but robust anyway)
        self._seed_trade(conn, "AAPL", side="buy")
        conn.execute(
            "INSERT INTO trades (order_id, symbol, side, qty, fill_price, notional, session, run_timestamp)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("ord2", "AAPL", "buy", 5, 101.0, 505.0, "quant_1130", f"{TODAY}T11:30:00+00:00"),
        )
        conn.commit()
        held = pm.get_held_today(TODAY)
        assert held == {"AAPL"}


# ---------------------------------------------------------------------------
# manage_time_based_exits
# ---------------------------------------------------------------------------

class TestManageTimeBasedExits:
    def _seed_entry(self, conn, symbol="AAPL", date=TODAY, side="buy") -> None:
        conn.execute(
            """
            INSERT INTO trades (order_id, symbol, side, qty, fill_price, notional, session, run_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (f"ord_{symbol}_{date}", symbol, side, 10, 100.0, 1000.0,
             "quant_1030", f"{date}T10:30:00+00:00"),
        )
        conn.commit()

    def test_stale_position_with_no_gain_closed(self):
        """Position held 8 days with 0% gain → closed (max_hold_days=7)."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)

        entry_date = "2025-04-05"  # 8 days before TODAY (2025-04-13)
        self._seed_entry(conn, "AAPL", date=entry_date)

        position = make_position("AAPL", unrealized_plpc=0.0)  # no gain
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([position], run_ts)
        assert closed == 1
        broker.close_position.assert_called_once_with("AAPL")

    def test_stale_position_up_over_1pct_kept(self):
        """Position held 8 days but up 2% → NOT closed (hold exception)."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)

        entry_date = "2025-04-05"
        self._seed_entry(conn, "AAPL", date=entry_date)

        position = make_position("AAPL", unrealized_plpc=0.02)  # up 2%
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([position], run_ts)
        assert closed == 0
        broker.close_position.assert_not_called()

    def test_fresh_position_not_closed(self):
        """Position held 2 days (below max_hold_days=7) → not touched."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)

        entry_date = "2025-04-11"  # 2 days before TODAY
        self._seed_entry(conn, "AAPL", date=entry_date)

        position = make_position("AAPL", unrealized_plpc=0.0)
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([position], run_ts)
        assert closed == 0
        broker.close_position.assert_not_called()

    def test_no_entry_record_skipped_gracefully(self):
        """Position with no matching trade in DB → skipped without error."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)
        # No trade seeded for AAPL

        position = make_position("AAPL", unrealized_plpc=0.0)
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([position], run_ts)
        assert closed == 0

    def test_exactly_at_threshold_closed(self):
        """Position held exactly max_hold_days (7) → closed."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)

        entry_date = "2025-04-06"  # exactly 7 days before TODAY
        self._seed_entry(conn, "AAPL", date=entry_date)

        position = make_position("AAPL", unrealized_plpc=0.005)  # 0.5% — below 1%
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([position], run_ts)
        assert closed == 1

    def test_multiple_positions_independent(self):
        """Stale + fresh positions in same run — only stale one is closed."""
        s = make_settings(max_hold_days=7)
        pm, broker, conn = make_manager(settings=s)

        self._seed_entry(conn, "AAPL", date="2025-04-05")   # 8 days — stale
        self._seed_entry(conn, "MSFT", date="2025-04-11")   # 2 days — fresh

        stale = make_position("AAPL", unrealized_plpc=0.0)
        fresh = make_position("MSFT", unrealized_plpc=0.0)
        run_ts = f"{TODAY}T14:30:00+00:00"

        closed = pm.manage_time_based_exits([stale, fresh], run_ts)
        assert closed == 1
        broker.close_position.assert_called_once_with("AAPL")
