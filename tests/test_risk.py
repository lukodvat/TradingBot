"""
Tests for core/risk.py — all pure logic, zero network calls.

Covers every public method and all branch conditions.
"""

import pytest
from unittest.mock import MagicMock

from alpaca.trading.enums import OrderSide

from config.settings import Settings
from core.risk import CircuitBreakerResult, RejectionReason, RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    defaults = dict(
        alpaca_api_key="test",
        alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test",
        finnhub_api_key="test",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_rm(**setting_overrides) -> RiskManager:
    return RiskManager(make_settings(**setting_overrides))


def make_position(
    symbol: str,
    sector_value: float,
    qty: float = 10.0,
    unrealized_pct: float = 0.0,
) -> MagicMock:
    """Create a mock Alpaca Position object."""
    pos = MagicMock()
    pos.symbol = symbol
    pos.qty = str(qty)
    avg_entry = 100.0
    current_price = avg_entry * (1 + unrealized_pct)
    pos.market_value = str(sector_value)
    pos.cost_basis = str(avg_entry * qty)
    pos.current_price = str(current_price)
    pos.unrealized_pl = str((current_price - avg_entry) * qty)
    return pos


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:

    def test_not_triggered_within_limit(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=-250.0, equity=10_000.0)
        assert not result.triggered  # -2.5% < -3% threshold

    def test_triggered_at_exact_threshold(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=-300.0, equity=10_000.0)
        assert result.triggered  # exactly -3%

    def test_triggered_beyond_threshold(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=-500.0, equity=10_000.0)
        assert result.triggered

    def test_not_triggered_on_profit(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=200.0, equity=10_000.0)
        assert not result.triggered

    def test_triggered_on_zero_equity(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=0.0, equity=0.0)
        assert result.triggered

    def test_pnl_pct_is_accurate(self):
        rm = make_rm()
        result = rm.check_circuit_breaker(daily_pnl=-200.0, equity=10_000.0)
        assert abs(result.daily_pnl_pct - (-0.02)) < 1e-9

    def test_custom_threshold_respected(self):
        rm = make_rm(daily_loss_limit_pct=0.05)  # 5% limit
        result = rm.check_circuit_breaker(daily_pnl=-400.0, equity=10_000.0)
        assert not result.triggered  # -4% is inside the 5% limit


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------

class TestPositionSizing:

    SECTOR_MAP = {"AAPL": "technology", "MSFT": "technology", "JPM": "finance"}

    def _size(self, rm, symbol="TSLA", sector="technology", side=OrderSide.BUY,
               price=100.0, equity=10_000.0, positions=None, buying_power=10_000.0):
        return rm.size_position(
            symbol=symbol,
            sector=sector,
            side=side,
            limit_price=price,
            equity=equity,
            open_positions=positions or [],
            sector_map=self.SECTOR_MAP,
            buying_power=buying_power,
        )

    def test_approved_with_no_positions(self):
        rm = make_rm()
        result = self._size(rm)
        assert result.approved
        assert result.qty > 0

    def test_notional_capped_at_10pct_equity(self):
        rm = make_rm()
        result = self._size(rm, equity=10_000.0, price=50.0)
        assert result.notional <= 10_000.0 * 0.10 + 0.01  # allow rounding

    def test_rejected_at_position_limit(self):
        rm = make_rm()
        positions = [make_position(f"SYM{i}", 500.0) for i in range(5)]
        # SECTOR_MAP doesn't cover SYM0-4 so they won't count toward sector
        result = self._size(rm, positions=positions)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.POSITION_LIMIT

    def test_rejected_if_already_holding(self):
        rm = make_rm()
        pos = make_position("TSLA", 1000.0)
        result = self._size(rm, symbol="TSLA", positions=[pos])
        assert not result.approved
        assert result.rejection_reason == RejectionReason.ALREADY_HELD

    def test_rejected_on_sector_concentration_breach(self):
        rm = make_rm()  # max_sector_pct = 30% = $3000
        # Already have $3000 in tech — adding any more should breach
        existing = make_position("AAPL", 3000.0)
        result = self._size(
            rm, symbol="GOOGL", sector="technology",
            positions=[existing],
        )
        assert not result.approved
        assert result.rejection_reason == RejectionReason.SECTOR_CONCENTRATION

    def test_sector_concentration_allows_partial_fill(self):
        rm = make_rm()  # max_sector_pct=30% → $3000 limit
        # $2000 already in tech → $1000 headroom; max_position = $1000
        existing = make_position("AAPL", 2000.0)
        result = self._size(
            rm, symbol="GOOGL", sector="technology", price=50.0,
            positions=[existing],
        )
        assert result.approved
        # Headroom = $3000 - $2000 = $1000; max_position = $1000; min = $1000
        assert result.notional <= 1000.0 + 0.01

    def test_rejected_on_insufficient_buying_power(self):
        rm = make_rm()
        result = self._size(rm, price=100.0, buying_power=1.0)
        assert not result.approved
        assert result.rejection_reason == RejectionReason.INSUFFICIENT_BUYING_POWER

    def test_stop_price_below_entry_for_long(self):
        rm = make_rm()
        result = self._size(rm, side=OrderSide.BUY, price=100.0)
        assert result.approved
        assert result.stop_price < result.limit_price

    def test_stop_price_above_entry_for_short(self):
        rm = make_rm()
        result = self._size(rm, side=OrderSide.SELL, price=100.0)
        assert result.approved
        assert result.stop_price > result.limit_price

    def test_stop_distance_equals_configured_pct(self):
        rm = make_rm(stop_loss_pct=0.02)
        result = self._size(rm, side=OrderSide.BUY, price=100.0)
        expected_stop = round(100.0 * (1 - 0.02), 2)
        assert abs(result.stop_price - expected_stop) < 0.01


# ---------------------------------------------------------------------------
# Stop price calculation
# ---------------------------------------------------------------------------

class TestStopPriceCalculation:

    def test_long_stop_is_below_entry(self):
        rm = make_rm(stop_loss_pct=0.02)
        stop = rm.compute_stop_price(100.0, OrderSide.BUY)
        assert stop == pytest.approx(98.0, abs=0.01)

    def test_short_stop_is_above_entry(self):
        rm = make_rm(stop_loss_pct=0.02)
        stop = rm.compute_stop_price(100.0, OrderSide.SELL)
        assert stop == pytest.approx(102.0, abs=0.01)

    def test_custom_stop_pct(self):
        rm = make_rm(stop_loss_pct=0.05)
        stop = rm.compute_stop_price(200.0, OrderSide.BUY)
        assert stop == pytest.approx(190.0, abs=0.01)


# ---------------------------------------------------------------------------
# Trailing stop activation
# ---------------------------------------------------------------------------

class TestTrailingStopActivation:

    def _make_long_position(self, avg_entry: float, current_price: float, qty: float = 10.0) -> MagicMock:
        pos = MagicMock()
        pos.qty = str(qty)
        pos.cost_basis = str(avg_entry * qty)
        pos.market_value = str(current_price * qty)
        return pos

    def test_not_activated_below_threshold(self):
        rm = make_rm(trail_activation_pct=0.015)
        pos = self._make_long_position(avg_entry=100.0, current_price=101.0)  # +1%
        result = rm.check_trailing_stop_activation(pos)
        assert not result.should_activate

    def test_activated_at_threshold(self):
        rm = make_rm(trail_activation_pct=0.015)
        pos = self._make_long_position(avg_entry=100.0, current_price=101.5)  # +1.5%
        result = rm.check_trailing_stop_activation(pos)
        assert result.should_activate

    def test_activated_above_threshold(self):
        rm = make_rm(trail_activation_pct=0.015)
        pos = self._make_long_position(avg_entry=100.0, current_price=103.0)  # +3%
        result = rm.check_trailing_stop_activation(pos)
        assert result.should_activate

    def test_not_activated_on_zero_cost_basis(self):
        rm = make_rm()
        pos = MagicMock()
        pos.cost_basis = "0"
        pos.market_value = "1000"
        pos.qty = "10"
        result = rm.check_trailing_stop_activation(pos)
        assert not result.should_activate

    def test_unrealized_pct_is_accurate(self):
        rm = make_rm(trail_activation_pct=0.015)
        pos = self._make_long_position(avg_entry=100.0, current_price=102.0)
        result = rm.check_trailing_stop_activation(pos)
        assert abs(result.unrealized_pct - 0.02) < 1e-9


# ---------------------------------------------------------------------------
# Overnight / flatten policy
# ---------------------------------------------------------------------------

class TestFlattenPolicy:

    def test_flatten_triggered_within_window_no_exception(self):
        rm = make_rm(flatten_before_close_minutes=10)
        assert rm.should_flatten_now(
            minutes_to_close=5, unrealized_pct=0.0, sentiment_positive=False
        )

    def test_no_flatten_outside_window(self):
        rm = make_rm(flatten_before_close_minutes=10)
        assert not rm.should_flatten_now(
            minutes_to_close=30, unrealized_pct=0.02, sentiment_positive=False
        )

    def test_hold_exception_applied_when_profitable_and_positive_sentiment(self):
        rm = make_rm(flatten_before_close_minutes=10)
        # Within window but meets hold exception
        assert not rm.should_flatten_now(
            minutes_to_close=5, unrealized_pct=0.015, sentiment_positive=True
        )

    def test_flatten_if_profitable_but_negative_sentiment(self):
        rm = make_rm(flatten_before_close_minutes=10)
        assert rm.should_flatten_now(
            minutes_to_close=5, unrealized_pct=0.02, sentiment_positive=False
        )

    def test_flatten_if_positive_sentiment_but_not_profitable_enough(self):
        rm = make_rm(flatten_before_close_minutes=10)
        # Sentiment positive but gain < 1%
        assert rm.should_flatten_now(
            minutes_to_close=5, unrealized_pct=0.005, sentiment_positive=True
        )


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

class TestPortfolioHelpers:

    def test_can_add_position_when_under_limit(self):
        rm = make_rm()  # max_positions=5
        positions = [make_position(f"SYM{i}", 500.0) for i in range(4)]
        assert rm.can_add_position(positions)

    def test_cannot_add_position_at_limit(self):
        rm = make_rm()
        positions = [make_position(f"SYM{i}", 500.0) for i in range(5)]
        assert not rm.can_add_position(positions)

    def test_sector_notional_sums_correctly(self):
        rm = make_rm()
        positions = [
            make_position("AAPL", 1000.0),
            make_position("MSFT", 1500.0),
            make_position("JPM", 2000.0),
        ]
        sector_map = {"AAPL": "technology", "MSFT": "technology", "JPM": "finance"}
        tech_notional = rm.sector_notional_for("technology", positions, sector_map)
        assert tech_notional == pytest.approx(2500.0)

    def test_sector_has_room_when_under_limit(self):
        rm = make_rm()  # max_sector_pct=30% → $3000 on $10k
        pos = make_position("AAPL", 1000.0)
        assert rm.sector_has_room(
            "technology", equity=10_000.0,
            positions=[pos], sector_map={"AAPL": "technology"}
        )

    def test_sector_no_room_when_at_limit(self):
        rm = make_rm()
        pos = make_position("AAPL", 3000.0)
        assert not rm.sector_has_room(
            "technology", equity=10_000.0,
            positions=[pos], sector_map={"AAPL": "technology"}
        )
