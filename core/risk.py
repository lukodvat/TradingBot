"""
Hard-coded risk management layer.

All guardrails live here. No strategy logic, no LLM calls, no I/O.
Every method is a pure function or takes only typed inputs — fully testable
without mocking anything.

Rules enforced:
- Max position size: 10% of equity per ticker
- Max sector concentration: 25% of equity
- Max 5 open positions simultaneously
- Fixed stop-loss: 2% from entry
- Trailing stop activation: unrealized gain >= 1.5%
- Daily loss circuit breaker: realized+unrealized P&L < -3% of equity
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from alpaca.trading.enums import OrderSide
from alpaca.trading.models import Position

from config.settings import Settings


class RejectionReason(str, Enum):
    POSITION_LIMIT = "max_positions_reached"
    SECTOR_CONCENTRATION = "sector_concentration_exceeded"
    INSUFFICIENT_BUYING_POWER = "insufficient_buying_power"
    ZERO_SHARES = "computed_qty_is_zero"
    CIRCUIT_BREAKER = "daily_circuit_breaker_triggered"
    ALREADY_HELD = "ticker_already_held"


@dataclass
class SizingResult:
    approved: bool
    qty: float                        # shares to buy/sell
    limit_price: float
    stop_price: float
    notional: float                   # qty * limit_price
    take_profit_price: Optional[float] = None   # None when not approved
    rejection_reason: Optional[RejectionReason] = None


@dataclass
class CircuitBreakerResult:
    triggered: bool
    daily_pnl: float
    daily_pnl_pct: float
    threshold_pct: float


@dataclass
class TrailingStopCheck:
    should_activate: bool
    unrealized_pct: float
    activation_threshold_pct: float


class RiskManager:
    """
    Stateless risk evaluator. All inputs are passed in; no internal state.

    Usage pattern:
        rm = RiskManager(settings)
        cb = rm.check_circuit_breaker(daily_pnl, equity)
        if cb.triggered:
            broker.close_all_positions()
            return
        result = rm.size_position(symbol, sector, side, price, equity, positions)
        if result.approved:
            broker.submit_bracket_order(...)
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    # -------------------------------------------------------------------------
    # Circuit breaker
    # -------------------------------------------------------------------------

    def check_circuit_breaker(
        self, daily_pnl: float, equity: float
    ) -> CircuitBreakerResult:
        """
        Returns triggered=True if today's P&L has breached the daily loss limit.

        daily_pnl: realized + unrealized P&L since previous close (negative = loss).
        equity:    current account equity.
        """
        if equity <= 0:
            return CircuitBreakerResult(
                triggered=True,
                daily_pnl=daily_pnl,
                daily_pnl_pct=0.0,
                threshold_pct=self._s.daily_loss_limit_pct,
            )

        daily_pnl_pct = daily_pnl / equity
        triggered = daily_pnl_pct <= -abs(self._s.daily_loss_limit_pct)

        return CircuitBreakerResult(
            triggered=triggered,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            threshold_pct=self._s.daily_loss_limit_pct,
        )

    # -------------------------------------------------------------------------
    # Position sizing and pre-trade checks
    # -------------------------------------------------------------------------

    def size_position(
        self,
        symbol: str,
        sector: str,
        side: OrderSide,
        limit_price: float,
        equity: float,
        open_positions: list[Position],
        sector_map: dict[str, str],       # symbol -> sector for open positions
        buying_power: float,
    ) -> SizingResult:
        """
        Compute approved position size after all risk pre-checks.

        Checks (in order):
        1. Already holding this ticker?
        2. Open position count < max_positions?
        3. Adding this position would breach sector concentration?
        4. Computed notional fits within buying power?

        Returns SizingResult with approved=False and rejection_reason if any
        check fails.
        """
        # 1. Already holding?
        held_symbols = {str(p.symbol) for p in open_positions}
        if symbol in held_symbols:
            return self._reject(0, limit_price, RejectionReason.ALREADY_HELD)

        # 2. Position count gate
        if len(open_positions) >= self._s.max_positions:
            return self._reject(0, limit_price, RejectionReason.POSITION_LIMIT)

        # 3. Sector concentration check
        sector_notional = self._sector_notional(sector, open_positions, sector_map)
        max_sector_notional = equity * self._s.max_sector_pct
        max_position_notional = equity * self._s.max_position_pct

        # How much room remains in this sector?
        sector_headroom = max_sector_notional - sector_notional
        if sector_headroom <= 0:
            return self._reject(0, limit_price, RejectionReason.SECTOR_CONCENTRATION)

        # Notional is the smaller of: max position size OR sector headroom
        target_notional = min(max_position_notional, sector_headroom)
        qty = target_notional / limit_price if limit_price > 0 else 0
        qty = self._floor_shares(qty)

        if qty <= 0:
            return self._reject(0, limit_price, RejectionReason.ZERO_SHARES)

        actual_notional = qty * limit_price

        # 4. Buying power
        if actual_notional > buying_power:
            return self._reject(qty, limit_price, RejectionReason.INSUFFICIENT_BUYING_POWER)

        stop_price = self._compute_stop(limit_price, side)
        take_profit_price = self._compute_take_profit(limit_price, side)

        return SizingResult(
            approved=True,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            notional=actual_notional,
            take_profit_price=take_profit_price,
        )

    # -------------------------------------------------------------------------
    # Stop price calculation
    # -------------------------------------------------------------------------

    def compute_stop_price(self, entry_price: float, side: OrderSide) -> float:
        """
        Calculate the initial stop-loss price.

        Long:  stop = entry * (1 - stop_loss_pct)
        Short: stop = entry * (1 + stop_loss_pct)
        """
        return self._compute_stop(entry_price, side)

    # -------------------------------------------------------------------------
    # Trailing stop activation check
    # -------------------------------------------------------------------------

    def check_trailing_stop_activation(self, position: Position) -> TrailingStopCheck:
        """
        Returns should_activate=True if this position's unrealized gain has
        reached the activation threshold and the trailing stop should replace
        the fixed stop.

        Only meaningful for positions with unrealized gains.
        """
        cost_basis = float(position.cost_basis or 0)
        market_value = float(position.market_value or 0)
        qty = float(position.qty or 0)

        if cost_basis <= 0 or qty == 0:
            return TrailingStopCheck(
                should_activate=False,
                unrealized_pct=0.0,
                activation_threshold_pct=self._s.trail_activation_pct,
            )

        avg_entry = cost_basis / qty
        current_price = market_value / qty if qty != 0 else 0

        # For longs: positive pnl = price went up
        # For shorts: qty is negative in Alpaca, market_value reflects short value
        side_sign = 1 if float(position.qty or 0) > 0 else -1
        unrealized_pct = side_sign * (current_price - avg_entry) / avg_entry if avg_entry > 0 else 0.0

        should_activate = unrealized_pct >= self._s.trail_activation_pct

        return TrailingStopCheck(
            should_activate=should_activate,
            unrealized_pct=unrealized_pct,
            activation_threshold_pct=self._s.trail_activation_pct,
        )

    # -------------------------------------------------------------------------
    # Overnight / flatten check
    # -------------------------------------------------------------------------

    def should_flatten_now(
        self,
        minutes_to_close: int,
        unrealized_pct: float,
        sentiment_positive: bool,
    ) -> bool:
        """
        Returns True if this position should be closed before market close.

        Overnight policy (from CLAUDE.md):
        - Default: close if within flatten_before_close_minutes of close
        - Exception: hold if unrealized gain >= 1% AND sentiment is positive
        """
        within_flatten_window = minutes_to_close <= self._s.flatten_before_close_minutes
        if not within_flatten_window:
            return False

        hold_exception = unrealized_pct >= 0.01 and sentiment_positive
        return not hold_exception

    # -------------------------------------------------------------------------
    # Portfolio-level helpers
    # -------------------------------------------------------------------------

    def open_position_count(self, positions: list[Position]) -> int:
        return len(positions)

    def can_add_position(self, positions: list[Position]) -> bool:
        return len(positions) < self._s.max_positions

    def sector_notional_for(
        self,
        sector: str,
        positions: list[Position],
        sector_map: dict[str, str],
    ) -> float:
        """Return total notional currently deployed in a sector."""
        return self._sector_notional(sector, positions, sector_map)

    def sector_has_room(
        self,
        sector: str,
        equity: float,
        positions: list[Position],
        sector_map: dict[str, str],
    ) -> bool:
        used = self._sector_notional(sector, positions, sector_map)
        return used < equity * self._s.max_sector_pct

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _compute_stop(self, entry_price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return round(entry_price * (1 - self._s.stop_loss_pct), 2)
        return round(entry_price * (1 + self._s.stop_loss_pct), 2)

    def _compute_take_profit(self, entry_price: float, side: OrderSide) -> float:
        if side == OrderSide.BUY:
            return round(entry_price * (1 + self._s.take_profit_pct), 2)
        return round(entry_price * (1 - self._s.take_profit_pct), 2)

    def _sector_notional(
        self,
        sector: str,
        positions: list[Position],
        sector_map: dict[str, str],
    ) -> float:
        total = 0.0
        for pos in positions:
            if sector_map.get(str(pos.symbol), "").lower() == sector.lower():
                total += abs(float(pos.market_value or 0))
        return total

    @staticmethod
    def _floor_shares(qty: float) -> float:
        """Floor to 2 decimal places (Alpaca supports fractional shares)."""
        return float(int(qty * 100) / 100)

    def _reject(
        self, qty: float, price: float, reason: RejectionReason
    ) -> SizingResult:
        return SizingResult(
            approved=False,
            qty=qty,
            limit_price=price,
            stop_price=0.0,
            notional=qty * price,
            rejection_reason=reason,
        )
