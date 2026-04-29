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
    PORTFOLIO_HEAT = "portfolio_heat_exceeded"


@dataclass
class SizingResult:
    approved: bool
    qty: float                        # shares to buy/sell
    limit_price: float
    stop_price: float
    notional: float                   # qty * limit_price
    take_profit_price: Optional[float] = None   # None when not approved
    rejection_reason: Optional[RejectionReason] = None
    stop_pct: float = 0.0             # actual stop distance used (ATR-derived or fixed)
    risk_dollars: float = 0.0         # qty × (entry - stop) — open risk this trade contributes


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
        atr_pct: Optional[float] = None,  # ATR/price ratio; enables ATR-based sizing
        sentiment_bias: Optional[str] = None,  # BULLISH|BEARISH|NEUTRAL — scales notional when matching
    ) -> SizingResult:
        """
        Compute approved position size after all risk pre-checks.

        Sizing model:
          - When atr_pct is provided: ATR-based. Stop distance = max(
              atr_stop_multiplier × atr_pct, min_stop_pct). Qty derived so that
              qty × stop_distance ≈ risk_per_trade_pct × equity. Each trade
              contributes the same dollar-risk regardless of ticker volatility.
          - When atr_pct is None: legacy fixed sizing at max_position_pct,
              stop_loss_pct from settings.

        Checks (in order):
        1. Already holding this ticker?
        2. Open position count < max_positions?
        3. Sector concentration headroom (caps notional).
        4. Portfolio heat cap — aggregate open dollar-risk ≤ max_portfolio_heat × equity.
        5. Computed notional fits within buying power?
        """
        # 1. Already holding?
        held_symbols = {str(p.symbol) for p in open_positions}
        if symbol in held_symbols:
            return self._reject(0, limit_price, RejectionReason.ALREADY_HELD)

        # 2. Position count gate
        if len(open_positions) >= self._s.max_positions:
            return self._reject(0, limit_price, RejectionReason.POSITION_LIMIT)

        # 3. Sector concentration headroom
        sector_notional = self._sector_notional(sector, open_positions, sector_map)
        max_sector_notional = equity * self._s.max_sector_pct
        max_position_notional = equity * self._s.max_position_pct
        sector_headroom = max_sector_notional - sector_notional
        if sector_headroom <= 0:
            return self._reject(0, limit_price, RejectionReason.SECTOR_CONCENTRATION)

        # Resolve effective stop distance (ATR-based or fallback fixed)
        stop_pct = self._resolve_stop_pct(atr_pct)

        # Risk-based target notional (if atr_pct given) vs flat max-position notional
        if atr_pct is not None and stop_pct > 0:
            risk_dollars_target = self._s.risk_per_trade_pct * equity
            risk_based_notional = risk_dollars_target / stop_pct
            target_notional = min(risk_based_notional, max_position_notional, sector_headroom)
        else:
            target_notional = min(max_position_notional, sector_headroom)

        # Sentiment-as-sizer: scale up when bias matches the trade direction.
        # Cap above by max_position_notional and sector_headroom so we never breach hard limits.
        if sentiment_bias and self._s.sentiment_size_multiplier > 1.0:
            matches = (
                (side == OrderSide.BUY and sentiment_bias == "BULLISH")
                or (side == OrderSide.SELL and sentiment_bias == "BEARISH")
            )
            if matches:
                target_notional = min(
                    target_notional * self._s.sentiment_size_multiplier,
                    max_position_notional,
                    sector_headroom,
                )

        qty = self._floor_shares(target_notional / limit_price) if limit_price > 0 else 0
        if qty <= 0:
            return self._reject(0, limit_price, RejectionReason.ZERO_SHARES)

        actual_notional = qty * limit_price
        new_risk = qty * limit_price * stop_pct

        # 4. Portfolio heat cap — aggregate open risk including this trade
        existing_risk = self._estimate_open_risk(open_positions)
        if existing_risk + new_risk > self._s.max_portfolio_heat * equity:
            return self._reject(0, limit_price, RejectionReason.PORTFOLIO_HEAT)

        # 5. Buying power
        if actual_notional > buying_power:
            return self._reject(qty, limit_price, RejectionReason.INSUFFICIENT_BUYING_POWER)

        stop_price = self._compute_stop(limit_price, side, stop_pct)
        take_profit_price = self._compute_take_profit(limit_price, side)

        return SizingResult(
            approved=True,
            qty=qty,
            limit_price=limit_price,
            stop_price=stop_price,
            notional=actual_notional,
            take_profit_price=take_profit_price,
            stop_pct=stop_pct,
            risk_dollars=new_risk,
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

    def _compute_stop(
        self,
        entry_price: float,
        side: OrderSide,
        stop_pct: Optional[float] = None,
    ) -> float:
        pct = stop_pct if stop_pct is not None else self._s.stop_loss_pct
        if side == OrderSide.BUY:
            return round(entry_price * (1 - pct), 2)
        return round(entry_price * (1 + pct), 2)

    def _resolve_stop_pct(self, atr_pct: Optional[float]) -> float:
        """ATR-derived stop distance, floored at min_stop_pct. Falls back to fixed."""
        if atr_pct is None or atr_pct <= 0:
            return self._s.stop_loss_pct
        return max(self._s.atr_stop_multiplier * atr_pct, self._s.min_stop_pct)

    def _estimate_open_risk(self, positions: list[Position]) -> float:
        """
        Approximate aggregate open dollar-risk across current positions.

        We don't store per-position stop prices yet, so we use cost_basis ×
        stop_loss_pct as a conservative-ish proxy. Tighter ATR stops will
        under-estimate; wider ATR stops will under-estimate. Acceptable for
        the heat cap's purpose — preventing pile-on of new risk.
        """
        total = 0.0
        for pos in positions:
            try:
                cb = abs(float(pos.cost_basis or 0))
            except (TypeError, ValueError):
                cb = 0.0
            total += cb * self._s.stop_loss_pct
        return total

    def estimate_open_risk(self, positions: list[Position]) -> float:
        """Public wrapper around _estimate_open_risk for callers that want to log."""
        return self._estimate_open_risk(positions)

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
        """Floor to whole shares — Alpaca rejects fractional bracket orders."""
        return float(int(qty))

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
