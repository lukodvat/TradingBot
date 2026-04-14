"""
Alpaca paper trading client.

This is the ONLY module that touches the Alpaca API. All order submission,
position reads, and account queries go through here.

The live-key guard runs in __init__ and raises immediately — no Alpaca client
is ever constructed if the URL is not paper.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderClass, OrderSide, TimeInForce
from alpaca.trading.models import Order, Position, TradeAccount
from alpaca.trading.requests import (
    ClosePositionRequest,
    GetOrdersRequest,
    LimitOrderRequest,
    MarketOrderRequest,
    StopLossRequest,
    TrailingStopOrderRequest,
)

from config.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class AccountSnapshot:
    """Minimal account state needed for risk checks each run."""
    equity: float
    cash: float
    portfolio_value: float
    buying_power: float
    positions: list[Position]
    daily_pnl: float  # unrealized + realized vs. previous close
    open_order_count: int


class LiveKeyError(RuntimeError):
    """Raised if a non-paper Alpaca URL is detected."""


class BrokerClient:
    """
    Thin wrapper around alpaca-py TradingClient.

    Responsibilities:
    - Enforce paper-only constraint
    - Provide typed helpers for account snapshot, order submission, stop updates
    - Never make strategy decisions — purely mechanical execution
    """

    def __init__(self, settings: Settings) -> None:
        self._guard_paper_url(settings.alpaca_base_url)
        self._settings = settings
        self._client = TradingClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            paper=True,  # explicit — alpaca-py uses this to set the correct endpoint
        )
        log.info("BrokerClient initialized (paper=True, url=%s)", settings.alpaca_base_url)

    # -------------------------------------------------------------------------
    # Guard
    # -------------------------------------------------------------------------

    @staticmethod
    def _guard_paper_url(url: str) -> None:
        """
        Belt-and-suspenders check separate from Settings validation.
        Settings catches misconfigured .env; this catches programmatic misuse.
        """
        if "paper" not in url.lower():
            raise LiveKeyError(
                f"BrokerClient refuses to initialize: URL does not contain 'paper'.\n"
                f"  Got: {url}\n"
                f"  This system is paper-trading only. Aborting."
            )

    # -------------------------------------------------------------------------
    # Account / portfolio reads
    # -------------------------------------------------------------------------

    def get_account(self) -> TradeAccount:
        return self._client.get_account()

    def get_positions(self) -> list[Position]:
        return self._client.get_all_positions()

    def get_open_orders(self) -> list[Order]:
        req = GetOrdersRequest(status="open")  # type: ignore[call-arg]
        result = self._client.get_orders(filter=req)
        return list(result) if result else []

    def snapshot(self) -> AccountSnapshot:
        """
        Single call that returns everything the run loop needs for risk checks.
        Fetches account + positions in two API calls.
        """
        account = self.get_account()
        positions = self.get_positions()
        open_orders = self.get_open_orders()

        equity = float(account.equity or 0)
        prev_equity = float(account.last_equity or equity)
        daily_pnl = equity - prev_equity

        return AccountSnapshot(
            equity=equity,
            cash=float(account.cash or 0),
            portfolio_value=float(account.portfolio_value or 0),
            buying_power=float(account.buying_power or 0),
            positions=positions,
            daily_pnl=daily_pnl,
            open_order_count=len(open_orders),
        )

    # -------------------------------------------------------------------------
    # Order submission
    # -------------------------------------------------------------------------

    def submit_bracket_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        limit_price: float,
        stop_price: float,
    ) -> Order:
        """
        Submit a bracket order: limit entry + stop-loss leg.

        No take-profit leg — exits are handled by the trailing stop update
        that runs at the top of each session once the position is up 1.5%.

        Args:
            symbol:      Ticker symbol.
            side:        OrderSide.BUY or OrderSide.SELL.
            qty:         Number of shares (will be rounded to 2 decimal places).
            limit_price: Entry limit price.
            stop_price:  Initial stop-loss price (2% below entry for longs).
        """
        qty = round(qty, 2)
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=round(limit_price, 2),
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
        )
        order = self._client.submit_order(req)
        log.info(
            "Bracket order submitted: %s %s x%.2f @ limit=%.2f stop=%.2f id=%s",
            side.value, symbol, qty, limit_price, stop_price, order.id,
        )
        return order

    def submit_market_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
    ) -> Order:
        """Market order used for emergency liquidations (circuit breaker)."""
        qty = round(qty, 2)
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = self._client.submit_order(req)
        log.info(
            "Market order submitted: %s %s x%.2f id=%s",
            side.value, symbol, qty, order.id,
        )
        return order

    def replace_stop_with_trailing(
        self,
        symbol: str,
        side: OrderSide,
        qty: float,
        trail_pct: float,
    ) -> Order:
        """
        Cancel the existing stop-loss order for a position and submit a
        trailing stop in its place.

        Called by the position management loop once unrealized gain >= trail_activation_pct.

        Args:
            symbol:    Ticker symbol.
            side:      OrderSide.SELL for long positions.
            qty:       Shares to protect (usually current position qty).
            trail_pct: Trail distance as a decimal (e.g. 0.03 for 3%).
        """
        # Cancel any open stop orders for this symbol first
        self._cancel_open_stops_for(symbol)

        req = TrailingStopOrderRequest(
            symbol=symbol,
            qty=round(qty, 2),
            side=side,
            time_in_force=TimeInForce.DAY,
            trail_percent=round(trail_pct * 100, 2),  # alpaca-py wants percentage
        )
        order = self._client.submit_order(req)
        log.info(
            "Trailing stop submitted: %s %s trail=%.1f%% id=%s",
            symbol, side.value, trail_pct * 100, order.id,
        )
        return order

    # -------------------------------------------------------------------------
    # Liquidation helpers
    # -------------------------------------------------------------------------

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close a single position at market. Returns None if no position exists."""
        try:
            order = self._client.close_position(symbol)
            log.info("Closed position: %s", symbol)
            return order
        except Exception as exc:
            log.warning("Failed to close position %s: %s", symbol, exc)
            return None

    def close_all_positions(self) -> list[Order]:
        """
        Market-close all open positions.
        Used by the daily circuit breaker and end-of-day flatten.
        """
        orders = self._client.close_all_positions(cancel_orders=True)
        result = list(orders) if orders else []
        log.warning("Closed all positions (%d orders submitted)", len(result))
        return result

    def cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
            log.info("Cancelled order: %s", order_id)
        except Exception as exc:
            log.warning("Failed to cancel order %s: %s", order_id, exc)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _cancel_open_stops_for(self, symbol: str) -> None:
        """Cancel all open stop orders for a symbol before replacing with a trail."""
        open_orders = self.get_open_orders()
        for order in open_orders:
            if (
                order.symbol == symbol
                and order.order_type is not None
                and "stop" in str(order.order_type).lower()
            ):
                self.cancel_order(str(order.id))
