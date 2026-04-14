"""
Portfolio management for Job B — position monitoring and lifecycle.

Responsibilities:
  1. Trailing stop activation — upgrade fixed stops to trailing stops once a
     position reaches the activation threshold (default +1.5% unrealized gain).
  2. Overnight flatten — close positions before market close per policy:
       - Default: flatten all near close (within flatten_before_close_minutes).
       - Hold exception: keep if unrealized >= +1% AND today's bias is BULLISH/BEARISH-match.
       - Friday override: always flatten everything (no exceptions).
  3. Equity snapshot — record account state to SQLite after each quant run.
  4. Held-today tracking — return the set of tickers already entered today so
     the signal scanner can enforce the same-ticker cooldown.

This module makes broker API calls (trailing stop replacement, position close).
It does NOT submit new orders — that belongs to the signal execution logic in main.py.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from alpaca.trading.models import Position

from config.settings import Settings
from core.broker import BrokerClient, AccountSnapshot
from core.risk import RiskManager

from db import store

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")
_MARKET_CLOSE_HOUR = 16   # NYSE closes at 16:00 ET


class PortfolioManager:
    """
    Stateful only in that it holds references to broker/risk/db.
    All decision logic lives in RiskManager — this class orchestrates calls.

    Usage (inside Job B):
        pm = PortfolioManager(settings, broker, risk, conn)
        pm.manage_trailing_stops(positions, run_timestamp)
        pm.manage_flattens(positions, biases, run_dt, run_timestamp)
        pm.record_snapshot(snap, run_timestamp, session="quant_1330")
        held = pm.get_held_today(date="2025-04-13")
    """

    def __init__(
        self,
        settings: Settings,
        broker: BrokerClient,
        risk: RiskManager,
        conn: sqlite3.Connection,
    ) -> None:
        self._s = settings
        self._broker = broker
        self._risk = risk
        self._conn = conn

    # -------------------------------------------------------------------------
    # Trailing stop activation
    # -------------------------------------------------------------------------

    def manage_trailing_stops(
        self,
        positions: list[Position],
        run_timestamp: str,
    ) -> int:
        """
        For each position, check if the unrealized gain has reached the activation
        threshold. If so, replace the fixed bracket stop with a trailing stop.

        Returns the number of stops upgraded this run.
        """
        upgraded = 0
        for position in positions:
            symbol = position.symbol
            check = self._risk.check_trailing_stop_activation(position)

            if not check.should_activate:
                log.debug(
                    "Trail check %s: unrealized=%.2f%% < threshold=%.2f%%",
                    symbol, check.unrealized_pct * 100, check.activation_threshold_pct * 100,
                )
                continue

            log.info(
                "Activating trailing stop for %s (unrealized=%.2f%%)",
                symbol, check.unrealized_pct * 100,
            )
            try:
                self._broker.replace_stop_with_trailing(symbol)
                upgraded += 1
                log.info("Trailing stop activated for %s", symbol)
            except Exception as exc:
                log.error("Failed to activate trailing stop for %s: %s", symbol, exc)

        return upgraded

    # -------------------------------------------------------------------------
    # Overnight flatten
    # -------------------------------------------------------------------------

    def manage_flattens(
        self,
        positions: list[Position],
        biases: dict[str, str],     # {symbol: 'BULLISH'|'NEUTRAL'|'BEARISH'}
        run_dt: datetime,           # timezone-aware, used to compute minutes-to-close
        run_timestamp: str,
        is_friday: bool = False,
    ) -> int:
        """
        Close positions that should not be held overnight.

        Friday override: close all regardless of P&L or sentiment.
        Normal policy: close unless unrealized >= +1% AND bias favours the direction.

        Returns the number of positions closed.
        """
        closed = 0
        minutes_to_close = _minutes_to_market_close(run_dt)

        for position in positions:
            symbol = position.symbol

            # Friday: always flatten
            if is_friday:
                log.info("Friday flatten: closing %s", symbol)
                self._close(symbol)
                closed += 1
                continue

            unrealized_pct = _unrealized_pct(position)

            # Sentiment positive = bias matches position direction
            qty = float(position.qty or 0)
            bias = biases.get(symbol, "NEUTRAL")
            sentiment_positive = (
                (qty > 0 and bias == "BULLISH") or
                (qty < 0 and bias == "BEARISH")
            )

            if self._risk.should_flatten_now(minutes_to_close, unrealized_pct, sentiment_positive):
                log.info(
                    "Flattening %s: min_to_close=%d unrealized=%.2f%% bias=%s",
                    symbol, minutes_to_close, unrealized_pct * 100, bias,
                )
                self._close(symbol)
                closed += 1
            else:
                log.debug(
                    "Holding %s: min_to_close=%d unrealized=%.2f%% bias=%s",
                    symbol, minutes_to_close, unrealized_pct * 100, bias,
                )

        return closed

    # -------------------------------------------------------------------------
    # Snapshot recording
    # -------------------------------------------------------------------------

    def record_snapshot(
        self,
        snap: AccountSnapshot,
        run_timestamp: str,
        session: str,
        date: str,
    ) -> None:
        """
        Write equity snapshot and upsert today's running P&L to SQLite.
        Call once at the end of each quant run.
        """
        store.record_equity_snapshot(
            self._conn,
            run_timestamp=run_timestamp,
            session=session,
            equity=snap.equity,
            cash=snap.cash,
            portfolio_value=snap.portfolio_value,
            open_positions=len(snap.positions),
            daily_pnl=snap.daily_pnl,
        )

        unrealized = sum(
            float(p.unrealized_pl or 0) for p in snap.positions
        )
        store.upsert_daily_pnl(
            self._conn,
            date=date,
            unrealized_pnl=unrealized,
            total_pnl=snap.daily_pnl,
        )

        log.info(
            "Snapshot recorded: equity=%.2f cash=%.2f positions=%d daily_pnl=%.2f",
            snap.equity, snap.cash, len(snap.positions), snap.daily_pnl,
        )

    # -------------------------------------------------------------------------
    # Same-ticker cooldown
    # -------------------------------------------------------------------------

    def get_held_today(self, date: str) -> set[str]:
        """
        Return the set of tickers for which an entry trade was recorded today.
        Used by the signal scanner to enforce the same-ticker-per-day cooldown.

        date: 'YYYY-MM-DD'
        """
        rows = self._conn.execute(
            """
            SELECT DISTINCT symbol
            FROM trades
            WHERE DATE(run_timestamp) = ?
              AND side IN ('buy', 'sell_short')
            """,
            (date,),
        ).fetchall()
        tickers = {r["symbol"] for r in rows}
        log.debug("Held today (%s): %s", date, tickers or "none")
        return tickers

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _close(self, symbol: str) -> None:
        try:
            self._broker.close_position(symbol)
        except Exception as exc:
            log.error("Failed to close position %s: %s", symbol, exc)


# ---------------------------------------------------------------------------
# Pure helpers (no broker state)
# ---------------------------------------------------------------------------

def _minutes_to_market_close(run_dt: datetime) -> int:
    """
    Compute integer minutes between run_dt and 16:00 ET today.
    Returns 0 if past close.
    """
    close_dt = run_dt.astimezone(_ET).replace(
        hour=_MARKET_CLOSE_HOUR, minute=0, second=0, microsecond=0,
    )
    delta = close_dt - run_dt.astimezone(_ET)
    return max(0, int(delta.total_seconds() // 60))


def _unrealized_pct(position: Position) -> float:
    """
    Return unrealized P&L as a fraction (e.g. 0.015 = +1.5%).
    Uses Alpaca's unrealized_plpc field when available; falls back to manual calc.
    """
    plpc = position.unrealized_plpc
    if plpc is not None:
        return float(plpc)

    cost_basis = float(position.cost_basis or 0)
    market_value = float(position.market_value or 0)
    if cost_basis <= 0:
        return 0.0
    return (market_value - cost_basis) / cost_basis
