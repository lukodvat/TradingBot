"""
Backtest performance metrics — pure functions, no I/O.

All functions accept a list of BacktestTrade objects and/or a daily equity Series.
Nothing here touches the network, disk, or Alpaca.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd


@dataclass
class BacktestTrade:
    symbol: str
    entry_date: object          # datetime.date
    exit_date: object           # datetime.date
    side: str                   # 'long' | 'short'
    entry_price: float
    exit_price: float
    qty: float
    pnl: float                  # absolute USD P&L
    pnl_pct: float              # return as fraction (0.02 = +2%)
    exit_reason: str            # 'stop_loss' | 'trailing_stop' | 'eod_flatten' | 'signal_exit'


@dataclass
class BacktestMetrics:
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    avg_win_pct: float
    avg_loss_pct: float
    expectancy_pct: float       # (win_rate * avg_win) + (loss_rate * avg_loss)
    profit_factor: float        # gross_profit / abs(gross_loss)
    total_trades: int
    winning_trades: int
    losing_trades: int
    trading_days: int
    passed_gate: bool           # True iff expectancy_pct > 0 and total_trades >= min_trades


MIN_TRADES_FOR_GATE = 10       # refuse to pass gate on tiny sample sizes


def compute_metrics(
    trades: list[BacktestTrade],
    equity_curve: pd.Series,    # index=date, values=equity (float)
    initial_equity: float,
    min_trades: int = MIN_TRADES_FOR_GATE,
) -> BacktestMetrics:
    """
    Compute all performance metrics from a completed backtest.

    Args:
        trades:         All completed trades from the simulation.
        equity_curve:   Daily equity values (one per trading day simulated).
        initial_equity: Starting account value.
        min_trades:     Minimum trades required for the gate to pass.

    Returns:
        BacktestMetrics dataclass with passed_gate indicating deployment readiness.
    """
    trading_days = len(equity_curve)
    total_trades = len(trades)

    # --- Trade-level stats ---
    if total_trades == 0:
        return _empty_metrics(initial_equity, equity_curve, trading_days)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    win_rate = len(wins) / total_trades
    avg_win_pct = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss_pct = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

    loss_rate = 1.0 - win_rate
    expectancy_pct = (win_rate * avg_win_pct) + (loss_rate * avg_loss_pct)

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # --- Equity curve stats ---
    final_equity = float(equity_curve.iloc[-1]) if not equity_curve.empty else initial_equity
    total_return_pct = (final_equity - initial_equity) / initial_equity

    sharpe = _compute_sharpe(equity_curve)
    max_dd = _compute_max_drawdown(equity_curve)
    annualized = _annualize_return(total_return_pct, trading_days)

    # --- Gate ---
    passed_gate = (
        expectancy_pct > 0
        and total_trades >= min_trades
        and sharpe > 0
    )

    return BacktestMetrics(
        total_return_pct=total_return_pct,
        annualized_return_pct=annualized,
        sharpe_ratio=sharpe,
        max_drawdown_pct=max_dd,
        win_rate=win_rate,
        avg_win_pct=avg_win_pct,
        avg_loss_pct=avg_loss_pct,
        expectancy_pct=expectancy_pct,
        profit_factor=profit_factor,
        total_trades=total_trades,
        winning_trades=len(wins),
        losing_trades=len(losses),
        trading_days=trading_days,
        passed_gate=passed_gate,
    )


# ---------------------------------------------------------------------------
# Individual metric functions (also exported for unit tests)
# ---------------------------------------------------------------------------

def compute_sharpe(equity_curve: pd.Series) -> float:
    return _compute_sharpe(equity_curve)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    return _compute_max_drawdown(equity_curve)


def compute_expectancy(
    trades: list[BacktestTrade],
) -> float:
    if not trades:
        return 0.0
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades)
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _compute_sharpe(equity_curve: pd.Series) -> float:
    """
    Annualized Sharpe ratio (assuming 0% risk-free rate).

    Uses daily returns from the equity curve.
    Returns 0.0 if insufficient data or zero variance.
    """
    if len(equity_curve) < 2:
        return 0.0

    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0

    sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
    return float(sharpe)


def _compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown as a positive fraction (e.g. 0.15 = 15% drawdown).

    Uses the standard rolling-peak approach.
    Returns 0.0 for empty or single-point curves.
    """
    if len(equity_curve) < 2:
        return 0.0

    rolling_peak = equity_curve.cummax()
    drawdown = (equity_curve - rolling_peak) / rolling_peak
    return float(abs(drawdown.min()))


def _annualize_return(total_return: float, trading_days: int) -> float:
    """Compound annualization: (1 + r)^(252/n) - 1."""
    if trading_days <= 0:
        return 0.0
    return float((1 + total_return) ** (252 / trading_days) - 1)


def _empty_metrics(
    initial_equity: float,
    equity_curve: pd.Series,
    trading_days: int,
) -> BacktestMetrics:
    return BacktestMetrics(
        total_return_pct=0.0,
        annualized_return_pct=0.0,
        sharpe_ratio=0.0,
        max_drawdown_pct=0.0,
        win_rate=0.0,
        avg_win_pct=0.0,
        avg_loss_pct=0.0,
        expectancy_pct=0.0,
        profit_factor=0.0,
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        trading_days=trading_days,
        passed_gate=False,
    )
