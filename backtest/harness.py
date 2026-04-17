"""
Backtest harness — Phase 1 (quant-only) replay engine.

Replays daily OHLCV bars day by day applying the same entry criteria,
position sizing, and stop-loss logic as the live system — without any
LLM or sentiment involvement.

Phase 1 purpose: establish that the quant signals have positive expectancy
BEFORE enabling sentiment filtering and paper trading. The gate check in
main.py reads the report produced here.

Lookahead prevention contract (unit-testable):
  - Signal on day N uses only bars with index <= N.
  - Entry executes at day N+1 OPEN price.
  - Stop/trail checked against day N+1 HIGH and LOW.
  - No future price information crosses the signal boundary.

Slippage model:
  - Entry:  price * (1 + slippage_bps / 10_000) for longs
  - Exit:   price * (1 - slippage_bps / 10_000) for longs
  - Inverse for shorts (not implemented in Phase 1 — longs only).
"""

import json
import logging
import os
from dataclasses import asdict
from datetime import date, datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as pta   # functional API — avoids DataFrame accessor ambiguity

from backtest.metrics import BacktestMetrics, BacktestTrade, compute_metrics
from config.settings import Settings
from data.market import load_sector_map

log = logging.getLogger(__name__)


def _load_macro_blackout_dates(path: str) -> set[str]:
    """Load YYYY-MM-DD blackout dates from macro_events.yaml. Returns empty set on error."""
    import yaml
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return {str(e["date"]) for e in (data or {}).get("events", []) if "date" in e}
    except Exception as exc:
        log.warning("Backtest: could not load macro events from %s: %s", path, exc)
        return set()

# Warm-up bars needed before any signal is valid.
# EMA(20) needs 20, RSI(14) needs 14, near-high lookback needs 63. Use 65 to match live
# system behaviour (which always has ≥80 bars loaded before scanning).
SIGNAL_WARMUP_BARS = 65


class BacktestHarness:
    """
    Day-by-day replay engine for the quant-only strategy.

    Usage:
        harness = BacktestHarness(settings, bars_dict)
        result = harness.run()
        harness.save_report(result, result.metrics)
    """

    def __init__(
        self,
        settings: Settings,
        bars: dict[str, pd.DataFrame],
        slippage_bps: Optional[int] = None,
        initial_equity: float = 10_000.0,
    ) -> None:
        self._settings = settings
        self._bars = dict(bars)  # copy so we can pop SPY without mutating caller's dict
        self._spy_bars: Optional[pd.DataFrame] = self._bars.pop("SPY", None)
        self._slippage_bps = slippage_bps if slippage_bps is not None else settings.backtest_slippage_bps
        self._slippage_mult = self._slippage_bps / 10_000
        self._initial_equity = initial_equity
        self._sector_map = load_sector_map(settings.watchlist_path)
        self._macro_blackouts = _load_macro_blackout_dates(settings.macro_events_path)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(self) -> tuple[list[BacktestTrade], pd.Series, BacktestMetrics]:
        """
        Execute the full backtest simulation.

        Returns:
            trades:       List of completed BacktestTrade objects.
            equity_curve: pd.Series (date index, float equity values).
            metrics:      Computed BacktestMetrics including passed_gate.
        """
        trades, equity_curve = self._simulate()
        metrics = compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            initial_equity=self._initial_equity,
        )
        log.info(
            "Backtest complete: %d trades | win_rate=%.1f%% | expectancy=%.2f%% "
            "| sharpe=%.2f | max_dd=%.1f%% | gate=%s",
            metrics.total_trades,
            metrics.win_rate * 100,
            metrics.expectancy_pct * 100,
            metrics.sharpe_ratio,
            metrics.max_drawdown_pct * 100,
            "PASS" if metrics.passed_gate else "FAIL",
        )
        return trades, equity_curve, metrics

    def save_report(
        self,
        trades: list[BacktestTrade],
        metrics: BacktestMetrics,
    ) -> str:
        """
        Persist the backtest report to reports/ as a JSON file.

        Filename includes a UTC timestamp so reruns don't overwrite history.
        Returns the file path written.
        """
        os.makedirs(self._settings.reports_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._settings.reports_dir, f"backtest_{ts}.json")

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "slippage_bps": self._slippage_bps,
            "initial_equity": self._initial_equity,
            "config_fingerprint": compute_config_fingerprint(self._settings),
            "metrics": asdict(metrics),
            "trades": [
                {
                    "symbol": t.symbol,
                    "entry_date": str(t.entry_date),
                    "exit_date": str(t.exit_date),
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": round(t.pnl, 4),
                    "pnl_pct": round(t.pnl_pct, 6),
                    "exit_reason": t.exit_reason,
                }
                for t in trades
            ],
        }

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

        log.info("Backtest report saved: %s", path)
        return path

    # -------------------------------------------------------------------------
    # Simulation core
    # -------------------------------------------------------------------------

    def _simulate(self) -> tuple[list[BacktestTrade], pd.Series]:
        """
        Main simulation loop. Processes each trading day in chronological order.
        """
        all_dates = self._get_trading_dates()
        equity = self._initial_equity
        open_positions: dict[str, _OpenPosition] = {}
        completed_trades: list[BacktestTrade] = []
        equity_curve: dict[date, float] = {}
        # Same-ticker cooldown: symbols that have already been entered on a given day.
        # Mirrors live system — once a ticker is traded today, no re-entry until tomorrow.
        held_today: dict[date, set[str]] = {}

        for i, today in enumerate(all_dates):
            # --- Step 1: Update existing positions against today's bars ---
            for symbol in list(open_positions.keys()):
                pos = open_positions[symbol]
                today_bar = self._get_bar(symbol, today)
                if today_bar is None:
                    continue  # no data today — carry position forward

                # Time-based exit: close stale positions with minimal gain
                time_exit = self._check_time_exit(pos, today, today_bar)
                if time_exit:
                    exit_price, exit_reason = time_exit
                    trade = self._close_position(pos, exit_price, today, exit_reason)
                    equity += trade.pnl
                    completed_trades.append(trade)
                    del open_positions[symbol]
                    continue

                exit_result = self._check_exits(pos, today_bar)
                if exit_result:
                    exit_price, exit_reason = exit_result
                    trade = self._close_position(pos, exit_price, today, exit_reason)
                    equity += trade.pnl
                    completed_trades.append(trade)
                    del open_positions[symbol]

            # --- Step 2: Scan for new entries (signal on today, entry tomorrow) ---
            next_day = all_dates[i + 1] if i + 1 < len(all_dates) else None

            # Macro blackout gate — block new entries on FOMC/CPI dates (entry date = next_day).
            macro_blocked = (
                next_day is not None
                and next_day.strftime("%Y-%m-%d") in self._macro_blackouts
            )

            if next_day is not None and not macro_blocked:
                today_entered = held_today.setdefault(next_day, set())
                candidates = self._find_candidates(today, open_positions, equity)
                for symbol, signal_price in candidates:
                    if len(open_positions) >= self._settings.max_positions:
                        break
                    # Same-ticker cooldown: skip if already entered today
                    if symbol in today_entered:
                        continue
                    # Sector concentration cap — reject if adding this fills the sector
                    if not self._sector_has_room(symbol, open_positions, equity):
                        continue
                    next_bar = self._get_bar(symbol, next_day)
                    if next_bar is None:
                        continue

                    # Entry at next day's OPEN + slippage (lookahead-free)
                    raw_entry = float(next_bar["open"])
                    entry_price = raw_entry * (1 + self._slippage_mult)
                    stop_price = entry_price * (1 - self._settings.stop_loss_pct)

                    qty = self._compute_qty(equity, entry_price, symbol)
                    if qty <= 0:
                        continue

                    open_positions[symbol] = _OpenPosition(
                        symbol=symbol,
                        entry_date=next_day,
                        entry_price=entry_price,
                        stop_price=stop_price,
                        qty=qty,
                        trailing_activated=False,
                        trail_stop_price=None,
                    )
                    today_entered.add(symbol)
                    log.debug(
                        "BT ENTRY %s @ %.2f stop=%.2f qty=%.2f on %s",
                        symbol, entry_price, stop_price, qty, next_day,
                    )

            equity_curve[today] = equity

        # Close any remaining open positions at last available close
        for symbol, pos in open_positions.items():
            last_bar = self._last_bar(symbol)
            if last_bar is not None:
                exit_price = float(last_bar["close"]) * (1 - self._slippage_mult)
                trade = self._close_position(pos, exit_price, all_dates[-1], "eod_flatten")
                equity += trade.pnl
                completed_trades.append(trade)

        equity_series = pd.Series(equity_curve)
        equity_series.index = pd.to_datetime(equity_series.index)
        return completed_trades, equity_series

    # -------------------------------------------------------------------------
    # Signal detection
    # -------------------------------------------------------------------------

    def _find_candidates(
        self,
        today: date,
        open_positions: dict,
        equity: float,
    ) -> list[tuple[str, float]]:
        """
        Scan all tickers for entry signals on `today`.

        Returns list of (symbol, close_price) for tickers with valid signals,
        ranked by signal strength (volume ratio, descending).
        """
        candidates: list[tuple[str, float, float]] = []  # (symbol, price, volume_ratio)

        spy_return_20d = self._compute_spy_return(today)

        for symbol, df in self._bars.items():
            if symbol in open_positions:
                continue

            # Bars up to and including today (lookahead prevention)
            history = df[df.index.date <= today]
            if len(history) < SIGNAL_WARMUP_BARS:
                continue

            signal = self._compute_signal(history, spy_return_20d=spy_return_20d)
            if signal:
                close = float(history["close"].iloc[-1])
                vol_ratio = signal
                candidates.append((symbol, close, vol_ratio))

        # Rank by volume ratio (strongest participation first)
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(sym, price) for sym, price, _ in candidates]

    def _sector_has_room(
        self,
        symbol: str,
        open_positions: dict,
        equity: float,
    ) -> bool:
        """
        Mirrors core.risk.RiskManager sector-cap logic.

        Returns True if adding one more position in `symbol`'s sector stays below
        max_sector_pct × equity. Positions with unknown sector always pass (sector
        map fallback). Position notional uses entry_price × qty at entry time.
        """
        sector = self._sector_map.get(symbol, "").lower()
        if not sector:
            return True  # unmapped ticker — defer to other caps
        used = 0.0
        for pos in open_positions.values():
            if self._sector_map.get(pos.symbol, "").lower() == sector:
                used += pos.entry_price * pos.qty
        return used < equity * self._settings.max_sector_pct

    def _compute_spy_return(self, today: date) -> Optional[float]:
        """
        Compute SPY's 20-day return ending on `today`, for the relative strength filter.

        Returns None if SPY bars are unavailable or there is insufficient history.
        """
        if self._spy_bars is None:
            return None
        spy_history = self._spy_bars[self._spy_bars.index.date <= today]
        if len(spy_history) < 21:
            return None
        spy_close = spy_history["close"]
        return (float(spy_close.iloc[-1]) - float(spy_close.iloc[-21])) / float(spy_close.iloc[-21])

    def _compute_signal(
        self,
        history: pd.DataFrame,
        spy_return_20d: Optional[float] = None,
    ) -> Optional[float]:
        """
        Compute entry signal on a history DataFrame ending at the signal bar.

        Entry criteria (longs only):
          1. Close above EMA(20)
          2. RSI(14) between rsi_min and rsi_max
          3. Volume > volume_multiplier × 20-bar average volume
          4. Near-high proximity: price within near_high_max_drawdown of 63-bar high
             (only applied when >= near_high_lookback bars available)
          5. Relative strength: 20d return > SPY 20d return
             (only applied when spy_return_20d is provided)

        Returns:
            volume ratio (float > 0) if all conditions met, else None.
            Returning the ratio rather than a bool allows ranking by signal strength.
        """
        s = self._settings

        close = history["close"]
        volume = history["volume"]

        # 1. EMA trend filter — use functional API to avoid DataFrame accessor ambiguity
        ema = pta.ema(close, length=s.ema_period)
        if ema is None or ema.empty or pd.isna(ema.iloc[-1]):
            return None
        if close.iloc[-1] <= float(ema.iloc[-1]):
            return None

        # 2. RSI momentum filter
        rsi = pta.rsi(close, length=s.rsi_period)
        if rsi is None or rsi.empty or pd.isna(rsi.iloc[-1]):
            return None
        rsi_val = float(rsi.iloc[-1])
        if not (s.rsi_min <= rsi_val <= s.rsi_max):
            return None

        # 3. Volume participation filter
        avg_vol = volume.rolling(20).mean().iloc[-1]
        if pd.isna(avg_vol) or avg_vol <= 0:
            return None
        vol_ratio = float(volume.iloc[-1] / avg_vol)
        if vol_ratio < s.volume_multiplier:
            return None

        price = float(close.iloc[-1])

        # 4. Near-high proximity filter: price must be within near_high_max_drawdown of
        #    the rolling high. Applied only when enough bars are available so the harness
        #    matches the live system (which always fetches 80 bars before scanning).
        lookback = s.near_high_lookback
        if len(history) >= lookback:
            high_n = float(history["high"].iloc[-lookback:].max())
            if high_n > 0 and price < high_n * (1 - s.near_high_max_drawdown):
                return None

        # 5. Relative strength vs SPY (longs only — backtest is longs-only Phase 1)
        if s.require_relative_strength and spy_return_20d is not None and len(close) >= 21:
            ticker_20d = (price - float(close.iloc[-21])) / float(close.iloc[-21])
            if ticker_20d <= spy_return_20d:
                return None

        return vol_ratio

    # -------------------------------------------------------------------------
    # Exit logic
    # -------------------------------------------------------------------------

    def _check_time_exit(
        self,
        pos: "_OpenPosition",
        today: date,
        bar: pd.Series,
    ) -> Optional[tuple[float, str]]:
        """
        Check if a stale position should be closed at today's close.

        Mirrors live Job B behaviour: close positions held ≥ max_hold_days with
        < 1% unrealized gain. Called before _check_exits each bar.

        Returns (exit_price, "time_exit") or None to hold.
        """
        days_held = (today - pos.entry_date).days
        if days_held < self._settings.max_hold_days:
            return None
        bar_close = float(bar["close"])
        unrealized_pct = (bar_close - pos.entry_price) / pos.entry_price
        if unrealized_pct >= 0.01:
            return None  # up ≥1% — keep holding
        exit_price = bar_close * (1 - self._slippage_mult)
        return exit_price, "time_exit"

    def _check_exits(
        self,
        pos: "_OpenPosition",
        bar: pd.Series,
    ) -> Optional[tuple[float, str]]:
        """
        Check if this position should exit on `bar` via price-based triggers.

        Checks in order:
          1. Take-profit — did the high reach entry + take_profit_pct?
          2. Trailing stop activation — did the high reach +trail_activation_pct?
          3. Stop-loss (fixed or trailing) — did the low breach it?

        Returns (exit_price, reason) or None to hold.

        Take-profit is checked first: on a day where both TP and stop are hit,
        TP wins — consistent with bracket order semantics (limit fills before stop).
        """
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        s = self._settings

        # 1. Take-profit: exit if high reaches target price
        take_profit_price = pos.entry_price * (1 + s.take_profit_pct)
        if bar_high >= take_profit_price:
            exit_price = take_profit_price * (1 - self._slippage_mult)
            return exit_price, "take_profit"

        # 2. Check trailing stop activation BEFORE checking if stop was hit.
        #    High may have triggered trail; low may have then hit the trail.
        if not pos.trailing_activated:
            gain_pct = (bar_high - pos.entry_price) / pos.entry_price
            if gain_pct >= s.trail_activation_pct:
                # Trailing stop activates — set trail below high
                trail_price = bar_high * (1 - s.trail_pct)
                pos.trailing_activated = True
                pos.trail_stop_price = trail_price
                pos.stop_price = trail_price  # update effective stop

        effective_stop = pos.trail_stop_price if pos.trailing_activated else pos.stop_price

        # 3. Stop-loss (fixed or trailing)
        if bar_low <= effective_stop:
            # Exit at stop price with slippage (slippage works against us on exits too)
            exit_price = effective_stop * (1 - self._slippage_mult)
            reason = "trailing_stop" if pos.trailing_activated else "stop_loss"
            return exit_price, reason

        return None  # hold

    # -------------------------------------------------------------------------
    # Position helpers
    # -------------------------------------------------------------------------

    def _compute_qty(self, equity: float, entry_price: float, symbol: str) -> float:
        """Size position to max_position_pct of equity, floored to 2dp."""
        max_notional = equity * self._settings.max_position_pct
        qty = max_notional / entry_price
        return float(int(qty * 100) / 100)  # floor to 2dp

    def _close_position(
        self,
        pos: "_OpenPosition",
        exit_price: float,
        exit_date: date,
        reason: str,
    ) -> BacktestTrade:
        pnl = (exit_price - pos.entry_price) * pos.qty
        pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        log.debug(
            "BT EXIT %s @ %.2f (entry %.2f) pnl=%.2f (%.1f%%) reason=%s",
            pos.symbol, exit_price, pos.entry_price, pnl, pnl_pct * 100, reason,
        )
        return BacktestTrade(
            symbol=pos.symbol,
            entry_date=pos.entry_date,
            exit_date=exit_date,
            side="long",
            entry_price=pos.entry_price,
            exit_price=exit_price,
            qty=pos.qty,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
        )

    # -------------------------------------------------------------------------
    # Bar access helpers
    # -------------------------------------------------------------------------

    def _get_trading_dates(self) -> list[date]:
        """Union of all dates across all tickers, sorted ascending."""
        all_dates: set[date] = set()
        for df in self._bars.values():
            all_dates.update(df.index.date)
        return sorted(all_dates)

    def _get_bar(self, symbol: str, trading_date: date) -> Optional[pd.Series]:
        """Return the OHLCV bar for `symbol` on `trading_date`, or None."""
        df = self._bars.get(symbol)
        if df is None:
            return None
        mask = df.index.date == trading_date
        rows = df.loc[mask]
        if rows.empty:
            return None
        return rows.iloc[0]

    def _last_bar(self, symbol: str) -> Optional[pd.Series]:
        df = self._bars.get(symbol)
        if df is None or df.empty:
            return None
        return df.iloc[-1]


# ---------------------------------------------------------------------------
# Internal position state
# ---------------------------------------------------------------------------

class _OpenPosition:
    """Mutable position state during simulation."""

    __slots__ = (
        "symbol", "entry_date", "entry_price", "stop_price",
        "qty", "trailing_activated", "trail_stop_price",
    )

    def __init__(
        self,
        symbol: str,
        entry_date: date,
        entry_price: float,
        stop_price: float,
        qty: float,
        trailing_activated: bool,
        trail_stop_price: Optional[float],
    ) -> None:
        self.symbol = symbol
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.qty = qty
        self.trailing_activated = trailing_activated
        self.trail_stop_price = trail_stop_price


# ---------------------------------------------------------------------------
# Gate check utility (used by main.py)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Config fingerprint — ties a saved report to the exact strategy config it tested
# ---------------------------------------------------------------------------

# Fields that materially change strategy behavior. Changing any of these invalidates
# prior backtest reports: the gate must be re-run against the new config.
_FINGERPRINT_FIELDS = (
    "ema_period", "rsi_period", "rsi_min", "rsi_max",
    "volume_multiplier", "stop_loss_pct", "take_profit_pct",
    "trail_pct", "trail_activation_pct", "max_hold_days",
    "near_high_lookback", "near_high_max_drawdown",
    "require_relative_strength",
    "max_position_pct", "max_sector_pct", "max_positions",
    "vol_atr_threshold", "vol_realized_threshold",
    "backtest_slippage_bps",
)


def compute_config_fingerprint(settings: Settings) -> str:
    """Return a short hex digest of the strategy-relevant settings."""
    import hashlib
    payload = {f: getattr(settings, f) for f in _FINGERPRINT_FIELDS}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def check_backtest_gate(
    reports_dir: str,
    settings: Optional[Settings] = None,
) -> tuple[bool, Optional[str]]:
    """
    Verify that a valid, positive-expectancy backtest report exists in reports_dir.

    If `settings` is provided, the report's config fingerprint must match the
    current settings — prevents running with a config that was never backtested.

    Returns:
        (True, path)  if a passing, fingerprint-matching report exists.
        (False, None) if no qualifying reports found.
    """
    import glob

    pattern = os.path.join(reports_dir, "backtest_*.json")
    report_files = sorted(glob.glob(pattern), reverse=True)  # newest first

    if not report_files:
        return False, None

    want_fp = compute_config_fingerprint(settings) if settings is not None else None

    for path in report_files:
        try:
            with open(path) as f:
                report = json.load(f)
            if not report.get("metrics", {}).get("passed_gate", False):
                continue
            if want_fp is not None:
                got_fp = report.get("config_fingerprint")
                if got_fp != want_fp:
                    log.debug(
                        "Report %s fingerprint %s != current %s — skipping",
                        path, got_fp, want_fp,
                    )
                    continue
            return True, path
        except Exception as exc:
            log.warning("Could not read backtest report %s: %s", path, exc)

    return False, None
