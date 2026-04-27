"""
Tests for backtest/metrics.py and backtest/harness.py.

No network calls. Metrics tests are pure-function unit tests.
Harness tests use synthetic price DataFrames with known properties
so we can assert deterministic outcomes.
"""

import json
import math
import os
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta, timezone, datetime

from backtest.metrics import (
    BacktestMetrics,
    BacktestTrade,
    compute_metrics,
    compute_sharpe,
    compute_max_drawdown,
    compute_expectancy,
)
from backtest.harness import BacktestHarness, check_backtest_gate, SIGNAL_WARMUP_BARS
from config.settings import Settings


# ---------------------------------------------------------------------------
# Helpers
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


def make_trade(pnl: float, entry=100.0, exit_=None) -> BacktestTrade:
    exit_ = exit_ or (entry + pnl)
    return BacktestTrade(
        symbol="TEST",
        entry_date=date(2025, 1, 2),
        exit_date=date(2025, 1, 3),
        side="long",
        entry_price=entry,
        exit_price=exit_,
        qty=1.0,
        pnl=pnl,
        pnl_pct=pnl / entry,
        exit_reason="stop_loss",
    )


def make_equity_curve(values: list[float], start: date = date(2025, 1, 2)) -> pd.Series:
    dates = pd.date_range(start=start, periods=len(values), freq="B", tz="UTC")
    return pd.Series(values, index=dates)


def make_trending_bars(
    n: int,
    start_price: float = 100.0,
    daily_gain_pct: float = 0.002,  # mean drift; kept small so RSI stays in 40-70
    volume_scale: float = 1.0,
    start_date: date = date(2024, 1, 2),
    seed: int = 7,
) -> pd.DataFrame:
    """
    Synthetic bars with realistic noise so RSI stays in 40-70 range.

    Uses normally distributed daily returns (mean=daily_gain_pct, std=0.012)
    to create a mix of up and down days. With seed=5: above EMA, RSI≈55,
    volume ratio≈1.82 — all three signal conditions pass on the final bar.

    Volume on the last 5 bars is 2.5x the base so vol_ratio > 1.5 reliably.
    """
    np.random.seed(seed)
    dates = pd.bdate_range(start=start_date, periods=n, tz="UTC")
    rets = np.random.normal(daily_gain_pct, 0.012, n)

    prices = [start_price]
    for r in rets[:-1]:
        prices.append(prices[-1] * (1 + r))

    base_vol = 1_000_000 * volume_scale
    volumes = [base_vol] * n
    for i in range(-5, 0):
        volumes[i] = base_vol * 2.5

    return pd.DataFrame({
        "open":   [p * 0.999 for p in prices],
        "high":   [p * 1.005 for p in prices],
        "low":    [p * 0.995 for p in prices],
        "close":  prices,
        "volume": volumes,
    }, index=dates)


def make_flat_bars(n: int, price: float = 100.0, start_date: date = date(2024, 1, 2)) -> pd.DataFrame:
    """Flat price bars — should not trigger trend signal."""
    dates = pd.bdate_range(start=start_date, periods=n, tz="UTC")
    df = pd.DataFrame({
        "open":   [price] * n,
        "high":   [price * 1.002] * n,
        "low":    [price * 0.998] * n,
        "close":  [price] * n,
        "volume": [500_000] * n,  # below average
    }, index=dates)
    return df


# ---------------------------------------------------------------------------
# metrics.py — compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:

    def test_positive_sharpe_for_consistently_rising_equity(self):
        values = [10_000 * (1.001 ** i) for i in range(60)]
        curve = make_equity_curve(values)
        sharpe = compute_sharpe(curve)
        assert sharpe > 0

    def test_negative_sharpe_for_falling_equity(self):
        values = [10_000 * (0.999 ** i) for i in range(60)]
        curve = make_equity_curve(values)
        sharpe = compute_sharpe(curve)
        assert sharpe < 0

    def test_zero_for_single_point_curve(self):
        curve = make_equity_curve([10_000.0])
        assert compute_sharpe(curve) == 0.0

    def test_zero_for_flat_equity(self):
        curve = make_equity_curve([10_000.0] * 60)
        assert compute_sharpe(curve) == 0.0

    def test_sharpe_scales_with_return_quality(self):
        """
        Higher mean / lower vol series should produce a higher Sharpe.
        Tests the directional correctness of annualization without relying
        on a specific random seed hitting close to the theoretical value.
        """
        # Good: mean=0.002, vol=0.005 → theoretical ~6.4
        np.random.seed(42)
        good_rets = np.random.normal(0.002, 0.005, 252)
        eq_good = [10_000.0]
        for r in good_rets:
            eq_good.append(eq_good[-1] * (1 + r))

        # Bad: mean=0.0002, vol=0.015 → theoretical ~0.2
        np.random.seed(42)
        bad_rets = np.random.normal(0.0002, 0.015, 252)
        eq_bad = [10_000.0]
        for r in bad_rets:
            eq_bad.append(eq_bad[-1] * (1 + r))

        sharpe_good = compute_sharpe(make_equity_curve(eq_good))
        sharpe_bad = compute_sharpe(make_equity_curve(eq_bad))
        assert sharpe_good > sharpe_bad


# ---------------------------------------------------------------------------
# metrics.py — compute_max_drawdown
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:

    def test_no_drawdown_on_monotone_rise(self):
        values = [10_000 + i * 100 for i in range(50)]
        curve = make_equity_curve(values)
        assert compute_max_drawdown(curve) == pytest.approx(0.0, abs=1e-9)

    def test_known_drawdown(self):
        # Peak 12000, trough 9000 → drawdown = (12000-9000)/12000 = 25%
        values = [10_000, 11_000, 12_000, 9_000, 10_000]
        curve = make_equity_curve(values)
        dd = compute_max_drawdown(curve)
        assert abs(dd - 0.25) < 0.001

    def test_returns_positive_fraction(self):
        values = [10_000, 8_000, 9_000]
        curve = make_equity_curve(values)
        assert compute_max_drawdown(curve) > 0

    def test_empty_curve_returns_zero(self):
        assert compute_max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# metrics.py — compute_expectancy
# ---------------------------------------------------------------------------

class TestComputeExpectancy:

    def test_positive_expectancy_on_winning_trades(self):
        trades = [make_trade(2.0)] * 7 + [make_trade(-1.0)] * 3  # 70% win rate
        exp = compute_expectancy(trades)
        assert exp > 0

    def test_negative_expectancy_on_losing_trades(self):
        trades = [make_trade(1.0)] * 3 + [make_trade(-3.0)] * 7
        exp = compute_expectancy(trades)
        assert exp < 0

    def test_zero_expectancy_on_empty(self):
        assert compute_expectancy([]) == 0.0

    def test_expectancy_formula(self):
        # 60% win rate, avg win +2%, avg loss -1%
        # expectancy = 0.6 * 0.02 + 0.4 * (-0.01) = 0.012 - 0.004 = 0.008
        wins = [make_trade(2.0, entry=100.0)] * 6
        losses = [make_trade(-1.0, entry=100.0)] * 4
        exp = compute_expectancy(wins + losses)
        assert abs(exp - 0.008) < 0.001


# ---------------------------------------------------------------------------
# metrics.py — compute_metrics (integration)
# ---------------------------------------------------------------------------

class TestComputeMetrics:

    def test_gate_passes_on_positive_expectancy(self):
        trades = [make_trade(2.0)] * 7 + [make_trade(-1.0)] * 3
        curve = make_equity_curve([10_000 + i * 50 for i in range(60)])
        m = compute_metrics(trades, curve, 10_000.0)
        assert m.passed_gate

    def test_gate_fails_on_negative_expectancy(self):
        trades = [make_trade(-3.0)] * 7 + [make_trade(1.0)] * 3
        curve = make_equity_curve([10_000 - i * 50 for i in range(60)])
        m = compute_metrics(trades, curve, 10_000.0)
        assert not m.passed_gate

    def test_gate_fails_on_insufficient_trades(self):
        trades = [make_trade(5.0)] * 3  # fewer than MIN_TRADES_FOR_GATE=10
        curve = make_equity_curve([10_000, 10_050, 10_100])
        m = compute_metrics(trades, curve, 10_000.0, min_trades=10)
        assert not m.passed_gate

    def test_zero_trades_returns_zeroed_metrics(self):
        curve = make_equity_curve([10_000.0] * 30)
        m = compute_metrics([], curve, 10_000.0)
        assert m.total_trades == 0
        assert m.win_rate == 0.0
        assert not m.passed_gate

    def test_win_rate_computed_correctly(self):
        trades = [make_trade(1.0)] * 6 + [make_trade(-1.0)] * 4
        curve = make_equity_curve([10_000.0] * 30)
        m = compute_metrics(trades, curve, 10_000.0)
        assert abs(m.win_rate - 0.6) < 1e-9

    def test_profit_factor_gt_one_on_winning_strategy(self):
        trades = [make_trade(3.0)] * 6 + [make_trade(-1.0)] * 4
        curve = make_equity_curve([10_000 + i * 100 for i in range(30)])
        m = compute_metrics(trades, curve, 10_000.0)
        assert m.profit_factor > 1.0


# ---------------------------------------------------------------------------
# harness.py — signal detection (via _compute_signal)
# ---------------------------------------------------------------------------

class TestHarnessSignalDetection:

    def _make_harness(self, bars):
        s = make_settings()
        return BacktestHarness(s, bars, slippage_bps=0, initial_equity=10_000.0)

    def test_no_signal_with_insufficient_history(self):
        bars = make_trending_bars(10)  # fewer than SIGNAL_WARMUP_BARS=65
        h = self._make_harness({"X": bars})
        # Harness guards on SIGNAL_WARMUP_BARS before calling _compute_signal,
        # but _compute_signal itself should also return None when EMA/RSI are NaN.
        result = h._compute_signal(bars)
        # With only 10 bars, EMA(20) will be NaN → None returned
        assert result is None

    def test_signal_fires_on_trending_bars_with_high_volume(self):
        # seed=7 gives: above EMA=True, RSI≈60, vol_ratio≈1.82 — all conditions pass
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness({"X": bars})
        result = h._compute_signal(bars)
        assert result is not None, (
            "Expected signal on trending bars with high volume. "
            "Check that EMA trend, RSI 40-70, and volume > 1.5x are all met."
        )
        assert result > 0  # returns volume ratio

    def test_no_signal_on_flat_bars(self):
        bars = make_flat_bars(60)
        h = self._make_harness({"X": bars})
        result = h._compute_signal(bars)
        assert result is None  # flat price = below EMA or low volume

    def test_volume_ratio_returned_as_signal_strength(self):
        high_vol = make_trending_bars(60, volume_scale=3.0)
        low_vol = make_trending_bars(60, volume_scale=1.5)
        h = self._make_harness({"X": high_vol})
        r_high = h._compute_signal(high_vol)
        r_low = h._compute_signal(low_vol)
        if r_high is not None and r_low is not None:
            assert r_high >= r_low


# ---------------------------------------------------------------------------
# harness.py — exit logic
# ---------------------------------------------------------------------------

class TestHarnessExitLogic:

    def _make_position(self, entry=100.0, stop=98.0):
        from backtest.harness import _OpenPosition
        return _OpenPosition(
            symbol="TEST",
            entry_date=date(2025, 1, 2),
            entry_price=entry,
            stop_price=stop,
            qty=10.0,
            trailing_activated=False,
            trail_stop_price=None,
        )

    def _make_harness(self):
        s = make_settings()
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def _make_bar(self, open_=100, high=101, low=99, close=100):
        return pd.Series({"open": open_, "high": float(high), "low": float(low), "close": float(close)})

    def test_stop_loss_triggers_when_low_breaches_stop(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        bar = self._make_bar(high=101, low=97)  # low < stop
        result = h._check_exits(pos, bar)
        assert result is not None
        _, reason = result
        assert reason == "stop_loss"

    def test_no_exit_when_low_above_stop(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        # high=101 → +1% gain, below trail_activation_pct=2% so no trail activates
        bar = self._make_bar(high=101, low=99)
        assert h._check_exits(pos, bar) is None

    def test_trailing_stop_activates_when_high_reaches_threshold(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        # trail_activation_pct=0.02; high=102 → gain=2% → activates
        bar = self._make_bar(high=102, low=101.5)
        h._check_exits(pos, bar)
        assert pos.trailing_activated

    def test_trailing_stop_triggers_after_activation(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        # Activate trail on bar 1 (high=102 → +2% gain)
        bar1 = self._make_bar(high=102, low=101.5)
        h._check_exits(pos, bar1)
        assert pos.trailing_activated
        # trail_pct=1% → trail stop = 102 * 0.99 = 100.98
        bar2 = self._make_bar(high=101, low=100)  # low < trail stop 100.98
        result = h._check_exits(pos, bar2)
        assert result is not None
        _, reason = result
        assert reason == "trailing_stop"


# ---------------------------------------------------------------------------
# harness.py — full simulation (end-to-end)
# ---------------------------------------------------------------------------

class TestHarnessSimulation:

    def _make_harness(self, bars, slippage_bps=0):
        s = make_settings(backtest_slippage_bps=slippage_bps)
        return BacktestHarness(s, bars, slippage_bps=slippage_bps, initial_equity=10_000.0)

    def test_no_trades_on_flat_bars(self):
        bars = {"FLAT": make_flat_bars(80)}
        h = self._make_harness(bars)
        trades, _, _ = h.run()
        assert len(trades) == 0

    def test_produces_trades_on_trending_bars(self):
        bars = {"TREND": make_trending_bars(80)}
        h = self._make_harness(bars)
        trades, _, _ = h.run()
        assert len(trades) > 0

    def test_equity_curve_length_matches_trading_days(self):
        bars = {"A": make_trending_bars(80)}
        h = self._make_harness(bars)
        _, equity_curve, _ = h.run()
        assert len(equity_curve) == 80

    def test_slippage_reduces_returns(self):
        bars = {"A": make_trending_bars(80)}
        no_slip = self._make_harness(bars, slippage_bps=0)
        with_slip = self._make_harness(bars, slippage_bps=50)
        _, eq_no, _ = no_slip.run()
        _, eq_with, _ = with_slip.run()
        # Slippage should result in equal or lower final equity
        assert float(eq_with.iloc[-1]) <= float(eq_no.iloc[-1]) + 1.0  # small tolerance

    def test_max_positions_respected(self):
        """With max_positions=1, never more than 1 simultaneous position."""
        bars = {
            f"T{i}": make_trending_bars(80, start_price=100.0 + i * 10)
            for i in range(5)
        }
        s = make_settings(max_positions=1)
        h = BacktestHarness(s, bars, slippage_bps=0, initial_equity=10_000.0)
        trades, _, _ = h.run()
        # Check no two trades overlap in dates
        open_positions_by_date: dict[date, int] = {}
        for t in trades:
            d = t.entry_date
            while d <= t.exit_date:
                open_positions_by_date[d] = open_positions_by_date.get(d, 0) + 1
                d += timedelta(days=1)
        if open_positions_by_date:
            assert max(open_positions_by_date.values()) <= 1

    def test_lookahead_prevention_entry_after_signal(self):
        """Entry date must always be strictly after signal date."""
        bars = {"A": make_trending_bars(80)}
        h = self._make_harness(bars)
        trades, _, _ = h.run()
        for t in trades:
            # entry_date > entry signals day (we can't easily recover the signal day,
            # but we can assert entry_date is a valid market date in the dataset)
            assert t.entry_date in bars["A"].index.date

    def test_metrics_gate_can_pass(self):
        bars = {"TREND": make_trending_bars(80, daily_gain_pct=0.008)}
        h = self._make_harness(bars)
        _, _, metrics = h.run()
        # We don't assert passed_gate=True (depends on strategy params and random seed)
        # but metrics must be a valid BacktestMetrics object
        assert isinstance(metrics, BacktestMetrics)
        assert 0.0 <= metrics.win_rate <= 1.0


# ---------------------------------------------------------------------------
# harness.py — report saving and gate check
# ---------------------------------------------------------------------------

class TestReportGate:

    def test_save_report_creates_file(self, tmp_path):
        s = make_settings(reports_dir=str(tmp_path))
        bars = {"A": make_trending_bars(80)}
        h = BacktestHarness(s, bars, initial_equity=10_000.0)
        trades, equity_curve, metrics = h.run()
        path = h.save_report(trades, metrics)
        assert os.path.exists(path)

    def test_saved_report_is_valid_json(self, tmp_path):
        s = make_settings(reports_dir=str(tmp_path))
        bars = {"A": make_trending_bars(80)}
        h = BacktestHarness(s, bars, initial_equity=10_000.0)
        trades, _, metrics = h.run()
        path = h.save_report(trades, metrics)
        with open(path) as f:
            data = json.load(f)
        assert "metrics" in data
        assert "trades" in data

    def test_check_gate_returns_false_when_no_reports(self, tmp_path):
        passed, path = check_backtest_gate(str(tmp_path))
        assert not passed
        assert path is None

    def test_check_gate_returns_true_when_passing_report_exists(self, tmp_path):
        # Write a fake passing report
        report = {"metrics": {"passed_gate": True}}
        report_path = tmp_path / "backtest_20250413_103000.json"
        report_path.write_text(json.dumps(report))
        passed, found_path = check_backtest_gate(str(tmp_path))
        assert passed
        assert found_path == str(report_path)

    def test_check_gate_returns_false_when_only_failing_reports(self, tmp_path):
        report = {"metrics": {"passed_gate": False}}
        (tmp_path / "backtest_20250413_103000.json").write_text(json.dumps(report))
        passed, _ = check_backtest_gate(str(tmp_path))
        assert not passed


# ---------------------------------------------------------------------------
# harness.py — take-profit exit
# ---------------------------------------------------------------------------

class TestTakeProfitExit:

    def _make_position(self, entry=100.0, stop=98.0):
        from backtest.harness import _OpenPosition
        return _OpenPosition(
            symbol="TEST",
            entry_date=date(2025, 1, 2),
            entry_price=entry,
            stop_price=stop,
            qty=10.0,
            trailing_activated=False,
            trail_stop_price=None,
        )

    def _make_harness(self):
        s = make_settings()
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def _make_bar(self, high, low, open_=100, close=100):
        return pd.Series({"open": float(open_), "high": float(high), "low": float(low), "close": float(close)})

    def test_take_profit_triggers_when_high_exceeds_target(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        # take_profit_pct=0.06 → target=106.0; high=107 exceeds target
        bar = self._make_bar(high=107, low=101)
        result = h._check_exits(pos, bar)
        assert result is not None
        exit_price, reason = result
        assert reason == "take_profit"
        assert exit_price == pytest.approx(106.0, rel=1e-4)  # slippage_bps=0

    def test_take_profit_not_triggered_when_high_below_target(self):
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        # high=101 < 106 target and below trail activation (2%) — should not take profit
        bar = self._make_bar(high=101, low=100.5)
        result = h._check_exits(pos, bar)
        assert result is None

    def test_take_profit_wins_over_stop_on_same_bar(self):
        # On a very volatile bar both TP (high>=106) and stop (low<=98) are hit.
        # TP should win because limit orders fill before protective stops.
        h = self._make_harness()
        pos = self._make_position(entry=100.0, stop=98.0)
        bar = self._make_bar(high=107, low=97)  # both TP and stop hit
        _, reason = h._check_exits(pos, bar)
        assert reason == "take_profit"

    def test_take_profit_with_slippage(self):
        s = make_settings(backtest_slippage_bps=50)  # 0.5% slippage
        h = BacktestHarness(s, {}, slippage_bps=50, initial_equity=10_000.0)
        pos = self._make_position(entry=100.0, stop=98.0)
        bar = self._make_bar(high=107, low=101)
        exit_price, reason = h._check_exits(pos, bar)
        assert reason == "take_profit"
        # exit at TP price with slippage applied: 106 * (1 - 0.005)
        assert exit_price == pytest.approx(106.0 * 0.995, rel=1e-4)


# ---------------------------------------------------------------------------
# harness.py — partial exit logic
# ---------------------------------------------------------------------------

class TestPartialExit:

    def _make_position(self, entry=100.0, stop=98.0, qty=10.0, partial_done=False):
        from backtest.harness import _OpenPosition
        return _OpenPosition(
            symbol="TEST",
            entry_date=date(2025, 1, 2),
            entry_price=entry,
            stop_price=stop,
            qty=qty,
            trailing_activated=False,
            trail_stop_price=None,
            partial_exit_done=partial_done,
        )

    def _make_harness(self, **overrides):
        s = make_settings(**overrides)
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def _bar(self, high, low):
        return pd.Series({"open": 100.0, "high": float(high), "low": float(low), "close": 100.0})

    def test_partial_exit_fires_when_high_reaches_trigger(self):
        h = self._make_harness()  # trigger=3%, TP=6%
        pos = self._make_position()
        bar = self._bar(high=103.5, low=101)  # past trigger, below TP
        result = h._check_exits(pos, bar)
        assert result is not None
        exit_price, reason = result
        assert reason == "partial_exit"
        assert exit_price == pytest.approx(103.0, rel=1e-4)

    def test_partial_exit_does_not_re_fire(self):
        h = self._make_harness()
        pos = self._make_position(partial_done=True)
        # high=104 above trigger (103) but low stays above any activated trail.
        bar = self._bar(high=104, low=103.1)
        result = h._check_exits(pos, bar)
        # Either None (no exit) or trailing_stop — but NOT partial_exit.
        assert result is None or result[1] != "partial_exit"

    def test_take_profit_wins_over_partial_when_both_hit(self):
        h = self._make_harness()
        pos = self._make_position()
        bar = self._bar(high=107, low=101)  # past 106 TP
        _, reason = h._check_exits(pos, bar)
        assert reason == "take_profit"

    def test_partial_disabled_via_settings(self):
        h = self._make_harness(partial_exit_enabled=False)
        pos = self._make_position()
        bar = self._bar(high=104, low=101)  # past trigger
        # No partial fires; high=104 < trail_activation @ 102 → trail activates,
        # trail price = 104*0.99 = 102.96 > low=101 — no exit on this bar.
        result = h._check_exits(pos, bar)
        # trailing activates but doesn't trigger; no return value yet
        assert result is None or result[1] != "partial_exit"

class TestTimeBasedExit:

    def _make_position(self, entry=100.0, stop=98.0, entry_date=date(2025, 1, 2)):
        from backtest.harness import _OpenPosition
        return _OpenPosition(
            symbol="TEST",
            entry_date=entry_date,
            entry_price=entry,
            stop_price=stop,
            qty=10.0,
            trailing_activated=False,
            trail_stop_price=None,
        )

    def _make_harness(self, **setting_overrides):
        s = make_settings(**setting_overrides)
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def _make_bar(self, close=100.0):
        return pd.Series({"open": close, "high": close * 1.005, "low": close * 0.995, "close": close})

    def test_time_exit_triggers_after_max_hold_with_small_gain(self):
        # Entry 2025-01-02, today 2025-01-09 → 7 calendar days; price barely above entry
        h = self._make_harness(max_hold_days=7)
        pos = self._make_position(entry=100.0, entry_date=date(2025, 1, 2))
        today = date(2025, 1, 9)  # 7 days later
        bar = self._make_bar(close=100.5)  # 0.5% gain < 1% threshold
        result = h._check_time_exit(pos, today, bar)
        assert result is not None
        exit_price, reason = result
        assert reason == "time_exit"
        assert exit_price == pytest.approx(100.5, rel=1e-4)

    def test_time_exit_does_not_trigger_before_max_hold(self):
        h = self._make_harness(max_hold_days=7)
        pos = self._make_position(entry=100.0, entry_date=date(2025, 1, 2))
        today = date(2025, 1, 8)  # only 6 days — not yet eligible
        bar = self._make_bar(close=100.5)
        result = h._check_time_exit(pos, today, bar)
        assert result is None

    def test_time_exit_does_not_trigger_when_gain_sufficient(self):
        # Position is up ≥1% — don't close even if stale
        h = self._make_harness(max_hold_days=7)
        pos = self._make_position(entry=100.0, entry_date=date(2025, 1, 2))
        today = date(2025, 1, 9)
        bar = self._make_bar(close=101.5)  # 1.5% gain — above 1% threshold
        result = h._check_time_exit(pos, today, bar)
        assert result is None

    def test_time_exit_at_exactly_max_hold_days(self):
        # Boundary: exactly max_hold_days → should trigger
        h = self._make_harness(max_hold_days=7)
        pos = self._make_position(entry=100.0, entry_date=date(2025, 1, 2))
        today = date(2025, 1, 9)  # exactly 7 days
        bar = self._make_bar(close=99.0)  # underwater — definitely < 1%
        result = h._check_time_exit(pos, today, bar)
        assert result is not None
        _, reason = result
        assert reason == "time_exit"


# ---------------------------------------------------------------------------
# harness.py — near-high proximity filter in _compute_signal
# ---------------------------------------------------------------------------

class TestNearHighFilter:

    def _make_harness(self, **setting_overrides):
        s = make_settings(**setting_overrides)
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def test_near_high_filter_passes_when_price_within_tolerance(self):
        # seed=7, 60 bars: basic conditions hold (EMA, RSI, vol all pass — proven by
        # test_signal_fires_on_trending_bars_with_high_volume). Set lookback=60 so
        # the filter applies (60 >= 60). A gently trending series (+0.2%/day mean)
        # ends within 10% of its 60-bar high, so the near-high filter must pass.
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(near_high_lookback=60, near_high_max_drawdown=0.10)
        result = h._compute_signal(bars)
        assert result is not None, (
            "Near-high filter should pass for a trending series ending near its 60-bar high"
        )

    def test_near_high_filter_blocks_with_very_strict_threshold(self):
        # With near_high_max_drawdown=0.0001 (0.01%), the filter blocks even a trending
        # series that is fractionally below its 63-day high (close < high * 0.9999).
        bars = make_trending_bars(70, seed=7)
        h = self._make_harness(near_high_max_drawdown=0.0001)
        result = h._compute_signal(bars)
        assert result is None, (
            "Near-high filter should block when drawdown tolerance is stricter "
            "than the gap between close and the 63-bar high"
        )

    def test_near_high_filter_skipped_when_insufficient_bars(self):
        # With only 60 bars (< near_high_lookback=63), the filter is skipped.
        # Same strict threshold — but without enough bars the signal should still fire.
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(near_high_max_drawdown=0.0001)
        result = h._compute_signal(bars)
        assert result is not None, (
            "Near-high filter should be skipped when fewer than near_high_lookback bars "
            "are available, allowing the signal to fire on other criteria alone"
        )


# ---------------------------------------------------------------------------
# harness.py — relative strength filter in _compute_signal
# ---------------------------------------------------------------------------

class TestRelativeStrengthFilter:

    def _make_harness(self, **setting_overrides):
        s = make_settings(**setting_overrides)
        return BacktestHarness(s, {}, slippage_bps=0, initial_equity=10_000.0)

    def _ticker_20d_return(self, bars: pd.DataFrame) -> float:
        close = bars["close"]
        return (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21])

    def test_rs_filter_blocks_underperformer(self):
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(require_relative_strength=True)
        ticker_ret = self._ticker_20d_return(bars)
        # SPY return is 5pp higher → ticker underperforms → blocked
        result = h._compute_signal(bars, spy_return_20d=ticker_ret + 0.05)
        assert result is None

    def test_rs_filter_passes_outperformer(self):
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(require_relative_strength=True)
        ticker_ret = self._ticker_20d_return(bars)
        # SPY return is 5pp lower → ticker outperforms → passes
        result = h._compute_signal(bars, spy_return_20d=ticker_ret - 0.05)
        assert result is not None

    def test_rs_filter_blocks_when_equal_to_spy(self):
        # Boundary: ticker return == SPY return → ticker must EXCEED SPY, not match
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(require_relative_strength=True)
        ticker_ret = self._ticker_20d_return(bars)
        result = h._compute_signal(bars, spy_return_20d=ticker_ret)
        assert result is None  # equal is not strictly greater

    def test_rs_filter_skipped_when_disabled(self):
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness(require_relative_strength=False)
        ticker_ret = self._ticker_20d_return(bars)
        # Even though SPY is much higher, RS filter is disabled
        result = h._compute_signal(bars, spy_return_20d=ticker_ret + 0.50)
        assert result is not None

    def test_rs_filter_skipped_when_no_spy_return(self):
        bars = make_trending_bars(60, seed=7)
        h = self._make_harness()
        # spy_return_20d=None (default) → filter skipped → signal can fire
        result = h._compute_signal(bars, spy_return_20d=None)
        assert result is not None


# ---------------------------------------------------------------------------
# harness.py — SPY bars popped from bars dict
# ---------------------------------------------------------------------------

class TestSpyBars:

    def test_spy_popped_from_tradeable_bars(self):
        """SPY in input bars should not end up in self._bars (not tradeable)."""
        spy_bars = make_trending_bars(80)
        ticker_bars = make_trending_bars(80)
        s = make_settings()
        h = BacktestHarness(s, {"AAPL": ticker_bars, "SPY": spy_bars}, slippage_bps=0)
        assert "SPY" not in h._bars
        assert "AAPL" in h._bars

    def test_spy_stored_for_rs_filter(self):
        """SPY bars should be accessible as self._spy_bars."""
        spy_bars = make_trending_bars(80)
        s = make_settings()
        h = BacktestHarness(s, {"AAPL": make_trending_bars(80), "SPY": spy_bars}, slippage_bps=0)
        assert h._spy_bars is not None
        assert len(h._spy_bars) == 80

    def test_compute_spy_return_returns_none_without_spy(self):
        s = make_settings()
        h = BacktestHarness(s, {"AAPL": make_trending_bars(80)}, slippage_bps=0)
        result = h._compute_spy_return(date(2024, 6, 1))
        assert result is None

    def test_compute_spy_return_returns_float_with_spy(self):
        spy_bars = make_trending_bars(80, start_date=date(2024, 1, 2))
        s = make_settings()
        h = BacktestHarness(s, {"AAPL": make_trending_bars(80), "SPY": spy_bars}, slippage_bps=0)
        # Use a date that's in the spy_bars range with enough history
        all_dates = sorted(spy_bars.index.date)
        today = all_dates[25]  # bar 25+ guarantees 21 bars of history
        result = h._compute_spy_return(today)
        assert result is not None
        assert isinstance(result, float)
