"""
Tests for analysis/volatility.py.

All tests use synthetic price DataFrames — no API calls, no files.
Covers: pass/fail thresholds, edge cases (insufficient data, NaN, zero price),
and the convenience helpers.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from analysis.volatility import (
    VolatilityResult,
    _compute_atr_ratio,
    _compute_realized_vol,
    filter_watchlist,
    passing_tickers,
)
from config.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures and helpers
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


def make_bars(
    n: int = 30,
    close_start: float = 100.0,
    daily_move_pct: float = 0.005,   # 0.5% daily moves → low vol
    high_low_spread_pct: float = 0.01,
) -> pd.DataFrame:
    """
    Generate a synthetic daily OHLCV DataFrame with `n` bars.

    daily_move_pct:      drives realized volatility
    high_low_spread_pct: drives ATR (high-low range relative to close)
    """
    # Anchor end to the most recent business day so `freq=B` returns exactly n bars
    # (date_range with end on a weekend snaps and drops the final period).
    end_anchor = pd.Timestamp(datetime.now(timezone.utc).replace(
        hour=20, minute=0, second=0, microsecond=0
    ))
    if end_anchor.weekday() >= 5:  # Sat/Sun
        end_anchor -= pd.tseries.offsets.BDay(1)
    dates = pd.date_range(end=end_anchor, periods=n, freq="B", tz="UTC")
    np.random.seed(42)
    pct_changes = np.random.normal(0, daily_move_pct, n)
    closes = close_start * np.cumprod(1 + pct_changes)
    highs = closes * (1 + high_low_spread_pct)
    lows = closes * (1 - high_low_spread_pct)
    opens = closes * (1 + np.random.normal(0, daily_move_pct / 2, n))

    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1_000_000.0},
        index=dates,
    )


def make_volatile_bars(n: int = 30) -> pd.DataFrame:
    """High-volatility bars: daily moves of 4%, high-low spread of 6%."""
    return make_bars(n=n, daily_move_pct=0.04, high_low_spread_pct=0.06)


def make_calm_bars(n: int = 30) -> pd.DataFrame:
    """Low-volatility bars: daily moves of 0.3%, high-low spread of 0.5%."""
    return make_bars(n=n, daily_move_pct=0.003, high_low_spread_pct=0.005)


# ---------------------------------------------------------------------------
# _compute_atr_ratio
# ---------------------------------------------------------------------------

class TestComputeAtrRatio:

    def test_returns_float_for_valid_bars(self):
        df = make_calm_bars(30)
        ratio = _compute_atr_ratio(df, period=14)
        assert ratio is not None
        assert isinstance(ratio, float)
        assert ratio > 0

    def test_ratio_is_relative_to_price(self):
        """ATR/price should be dimensionless and roughly equal to high-low spread."""
        df = make_bars(30, close_start=100.0, high_low_spread_pct=0.01)
        ratio = _compute_atr_ratio(df, period=14)
        # ATR should be in the ballpark of the spread (not exact due to Wilder smoothing)
        assert 0.005 < ratio < 0.05

    def test_returns_none_for_insufficient_bars(self):
        df = make_calm_bars(5)  # fewer than ATR period
        ratio = _compute_atr_ratio(df, period=14)
        # Either None or NaN result — both acceptable; we check for None
        # (pandas-ta will return NaN for early rows; iloc[-1] might be NaN on tiny df)
        # With 5 bars and period=14, the last ATR value will be NaN
        assert ratio is None

    def test_volatile_bars_have_higher_ratio_than_calm(self):
        calm_ratio = _compute_atr_ratio(make_calm_bars(30), period=14)
        volatile_ratio = _compute_atr_ratio(make_volatile_bars(30), period=14)
        assert volatile_ratio > calm_ratio


# ---------------------------------------------------------------------------
# _compute_realized_vol
# ---------------------------------------------------------------------------

class TestComputeRealizedVol:

    def test_returns_float_for_valid_bars(self):
        df = make_calm_bars(30)
        vol = _compute_realized_vol(df, window=20)
        assert vol is not None
        assert isinstance(vol, float)
        assert vol > 0

    def test_annualized_vol_is_in_reasonable_range(self):
        """0.3% daily moves → ~4.8% annualized. Should be well below 50% threshold."""
        df = make_bars(30, daily_move_pct=0.003)
        vol = _compute_realized_vol(df, window=20)
        # 0.003 * sqrt(252) ≈ 0.048; allow generous band for random seed effects
        assert 0.01 < vol < 0.20

    def test_high_vol_bars_produce_higher_reading(self):
        calm_vol = _compute_realized_vol(make_calm_bars(30), window=20)
        volatile_vol = _compute_realized_vol(make_volatile_bars(30), window=20)
        assert volatile_vol > calm_vol

    def test_returns_none_for_insufficient_bars(self):
        df = make_calm_bars(10)  # fewer than window=20
        vol = _compute_realized_vol(df, window=20)
        assert vol is None

    def test_vol_is_annualized(self):
        """Daily std * sqrt(252) should roughly equal our input daily_move_pct * sqrt(252)."""
        df = make_bars(60, daily_move_pct=0.01)
        vol = _compute_realized_vol(df, window=20)
        expected_approx = 0.01 * np.sqrt(252)
        assert abs(vol - expected_approx) < 0.05  # within 5% annualized


# ---------------------------------------------------------------------------
# filter_watchlist
# ---------------------------------------------------------------------------

class TestFilterWatchlist:

    def test_calm_ticker_passes(self):
        settings = make_settings()  # default thresholds
        bars = {"AAPL": make_calm_bars(30)}
        results = filter_watchlist(bars, settings)
        assert results["AAPL"].passed

    def test_volatile_ticker_fails(self):
        settings = make_settings(
            vol_atr_threshold=0.01,     # very tight — volatile bars will exceed this
            vol_realized_threshold=0.50,
        )
        bars = {"TSLA": make_volatile_bars(30)}
        results = filter_watchlist(bars, settings)
        assert not results["TSLA"].passed

    def test_fail_reason_atr_too_high(self):
        settings = make_settings(vol_atr_threshold=0.001)  # impossibly tight
        bars = {"X": make_calm_bars(30)}
        results = filter_watchlist(bars, settings)
        assert results["X"].fail_reason == "atr_too_high"

    def test_fail_reason_vol_too_high(self):
        settings = make_settings(
            vol_atr_threshold=0.99,        # always pass ATR
            vol_realized_threshold=0.001,  # impossibly tight
        )
        bars = {"X": make_calm_bars(30)}
        results = filter_watchlist(bars, settings)
        assert results["X"].fail_reason == "vol_too_high"

    def test_insufficient_data_fails(self):
        settings = make_settings()
        bars = {"SHORT": make_calm_bars(5)}
        results = filter_watchlist(bars, settings)
        assert not results["SHORT"].passed
        assert results["SHORT"].fail_reason == "insufficient_data"

    def test_empty_bars_dict_returns_empty(self):
        settings = make_settings()
        results = filter_watchlist({}, settings)
        assert results == {}

    def test_mixed_watchlist(self):
        settings = make_settings(
            vol_atr_threshold=0.02,
            vol_realized_threshold=0.30,
        )
        bars = {
            "CALM": make_calm_bars(30),
            "WILD": make_volatile_bars(30),
        }
        results = filter_watchlist(bars, settings)
        assert results["CALM"].passed
        assert not results["WILD"].passed

    def test_result_contains_computed_metrics(self):
        settings = make_settings()
        bars = {"AAPL": make_calm_bars(30)}
        results = filter_watchlist(bars, settings)
        r = results["AAPL"]
        assert r.atr_price_ratio is not None
        assert r.realized_vol is not None
        assert r.current_price is not None
        assert r.current_price > 0

    def test_atr_threshold_from_settings_is_respected(self):
        """Tighten ATR threshold so calm bars fail; relax and they pass."""
        bars = {"X": make_calm_bars(30)}

        tight = make_settings(vol_atr_threshold=0.0001, vol_realized_threshold=0.99)
        assert not filter_watchlist(bars, tight)["X"].passed

        loose = make_settings(vol_atr_threshold=0.99, vol_realized_threshold=0.99)
        assert filter_watchlist(bars, loose)["X"].passed


# ---------------------------------------------------------------------------
# passing_tickers
# ---------------------------------------------------------------------------

class TestPassingTickers:

    def test_returns_only_passed_symbols(self):
        results = {
            "AAPL": VolatilityResult("AAPL", passed=True, atr_price_ratio=0.01,
                                      realized_vol=0.15, current_price=150.0, fail_reason=None),
            "TSLA": VolatilityResult("TSLA", passed=False, atr_price_ratio=0.08,
                                      realized_vol=0.80, current_price=200.0, fail_reason="atr_too_high"),
        }
        passed = passing_tickers(results)
        assert passed == ["AAPL"]

    def test_empty_results_returns_empty_list(self):
        assert passing_tickers({}) == []

    def test_all_pass(self):
        results = {
            s: VolatilityResult(s, passed=True, atr_price_ratio=0.01,
                                 realized_vol=0.10, current_price=100.0, fail_reason=None)
            for s in ["A", "B", "C"]
        }
        assert set(passing_tickers(results)) == {"A", "B", "C"}

    def test_none_pass(self):
        results = {
            s: VolatilityResult(s, passed=False, atr_price_ratio=0.10,
                                 realized_vol=0.90, current_price=100.0, fail_reason="atr_too_high")
            for s in ["A", "B"]
        }
        assert passing_tickers(results) == []
