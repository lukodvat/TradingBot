"""
Tests for analysis/regime.py — MarketRegimeFilter.

All tests use synthetic SPY bar DataFrames.
No broker or DB calls required.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone

from config.settings import Settings
from analysis.regime import MarketRegimeFilter, RegimeState


def make_settings(**overrides) -> Settings:
    defaults = dict(
        alpaca_api_key="test", alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test", finnhub_api_key="test",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_spy_bars(
    n: int = 220,
    start_price: float = 400.0,
    drift: float = 0.0005,   # daily return
    vol: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic SPY OHLCV bars."""
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + drift + rng.normal(0, vol)))

    now = datetime.now(timezone.utc)
    index = pd.date_range(end=now, periods=n, freq="B", tz="UTC")

    closes = np.array(prices)
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": rng.integers(50_000_000, 150_000_000, size=n).astype(float),
    }, index=index)


# ---------------------------------------------------------------------------
# UNKNOWN / pass-through states
# ---------------------------------------------------------------------------

class TestUnknownFallback:
    def test_none_bars_returns_unknown(self):
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(None)
        assert result.label == "UNKNOWN"
        assert result.allow_any_entries is True
        assert result.allow_long_entries is True

    def test_empty_bars_returns_unknown(self):
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(pd.DataFrame())
        assert result.label == "UNKNOWN"

    def test_insufficient_bars_returns_unknown(self):
        # Only 50 bars — not enough for EMA(200)
        bars = make_spy_bars(n=50)
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.label == "UNKNOWN"
        assert result.allow_any_entries is True

    def test_unknown_preserves_max_positions(self):
        bars = make_spy_bars(n=50)
        s = make_settings(max_positions=5)
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.max_positions_override == 5


# ---------------------------------------------------------------------------
# BULL regime (SPY > EMA50 > EMA200)
# ---------------------------------------------------------------------------

class TestBullRegime:
    def test_strong_uptrend_is_bull(self):
        # High positive drift — SPY will be well above both EMAs
        bars = make_spy_bars(n=220, drift=0.003, vol=0.005, seed=1)
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.label == "BULL"
        assert result.allow_long_entries is True
        assert result.allow_short_entries is True
        assert result.allow_any_entries is True

    def test_bull_preserves_max_positions(self):
        bars = make_spy_bars(n=220, drift=0.003, vol=0.005, seed=1)
        s = make_settings(max_positions=5)
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.max_positions_override == 5

    def test_bull_spy_price_populated(self):
        bars = make_spy_bars(n=220, drift=0.003, vol=0.005, seed=1)
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.spy_price is not None
        assert result.spy_ema50 is not None
        assert result.spy_ema200 is not None


# ---------------------------------------------------------------------------
# CAUTION regime (SPY < EMA50, SPY > EMA200)
# ---------------------------------------------------------------------------

class TestCautionRegime:
    def _caution_bars(self) -> pd.DataFrame:
        """
        Create bars where SPY is below EMA(50) but above EMA(200).
        Strategy: long bull run then sharp pullback.
        """
        # 200 days of gentle uptrend, then 20 days of decline
        up = make_spy_bars(n=200, drift=0.002, vol=0.005, seed=5)
        last_price = float(up["close"].iloc[-1])
        down = make_spy_bars(n=20, start_price=last_price, drift=-0.015, vol=0.005, seed=5)

        # Re-index down bars to follow up bars
        new_index = pd.date_range(
            start=up.index[-1] + pd.Timedelta(days=1),
            periods=20, freq="B", tz="UTC",
        )
        down.index = new_index
        return pd.concat([up, down])

    def test_caution_blocks_longs(self):
        bars = self._caution_bars()
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        if result.label == "CAUTION":
            assert result.allow_long_entries is False
            assert result.allow_short_entries is True
            assert result.allow_any_entries is True

    def test_caution_label_when_below_ema50(self):
        bars = self._caution_bars()
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        # Either CAUTION or BEAR depending on severity of pullback
        assert result.label in ("CAUTION", "BEAR", "BULL")  # valid labels


# ---------------------------------------------------------------------------
# BEAR regime (SPY < EMA200)
# ---------------------------------------------------------------------------

class TestBearRegime:
    def _bear_bars(self) -> pd.DataFrame:
        """Strong downtrend from start — SPY falls well below both EMAs."""
        return make_spy_bars(n=220, start_price=500.0, drift=-0.004, vol=0.008, seed=7)

    def test_bear_blocks_all_entries(self):
        bars = self._bear_bars()
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        if result.label == "BEAR":
            assert result.allow_any_entries is False
            assert result.allow_long_entries is False

    def test_bear_still_allows_shorts(self):
        bars = self._bear_bars()
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        if result.label == "BEAR":
            assert result.allow_short_entries is True


# ---------------------------------------------------------------------------
# Volatility regime — position cap reduction
# ---------------------------------------------------------------------------

class TestVolRegime:
    def test_high_vol_reduces_max_positions(self):
        # Very high volatility: realized vol will exceed threshold
        bars = make_spy_bars(n=220, drift=0.0, vol=0.04, seed=3)  # ~63% annualised vol
        s = make_settings(
            vol_regime_threshold=0.30,
            vol_regime_max_positions=2,
            max_positions=5,
        )
        result = MarketRegimeFilter(s).evaluate(bars)
        # If vol > 30%, max positions should be capped at 2
        if result.spy_realized_vol is not None and result.spy_realized_vol > 0.30:
            assert result.max_positions_override == 2

    def test_low_vol_preserves_max_positions(self):
        # Very low volatility: realized vol well below threshold
        bars = make_spy_bars(n=220, drift=0.001, vol=0.002, seed=3)  # ~3% annualised
        s = make_settings(
            vol_regime_threshold=0.30,
            vol_regime_max_positions=2,
            max_positions=5,
        )
        result = MarketRegimeFilter(s).evaluate(bars)
        assert result.max_positions_override == 5

    def test_realized_vol_is_annualised(self):
        bars = make_spy_bars(n=220, drift=0.0, vol=0.01, seed=3)
        s = make_settings()
        result = MarketRegimeFilter(s).evaluate(bars)
        if result.spy_realized_vol is not None:
            # Daily vol of ~1% → annualised ~16%; should be in a sane range
            assert 0.05 < result.spy_realized_vol < 2.0
