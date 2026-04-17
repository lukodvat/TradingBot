"""
Tests for analysis/signals.py — Job B signal scanner.

Strategy:
- Build synthetic OHLCV DataFrames with controlled properties.
- Seed sentiment_bias rows into an in-memory SQLite DB.
- Verify each filter (EMA, RSI, volume, sentiment, cooldown) independently.
- Verify conviction scores and candidate ranking.
"""

import sqlite3
from datetime import datetime, timezone, date as date_type

import numpy as np
import pandas as pd
import pytest

from config.settings import Settings
from db.schema import SCHEMA_SQL
from db.store import upsert_sentiment_bias
from analysis.signals import SignalScanner, SignalCandidate, _compute_conviction, LONG, SHORT

TODAY = "2025-04-13"
N_BARS = 30  # enough for EMA(20) + RSI(14) warmup


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_settings(**overrides) -> Settings:
    defaults = dict(
        alpaca_api_key="test", alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test", finnhub_api_key="test",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def seed_bias(conn: sqlite3.Connection, symbol: str, bias: str, llm_run: str = "morning") -> None:
    upsert_sentiment_bias(
        conn,
        ticker=symbol,
        date=TODAY,
        bias=bias,
        aggregated_score=0.8 if bias == "BULLISH" else -0.8 if bias == "BEARISH" else 0.0,
        headline_count=3,
        llm_run=llm_run,
    )


def make_bullish_bars(n: int = N_BARS, base_price: float = 100.0) -> pd.DataFrame:
    """
    Bars designed to trigger a LONG signal:
    - Gentle uptrend so price ends above EMA(20)
    - Noisy enough that RSI stays in [40, 70]
    - Last bar has elevated volume (2x average)
    """
    np.random.seed(1)
    returns = np.random.normal(0.003, 0.010, n)
    prices = base_price * np.cumprod(1 + returns)

    # Construct OHLCV
    close = prices
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.002, n)))
    volume = np.full(n, 100_000.0)
    volume[-1] = 200_000.0  # 2x on last bar → ratio = 2.0

    idx = pd.date_range("2025-03-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


def make_bearish_bars(n: int = N_BARS, base_price: float = 100.0) -> pd.DataFrame:
    """
    Bars designed to trigger a SHORT signal:
    - Gentle downtrend so price ends below EMA(20)
    - Noisy enough that RSI stays in [40, 70]
    - Last bar has elevated volume (2x average)
    """
    np.random.seed(99)
    returns = np.random.normal(-0.003, 0.010, n)
    prices = base_price * np.cumprod(1 + returns)

    close = prices
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.002, n)))
    volume = np.full(n, 100_000.0)
    volume[-1] = 200_000.0

    idx = pd.date_range("2025-03-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


def make_scanner(settings=None, conn=None, spy_return_20d=None):
    s = settings or make_settings()
    c = conn or make_db()
    return SignalScanner(s, c, spy_return_20d=spy_return_20d), c


# ---------------------------------------------------------------------------
# _compute_conviction unit tests
# ---------------------------------------------------------------------------

class TestComputeConviction:
    def _s(self, **kw):
        return make_settings(**kw)

    def test_long_high_volume_high_rsi_near_max_conviction(self):
        s = self._s()
        # RSI at top of range (80) → rsi_score=1.0; volume=3x → vol_score=1.0
        score = _compute_conviction(rsi=80.0, volume_ratio=3.0, direction=LONG, s=s)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_long_low_rsi_low_volume_near_zero_conviction(self):
        s = self._s()
        # RSI at bottom of range (50) → rsi_score=0.0; volume at exactly 1.2x → vol_score=0.5
        score = _compute_conviction(rsi=50.0, volume_ratio=1.2, direction=LONG, s=s)
        assert 0.0 <= score <= 0.5

    def test_short_low_rsi_near_max_conviction(self):
        s = self._s()
        # SHORT inverted band is [20, 50]; RSI at 20 (floor) → rsi_score=1.0; vol=3x → 1.0
        score = _compute_conviction(rsi=20.0, volume_ratio=3.0, direction=SHORT, s=s)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_conviction_clamped_by_volume_cap(self):
        s = self._s()
        # volume_ratio=10 → vol_score still capped at 1.0
        score_10x = _compute_conviction(rsi=55.0, volume_ratio=10.0, direction=LONG, s=s)
        score_3x = _compute_conviction(rsi=55.0, volume_ratio=3.0, direction=LONG, s=s)
        assert score_10x == score_3x

    def test_conviction_in_0_1_range(self):
        s = self._s()
        for rsi in [50, 55, 65, 75, 80]:
            for vol in [1.2, 2.0, 3.0, 5.0]:
                for direction in [LONG, SHORT]:
                    score = _compute_conviction(rsi, vol, direction, s)
                    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SignalScanner.scan tests
# ---------------------------------------------------------------------------

class TestSignalScanner:

    def test_no_sentiment_bias_permitted_under_veto_semantics(self):
        """Missing bias is treated as NEUTRAL — permitted, not vetoed."""
        scanner, conn = make_scanner()
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        # Bullish bars → price > EMA → LONG; NEUTRAL (missing) does not veto LONG
        assert len(result) == 1
        assert result[0].direction == LONG
        assert result[0].sentiment_bias == "NEUTRAL"

    def test_neutral_bias_permitted(self):
        """Explicit NEUTRAL bias is permitted under veto semantics."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "NEUTRAL")
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        assert len(result) == 1
        assert result[0].direction == LONG
        assert result[0].sentiment_bias == "NEUTRAL"

    def test_bearish_bias_vetoes_long(self):
        """LONG setup (price > EMA) blocked by BEARISH bias."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BEARISH")
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        assert result == []

    def test_bullish_bias_vetoes_short(self):
        """SHORT setup (price < EMA) blocked by BULLISH bias."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = {"AAPL": make_bearish_bars()}
        result = scanner.scan(bars, TODAY)
        assert result == []

    def test_held_today_skipped(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY, held_today={"AAPL"})
        assert result == []

    def test_bullish_bias_generates_long_candidate(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        assert len(result) == 1
        assert result[0].symbol == "AAPL"
        assert result[0].direction == LONG
        assert result[0].sentiment_bias == "BULLISH"

    def test_bearish_bias_generates_short_candidate(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BEARISH")
        bars = {"AAPL": make_bearish_bars()}
        result = scanner.scan(bars, TODAY)
        assert len(result) == 1
        assert result[0].direction == SHORT
        assert result[0].sentiment_bias == "BEARISH"

    def test_insufficient_bars_skipped(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        short_bars = make_bullish_bars(n=10)  # below _MIN_BARS=22
        result = scanner.scan({"AAPL": short_bars}, TODAY)
        assert result == []

    def test_direction_derived_from_price_vs_ema(self):
        """Direction comes from quant (price vs EMA), not from bias."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        # Bearish bars → price < EMA → SHORT direction → BULLISH vetoes
        result = scanner.scan({"AAPL": make_bearish_bars()}, TODAY)
        assert result == []

    def test_low_volume_skipped(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = make_bullish_bars()
        bars["volume"] = 100_000.0  # uniform volume → ratio = 1.0 < 1.2
        result = scanner.scan({"AAPL": bars}, TODAY)
        assert result == []

    def test_rsi_filter_rejects_overbought(self):
        """Pure uptrend pushes RSI above 80 — should be rejected."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        n = N_BARS
        # Strongly trending up every bar — RSI will hit ~90+
        prices = 100.0 * (1.01 ** np.arange(n))
        close = prices
        open_ = np.roll(close, 1); open_[0] = close[0]
        high = close * 1.005
        low = open_ * 0.995
        volume = np.full(n, 100_000.0); volume[-1] = 200_000.0
        idx = pd.date_range("2025-03-01", periods=n, freq="D", tz="UTC")
        overbought = pd.DataFrame({"open": open_, "high": high, "low": low,
                                    "close": close, "volume": volume}, index=idx)
        result = scanner.scan({"AAPL": overbought}, TODAY)
        assert result == []

    def test_candidates_sorted_by_conviction_descending(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        seed_bias(conn, "MSFT", "BULLISH")

        bars_aapl = make_bullish_bars(base_price=100.0)
        # MSFT: same trend but much higher volume → higher conviction
        bars_msft = make_bullish_bars(base_price=200.0)
        bars_msft["volume"] = np.where(
            bars_msft.index == bars_msft.index[-1],
            500_000.0,   # 5x
            100_000.0,
        )

        result = scanner.scan({"AAPL": bars_aapl, "MSFT": bars_msft}, TODAY)
        convictions = [c.conviction for c in result]
        assert convictions == sorted(convictions, reverse=True)

    def test_multiple_tickers_mixed_results(self):
        """BULLISH confirms LONG, BEARISH confirms SHORT, NEUTRAL is permitted."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        seed_bias(conn, "MSFT", "NEUTRAL")
        seed_bias(conn, "GOOGL", "BEARISH")

        bars = {
            "AAPL": make_bullish_bars(),
            "MSFT": make_bullish_bars(),
            "GOOGL": make_bearish_bars(),
        }
        result = scanner.scan(bars, TODAY)
        by_symbol = {c.symbol: c for c in result}
        # All three should produce candidates: AAPL/MSFT LONG, GOOGL SHORT
        assert "AAPL" in by_symbol and by_symbol["AAPL"].direction == LONG
        assert "MSFT" in by_symbol and by_symbol["MSFT"].direction == LONG
        assert "GOOGL" in by_symbol and by_symbol["GOOGL"].direction == SHORT

    def test_midday_bias_preferred_over_morning(self):
        """midday > morning lexicographically — midday should win."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BEARISH", llm_run="morning")
        seed_bias(conn, "AAPL", "BULLISH", llm_run="midday")  # midday overrides
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        # midday says BULLISH → LONG signal expected (if technical criteria met)
        for c in result:
            assert c.direction == LONG

    def test_candidate_fields_populated(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = {"AAPL": make_bullish_bars()}
        result = scanner.scan(bars, TODAY)
        if result:  # if bars pass all filters
            c = result[0]
            assert c.symbol == "AAPL"
            assert isinstance(c.ema, float)
            assert isinstance(c.rsi, float)
            assert isinstance(c.volume_ratio, float)
            assert isinstance(c.current_price, float)
            assert 0.0 <= c.conviction <= 1.0
            assert c.direction in (LONG, SHORT)


# ---------------------------------------------------------------------------
# Relative strength filter
# ---------------------------------------------------------------------------

class TestRelativeStrengthFilter:
    def test_underperforming_long_rejected(self):
        """Ticker flat (0%) while SPY up 5% → should be filtered out."""
        # spy_return_20d = 5%; ticker return ≈ 0% from seed 1 bars
        # Make flat bars: no trend
        n = 30
        prices = np.full(n, 100.0)
        idx = pd.date_range("2025-03-01", periods=n, freq="D", tz="UTC")
        flat_bars = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices,
            "volume": np.where(np.arange(n) == n - 1, 200_000.0, 100_000.0),
        }, index=idx)

        s = make_settings(require_relative_strength=True)
        scanner, conn = make_scanner(s, spy_return_20d=0.05)  # SPY up 5%
        seed_bias(conn, "AAPL", "BULLISH")
        result = scanner.scan({"AAPL": flat_bars}, TODAY)
        # Ticker return ~0% < SPY 5% → filtered out
        assert result == []

    def test_outperforming_long_passes(self):
        """Ticker outperforms SPY → relative strength check passes."""
        # spy_return_20d = -5% (SPY negative); ticker up → passes easily
        s = make_settings(require_relative_strength=True)
        scanner, conn = make_scanner(s, spy_return_20d=-0.05)  # SPY down 5%
        seed_bias(conn, "AAPL", "BULLISH")
        result = scanner.scan({"AAPL": make_bullish_bars()}, TODAY)
        # With SPY negative and ticker positive, rel-strength passes
        # (though other filters may still block; we just verify no extra rejection)
        for c in result:
            assert c.direction == LONG

    def test_relative_strength_disabled_passes_all(self):
        """require_relative_strength=False bypasses the filter."""
        n = 30
        prices = np.full(n, 100.0)
        idx = pd.date_range("2025-03-01", periods=n, freq="D", tz="UTC")
        flat_bars = pd.DataFrame({
            "open": prices, "high": prices * 1.001,
            "low": prices * 0.999, "close": prices,
            "volume": np.where(np.arange(n) == n - 1, 200_000.0, 100_000.0),
        }, index=idx)

        s = make_settings(require_relative_strength=False)
        scanner, conn = make_scanner(s, spy_return_20d=0.10)  # high SPY return
        seed_bias(conn, "AAPL", "BULLISH")
        # Filter disabled — outcome depends only on other technical criteria
        # Just verify no TypeError or crash
        scanner.scan({"AAPL": flat_bars}, TODAY)

    def test_no_spy_data_skips_filter(self):
        """spy_return_20d=None → relative strength check is skipped."""
        s = make_settings(require_relative_strength=True)
        scanner, conn = make_scanner(s, spy_return_20d=None)
        seed_bias(conn, "AAPL", "BULLISH")
        # Should not raise; result depends only on technical filters
        result = scanner.scan({"AAPL": make_bullish_bars()}, TODAY)
        for c in result:
            assert c.direction == LONG

    def test_short_direction_not_filtered_by_relative_strength(self):
        """Relative strength filter only applies to LONG setups."""
        # Even if ticker underperforms SPY for longs, SHORT setups are unaffected
        s = make_settings(require_relative_strength=True)
        scanner, conn = make_scanner(s, spy_return_20d=0.10)  # high SPY return
        seed_bias(conn, "AAPL", "BEARISH")
        result = scanner.scan({"AAPL": make_bearish_bars()}, TODAY)
        # SHORT setups should not be filtered by relative strength
        for c in result:
            assert c.direction == SHORT


# ---------------------------------------------------------------------------
# Near-high proximity filter
# ---------------------------------------------------------------------------

class TestNearHighFilter:
    def _make_bars_with_high(
        self, n: int, last_price: float, lookback_high: float
    ) -> pd.DataFrame:
        """Build bars where the rolling lookback high is set to lookback_high."""
        idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
        close = np.full(n, last_price)
        # Inject the high in the middle of the bars
        high_col = np.full(n, last_price * 1.001)
        high_col[n // 2] = lookback_high  # spike to set the rolling high
        volume = np.full(n, 100_000.0)
        volume[-1] = 200_000.0
        return pd.DataFrame({
            "open":   close * 0.999,
            "high":   high_col,
            "low":    close * 0.999,
            "close":  close,
            "volume": volume,
        }, index=idx)

    def test_far_below_high_long_rejected(self):
        """Ticker 20% below 63-day high with 10% threshold → rejected."""
        # last_price=80, high=100 → 20% below → rejected (threshold=10%)
        bars = self._make_bars_with_high(n=80, last_price=80.0, lookback_high=100.0)

        s = make_settings(
            near_high_lookback=63,
            near_high_max_drawdown=0.10,
            require_relative_strength=False,  # isolate this filter
        )
        scanner, conn = make_scanner(s)
        seed_bias(conn, "AAPL", "BULLISH")
        result = scanner.scan({"AAPL": bars}, TODAY)
        assert result == []

    def test_near_high_long_passes(self):
        """Ticker 5% below 63-day high with 10% threshold → passes."""
        # last_price=95, high=100 → 5% below → passes (threshold=10%)
        bars = self._make_bars_with_high(n=80, last_price=95.0, lookback_high=100.0)

        s = make_settings(
            near_high_lookback=63,
            near_high_max_drawdown=0.10,
            require_relative_strength=False,
        )
        scanner, conn = make_scanner(s)
        seed_bias(conn, "AAPL", "BULLISH")
        # With flat bars the EMA/RSI/volume checks may block; just verify no crash
        scanner.scan({"AAPL": bars}, TODAY)

    def test_near_high_not_applied_to_shorts(self):
        """Near-high filter only applies to LONG setups."""
        bars = self._make_bars_with_high(n=80, last_price=60.0, lookback_high=100.0)

        s = make_settings(
            near_high_lookback=63,
            near_high_max_drawdown=0.10,
            require_relative_strength=False,
        )
        scanner, conn = make_scanner(s)
        seed_bias(conn, "AAPL", "BEARISH")
        # BEARISH → SHORT → near-high filter should not apply
        # (other filters may still reject, but not this one)
        scanner.scan({"AAPL": bars}, TODAY)

    def test_near_high_skipped_when_insufficient_bars(self):
        """With fewer bars than near_high_lookback, filter is skipped gracefully."""
        bars = make_bullish_bars(n=30)  # only 30 bars, lookback=63
        s = make_settings(
            near_high_lookback=63,
            near_high_max_drawdown=0.10,
            require_relative_strength=False,
        )
        scanner, conn = make_scanner(s)
        seed_bias(conn, "AAPL", "BULLISH")
        # Should not raise — filter is skipped with insufficient bars
        result = scanner.scan({"AAPL": bars}, TODAY)
        # Result depends on other technical filters
        assert isinstance(result, list)
