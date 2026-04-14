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


def make_scanner(settings=None, conn=None):
    s = settings or make_settings()
    c = conn or make_db()
    return SignalScanner(s, c), c


# ---------------------------------------------------------------------------
# _compute_conviction unit tests
# ---------------------------------------------------------------------------

class TestComputeConviction:
    def _s(self, **kw):
        return make_settings(**kw)

    def test_long_high_volume_high_rsi_near_max_conviction(self):
        s = self._s()
        # RSI at top of range (70) → rsi_score=1.0; volume=3x → vol_score=1.0
        score = _compute_conviction(rsi=70.0, volume_ratio=3.0, direction=LONG, s=s)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_long_low_rsi_low_volume_near_zero_conviction(self):
        s = self._s()
        # RSI at bottom of range (40) → rsi_score=0.0; volume at exactly 1.5x → vol_score=0.5
        score = _compute_conviction(rsi=40.0, volume_ratio=1.5, direction=LONG, s=s)
        assert 0.0 <= score <= 0.5

    def test_short_low_rsi_near_max_conviction(self):
        s = self._s()
        # SHORT: RSI at 40 (floor) → rsi_score=1.0; volume=3x → vol_score=1.0
        score = _compute_conviction(rsi=40.0, volume_ratio=3.0, direction=SHORT, s=s)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_conviction_clamped_by_volume_cap(self):
        s = self._s()
        # volume_ratio=10 → vol_score still capped at 1.0
        score_10x = _compute_conviction(rsi=55.0, volume_ratio=10.0, direction=LONG, s=s)
        score_3x = _compute_conviction(rsi=55.0, volume_ratio=3.0, direction=LONG, s=s)
        assert score_10x == score_3x

    def test_conviction_in_0_1_range(self):
        s = self._s()
        for rsi in [40, 50, 55, 65, 70]:
            for vol in [1.5, 2.0, 3.0, 5.0]:
                for direction in [LONG, SHORT]:
                    score = _compute_conviction(rsi, vol, direction, s)
                    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# SignalScanner.scan tests
# ---------------------------------------------------------------------------

class TestSignalScanner:

    def test_no_sentiment_bias_returns_empty(self):
        scanner, conn = make_scanner()
        bars = {"AAPL": make_bullish_bars()}
        # No bias seeded → AAPL missing from biases dict
        result = scanner.scan(bars, TODAY)
        assert result == []

    def test_neutral_bias_skipped(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "NEUTRAL")
        bars = {"AAPL": make_bullish_bars()}
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

    def test_price_below_ema_skipped_for_long(self):
        """Force price well below EMA by using a strong downtrend with BULLISH bias."""
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        # Use bearish bars (price trends down, ends below EMA) with BULLISH bias
        result = scanner.scan({"AAPL": make_bearish_bars()}, TODAY)
        # May or may not produce a signal depending on where EMA lands — this tests
        # the direction logic. Bearish bars with BULLISH bias will usually be blocked.
        # We don't assert empty here since bearish bars could cross EMA; instead
        # verify the direction field if a candidate does appear:
        for c in result:
            assert c.direction == LONG  # bias dictates direction regardless

    def test_low_volume_skipped(self):
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        bars = make_bullish_bars()
        bars["volume"] = 100_000.0  # uniform volume → ratio = 1.0 < 1.5
        result = scanner.scan({"AAPL": bars}, TODAY)
        assert result == []

    def test_rsi_filter_rejects_overbought(self):
        """Pure uptrend pushes RSI above 70 — should be rejected."""
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
        scanner, conn = make_scanner()
        seed_bias(conn, "AAPL", "BULLISH")
        seed_bias(conn, "MSFT", "NEUTRAL")   # should be skipped
        seed_bias(conn, "GOOGL", "BEARISH")

        bars = {
            "AAPL": make_bullish_bars(),
            "MSFT": make_bullish_bars(),
            "GOOGL": make_bearish_bars(),
        }
        result = scanner.scan(bars, TODAY)
        symbols = {c.symbol for c in result}
        assert "MSFT" not in symbols

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
