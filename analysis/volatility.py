"""
Volatility filter — the quant gate.

Runs on daily OHLCV bars already fetched by data/market.py.
Zero API calls. Zero LLM cost. Pure pandas + pandas-ta.

Two metrics are computed per ticker:

  ATR/price ratio  — ATR(14) divided by the latest close price.
                     Measures volatility relative to price level.
                     High ratio → wide intraday swings → stop-loss too likely to
                     trigger on noise before the trade has time to work.

  Realized vol     — 20-day annualized standard deviation of log returns.
                     Measures recent historical volatility.
                     High realized vol → unpredictable price path → position
                     sizing assumptions break down.

A ticker must pass BOTH filters to reach the signal scanner.
Results are logged to the volatility_filter_log SQLite table by the caller.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta  # noqa: F401 — registers the .ta accessor on DataFrames

from config.settings import Settings

log = logging.getLogger(__name__)

MIN_BARS_REQUIRED = 21  # need 14 for ATR + 20 for realized vol, 21 covers both


@dataclass
class VolatilityResult:
    symbol: str
    passed: bool
    atr_price_ratio: Optional[float]    # None if insufficient data
    realized_vol: Optional[float]       # None if insufficient data
    current_price: Optional[float]
    fail_reason: Optional[str]          # 'insufficient_data' | 'atr_too_high' | 'vol_too_high'


def filter_watchlist(
    bars: dict[str, pd.DataFrame],
    settings: Settings,
) -> dict[str, VolatilityResult]:
    """
    Apply the volatility filter to every ticker in the bars dict.

    Args:
        bars:     Output of MarketDataClient.get_daily_bars() —
                  {symbol: DataFrame with [open,high,low,close,volume]}.
        settings: Loaded Settings object (thresholds come from here).

    Returns:
        dict mapping symbol -> VolatilityResult.
        Tickers absent from bars are not present in the output.
    """
    results: dict[str, VolatilityResult] = {}

    for symbol, df in bars.items():
        result = _evaluate_ticker(symbol, df, settings)
        results[symbol] = result

        status = "PASS" if result.passed else f"FAIL ({result.fail_reason})"
        log.debug(
            "Vol filter %s %s | ATR/price=%.4f | realized_vol=%.4f",
            symbol,
            status,
            result.atr_price_ratio or 0,
            result.realized_vol or 0,
        )

    passed = sum(1 for r in results.values() if r.passed)
    log.info(
        "Volatility filter: %d/%d tickers passed", passed, len(results)
    )
    return results


def passing_tickers(results: dict[str, VolatilityResult]) -> list[str]:
    """Convenience — return only the symbols that passed the filter."""
    return [symbol for symbol, r in results.items() if r.passed]


# ---------------------------------------------------------------------------
# Per-ticker evaluation
# ---------------------------------------------------------------------------

def _evaluate_ticker(
    symbol: str,
    df: pd.DataFrame,
    settings: Settings,
) -> VolatilityResult:
    if len(df) < MIN_BARS_REQUIRED:
        log.warning("%s: only %d bars, need %d — skipping", symbol, len(df), MIN_BARS_REQUIRED)
        return VolatilityResult(
            symbol=symbol,
            passed=False,
            atr_price_ratio=None,
            realized_vol=None,
            current_price=None,
            fail_reason="insufficient_data",
        )

    current_price = float(df["close"].iloc[-1])

    atr_price_ratio = _compute_atr_ratio(df, settings.atr_period)
    realized_vol = _compute_realized_vol(df)

    if atr_price_ratio is None or realized_vol is None:
        return VolatilityResult(
            symbol=symbol,
            passed=False,
            atr_price_ratio=atr_price_ratio,
            realized_vol=realized_vol,
            current_price=current_price,
            fail_reason="insufficient_data",
        )

    # Apply thresholds — ATR check first (cheaper to compute and more common failure)
    if atr_price_ratio > settings.vol_atr_threshold:
        return VolatilityResult(
            symbol=symbol,
            passed=False,
            atr_price_ratio=atr_price_ratio,
            realized_vol=realized_vol,
            current_price=current_price,
            fail_reason="atr_too_high",
        )

    if realized_vol > settings.vol_realized_threshold:
        return VolatilityResult(
            symbol=symbol,
            passed=False,
            atr_price_ratio=atr_price_ratio,
            realized_vol=realized_vol,
            current_price=current_price,
            fail_reason="vol_too_high",
        )

    return VolatilityResult(
        symbol=symbol,
        passed=True,
        atr_price_ratio=atr_price_ratio,
        realized_vol=realized_vol,
        current_price=current_price,
        fail_reason=None,
    )


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_atr_pct(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Public wrapper exposing ATR(period)/close as a fraction.

    Used by sizing layers (core/risk.py + backtest harness) to derive a
    volatility-aware stop distance. Returns None when bars are insufficient.
    """
    if df is None or len(df) < period + 1:
        return None
    return _compute_atr_ratio(df, period)


def _compute_atr_ratio(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """
    Compute ATR(period) / latest close price.

    Uses pandas-ta's ATR implementation (Wilder smoothing by default).
    Returns None if the ATR series contains only NaN at the last position.
    """
    try:
        atr_series = df.ta.atr(length=period)
        if atr_series is None or atr_series.empty:
            return None

        last_atr = atr_series.iloc[-1]
        last_close = df["close"].iloc[-1]

        if pd.isna(last_atr) or last_close <= 0:
            return None

        return float(last_atr / last_close)
    except Exception as exc:
        log.warning("ATR computation failed: %s", exc)
        return None


def _compute_realized_vol(df: pd.DataFrame, window: int = 20) -> Optional[float]:
    """
    Compute annualized realized volatility from the last `window` daily log returns.

    Formula: std(log(close_t / close_{t-1}), window=window) * sqrt(252)

    Returns None if there are fewer than window+1 rows or the result is NaN.
    """
    try:
        close = df["close"]
        log_returns = np.log(close / close.shift(1))
        rolling_vol = log_returns.rolling(window=window).std()
        last_vol = rolling_vol.iloc[-1]

        if pd.isna(last_vol):
            return None

        return float(last_vol * np.sqrt(252))
    except Exception as exc:
        log.warning("Realized vol computation failed: %s", exc)
        return None
