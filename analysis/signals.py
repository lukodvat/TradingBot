"""
Job B signal scanner — quant entry criteria.

Accepts a dict of {symbol: OHLCV DataFrame} (already volatility-filtered by the
caller) and a date string, reads the day's sentiment bias from SQLite, then applies
technical entry criteria. Returns a ranked list of SignalCandidate objects.

Entry criteria (ALL must hold for LONG; inverse for SHORT):
  - Sentiment bias is BULLISH (LONG) or BEARISH (SHORT) — NEUTRAL → skip
  - Price above EMA(20)  [LONG] / below EMA(20)  [SHORT]
  - RSI(14) in [rsi_min, rsi_max]               (same check for both directions)
  - Volume > volume_multiplier × 20-day avg volume

Conviction score (0..1): weighted average of volume strength and RSI momentum.
Candidates are returned sorted descending by conviction (highest first).

Same-ticker cooldown: caller passes a set of tickers already entered today;
these are skipped before any indicator computation.
"""

import logging
from dataclasses import dataclass
from typing import Optional
import sqlite3

import pandas as pd
import pandas_ta as pta

from config.settings import Settings
from db.store import get_all_sentiment_biases_for_date

log = logging.getLogger(__name__)

LONG = "LONG"
SHORT = "SHORT"

# Minimum bars needed to compute all indicators reliably
_MIN_BARS = 22  # EMA(20) + 2 warm-up buffer


@dataclass
class SignalCandidate:
    symbol: str
    direction: str          # LONG | SHORT
    conviction: float       # 0..1, higher = stronger setup
    sentiment_bias: str     # BULLISH | BEARISH
    ema: float              # EMA(20) value at last bar
    rsi: float              # RSI(14) value at last bar
    volume_ratio: float     # current vol / 20-day avg vol
    current_price: float    # close of last bar


class SignalScanner:
    """
    Stateless quant signal scanner for Job B.

    Usage:
        scanner = SignalScanner(settings, conn, spy_return_20d=spy_ret)
        candidates = scanner.scan(bars, date="2025-04-13", held_today={"AAPL"})
    """

    def __init__(
        self,
        settings: Settings,
        conn: sqlite3.Connection,
        spy_return_20d: Optional[float] = None,
    ) -> None:
        self._s = settings
        self._conn = conn
        self._spy_return_20d = spy_return_20d  # used for relative-strength filter

    def scan(
        self,
        bars: dict[str, pd.DataFrame],
        date: str,
        held_today: Optional[set[str]] = None,
        volume_multiplier_override: Optional[float] = None,
    ) -> list[SignalCandidate]:
        """
        Scan vol-filtered bars for entry setups.

        Args:
            bars:       {symbol: OHLCV DataFrame} — already volatility-filtered.
                        DataFrame must have columns: open, high, low, close, volume.
                        Index should be DatetimeIndex (UTC), sorted ascending.
            date:       'YYYY-MM-DD' string — used to look up today's sentiment bias.
            held_today: tickers already entered today (same-ticker cooldown). Skipped.

        Returns:
            List of SignalCandidate, sorted by conviction descending.
        """
        if held_today is None:
            held_today = set()

        biases = get_all_sentiment_biases_for_date(self._conn, date)
        log.debug("Sentiment biases for %s: %s", date, biases)

        candidates: list[SignalCandidate] = []

        for symbol, df in bars.items():
            if symbol in held_today:
                log.debug("SKIP %s — already held today", symbol)
                continue

            # Veto semantics: bias is a gate, not a signal. Direction comes from
            # price vs EMA (set inside _evaluate). Only a contradicting bias vetoes:
            # BEARISH blocks LONG, BULLISH blocks SHORT. NEUTRAL/missing is permitted.
            bias = biases.get(symbol) or "NEUTRAL"

            candidate = self._evaluate(symbol, df, bias, volume_multiplier_override)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda c: c.conviction, reverse=True)
        log.info(
            "Signal scan complete: %d candidates from %d tickers",
            len(candidates), len(bars),
        )
        return candidates

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _evaluate(
        self,
        symbol: str,
        df: pd.DataFrame,
        bias: str,
        volume_multiplier_override: Optional[float] = None,
    ) -> Optional[SignalCandidate]:
        """
        Apply technical criteria for one ticker. Returns None if criteria not met.
        """
        if len(df) < _MIN_BARS:
            log.debug("SKIP %s — insufficient bars (%d < %d)", symbol, len(df), _MIN_BARS)
            return None

        s = self._s
        close = df["close"]
        volume = df["volume"]

        # --- Indicators ---
        ema_series = pta.ema(close, length=s.ema_period)
        rsi_series = pta.rsi(close, length=s.rsi_period)

        if ema_series is None or rsi_series is None:
            log.debug("SKIP %s — indicator computation returned None", symbol)
            return None

        ema_val = float(ema_series.iloc[-1])
        rsi_val = float(rsi_series.iloc[-1])
        price = float(close.iloc[-1])

        if pd.isna(ema_val) or pd.isna(rsi_val):
            log.debug("SKIP %s — NaN indicator (ema=%.2f rsi=%.2f)", symbol, ema_val, rsi_val)
            return None

        # --- Volume ratio ---
        avg_vol = float(volume.iloc[-21:-1].mean())  # 20-bar average, excluding current
        curr_vol = float(volume.iloc[-1])
        volume_ratio = curr_vol / avg_vol if avg_vol > 0 else 0.0

        # --- Direction from quant trend (price vs EMA) ---
        if price > ema_val:
            direction = LONG
        elif price < ema_val:
            direction = SHORT
        else:
            log.debug("SKIP %s — price == EMA, no directional bias", symbol)
            return None

        # --- Sentiment veto (only a contradicting bias blocks) ---
        if direction == LONG and bias == "BEARISH":
            log.debug("SKIP %s — LONG vetoed by BEARISH bias", symbol)
            return None
        if direction == SHORT and bias == "BULLISH":
            log.debug("SKIP %s — SHORT vetoed by BULLISH bias", symbol)
            return None

        # --- Direction-aware RSI filter ---
        # LONG wants RSI in [rsi_min, rsi_max] (momentum into strength).
        # SHORT inverts: RSI in [100-rsi_max, 100-rsi_min] (momentum into weakness).
        if direction == LONG:
            rsi_lo, rsi_hi = s.rsi_min, s.rsi_max
        else:
            rsi_lo, rsi_hi = 100.0 - s.rsi_max, 100.0 - s.rsi_min
        if not (rsi_lo <= rsi_val <= rsi_hi):
            log.debug("SKIP %s — RSI %.1f outside [%.0f, %.0f] for %s", symbol, rsi_val, rsi_lo, rsi_hi, direction)
            return None

        # --- Volume filter (with optional session-aware override) ---
        vol_threshold = (
            volume_multiplier_override
            if volume_multiplier_override is not None
            else s.volume_multiplier
        )
        if volume_ratio < vol_threshold:
            log.debug("SKIP %s — volume ratio %.2f < %.1f", symbol, volume_ratio, vol_threshold)
            return None

        # --- Relative strength vs SPY (longs only) ---
        if (
            s.require_relative_strength
            and self._spy_return_20d is not None
            and len(close) >= 21
        ):
            ticker_20d = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21])
            if direction == LONG and ticker_20d <= self._spy_return_20d:
                log.debug(
                    "SKIP %s — relative strength %.2f%% <= SPY %.2f%%",
                    symbol, ticker_20d * 100, self._spy_return_20d * 100,
                )
                return None

        # --- Near-high proximity filter (longs only; only when enough bars) ---
        lookback = s.near_high_lookback
        if direction == LONG and len(df) >= lookback:
            high_n = float(df["high"].iloc[-lookback:].max())
            if high_n > 0 and price < high_n * (1 - s.near_high_max_drawdown):
                log.debug(
                    "SKIP %s — price %.2f is >%.0f%% below %d-day high %.2f",
                    symbol, price, s.near_high_max_drawdown * 100, lookback, high_n,
                )
                return None

        conviction = _compute_conviction(rsi_val, volume_ratio, direction, s)

        log.info(
            "SIGNAL %s %s | price=%.2f ema=%.2f rsi=%.1f vol_ratio=%.2f conv=%.2f",
            symbol, direction, price, ema_val, rsi_val, volume_ratio, conviction,
        )

        return SignalCandidate(
            symbol=symbol,
            direction=direction,
            conviction=conviction,
            sentiment_bias=bias,
            ema=round(ema_val, 4),
            rsi=round(rsi_val, 2),
            volume_ratio=round(volume_ratio, 3),
            current_price=round(price, 4),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_conviction(
    rsi: float,
    volume_ratio: float,
    direction: str,
    s: Settings,
) -> float:
    """
    Conviction score in [0, 1].

    Volume component (50%): saturates at 3× average volume.
    RSI momentum component (50%):
        LONG  — higher RSI within range = more momentum (distance from floor).
        SHORT — lower RSI within range = more bearish momentum (distance from ceiling).
    """
    rsi_range = s.rsi_max - s.rsi_min  # typically 30

    if direction == LONG:
        rsi_lo, rsi_hi = s.rsi_min, s.rsi_max
        rsi_score = (rsi - rsi_lo) / rsi_range
    else:
        # SHORT uses the inverted band [100-rsi_max, 100-rsi_min]; lower RSI = stronger.
        rsi_lo, rsi_hi = 100.0 - s.rsi_max, 100.0 - s.rsi_min
        rsi_score = (rsi_hi - rsi) / rsi_range

    rsi_score = max(0.0, min(1.0, rsi_score))
    vol_score = min(volume_ratio / 3.0, 1.0)

    return round(0.5 * rsi_score + 0.5 * vol_score, 4)
