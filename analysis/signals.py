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
        scanner = SignalScanner(settings, conn)
        candidates = scanner.scan(bars, date="2025-04-13", held_today={"AAPL"})
    """

    def __init__(self, settings: Settings, conn: sqlite3.Connection) -> None:
        self._s = settings
        self._conn = conn

    def scan(
        self,
        bars: dict[str, pd.DataFrame],
        date: str,
        held_today: Optional[set[str]] = None,
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

            bias = biases.get(symbol)
            if bias not in ("BULLISH", "BEARISH"):
                log.debug("SKIP %s — sentiment is %s", symbol, bias or "missing")
                continue

            candidate = self._evaluate(symbol, df, bias)
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

        # --- Trend filter ---
        if bias == "BULLISH" and price <= ema_val:
            log.debug("SKIP %s — LONG: price %.2f <= EMA %.2f", symbol, price, ema_val)
            return None
        if bias == "BEARISH" and price >= ema_val:
            log.debug("SKIP %s — SHORT: price %.2f >= EMA %.2f", symbol, price, ema_val)
            return None

        # --- RSI filter ---
        if not (s.rsi_min <= rsi_val <= s.rsi_max):
            log.debug("SKIP %s — RSI %.1f outside [%.0f, %.0f]", symbol, rsi_val, s.rsi_min, s.rsi_max)
            return None

        # --- Volume filter ---
        if volume_ratio < s.volume_multiplier:
            log.debug("SKIP %s — volume ratio %.2f < %.1f", symbol, volume_ratio, s.volume_multiplier)
            return None

        direction = LONG if bias == "BULLISH" else SHORT
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
        rsi_score = (rsi - s.rsi_min) / rsi_range
    else:
        rsi_score = (s.rsi_max - rsi) / rsi_range

    vol_score = min(volume_ratio / 3.0, 1.0)

    return round(0.5 * rsi_score + 0.5 * vol_score, 4)
