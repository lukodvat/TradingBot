"""
Market regime filter — determines if macro conditions favour new position entries.

Evaluated once per Job B run, before the signal scan. Returns a RegimeState that
the run loop uses to gate entries and optionally cap max positions.

Two-tier check
--------------
1. SPY trend (EMA-based):
     SPY > EMA(50)  → BULL:    all entries allowed.
     SPY < EMA(50)  → CAUTION: longs suppressed; shorts still allowed.
     SPY < EMA(200) → BEAR:    all new entries halted.
     No SPY data    → UNKNOWN: pass-through (don't block, log a warning).

2. Volatility regime (SPY 20-day realized vol, annualised):
     vol > vol_regime_threshold → reduce max_positions to vol_regime_max_positions.

Both are configurable via Settings and can be toggled off by setting the
thresholds to extreme values (0 for vol, very large for EMA periods).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import pandas_ta as pta

from config.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class RegimeState:
    allow_long_entries: bool
    allow_short_entries: bool
    allow_any_entries: bool          # False only when SPY < EMA200
    max_positions_override: int      # Regime-adjusted cap (== settings.max_positions unless high vol)
    label: str                       # 'BULL' | 'CAUTION' | 'BEAR' | 'UNKNOWN'
    spy_price: Optional[float]
    spy_ema50: Optional[float]
    spy_ema200: Optional[float]
    spy_realized_vol: Optional[float]


class MarketRegimeFilter:
    """
    Stateless regime evaluator.

    Usage:
        spy_bars = market_data.get_daily_bars(["SPY"], lookback_days=80)["SPY"]
        regime = MarketRegimeFilter(settings).evaluate(spy_bars)
        if not regime.allow_any_entries:
            log.info("Regime BEAR — skipping signal scan")
            return
    """

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def evaluate(self, spy_bars: Optional[pd.DataFrame]) -> RegimeState:
        """
        Evaluate the current market regime from SPY daily bars.

        If spy_bars is None or has insufficient history, returns UNKNOWN
        (pass-through: entries are allowed, no position cap change).
        """
        s = self._s

        if spy_bars is None or spy_bars.empty:
            log.warning("Regime: no SPY bars — returning UNKNOWN (pass-through)")
            return _unknown(s.max_positions)

        close = spy_bars["close"]
        n = len(close)

        # Need enough bars for EMA(200)
        if n < s.spy_ema_long:
            log.warning(
                "Regime: only %d SPY bars, need %d for EMA(%d) — returning UNKNOWN",
                n, s.spy_ema_long, s.spy_ema_long,
            )
            return _unknown(s.max_positions)

        ema50_series = pta.ema(close, length=s.spy_ema_short)
        ema200_series = pta.ema(close, length=s.spy_ema_long)

        if ema50_series is None or ema200_series is None:
            log.warning("Regime: EMA computation returned None — UNKNOWN")
            return _unknown(s.max_positions)

        spy_price = float(close.iloc[-1])
        ema50 = float(ema50_series.iloc[-1])
        ema200 = float(ema200_series.iloc[-1])

        if pd.isna(ema50) or pd.isna(ema200):
            log.warning("Regime: NaN EMA values — UNKNOWN")
            return _unknown(s.max_positions)

        # 20-day realized vol (annualised, same method as volatility.py)
        realized_vol: Optional[float] = None
        if n >= 21:
            daily_returns = close.pct_change().iloc[-21:].dropna()
            if len(daily_returns) >= 15:
                realized_vol = float(daily_returns.std() * (252 ** 0.5))

        # Determine trend label and entry permissions
        if spy_price < ema200:
            label = "BEAR"
            allow_long = False
            allow_short = True
            allow_any = False   # full halt on longs AND new shorts in deep bear
        elif spy_price < ema50:
            label = "CAUTION"
            allow_long = False
            allow_short = True
            allow_any = True
        else:
            label = "BULL"
            allow_long = True
            allow_short = True
            allow_any = True

        # Vol regime override for position cap
        max_pos = s.max_positions
        if (
            realized_vol is not None
            and realized_vol > s.vol_regime_threshold
        ):
            max_pos = min(max_pos, s.vol_regime_max_positions)
            log.info(
                "Regime: high vol (%.1f%% > %.0f%%) — max positions capped at %d",
                realized_vol * 100, s.vol_regime_threshold * 100, max_pos,
            )

        log.info(
            "Regime: %s | SPY=%.2f EMA50=%.2f EMA200=%.2f vol=%s max_pos=%d",
            label, spy_price, ema50, ema200,
            f"{realized_vol*100:.1f}%" if realized_vol is not None else "N/A",
            max_pos,
        )

        return RegimeState(
            allow_long_entries=allow_long,
            allow_short_entries=allow_short,
            allow_any_entries=allow_any,
            max_positions_override=max_pos,
            label=label,
            spy_price=spy_price,
            spy_ema50=ema50,
            spy_ema200=ema200,
            spy_realized_vol=realized_vol,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unknown(max_positions: int) -> RegimeState:
    return RegimeState(
        allow_long_entries=True,
        allow_short_entries=True,
        allow_any_entries=True,
        max_positions_override=max_positions,
        label="UNKNOWN",
        spy_price=None,
        spy_ema50=None,
        spy_ema200=None,
        spy_realized_vol=None,
    )
