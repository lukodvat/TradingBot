"""
Historical bar loader for the backtest harness.

Thin wrapper around MarketDataClient that accepts explicit date ranges
and enforces the minimum bar count required by the harness.

SPY is always loaded alongside watchlist tickers so the harness can compute
the relative-strength filter (ticker 20d return vs SPY 20d return).
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd

from config.settings import Settings
from data.market import MarketDataClient, load_sector_map, load_watchlist

log = logging.getLogger(__name__)

# Need ATR(14) + EMA(20) + realized_vol(20) + near-high(63) + warm-up buffer.
# 90 calendar days safely covers 65 trading days even with holidays.
MIN_CALENDAR_DAYS = 90

_ONE_YEAR_DAYS = 365


def load_bars_for_backtest(
    settings: Settings,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    symbols: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Load daily OHLCV bars for watchlist tickers (plus SPY) over [start_date, end_date].

    Args:
        settings:   Loaded Settings object (watchlist path, API credentials).
        start_date: First date of the backtest window (inclusive). Defaults to 1 year ago.
        end_date:   Last date of the backtest window (inclusive). Defaults to today.
        symbols:    Optional override ticker list. Defaults to config/watchlist.yaml.

    Returns:
        dict[symbol, DataFrame] — same schema as MarketDataClient.get_daily_bars().
        Always includes "SPY" if data is available (used for relative strength filter).
        Watchlist tickers with insufficient data are excluded and logged.

    Raises:
        ValueError: if end_date - start_date is less than MIN_CALENDAR_DAYS.
    """
    now_utc = datetime.now(timezone.utc)

    if end_date is None:
        end_date = now_utc
    if start_date is None:
        start_date = end_date - timedelta(days=_ONE_YEAR_DAYS)

    # Normalise to UTC-aware datetimes
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=timezone.utc)
    if end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    start = start_date.date()
    end = end_date.date()

    span = (end - start).days
    if span < MIN_CALENDAR_DAYS:
        raise ValueError(
            f"Backtest window is only {span} calendar days. "
            f"Need >= {MIN_CALENDAR_DAYS} to cover {settings.backtest_min_days} "
            f"trading days with indicator warm-up. Extend the date range."
        )

    tickers = symbols if symbols is not None else load_watchlist(settings.watchlist_path)
    # Always include SPY for the relative-strength filter; deduplicate in case it's
    # already in the symbols list.
    all_tickers = list(dict.fromkeys(tickers + ["SPY"]))

    client = MarketDataClient(settings)
    lookback_days = span + 5  # +5 for weekend buffer
    all_bars = client.get_daily_bars(all_tickers, lookback_days=lookback_days)

    result: dict[str, pd.DataFrame] = {}
    for symbol, df in all_bars.items():
        # Trim to [start, end]
        mask = (df.index.date >= start) & (df.index.date <= end)
        trimmed = df.loc[mask]

        if symbol == "SPY":
            # Include SPY regardless of bar count — it's a benchmark, not a traded ticker.
            result[symbol] = trimmed
            continue

        if len(trimmed) < settings.backtest_min_days:
            log.warning(
                "%s: only %d bars in backtest window (need %d) — excluded",
                symbol, len(trimmed), settings.backtest_min_days,
            )
            continue
        result[symbol] = trimmed

    log.info(
        "Loaded backtest bars: %d/%d tickers, window %s → %s (SPY=%s)",
        len(result), len(tickers), start, end,
        "yes" if "SPY" in result else "no",
    )
    return result
