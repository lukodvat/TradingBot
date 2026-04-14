"""
Historical bar loader for the backtest harness.

Thin wrapper around MarketDataClient that accepts explicit date ranges
and enforces the minimum bar count required by the harness.
"""

import logging
from datetime import date, datetime, timedelta, timezone

import pandas as pd

from config.settings import Settings
from data.market import MarketDataClient, load_sector_map, load_watchlist

log = logging.getLogger(__name__)

# Need ATR(14) + EMA(20) + realized_vol(20) + enough room to detect signals.
# 90 calendar days safely covers 60 trading days even with holidays.
MIN_CALENDAR_DAYS = 90


def load_bars_for_backtest(
    settings: Settings,
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """
    Load daily OHLCV bars for all watchlist tickers over [start, end].

    Args:
        settings: Loaded Settings object (watchlist path, API credentials).
        start:    First date of the backtest window (inclusive).
        end:      Last date of the backtest window (inclusive).

    Returns:
        dict[symbol, DataFrame] — same schema as MarketDataClient.get_daily_bars().
        Tickers with insufficient data are excluded and logged.

    Raises:
        ValueError: if end - start is less than MIN_CALENDAR_DAYS.
    """
    span = (end - start).days
    if span < MIN_CALENDAR_DAYS:
        raise ValueError(
            f"Backtest window is only {span} calendar days. "
            f"Need >= {MIN_CALENDAR_DAYS} to cover {settings.backtest_min_days} "
            f"trading days with indicator warm-up. Extend the date range."
        )

    tickers = load_watchlist(settings.watchlist_path)
    client = MarketDataClient(settings)

    # Convert date → datetime (UTC midnight) for the API
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_dt = datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=timezone.utc)

    lookback_days = (end_dt - start_dt).days + 5  # +5 for weekend buffer
    all_bars = client.get_daily_bars(tickers, lookback_days=lookback_days)

    # Filter to the requested window and drop tickers below minimum bar count
    result: dict[str, pd.DataFrame] = {}
    for symbol, df in all_bars.items():
        # Trim to [start, end]
        mask = (df.index.date >= start) & (df.index.date <= end)
        trimmed = df.loc[mask]
        if len(trimmed) < settings.backtest_min_days:
            log.warning(
                "%s: only %d bars in backtest window (need %d) — excluded",
                symbol, len(trimmed), settings.backtest_min_days,
            )
            continue
        result[symbol] = trimmed

    log.info(
        "Loaded backtest bars: %d/%d tickers, window %s → %s",
        len(result), len(tickers), start, end,
    )
    return result
