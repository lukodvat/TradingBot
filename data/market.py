"""
Market data fetching via alpaca-py StockHistoricalDataClient.

Two bar types are needed by the system:

  Daily bars  — used by the volatility filter (ATR, realized vol).
                Fetches last N calendar days; returns enough history for
                ATR(14) and 20-day realized vol. Caller should request >= 21 days.

  Hourly bars — used by the quant signal scanner (EMA, RSI, volume ratio).
                Fetches last N trading hours. 80 hours (~10 trading days) gives
                comfortable history for EMA(20) and RSI(14) on the hourly chart.
                Hourly bars mean the 10:30 and 11:30 quant windows see different
                data as the day progresses — this is the point of running hourly.

Both methods fetch all requested tickers in a single API call (Alpaca supports
multi-symbol requests). The caller fetches once per job run and passes the result
to the volatility filter and signal scanner — no double-fetching.

Returns: dict[symbol, pd.DataFrame] with lowercase columns:
  open, high, low, close, volume
  index: DatetimeIndex (UTC, tz-aware)

Missing or errored tickers are logged and excluded from the return dict — callers
must handle absent keys gracefully.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from config.settings import Settings

log = logging.getLogger(__name__)

# Alpaca free tier: 200 bars per symbol per request (multi-symbol).
# For daily bars we request <= 30 days so we're well within limits.
# For hourly bars 80 hours = ~10 trading days, also fine.
_MAX_TICKERS_PER_REQUEST = 50  # stay conservative; watchlist is 20


class MarketDataClient:
    """
    Thin wrapper around alpaca-py StockHistoricalDataClient.

    One instance per process. StockHistoricalDataClient uses the same API
    credentials as TradingClient but hits the market data endpoint, which
    is the same for both paper and live accounts.
    """

    def __init__(self, settings: Settings) -> None:
        self._client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )
        log.info("MarketDataClient initialized")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_daily_bars(
        self,
        tickers: list[str],
        lookback_days: int = 21,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch daily OHLCV bars for all tickers in one API call.

        Args:
            tickers:      List of symbol strings.
            lookback_days: Calendar days to look back. Use >= 21 to cover
                           ATR(14) + 20-day realized vol with buffer.

        Returns:
            dict mapping symbol -> DataFrame with columns
            [open, high, low, close, volume] and a UTC DatetimeIndex.
            Tickers that return no data are absent from the dict.
        """
        if not tickers:
            return {}

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 5)  # +5 for weekends/holidays

        return self._fetch_bars(tickers, TimeFrame.Day, start, end, "daily")

    def get_hourly_bars(
        self,
        tickers: list[str],
        lookback_hours: int = 80,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch 1-hour OHLCV bars for all tickers in one API call.

        80 hours ≈ 10 trading days, enough for EMA(20) and RSI(14)
        on the hourly timeframe with comfortable buffer.

        Args:
            tickers:       List of symbol strings.
            lookback_hours: Trading hours to look back.

        Returns:
            dict mapping symbol -> DataFrame (same schema as get_daily_bars).
        """
        if not tickers:
            return {}

        end = datetime.now(timezone.utc)
        # Convert trading hours to calendar days (conservative: 6.5h/day × 2 for weekends)
        lookback_calendar_days = max(int(lookback_hours / 6.5 * 2), lookback_hours // 6 + 5)
        start = end - timedelta(days=lookback_calendar_days)

        return self._fetch_bars(tickers, TimeFrame.Hour, start, end, "hourly")

    def get_latest_quotes(
        self, tickers: list[str]
    ) -> dict[str, dict]:
        """
        Fetch the latest bid/ask quote for each ticker.

        Used to compute a realistic limit price at order submission time
        (midpoint of bid/ask, or ask for buys).

        Returns:
            dict mapping symbol -> {"bid": float, "ask": float, "mid": float}
            Absent symbols had no quote data.
        """
        if not tickers:
            return {}

        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            raw = self._client.get_stock_latest_quote(req)
        except Exception as exc:
            log.error("Failed to fetch latest quotes for %s: %s", tickers, exc)
            return {}

        result: dict[str, dict] = {}
        for symbol, quote in raw.items():
            try:
                bid = float(quote.bid_price or 0)
                ask = float(quote.ask_price or 0)
                mid = round((bid + ask) / 2, 2) if bid and ask else 0.0
                result[symbol] = {"bid": bid, "ask": ask, "mid": mid}
            except Exception as exc:
                log.warning("Could not parse quote for %s: %s", symbol, exc)

        return result

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _fetch_bars(
        self,
        tickers: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        label: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Core fetch — handles chunking, parsing, and per-ticker error isolation.
        """
        results: dict[str, pd.DataFrame] = {}

        # Chunk to stay within API limits (watchlist is 20, chunking is a safety net)
        for chunk in _chunks(tickers, _MAX_TICKERS_PER_REQUEST):
            chunk_results = self._fetch_chunk(chunk, timeframe, start, end, label)
            results.update(chunk_results)

        log.debug(
            "Fetched %s bars: %d/%d tickers returned data",
            label, len(results), len(tickers),
        )
        missing = set(tickers) - set(results)
        if missing:
            log.warning("No %s bar data for: %s", label, sorted(missing))

        return results

    def _fetch_chunk(
        self,
        tickers: list[str],
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
        label: str,
    ) -> dict[str, pd.DataFrame]:
        try:
            req = StockBarsRequest(
                symbol_or_symbols=tickers,
                timeframe=timeframe,
                start=start,
                end=end,
                adjustment="split",  # adjust for stock splits
            )
            raw = self._client.get_stock_bars(req)
        except Exception as exc:
            log.error("Alpaca bars request failed (%s, %s): %s", label, tickers, exc)
            return {}

        result: dict[str, pd.DataFrame] = {}
        bar_dict = raw.data if hasattr(raw, "data") else {}

        for symbol, bars in bar_dict.items():
            if not bars:
                continue
            try:
                df = _bars_to_dataframe(bars)
                if not df.empty:
                    result[symbol] = df
            except Exception as exc:
                log.warning("Failed to parse %s bars for %s: %s", label, symbol, exc)

        return result


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def _bars_to_dataframe(bars: list) -> pd.DataFrame:
    """
    Convert a list of alpaca Bar objects to a clean DataFrame.

    Output columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC, tz-aware), sorted ascending.
    """
    records = []
    for bar in bars:
        records.append({
            "timestamp": bar.timestamp,
            "open":      float(bar.open),
            "high":      float(bar.high),
            "low":       float(bar.low),
            "close":     float(bar.close),
            "volume":    float(bar.volume),
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _chunks(lst: list, size: int):
    """Yield successive chunks of `size` from `lst`."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def load_watchlist(watchlist_path: str) -> list[str]:
    """
    Load ticker symbols from the watchlist YAML.
    Returns a flat list of symbol strings.
    """
    import yaml

    with open(watchlist_path) as f:
        data = yaml.safe_load(f)

    tickers = [t["symbol"] for t in data.get("tickers", [])]
    log.info("Loaded %d tickers from %s", len(tickers), watchlist_path)
    return tickers


def load_sector_map(watchlist_path: str) -> dict[str, str]:
    """
    Load {symbol: sector} mapping from the watchlist YAML.
    Used by the risk manager for sector concentration checks.
    """
    import yaml

    with open(watchlist_path) as f:
        data = yaml.safe_load(f)

    return {t["symbol"]: t["sector"] for t in data.get("tickers", [])}
