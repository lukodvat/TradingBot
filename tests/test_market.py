"""
Tests for data/market.py.

All Alpaca API calls are mocked — no network access required.
Tests focus on: DataFrame construction, error isolation, chunking,
watchlist loading, and quote parsing.
"""

import pytest
import pandas as pd
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from data.market import (
    MarketDataClient,
    _bars_to_dataframe,
    _chunks,
    load_watchlist,
    load_sector_map,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_settings():
    from config.settings import Settings
    return Settings(
        alpaca_api_key="test",
        alpaca_secret_key="test",
        alpaca_base_url="https://paper-api.alpaca.markets",
        anthropic_api_key="test",
        finnhub_api_key="test",
    )


def make_mock_bar(ts, o=100.0, h=105.0, l=99.0, c=102.0, v=1_000_000.0):
    bar = MagicMock()
    bar.timestamp = ts
    bar.open = o
    bar.high = h
    bar.low = l
    bar.close = c
    bar.volume = v
    return bar


def make_mock_bars_response(data: dict):
    """data: {symbol: [bar, bar, ...]}"""
    resp = MagicMock()
    resp.data = data
    return resp


# ---------------------------------------------------------------------------
# _bars_to_dataframe
# ---------------------------------------------------------------------------

class TestBarsToDataframe:

    def test_returns_dataframe_with_correct_columns(self):
        bars = [make_mock_bar(datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc))]
        df = _bars_to_dataframe(bars)
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_index_is_utc_datetimeindex(self):
        bars = [make_mock_bar(datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc))]
        df = _bars_to_dataframe(bars)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.tz is not None  # tz-aware

    def test_rows_sorted_ascending(self):
        bars = [
            make_mock_bar(datetime(2025, 4, 10, 15, 0, tzinfo=timezone.utc), c=103.0),
            make_mock_bar(datetime(2025, 4, 10, 13, 0, tzinfo=timezone.utc), c=101.0),
        ]
        df = _bars_to_dataframe(bars)
        assert df["close"].iloc[0] == 101.0
        assert df["close"].iloc[1] == 103.0

    def test_empty_input_returns_empty_dataframe(self):
        df = _bars_to_dataframe([])
        assert df.empty

    def test_values_are_floats(self):
        bars = [make_mock_bar(datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc))]
        df = _bars_to_dataframe(bars)
        for col in ["open", "high", "low", "close", "volume"]:
            assert df[col].dtype == float


# ---------------------------------------------------------------------------
# _chunks
# ---------------------------------------------------------------------------

class TestChunks:

    def test_even_split(self):
        result = list(_chunks([1, 2, 3, 4], 2))
        assert result == [[1, 2], [3, 4]]

    def test_uneven_split(self):
        result = list(_chunks([1, 2, 3], 2))
        assert result == [[1, 2], [3]]

    def test_single_chunk(self):
        result = list(_chunks([1, 2, 3], 10))
        assert result == [[1, 2, 3]]

    def test_empty_list(self):
        result = list(_chunks([], 5))
        assert result == []


# ---------------------------------------------------------------------------
# MarketDataClient.get_daily_bars
# ---------------------------------------------------------------------------

class TestGetDailyBars:

    def _make_client(self):
        with patch("data.market.StockHistoricalDataClient"):
            client = MarketDataClient(make_settings())
        return client

    def test_empty_tickers_returns_empty_dict(self):
        client = self._make_client()
        result = client.get_daily_bars([])
        assert result == {}

    def test_returns_dataframe_per_symbol(self):
        client = self._make_client()
        ts = datetime(2025, 4, 10, 20, 0, tzinfo=timezone.utc)
        mock_bars = [make_mock_bar(ts)]
        mock_response = make_mock_bars_response({"AAPL": mock_bars, "MSFT": mock_bars})

        client._client.get_stock_bars = MagicMock(return_value=mock_response)
        result = client.get_daily_bars(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" in result
        assert isinstance(result["AAPL"], pd.DataFrame)
        assert not result["AAPL"].empty

    def test_missing_ticker_absent_from_result(self):
        client = self._make_client()
        ts = datetime(2025, 4, 10, 20, 0, tzinfo=timezone.utc)
        mock_response = make_mock_bars_response({"AAPL": [make_mock_bar(ts)]})

        client._client.get_stock_bars = MagicMock(return_value=mock_response)
        result = client.get_daily_bars(["AAPL", "FAKE"])

        assert "AAPL" in result
        assert "FAKE" not in result

    def test_api_error_returns_empty_dict(self):
        client = self._make_client()
        client._client.get_stock_bars = MagicMock(side_effect=Exception("API error"))
        result = client.get_daily_bars(["AAPL"])
        assert result == {}

    def test_bad_bar_data_for_one_ticker_does_not_affect_others(self):
        """If one ticker's bar list is empty, others still return data."""
        client = self._make_client()
        ts = datetime(2025, 4, 10, 20, 0, tzinfo=timezone.utc)
        mock_response = make_mock_bars_response({
            "AAPL": [make_mock_bar(ts)],
            "MSFT": [],  # empty bar list
        })
        client._client.get_stock_bars = MagicMock(return_value=mock_response)
        result = client.get_daily_bars(["AAPL", "MSFT"])

        assert "AAPL" in result
        assert "MSFT" not in result


# ---------------------------------------------------------------------------
# MarketDataClient.get_hourly_bars
# ---------------------------------------------------------------------------

class TestGetHourlyBars:

    def _make_client(self):
        with patch("data.market.StockHistoricalDataClient"):
            client = MarketDataClient(make_settings())
        return client

    def test_returns_dataframe_per_symbol(self):
        client = self._make_client()
        ts = datetime(2025, 4, 10, 14, 0, tzinfo=timezone.utc)
        mock_response = make_mock_bars_response({"NVDA": [make_mock_bar(ts)]})
        client._client.get_stock_bars = MagicMock(return_value=mock_response)
        result = client.get_hourly_bars(["NVDA"])
        assert "NVDA" in result
        assert not result["NVDA"].empty

    def test_empty_tickers_returns_empty_dict(self):
        client = self._make_client()
        assert client.get_hourly_bars([]) == {}


# ---------------------------------------------------------------------------
# MarketDataClient.get_latest_quotes
# ---------------------------------------------------------------------------

class TestGetLatestQuotes:

    def _make_client(self):
        with patch("data.market.StockHistoricalDataClient"):
            client = MarketDataClient(make_settings())
        return client

    def _make_quote(self, bid: float, ask: float):
        q = MagicMock()
        q.bid_price = bid
        q.ask_price = ask
        return q

    def test_returns_bid_ask_mid(self):
        client = self._make_client()
        client._client.get_stock_latest_quote = MagicMock(
            return_value={"AAPL": self._make_quote(149.90, 150.10)}
        )
        result = client.get_latest_quotes(["AAPL"])
        assert "AAPL" in result
        assert abs(result["AAPL"]["bid"] - 149.90) < 0.001
        assert abs(result["AAPL"]["ask"] - 150.10) < 0.001
        assert abs(result["AAPL"]["mid"] - 150.00) < 0.001

    def test_empty_tickers_returns_empty_dict(self):
        client = self._make_client()
        assert client.get_latest_quotes([]) == {}

    def test_api_error_returns_empty_dict(self):
        client = self._make_client()
        client._client.get_stock_latest_quote = MagicMock(side_effect=Exception("timeout"))
        result = client.get_latest_quotes(["AAPL"])
        assert result == {}


# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

class TestWatchlistHelpers:

    def test_load_watchlist_returns_symbol_list(self):
        tickers = load_watchlist("config/watchlist.yaml")
        assert isinstance(tickers, list)
        assert len(tickers) > 0
        assert all(isinstance(t, str) for t in tickers)

    def test_load_watchlist_includes_expected_tickers(self):
        tickers = load_watchlist("config/watchlist.yaml")
        assert "AAPL" in tickers
        assert "JPM" in tickers

    def test_load_sector_map_returns_dict(self):
        sector_map = load_sector_map("config/watchlist.yaml")
        assert isinstance(sector_map, dict)
        assert len(sector_map) > 0

    def test_load_sector_map_correct_sectors(self):
        sector_map = load_sector_map("config/watchlist.yaml")
        assert sector_map["AAPL"] == "technology"
        assert sector_map["JPM"] == "finance"
        assert sector_map["XOM"] == "energy"

    def test_every_ticker_has_a_sector(self):
        tickers = load_watchlist("config/watchlist.yaml")
        sector_map = load_sector_map("config/watchlist.yaml")
        for ticker in tickers:
            assert ticker in sector_map, f"{ticker} missing from sector map"
