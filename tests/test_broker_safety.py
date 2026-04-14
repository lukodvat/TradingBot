"""
Tests for paper-trading safety guards.

These run with zero real API calls — they verify that misconfigurations
fail loudly and immediately, never silently.
"""

import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Settings-level guard (Pydantic model_validator)
# ---------------------------------------------------------------------------

class TestSettingsLiveKeyGuard:
    """Settings must reject any URL that doesn't contain 'paper'."""

    def _make_settings(self, base_url: str):
        from config.settings import Settings
        return Settings(
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
            alpaca_base_url=base_url,
            anthropic_api_key="test_anthropic",
            finnhub_api_key="test_finnhub",
        )

    def test_paper_url_accepted(self):
        s = self._make_settings("https://paper-api.alpaca.markets")
        assert "paper" in s.alpaca_base_url

    def test_live_url_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            self._make_settings("https://api.alpaca.markets")
        assert "paper" in str(exc_info.value).lower()

    def test_live_url_with_paper_subdomain_variant(self):
        """URL must contain the string 'paper' — not just be Alpaca's domain."""
        with pytest.raises(ValidationError):
            self._make_settings("https://api.alpaca.markets/v2")

    def test_paper_url_case_insensitive(self):
        """Guard is case-insensitive."""
        s = self._make_settings("https://PAPER-api.alpaca.markets")
        assert s.alpaca_base_url is not None

    def test_missing_required_fields_raises(self):
        from config.settings import Settings
        with pytest.raises(ValidationError):
            Settings()  # type: ignore[call-arg]  # missing required fields


# ---------------------------------------------------------------------------
# BrokerClient-level guard (belt-and-suspenders)
# ---------------------------------------------------------------------------

class TestBrokerClientLiveKeyGuard:
    """BrokerClient has its own guard independent of Settings."""

    def _make_paper_settings(self):
        from config.settings import Settings
        return Settings(
            alpaca_api_key="test_key",
            alpaca_secret_key="test_secret",
            alpaca_base_url="https://paper-api.alpaca.markets",
            anthropic_api_key="test_anthropic",
            finnhub_api_key="test_finnhub",
        )

    def test_broker_client_initializes_with_paper_settings(self):
        """BrokerClient should initialize when settings are valid."""
        from core.broker import BrokerClient
        settings = self._make_paper_settings()
        # Patch TradingClient so no real network call is made
        with patch("core.broker.TradingClient") as mock_tc:
            client = BrokerClient(settings)
            mock_tc.assert_called_once_with(
                api_key="test_key",
                secret_key="test_secret",
                paper=True,
            )

    def test_broker_static_guard_rejects_live_url(self):
        """_guard_paper_url must raise LiveKeyError for live URL."""
        from core.broker import BrokerClient, LiveKeyError
        with pytest.raises(LiveKeyError) as exc_info:
            BrokerClient._guard_paper_url("https://api.alpaca.markets")
        assert "paper" in str(exc_info.value).lower()

    def test_broker_static_guard_accepts_paper_url(self):
        """_guard_paper_url must not raise for paper URL."""
        from core.broker import BrokerClient
        BrokerClient._guard_paper_url("https://paper-api.alpaca.markets")

    def test_broker_init_raises_if_url_mutated_after_settings(self):
        """
        If settings somehow passes validation but URL is wrong at BrokerClient
        init time, BrokerClient should still catch it.
        """
        from core.broker import BrokerClient, LiveKeyError
        settings = self._make_paper_settings()
        settings.__dict__["alpaca_base_url"] = "https://api.alpaca.markets"
        with pytest.raises(LiveKeyError):
            BrokerClient(settings)


# ---------------------------------------------------------------------------
# Watchlist loading
# ---------------------------------------------------------------------------

class TestWatchlistLoading:
    """Watchlist YAML should load and contain valid data."""

    def test_watchlist_loads(self):
        import yaml
        with open("config/watchlist.yaml") as f:
            data = yaml.safe_load(f)
        assert "tickers" in data
        assert len(data["tickers"]) > 0

    def test_all_tickers_have_required_fields(self):
        import yaml
        with open("config/watchlist.yaml") as f:
            data = yaml.safe_load(f)
        for ticker in data["tickers"]:
            assert "symbol" in ticker, f"Missing 'symbol' in {ticker}"
            assert "sector" in ticker, f"Missing 'sector' in {ticker}"
            assert "name" in ticker, f"Missing 'name' in {ticker}"
            assert isinstance(ticker["symbol"], str)
            assert len(ticker["symbol"]) > 0

    def test_sector_concentration_is_manageable(self):
        """No single sector should have more than 40% of tickers (rough sanity check)."""
        import yaml
        with open("config/watchlist.yaml") as f:
            data = yaml.safe_load(f)
        tickers = data["tickers"]
        total = len(tickers)
        sector_counts: dict[str, int] = {}
        for t in tickers:
            sector_counts[t["sector"]] = sector_counts.get(t["sector"], 0) + 1
        for sector, count in sector_counts.items():
            assert count / total <= 0.40, (
                f"Sector '{sector}' has {count}/{total} tickers ({count/total:.0%}) "
                f"— reduce concentration"
            )
