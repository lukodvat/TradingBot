"""
Tests for data/news.py, llm/client.py, llm/budget.py, and analysis/sentiment.py.

All external API calls (Finnhub, Anthropic) are mocked.
DB calls use in-memory SQLite.
"""

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from data.news import Headline, FinnhubProvider, fetch_all_headlines, _parse_finnhub_datetime
from llm.client import LLMClient, LLMResponse, _compute_cost, _parse_json, HAIKU_MODEL, SONNET_MODEL
from llm.budget import check_budget, assert_budget_ok, BudgetStatus
from analysis.sentiment import (
    SentimentAnalyzer,
    tier1_filter,
    BULLISH, BEARISH, NEUTRAL,
)


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
    """In-memory DB with full schema."""
    from db.schema import SCHEMA_SQL
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA_SQL)
    return conn


def make_headline(
    id="h1", symbol="AAPL",
    headline="Apple reports record quarterly earnings, beats estimates by 15%",
    hours_ago=1,
) -> Headline:
    return Headline(
        id=id, symbol=symbol, headline=headline, source="Reuters",
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago),
        url=None,
    )


RUN_TS = "2025-04-13T10:00:00+00:00"


# ---------------------------------------------------------------------------
# data/news.py — _parse_finnhub_datetime
# ---------------------------------------------------------------------------

class TestParseFinnhubDatetime:
    def test_valid_timestamp(self):
        dt = _parse_finnhub_datetime(1_700_000_000)
        assert dt is not None
        assert dt.tzinfo is not None

    def test_zero_returns_epoch(self):
        dt = _parse_finnhub_datetime(0)
        assert dt is not None

    def test_invalid_returns_none(self):
        assert _parse_finnhub_datetime("bad") is None  # type: ignore


# ---------------------------------------------------------------------------
# data/news.py — FinnhubProvider
# ---------------------------------------------------------------------------

class TestFinnhubProvider:
    def _make_provider(self):
        with patch("data.news.finnhub.Client"):
            provider = FinnhubProvider(make_settings())
        return provider

    def _fake_news(self, n=2, hours_ago=1):
        now = datetime.now(timezone.utc)
        return [
            {
                "id": i,
                "headline": f"News item {i}",
                "source": "reuters",
                "datetime": int((now - timedelta(hours=hours_ago)).timestamp()),
                "url": f"http://example.com/{i}",
            }
            for i in range(n)
        ]

    def test_returns_headlines_within_lookback(self):
        provider = self._make_provider()
        provider._client.company_news = MagicMock(return_value=self._fake_news(3, hours_ago=2))
        provider._rate_limit = MagicMock()
        results = provider.get_headlines("AAPL", lookback_hours=4)
        assert len(results) == 3

    def test_filters_headlines_older_than_lookback(self):
        provider = self._make_provider()
        provider._client.company_news = MagicMock(return_value=self._fake_news(1, hours_ago=10))
        provider._rate_limit = MagicMock()
        results = provider.get_headlines("AAPL", lookback_hours=4)
        assert len(results) == 0

    def test_returns_empty_on_api_error(self):
        provider = self._make_provider()
        provider._client.company_news = MagicMock(side_effect=Exception("API error"))
        provider._rate_limit = MagicMock()
        results = provider.get_headlines("AAPL", lookback_hours=4)
        assert results == []

    def test_returns_empty_on_empty_response(self):
        provider = self._make_provider()
        provider._client.company_news = MagicMock(return_value=[])
        provider._rate_limit = MagicMock()
        assert provider.get_headlines("AAPL", lookback_hours=4) == []

    def test_headline_fields_populated(self):
        provider = self._make_provider()
        provider._client.company_news = MagicMock(return_value=self._fake_news(1, hours_ago=1))
        provider._rate_limit = MagicMock()
        results = provider.get_headlines("AAPL", lookback_hours=4)
        h = results[0]
        assert h.symbol == "AAPL"
        assert h.id == "0"
        assert h.headline == "News item 0"
        assert h.published_at.tzinfo is not None


class TestFetchAllHeadlines:
    def test_aggregates_across_symbols(self):
        mock_provider = MagicMock()
        mock_provider.get_headlines.side_effect = lambda sym, **kw: [
            make_headline(id=f"{sym}_1", symbol=sym),
            make_headline(id=f"{sym}_2", symbol=sym),
        ]
        results = fetch_all_headlines(mock_provider, ["AAPL", "MSFT"], lookback_hours=4)
        assert len(results) == 4

    def test_empty_symbols_returns_empty(self):
        mock_provider = MagicMock()
        assert fetch_all_headlines(mock_provider, [], lookback_hours=4) == []


# ---------------------------------------------------------------------------
# llm/client.py — cost and JSON parsing
# ---------------------------------------------------------------------------

class TestLLMCostComputation:
    def test_haiku_cost_is_cheaper_than_sonnet(self):
        haiku_cost = _compute_cost(HAIKU_MODEL, 1000, 200)
        sonnet_cost = _compute_cost(SONNET_MODEL, 1000, 200)
        assert haiku_cost < sonnet_cost

    def test_zero_tokens_zero_cost(self):
        assert _compute_cost(HAIKU_MODEL, 0, 0) == 0.0

    def test_known_haiku_cost(self):
        # 1000 input @ $0.80/MTok + 100 output @ $4.00/MTok
        expected = 1000 * 0.80e-6 + 100 * 4.00e-6
        assert abs(_compute_cost(HAIKU_MODEL, 1000, 100) - expected) < 1e-10


class TestJsonParsing:
    def test_valid_json_parsed(self):
        parsed, err = _parse_json('{"sentiment": 0.7, "confidence": 0.9}')
        assert parsed == {"sentiment": 0.7, "confidence": 0.9}
        assert err is None

    def test_invalid_json_returns_none_with_error(self):
        parsed, err = _parse_json("not json")
        assert parsed is None
        assert err is not None

    def test_strips_markdown_fences(self):
        text = '```json\n{"sentiment": 0.5}\n```'
        parsed, err = _parse_json(text)
        assert parsed == {"sentiment": 0.5}
        assert err is None

    def test_empty_string_returns_error(self):
        parsed, err = _parse_json("")
        assert parsed is None


class TestLLMClient:
    def _make_client(self):
        with patch("llm.client.anthropic.Anthropic"):
            client = LLMClient(make_settings())
        return client

    def _mock_response(self, text='{"sentiment": 0.8, "confidence": 0.9, "reason": "test"}'):
        resp = MagicMock()
        resp.usage.input_tokens = 200
        resp.usage.output_tokens = 50
        resp.content = [MagicMock(text=text)]
        return resp

    def test_call_haiku_returns_llm_response(self):
        client = self._make_client()
        client._client.messages.create = MagicMock(return_value=self._mock_response())
        result = client.call_haiku("system", [{"role": "user", "content": "test"}])
        assert isinstance(result, LLMResponse)
        assert result.model == HAIKU_MODEL
        assert result.parsed is not None
        assert result.cost_usd > 0

    def test_call_sonnet_uses_sonnet_model(self):
        client = self._make_client()
        client._client.messages.create = MagicMock(return_value=self._mock_response('{"direction":"LONG","conviction":0.8,"catalyst_strength":"HIGH","key_risk":"x","reasoning":"y"}'))
        result = client.call_sonnet("system", [{"role": "user", "content": "test"}])
        assert result.model == SONNET_MODEL

    def test_api_error_propagates(self):
        client = self._make_client()
        client._client.messages.create = MagicMock(side_effect=Exception("rate limit"))
        with pytest.raises(Exception, match="rate limit"):
            client.call_haiku("system", [{"role": "user", "content": "test"}])


# ---------------------------------------------------------------------------
# llm/budget.py
# ---------------------------------------------------------------------------

class TestBudgetGuard:
    def test_under_limit_not_over(self):
        conn = make_db()
        s = make_settings(llm_budget_monthly_usd=18.0)
        status = check_budget(conn, s)
        assert not status.over_limit
        assert status.monthly_spend == 0.0

    def test_over_limit_when_spend_exceeds(self):
        from db.store import record_llm_call
        conn = make_db()
        s = make_settings(llm_budget_monthly_usd=18.0)
        record_llm_call(
            conn, model=HAIKU_MODEL, tier=2, symbol="X", headline_id="h1",
            prompt_tokens=1000, completion_tokens=500, cost_usd=20.0,
            sentiment=0.5, confidence=0.8, response_json=None, run_timestamp=RUN_TS,
        )
        status = check_budget(conn, s)
        assert status.over_limit

    def test_near_limit_flag(self):
        from db.store import record_llm_call
        conn = make_db()
        s = make_settings(llm_budget_monthly_usd=18.0)
        record_llm_call(
            conn, model=HAIKU_MODEL, tier=2, symbol="X", headline_id="h1",
            prompt_tokens=1000, completion_tokens=500, cost_usd=16.0,
            sentiment=0.5, confidence=0.8, response_json=None, run_timestamp=RUN_TS,
        )
        status = check_budget(conn, s)
        assert status.near_limit
        assert not status.over_limit

    def test_assert_budget_ok_raises_when_over(self):
        from db.store import record_llm_call
        conn = make_db()
        s = make_settings(llm_budget_monthly_usd=1.0)
        record_llm_call(
            conn, model=HAIKU_MODEL, tier=2, symbol="X", headline_id="h1",
            prompt_tokens=1000, completion_tokens=500, cost_usd=2.0,
            sentiment=None, confidence=None, response_json=None, run_timestamp=RUN_TS,
        )
        with pytest.raises(RuntimeError, match="budget exhausted"):
            assert_budget_ok(conn, s)

    def test_assert_budget_ok_passes_when_under(self):
        conn = make_db()
        s = make_settings(llm_budget_monthly_usd=18.0)
        assert_budget_ok(conn, s)  # should not raise


# ---------------------------------------------------------------------------
# analysis/sentiment.py — Tier 1 filter
# ---------------------------------------------------------------------------

class TestTier1Filter:
    def test_passes_earnings_beat_headline(self):
        h = make_headline(headline="Apple beats earnings estimates by wide margin")
        assert tier1_filter(h).passed

    def test_passes_fda_approval(self):
        h = make_headline(headline="Pfizer receives FDA approval for new drug")
        assert tier1_filter(h).passed

    def test_passes_analyst_upgrade(self):
        h = make_headline(headline="Goldman upgrades Apple to Buy with $220 target")
        assert tier1_filter(h).passed

    def test_rejects_10k_filing(self):
        h = make_headline(headline="Apple files 10-K annual report with SEC")
        result = tier1_filter(h)
        assert not result.passed
        assert result.reason == "sec_filing"

    def test_rejects_proxy_statement(self):
        h = make_headline(headline="Company files proxy statement ahead of AGM")
        result = tier1_filter(h)
        assert not result.passed
        assert result.reason == "sec_filing"

    def test_rejects_routine_dividend(self):
        h = make_headline(headline="Apple declares quarterly dividend of $0.25 per share")
        result = tier1_filter(h)
        assert not result.passed
        assert result.reason == "routine_dividend"

    def test_rejects_insider_sale(self):
        h = make_headline(headline="CEO insider sold 10,000 shares last week")
        result = tier1_filter(h)
        assert not result.passed
        assert result.reason == "insider_trade"

    def test_rejects_price_target_unchanged(self):
        h = make_headline(headline="Morgan Stanley reiterates Buy with price target unchanged")
        result = tier1_filter(h)
        assert not result.passed

    def test_case_insensitive(self):
        h = make_headline(headline="SEC FORM 4 filed by director")
        result = tier1_filter(h)
        assert not result.passed


# ---------------------------------------------------------------------------
# analysis/sentiment.py — SentimentAnalyzer (integration)
# ---------------------------------------------------------------------------

class TestSentimentAnalyzer:
    def _make_analyzer(self, settings=None):
        s = settings or make_settings()
        conn = make_db()
        llm = MagicMock(spec=LLMClient)
        analyzer = SentimentAnalyzer(s, llm, conn)
        return analyzer, llm, conn

    def _mock_haiku(self, llm, sentiment=0.7, confidence=0.85):
        resp = LLMResponse(
            model=HAIKU_MODEL, prompt_tokens=200, completion_tokens=50,
            cost_usd=0.001,
            raw_text=json.dumps({"sentiment": sentiment, "confidence": confidence, "reason": "test"}),
            parsed={"sentiment": sentiment, "confidence": confidence, "reason": "test"},
            parse_error=None,
        )
        llm.call_haiku.return_value = resp

    def _mock_sonnet(self, llm, direction="LONG", conviction=0.8):
        resp = LLMResponse(
            model=SONNET_MODEL, prompt_tokens=400, completion_tokens=150,
            cost_usd=0.005,
            raw_text=json.dumps({"direction": direction, "conviction": conviction,
                                  "catalyst_strength": "HIGH", "key_risk": "macro", "reasoning": "strong"}),
            parsed={"direction": direction, "conviction": conviction,
                    "catalyst_strength": "HIGH", "key_risk": "macro", "reasoning": "strong"},
            parse_error=None,
        )
        llm.call_sonnet.return_value = resp

    def test_tier1_rejected_headline_no_llm_call(self):
        analyzer, llm, conn = self._make_analyzer()
        headlines = [make_headline(headline="Apple files 10-K with SEC")]
        biases = analyzer.analyze(headlines, RUN_TS, "morning")
        llm.call_haiku.assert_not_called()
        assert biases == {}

    def test_tier2_called_for_tier1_pass(self):
        analyzer, llm, conn = self._make_analyzer()
        self._mock_haiku(llm, sentiment=0.5, confidence=0.7)
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        llm.call_haiku.assert_called_once()

    def test_tier3_not_called_below_threshold(self):
        s = make_settings(tier2_confidence_threshold=0.75, tier3_sentiment_threshold=0.6)
        analyzer, llm, conn = self._make_analyzer(s)
        # confidence=0.6 < 0.75 threshold → no T3
        self._mock_haiku(llm, sentiment=0.8, confidence=0.6)
        analyzer.analyze([make_headline()], RUN_TS, "morning")
        llm.call_sonnet.assert_not_called()

    def test_tier3_called_above_threshold(self):
        s = make_settings(tier2_confidence_threshold=0.75, tier3_sentiment_threshold=0.6)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=0.8, confidence=0.9)
        self._mock_sonnet(llm)
        analyzer.analyze([make_headline()], RUN_TS, "morning")
        llm.call_sonnet.assert_called_once()

    def test_tier3_capped_at_max_per_run(self):
        s = make_settings(tier3_max_per_run=2, tier2_confidence_threshold=0.5, tier3_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=0.9, confidence=0.9)
        self._mock_sonnet(llm)
        headlines = [make_headline(id=f"h{i}") for i in range(5)]
        analyzer.analyze(headlines, RUN_TS, "morning")
        assert llm.call_sonnet.call_count == 2

    def test_bullish_bias_on_positive_sentiment(self):
        s = make_settings(tier2_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=0.8, confidence=0.9)
        self._mock_sonnet(llm, direction="LONG")
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        assert biases["AAPL"].bias == BULLISH

    def test_bearish_bias_on_negative_sentiment(self):
        s = make_settings(tier2_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=-0.7, confidence=0.85)
        self._mock_sonnet(llm, direction="SHORT")
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        assert biases["AAPL"].bias == BEARISH

    def test_neutral_bias_on_weak_sentiment(self):
        s = make_settings(tier2_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=0.2, confidence=0.9)
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        assert biases["AAPL"].bias == NEUTRAL

    def test_tier3_skip_overrides_to_neutral(self):
        s = make_settings(tier2_confidence_threshold=0.5, tier3_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        self._mock_haiku(llm, sentiment=0.9, confidence=0.9)
        self._mock_sonnet(llm, direction="SKIP")
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        assert biases["AAPL"].bias == NEUTRAL

    def test_haiku_parse_failure_skips_headline(self):
        analyzer, llm, conn = self._make_analyzer()
        bad_resp = LLMResponse(
            model=HAIKU_MODEL, prompt_tokens=100, completion_tokens=20, cost_usd=0.0001,
            raw_text="not json", parsed=None, parse_error="parse error",
        )
        llm.call_haiku.return_value = bad_resp
        biases = analyzer.analyze([make_headline()], RUN_TS, "morning")
        assert biases == {}

    def test_multiple_headlines_same_ticker_aggregated(self):
        s = make_settings(tier2_sentiment_threshold=0.3)
        analyzer, llm, conn = self._make_analyzer(s)
        # Two calls: first positive, second weakly positive → aggregated BULLISH
        resp1 = LLMResponse(model=HAIKU_MODEL, prompt_tokens=100, completion_tokens=30,
                             cost_usd=0.001, raw_text="",
                             parsed={"sentiment": 0.8, "confidence": 0.9, "reason": "good"},
                             parse_error=None)
        resp2 = LLMResponse(model=HAIKU_MODEL, prompt_tokens=100, completion_tokens=30,
                             cost_usd=0.001, raw_text="",
                             parsed={"sentiment": 0.5, "confidence": 0.7, "reason": "ok"},
                             parse_error=None)
        llm.call_haiku.side_effect = [resp1, resp2]
        self._mock_sonnet(llm, direction="LONG")
        headlines = [make_headline(id="h1"), make_headline(id="h2")]
        biases = analyzer.analyze(headlines, RUN_TS, "morning")
        assert biases["AAPL"].headline_count == 2
        assert biases["AAPL"].bias == BULLISH
