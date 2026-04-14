"""
Tiered LLM triage orchestrator — Job A pipeline.

Three tiers, in order:

  Tier 1 — Regex/keyword rejection (free, instant).
            Drops obviously irrelevant headlines before any LLM call.

  Tier 2 — Claude Haiku sentiment scoring.
            Returns {sentiment: -1..+1, confidence: 0..1}.
            Runs on every Tier 1 pass.

  Tier 3 — Claude Sonnet deep assessment.
            Returns {direction, conviction, catalyst_strength, key_risk}.
            Only runs when |sentiment| > threshold AND confidence > threshold.
            Hard-capped at tier3_max_per_run calls per scheduled run.

Output: per-ticker SentimentBias — written to sentiment_bias SQLite table
by the caller (job runner). This module returns the aggregated results.
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from data.news import Headline
from db import store
from llm.client import LLMClient, HAIKU_MODEL, SONNET_MODEL
from llm.prompts import (
    TIER2_SYSTEM,
    TIER3_SYSTEM,
    build_tier2_messages,
    build_tier3_messages,
)

log = logging.getLogger(__name__)

# Bias labels used in sentiment_bias table and by the quant scanner
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"


@dataclass
class SentimentBias:
    symbol: str
    bias: str                       # BULLISH | NEUTRAL | BEARISH
    aggregated_score: float         # mean weighted sentiment across headlines
    headline_count: int             # headlines that reached Tier 2
    tier3_direction: Optional[str]  # LONG | SHORT | SKIP | None


@dataclass
class _Tier1Result:
    passed: bool
    reason: str   # rejection keyword/pattern, or "passed" if through


@dataclass
class _Tier2Result:
    sentiment: float
    confidence: float
    reason: str


@dataclass
class _Tier3Result:
    direction: str          # LONG | SHORT | SKIP
    conviction: float
    catalyst_strength: str
    key_risk: str
    reasoning: str


# ---------------------------------------------------------------------------
# Tier 1 keyword filter
# ---------------------------------------------------------------------------

# Rejection patterns — matched case-insensitively against headline text.
# Match = noise, reject immediately. Ambiguous? Let the LLM decide.
_REJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("sec_filing",        re.compile(r"\b(10-K|10-Q|8-K|proxy statement|annual report|quarterly report|form 4)\b", re.I)),
    ("routine_dividend",  re.compile(r"\b(declares? (quarterly|monthly|annual) dividend|ex-dividend date|dividend payment)\b", re.I)),
    ("insider_trade",     re.compile(r"\b(insider (bought?|sold?|purchases?|sales?)|director (buys?|sells?))\b", re.I)),
    ("index_rebalance",   re.compile(r"\b(index (rebalance|reconstitution)|added to (S&P|Nasdaq|Dow))\b", re.I)),
    ("price_target_minor",re.compile(r"\b(reiterates? (buy|hold|sell)|maintains? (buy|hold|sell)|price target (unchanged|maintained))\b", re.I)),
]


def tier1_filter(headline: Headline) -> _Tier1Result:
    """
    Reject noisy/routine headlines with regex before spending on LLM.
    Returns passed=True if the headline should proceed to Tier 2.
    """
    text = headline.headline
    for label, pattern in _REJECTION_PATTERNS:
        if pattern.search(text):
            log.debug("T1 REJECT [%s]: %s", label, text[:80])
            return _Tier1Result(passed=False, reason=label)
    return _Tier1Result(passed=True, reason="passed")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Runs tiered LLM triage on a batch of headlines and aggregates per-ticker bias.

    Usage (inside Job A):
        analyzer = SentimentAnalyzer(settings, llm_client, db_conn)
        biases = analyzer.analyze(headlines, run_timestamp="...", llm_run="morning")
        for symbol, bias in biases.items():
            store.upsert_sentiment_bias(conn, ticker=symbol, ...)
    """

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        conn: sqlite3.Connection,
    ) -> None:
        self._s = settings
        self._llm = llm_client
        self._conn = conn

    def analyze(
        self,
        headlines: list[Headline],
        run_timestamp: str,
        llm_run: str,           # 'morning' | 'midday'
    ) -> dict[str, SentimentBias]:
        """
        Process all headlines through the three tiers.

        Returns:
            dict mapping symbol -> SentimentBias.
            Symbols with no Tier 2 passes are absent (caller uses NEUTRAL fallback).
        """
        tier3_remaining = self._s.tier3_max_per_run

        # symbol -> list of (sentiment, confidence) for aggregation
        scores: dict[str, list[tuple[float, float]]] = {}
        tier3_directions: dict[str, str] = {}

        for headline in headlines:
            # --- Tier 1 ---
            t1 = tier1_filter(headline)
            store.update_headline_triage(
                self._conn,
                headline_id=headline.id,
                tier1_pass=1 if t1.passed else 0,
                tier1_reason=t1.reason,
            )
            if not t1.passed:
                continue

            # --- Tier 2 ---
            t2 = self._run_tier2(headline, run_timestamp)
            if t2 is None:
                continue

            store.update_headline_triage(
                self._conn,
                headline_id=headline.id,
                tier2_sentiment=t2.sentiment,
                tier2_confidence=t2.confidence,
            )

            scores.setdefault(headline.symbol, []).append((t2.sentiment, t2.confidence))

            # --- Tier 3 ---
            should_escalate = (
                abs(t2.sentiment) > self._s.tier3_sentiment_threshold
                and t2.confidence > self._s.tier2_confidence_threshold
                and tier3_remaining > 0
            )

            if should_escalate:
                t3 = self._run_tier3(headline, t2, run_timestamp)
                tier3_remaining -= 1
                if t3 is not None:
                    store.update_headline_triage(
                        self._conn,
                        headline_id=headline.id,
                        tier3_assessment=json.dumps({
                            "direction": t3.direction,
                            "conviction": t3.conviction,
                            "catalyst_strength": t3.catalyst_strength,
                            "key_risk": t3.key_risk,
                            "reasoning": t3.reasoning,
                        }),
                    )
                    # If Sonnet says SKIP, override to NEUTRAL regardless of T2
                    if t3.direction == "SKIP":
                        tier3_directions[headline.symbol] = "SKIP"
                    elif headline.symbol not in tier3_directions:
                        tier3_directions[headline.symbol] = t3.direction

        return self._aggregate(scores, tier3_directions)

    # -------------------------------------------------------------------------
    # Tier runners
    # -------------------------------------------------------------------------

    def _run_tier2(self, headline: Headline, run_timestamp: str) -> Optional[_Tier2Result]:
        try:
            messages = build_tier2_messages(headline.symbol, headline.headline)
            resp = self._llm.call_haiku(TIER2_SYSTEM, messages)
        except Exception as exc:
            log.error("Tier 2 API call failed for %s: %s", headline.symbol, exc)
            return None

        # Log to SQLite
        store.record_llm_call(
            self._conn,
            model=resp.model,
            tier=2,
            symbol=headline.symbol,
            headline_id=headline.id,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=resp.cost_usd,
            sentiment=resp.parsed.get("sentiment") if resp.parsed else None,
            confidence=resp.parsed.get("confidence") if resp.parsed else None,
            response_json=resp.raw_text,
            run_timestamp=run_timestamp,
        )

        if resp.parsed is None:
            log.warning("Tier 2 JSON parse failed for %s: %s", headline.symbol, resp.parse_error)
            return None

        sentiment = float(resp.parsed.get("sentiment", 0.0))
        confidence = float(resp.parsed.get("confidence", 0.5))
        reason = str(resp.parsed.get("reason", ""))

        # Clamp to valid ranges
        sentiment = max(-1.0, min(1.0, sentiment))
        confidence = max(0.0, min(1.0, confidence))

        log.info(
            "T2 %s | sent=%.2f conf=%.2f | %s",
            headline.symbol, sentiment, confidence, headline.headline[:60],
        )
        return _Tier2Result(sentiment=sentiment, confidence=confidence, reason=reason)

    def _run_tier3(
        self,
        headline: Headline,
        t2: _Tier2Result,
        run_timestamp: str,
    ) -> Optional[_Tier3Result]:
        try:
            messages = build_tier3_messages(
                headline.symbol, headline.headline, t2.sentiment, t2.confidence
            )
            resp = self._llm.call_sonnet(TIER3_SYSTEM, messages)
        except Exception as exc:
            log.error("Tier 3 API call failed for %s: %s", headline.symbol, exc)
            return None

        store.record_llm_call(
            self._conn,
            model=resp.model,
            tier=3,
            symbol=headline.symbol,
            headline_id=headline.id,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cost_usd=resp.cost_usd,
            sentiment=None,
            confidence=None,
            response_json=resp.raw_text,
            run_timestamp=run_timestamp,
        )

        if resp.parsed is None:
            log.warning("Tier 3 JSON parse failed for %s: %s", headline.symbol, resp.parse_error)
            return None

        p = resp.parsed
        direction = str(p.get("direction", "SKIP")).upper()
        if direction not in ("LONG", "SHORT", "SKIP"):
            direction = "SKIP"

        log.info(
            "T3 %s | direction=%s conviction=%.2f | %s",
            headline.symbol, direction,
            float(p.get("conviction", 0.0)),
            headline.headline[:60],
        )

        return _Tier3Result(
            direction=direction,
            conviction=float(p.get("conviction", 0.0)),
            catalyst_strength=str(p.get("catalyst_strength", "LOW")),
            key_risk=str(p.get("key_risk", "")),
            reasoning=str(p.get("reasoning", "")),
        )

    # -------------------------------------------------------------------------
    # Aggregation
    # -------------------------------------------------------------------------

    def _aggregate(
        self,
        scores: dict[str, list[tuple[float, float]]],
        tier3_directions: dict[str, str],
    ) -> dict[str, SentimentBias]:
        """
        Aggregate per-ticker T2 scores into a single SentimentBias.

        Weighting: each score is weighted by its confidence.
        A Tier 3 SKIP overrides to NEUTRAL regardless of T2 scores.
        """
        biases: dict[str, SentimentBias] = {}
        s = self._s

        for symbol, score_list in scores.items():
            # Confidence-weighted mean sentiment
            total_weight = sum(conf for _, conf in score_list)
            if total_weight == 0:
                agg_score = 0.0
            else:
                agg_score = sum(sent * conf for sent, conf in score_list) / total_weight

            # Tier 3 SKIP → force NEUTRAL
            if tier3_directions.get(symbol) == "SKIP":
                bias = NEUTRAL
            elif agg_score > s.tier2_sentiment_threshold:
                bias = BULLISH
            elif agg_score < -s.tier2_sentiment_threshold:
                bias = BEARISH
            else:
                bias = NEUTRAL

            biases[symbol] = SentimentBias(
                symbol=symbol,
                bias=bias,
                aggregated_score=round(agg_score, 4),
                headline_count=len(score_list),
                tier3_direction=tier3_directions.get(symbol),
            )
            log.info(
                "Bias %s: %s (score=%.3f, %d headlines)",
                symbol, bias, agg_score, len(score_list),
            )

        return biases
