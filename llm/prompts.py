"""
Versioned prompt templates for Tier 2 (Haiku) and Tier 3 (Sonnet).

Keep prompts short — token cost is real. Every word costs money.
Templates use str.format() with named keys so callers are explicit.
"""

# ---------------------------------------------------------------------------
# Tier 2 — Claude Haiku: fast sentiment scoring
# ---------------------------------------------------------------------------

TIER2_SYSTEM = (
    "You are a financial news sentiment scorer. "
    "Respond ONLY with a valid JSON object — no markdown, no extra text."
)

TIER2_USER = (
    "Ticker: {symbol}\n"
    "Headline: {headline}\n\n"
    "Return JSON with:\n"
    '  "sentiment": float -1.0 (very bearish) to +1.0 (very bullish)\n'
    '  "confidence": float 0.0 (uncertain) to 1.0 (certain)\n'
    '  "reason": one sentence\n\n'
    "Consider only the likely short-term (intraday) price impact."
)

# ---------------------------------------------------------------------------
# Tier 3 — Claude Sonnet: deep setup assessment
# ---------------------------------------------------------------------------

TIER3_SYSTEM = (
    "You are a quantitative trader evaluating whether a news catalyst supports "
    "a momentum trade entry. Respond ONLY with a valid JSON object."
)

TIER3_USER = (
    "Ticker: {symbol}\n"
    "Headline: {headline}\n"
    "Haiku sentiment: {sentiment:.2f} (confidence: {confidence:.2f})\n\n"
    "Evaluate for an intraday momentum entry:\n"
    "1. Is the catalyst significant enough to drive sustained directional movement?\n"
    "2. Is the directional bias reliable, or is the outcome ambiguous?\n"
    "3. What is the main risk?\n\n"
    "Return JSON with:\n"
    '  "direction": "LONG" | "SHORT" | "SKIP"\n'
    '  "conviction": float 0.0-1.0\n'
    '  "catalyst_strength": "HIGH" | "MEDIUM" | "LOW"\n'
    '  "key_risk": string (one sentence)\n'
    '  "reasoning": string (2-3 sentences)'
)


def build_tier2_messages(symbol: str, headline: str) -> list[dict]:
    return [{"role": "user", "content": TIER2_USER.format(symbol=symbol, headline=headline)}]


def build_tier3_messages(
    symbol: str,
    headline: str,
    sentiment: float,
    confidence: float,
) -> list[dict]:
    return [
        {
            "role": "user",
            "content": TIER3_USER.format(
                symbol=symbol,
                headline=headline,
                sentiment=sentiment,
                confidence=confidence,
            ),
        }
    ]
