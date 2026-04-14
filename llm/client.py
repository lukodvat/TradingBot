"""
Anthropic SDK wrapper with per-call cost tracking.

Responsibilities:
- Make API calls to Claude Haiku (Tier 2) and Claude Sonnet (Tier 3).
- Calculate USD cost from token usage.
- Return a typed LLMResponse — does NOT write to SQLite (caller does that).

Cost is calculated from known pricing. If Anthropic changes prices, update
PRICING below. The budget guard reads from SQLite, not this module.
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import anthropic

from config.settings import Settings

log = logging.getLogger(__name__)

# Model IDs
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-6"

# Pricing in USD per token (input / output)
PRICING: dict[str, tuple[float, float]] = {
    HAIKU_MODEL:  (0.80 / 1_000_000, 4.00 / 1_000_000),
    SONNET_MODEL: (3.00 / 1_000_000, 15.00 / 1_000_000),
}

# Conservative max tokens for responses — keep costs predictable
TIER2_MAX_TOKENS = 150
TIER3_MAX_TOKENS = 400


@dataclass
class LLMResponse:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    raw_text: str                   # full response text for audit
    parsed: Optional[dict[str, Any]]  # parsed JSON, or None if parse failed
    parse_error: Optional[str]      # error message if JSON parsing failed


class LLMClient:
    """
    Thin Anthropic SDK wrapper. One instance per process.

    Usage:
        client = LLMClient(settings)
        response = client.call_haiku(system, messages)
        if response.parsed:
            sentiment = response.parsed["sentiment"]
    """

    def __init__(self, settings: Settings) -> None:
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def call_haiku(
        self,
        system: str,
        messages: list[dict],
    ) -> LLMResponse:
        return self._call(HAIKU_MODEL, system, messages, TIER2_MAX_TOKENS)

    def call_sonnet(
        self,
        system: str,
        messages: list[dict],
    ) -> LLMResponse:
        return self._call(SONNET_MODEL, system, messages, TIER3_MAX_TOKENS)

    # -------------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------------

    def _call(
        self,
        model: str,
        system: str,
        messages: list[dict],
        max_tokens: int,
    ) -> LLMResponse:
        try:
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:
            log.error("Anthropic API call failed (model=%s): %s", model, exc)
            raise

        prompt_tokens = resp.usage.input_tokens
        completion_tokens = resp.usage.output_tokens
        cost = _compute_cost(model, prompt_tokens, completion_tokens)
        raw_text = resp.content[0].text if resp.content else ""

        parsed, parse_error = _parse_json(raw_text)

        log.debug(
            "LLM call: model=%s prompt=%d completion=%d cost=$%.5f",
            model, prompt_tokens, completion_tokens, cost,
        )

        return LLMResponse(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            raw_text=raw_text,
            parsed=parsed,
            parse_error=parse_error,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    input_rate, output_rate = PRICING.get(model, (0.0, 0.0))
    return round(prompt_tokens * input_rate + completion_tokens * output_rate, 8)


def _parse_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    """
    Parse JSON from LLM response text.

    Handles the common case where the model wraps JSON in markdown fences
    (```json ... ```) despite being told not to.
    """
    text = text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    try:
        return json.loads(text), None
    except json.JSONDecodeError as exc:
        return None, str(exc)
