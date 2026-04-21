"""
LLM spend budget guard.

Thin wrapper around db/store.get_monthly_llm_spend() that compares against
the configured limit. The actual spend data lives in SQLite — this module
just enforces the threshold.

Hard-stop semantics:
  - >= $10 (llm_budget_monthly_usd): no more LLM calls this month.
    The LLM job exits; the quant scanner still runs using last known bias.
  - The entrypoint logs a CRITICAL warning at ~$8.33 (83.3% soft warning threshold).
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.settings import Settings
from db.store import get_monthly_llm_spend

log = logging.getLogger(__name__)

_SOFT_WARNING_FRACTION = 0.833  # warn at 83.3% of limit (~$8.33 on a $10 cap)


@dataclass
class BudgetStatus:
    monthly_spend: float
    monthly_limit: float
    remaining: float
    over_limit: bool
    near_limit: bool        # True if >= soft warning threshold
    year_month: str         # 'YYYY-MM'


def check_budget(conn: sqlite3.Connection, settings: Settings) -> BudgetStatus:
    """
    Read current month's LLM spend and compare against the configured limit.

    Always safe to call — never raises on DB errors (returns a conservative
    over_limit=True status instead so the caller stays safe).
    """
    year_month = datetime.now(timezone.utc).strftime("%Y-%m")

    try:
        spend = get_monthly_llm_spend(conn, year_month)
    except Exception as exc:
        log.error("Failed to read LLM budget from DB: %s — treating as over limit", exc)
        return BudgetStatus(
            monthly_spend=settings.llm_budget_monthly_usd,
            monthly_limit=settings.llm_budget_monthly_usd,
            remaining=0.0,
            over_limit=True,
            near_limit=True,
            year_month=year_month,
        )

    limit = settings.llm_budget_monthly_usd
    remaining = max(0.0, limit - spend)
    over_limit = spend >= limit
    near_limit = spend >= limit * _SOFT_WARNING_FRACTION

    if over_limit:
        log.critical(
            "LLM BUDGET EXCEEDED: spent $%.4f of $%.2f limit for %s. "
            "No more LLM calls this month — running quant-only.",
            spend, limit, year_month,
        )
    elif near_limit:
        log.warning(
            "LLM budget near limit: spent $%.4f of $%.2f (%.0f%%) for %s.",
            spend, limit, (spend / limit) * 100, year_month,
        )

    return BudgetStatus(
        monthly_spend=spend,
        monthly_limit=limit,
        remaining=remaining,
        over_limit=over_limit,
        near_limit=near_limit,
        year_month=year_month,
    )


def assert_budget_ok(conn: sqlite3.Connection, settings: Settings) -> None:
    """
    Raise RuntimeError if the monthly LLM budget is exhausted.
    Call this at the top of the LLM job before any API calls.
    """
    status = check_budget(conn, settings)
    if status.over_limit:
        raise RuntimeError(
            f"LLM budget exhausted: ${status.monthly_spend:.4f} spent "
            f"of ${status.monthly_limit:.2f} limit for {status.year_month}."
        )
