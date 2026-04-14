"""
News data provider.

Defines the NewsProvider ABC and the FinnhubProvider implementation.
The provider's job is purely fetching — de-duplication against SQLite
is handled upstream by the sentiment orchestrator.

Finnhub free tier: 60 API calls/minute. With 20 watchlist tickers and
two LLM runs per day, we make 40 calls total — well within limits.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import finnhub

from config.settings import Settings

log = logging.getLogger(__name__)


@dataclass
class Headline:
    id: str                     # unique — used for SQLite de-duplication
    symbol: str
    headline: str
    source: Optional[str]
    published_at: datetime      # UTC, tz-aware
    url: Optional[str]


class NewsProvider(ABC):
    """Interface for headline providers — swap implementations freely."""

    @abstractmethod
    def get_headlines(
        self,
        symbol: str,
        lookback_hours: int,
    ) -> list[Headline]:
        """
        Fetch headlines for `symbol` published within the last `lookback_hours`.

        Implementations must:
        - Return an empty list (not raise) if no headlines are found.
        - Filter out headlines older than lookback_hours at the source level.
        - Assign a stable, unique `id` per article.
        """
        ...


class FinnhubProvider(NewsProvider):
    """
    Finnhub company news endpoint wrapper.

    Rate-limits itself to avoid 429s on the free tier (60 req/min).
    Each call is per-symbol; with 20 tickers we make 20 sequential calls
    per LLM run, spaced by _min_call_interval_s.
    """

    _min_call_interval_s: float = 1.1  # 60 req/min → 1 req/sec with safety margin

    def __init__(self, settings: Settings) -> None:
        self._client = finnhub.Client(api_key=settings.finnhub_api_key)
        self._last_call_ts: float = 0.0

    def get_headlines(
        self,
        symbol: str,
        lookback_hours: int,
    ) -> list[Headline]:
        self._rate_limit()

        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(hours=lookback_hours)

        # Finnhub expects YYYY-MM-DD strings
        from_str = from_dt.strftime("%Y-%m-%d")
        to_str = now.strftime("%Y-%m-%d")

        try:
            raw = self._client.company_news(symbol, _from=from_str, to=to_str)
        except Exception as exc:
            log.warning("Finnhub news fetch failed for %s: %s", symbol, exc)
            return []

        if not raw:
            return []

        headlines: list[Headline] = []
        cutoff = now - timedelta(hours=lookback_hours)

        for item in raw:
            published_at = _parse_finnhub_datetime(item.get("datetime", 0))
            if published_at is None or published_at < cutoff:
                continue

            headline_id = str(item.get("id", ""))
            if not headline_id:
                continue

            headlines.append(
                Headline(
                    id=headline_id,
                    symbol=symbol,
                    headline=item.get("headline", "").strip(),
                    source=item.get("source"),
                    published_at=published_at,
                    url=item.get("url"),
                )
            )

        log.debug(
            "Finnhub: %d headlines for %s (lookback %dh)",
            len(headlines), symbol, lookback_hours,
        )
        return headlines

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_call_ts
        wait = self._min_call_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_call_ts = time.monotonic()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_finnhub_datetime(ts: int) -> Optional[datetime]:
    """Convert a Finnhub Unix timestamp (int) to a UTC datetime."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def fetch_all_headlines(
    provider: NewsProvider,
    symbols: list[str],
    lookback_hours: int,
) -> list[Headline]:
    """
    Fetch headlines for every symbol in one sweep.
    Returns a flat list, sorted by published_at descending (newest first).
    """
    all_headlines: list[Headline] = []
    for symbol in symbols:
        headlines = provider.get_headlines(symbol, lookback_hours=lookback_hours)
        all_headlines.extend(headlines)

    all_headlines.sort(key=lambda h: h.published_at, reverse=True)
    log.info(
        "Fetched %d total headlines across %d symbols",
        len(all_headlines), len(symbols),
    )
    return all_headlines
