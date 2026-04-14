"""
Central configuration via Pydantic BaseSettings.

All tunables live here. Values are loaded from environment variables / .env file.
Paper-only guard is enforced at model validation time — the process dies immediately
if a live API URL is detected.
"""

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Broker credentials
    # -------------------------------------------------------------------------
    alpaca_api_key: str = Field(..., description="Alpaca paper API key")
    alpaca_secret_key: str = Field(..., description="Alpaca paper secret key")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        description="Must contain 'paper' — live URLs are rejected at startup",
    )

    # -------------------------------------------------------------------------
    # External services
    # -------------------------------------------------------------------------
    anthropic_api_key: str = Field(..., description="Anthropic API key")
    finnhub_api_key: str = Field(..., description="Finnhub API key (free tier)")

    # -------------------------------------------------------------------------
    # Risk guardrails (hard limits — do NOT tune without re-running backtest)
    # -------------------------------------------------------------------------
    account_equity: float = Field(
        default=10_000.0, description="Starting account equity in USD"
    )
    max_positions: int = Field(default=5, description="Max simultaneous open positions")
    max_position_pct: float = Field(
        default=0.10, description="Max single position as fraction of equity"
    )
    max_sector_pct: float = Field(
        default=0.25, description="Max sector concentration as fraction of equity"
    )
    stop_loss_pct: float = Field(
        default=0.02, description="Fixed stop-loss distance from entry (2%)"
    )
    trail_pct: float = Field(
        default=0.03, description="Trailing stop distance once activated (3%)"
    )
    trail_activation_pct: float = Field(
        default=0.015, description="Unrealized gain required to activate trailing stop"
    )
    daily_loss_limit_pct: float = Field(
        default=0.03, description="Daily P&L circuit breaker threshold (-3% of equity)"
    )

    # -------------------------------------------------------------------------
    # Volatility filter thresholds
    # -------------------------------------------------------------------------
    vol_atr_threshold: float = Field(
        default=0.04, description="Max ATR(14)/price ratio to pass filter"
    )
    vol_realized_threshold: float = Field(
        default=0.50, description="Max 20-day annualized realized vol to pass filter"
    )

    # -------------------------------------------------------------------------
    # Signal entry criteria
    # -------------------------------------------------------------------------
    rsi_min: float = Field(default=40.0, description="RSI lower bound for long entries")
    rsi_max: float = Field(default=70.0, description="RSI upper bound for long entries")
    volume_multiplier: float = Field(
        default=1.5, description="Min current volume vs 20-day average"
    )
    ema_period: int = Field(default=20, description="EMA period for trend filter")
    rsi_period: int = Field(default=14, description="RSI period")
    atr_period: int = Field(default=14, description="ATR period")

    # -------------------------------------------------------------------------
    # LLM triage thresholds
    # -------------------------------------------------------------------------
    tier2_sentiment_threshold: float = Field(
        default=0.3,
        description="Minimum |sentiment| from Haiku to pass as actionable signal",
    )
    tier2_confidence_threshold: float = Field(
        default=0.75, description="Confidence required to escalate to Tier 3 (Sonnet)"
    )
    tier3_sentiment_threshold: float = Field(
        default=0.6, description="|sentiment| required alongside confidence for T3 escalation"
    )
    tier3_max_per_run: int = Field(
        default=5, description="Hard cap on Sonnet (Tier 3) calls per scheduled run"
    )

    # -------------------------------------------------------------------------
    # LLM budget
    # -------------------------------------------------------------------------
    llm_budget_monthly_usd: float = Field(
        default=18.0, description="Monthly LLM spend hard-stop in USD"
    )

    # -------------------------------------------------------------------------
    # News / data fetching
    # -------------------------------------------------------------------------
    news_lookback_morning_hours: int = Field(
        default=4, description="Hours of headline lookback for 10:30 ET run"
    )
    news_lookback_afternoon_hours: int = Field(
        default=2, description="Hours of headline lookback for 15:30 ET run"
    )
    max_news_age_hours: int = Field(
        default=6, description="Headlines older than this are discarded before triage"
    )

    # -------------------------------------------------------------------------
    # Scheduler windows (Eastern Time)
    # -------------------------------------------------------------------------
    run_hour_morning: int = Field(default=10)
    run_minute_morning: int = Field(default=30)
    run_hour_afternoon: int = Field(default=15)
    run_minute_afternoon: int = Field(default=30)

    # Execution time gates (minutes from market open/close)
    no_trade_after_open_minutes: int = Field(
        default=30, description="Skip first N minutes after market open"
    )
    no_trade_before_close_minutes: int = Field(
        default=10, description="Stop trading N minutes before market close"
    )
    flatten_before_close_minutes: int = Field(
        default=10, description="Flatten all positions N minutes before close"
    )

    # -------------------------------------------------------------------------
    # Paths
    # -------------------------------------------------------------------------
    db_path: str = Field(default="tradingbot.db")
    log_path: str = Field(default="logs/trading.jsonl")
    reports_dir: str = Field(default="reports")
    watchlist_path: str = Field(default="config/watchlist.yaml")

    # -------------------------------------------------------------------------
    # Backtest
    # -------------------------------------------------------------------------
    backtest_min_days: int = Field(
        default=60, description="Minimum trading days required in backtest"
    )
    backtest_slippage_bps: int = Field(
        default=10, description="Slippage model applied to every fill (basis points)"
    )

    # -------------------------------------------------------------------------
    # Email notifications
    # -------------------------------------------------------------------------
    email_enabled: bool = Field(
        default=False, description="Send daily summary email at market close"
    )
    email_recipient: str = Field(
        default="", description="Recipient address for daily summary email"
    )
    email_sender: str = Field(
        default="", description="Gmail address used to send notifications"
    )
    email_app_password: str = Field(
        default="", description="Gmail app password (not account password)"
    )

    # -------------------------------------------------------------------------
    # Guard: reject live API URLs at model construction time
    # -------------------------------------------------------------------------
    @model_validator(mode="after")
    def _reject_live_keys(self) -> "Settings":
        if "paper" not in self.alpaca_base_url.lower():
            raise ValueError(
                f"FATAL: ALPACA_BASE_URL does not contain 'paper'.\n"
                f"  Got: {self.alpaca_base_url}\n"
                f"  This bot is paper-trading only. Set ALPACA_BASE_URL to "
                f"https://paper-api.alpaca.markets and restart."
            )
        return self


# Module-level singleton — import and use directly.
# Raises immediately if .env is missing required fields or URL is live.
def load_settings() -> Settings:
    """Load and validate settings. Raises on misconfiguration."""
    return Settings()  # type: ignore[call-arg]
