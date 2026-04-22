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
        default=0.30, description="Max sector concentration as fraction of equity"
    )
    stop_loss_pct: float = Field(
        default=0.02,
        description="Fallback stop-loss distance from entry when ATR is unavailable",
    )
    risk_per_trade_pct: float = Field(
        default=0.005,
        description="Dollar risk per trade as fraction of equity (0.5% = $50 on $10k)",
    )
    atr_stop_multiplier: float = Field(
        default=1.5,
        description="Stop distance = atr_stop_multiplier × ATR/price (ATR-based sizing)",
    )
    min_stop_pct: float = Field(
        default=0.01,
        description="Floor on dynamic stop distance — prevents 0.1% noise stops",
    )
    max_portfolio_heat: float = Field(
        default=0.04,
        description="Max aggregate open dollar-risk as fraction of equity (4%)",
    )
    trail_pct: float = Field(
        default=0.01, description="Trailing stop distance once activated (1%)"
    )
    trail_activation_pct: float = Field(
        default=0.02, description="Unrealized gain required to activate trailing stop"
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
        default=0.80, description="Max 20-day annualized realized vol to pass filter"
    )

    # -------------------------------------------------------------------------
    # Signal entry criteria
    # -------------------------------------------------------------------------
    rsi_min: float = Field(default=50.0, description="RSI lower bound for long entries")
    rsi_max: float = Field(default=80.0, description="RSI upper bound for long entries")
    volume_multiplier: float = Field(
        default=1.2, description="Min current volume vs 20-day average"
    )
    volume_multiplier_open: float = Field(
        default=2.0,
        description="Stricter volume multiplier for the 10:30 ET opening session",
    )
    open_session_minute_marker: int = Field(
        default=30,
        description="Marker minute identifying the opening session (10:30 ET)",
    )
    no_new_entries_session_minute_marker: int = Field(
        default=30,
        description="Final scan minute (15:30 ET) — manage only, no new entries",
    )
    sentiment_size_multiplier: float = Field(
        default=1.25,
        description="Position-size multiplier when sentiment bias matches direction",
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
        default=12, description="Hard cap on Sonnet (Tier 3) calls per scheduled run"
    )

    # -------------------------------------------------------------------------
    # LLM budget
    # -------------------------------------------------------------------------
    llm_budget_monthly_usd: float = Field(
        default=10.0, description="Monthly LLM spend hard-stop in USD"
    )

    # -------------------------------------------------------------------------
    # News / data fetching
    # -------------------------------------------------------------------------
    news_lookback_morning_hours: int = Field(
        default=16, description="Hours of headline lookback for 10:00 ET morning run (covers yesterday's close through this morning)"
    )
    news_lookback_afternoon_hours: int = Field(
        default=4, description="Hours of headline lookback for 13:00 ET midday run"
    )
    max_news_age_hours: int = Field(
        default=20, description="Headlines older than this are discarded before triage (covers 16h morning lookback + buffer)"
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
        default=35, description="Flatten all positions N minutes before close"
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
    # Market regime filter (SPY trend + volatility gate)
    # -------------------------------------------------------------------------
    spy_ema_short: int = Field(
        default=50, description="EMA period for SPY trend gate (CAUTION threshold)"
    )
    spy_ema_long: int = Field(
        default=200, description="EMA period for SPY trend gate (BEAR/halt threshold)"
    )
    vol_regime_threshold: float = Field(
        default=0.30,
        description="SPY 20-day annualised vol above which to cap max positions",
    )
    vol_regime_max_positions: int = Field(
        default=2, description="Max positions when vol_regime_threshold is breached"
    )

    # -------------------------------------------------------------------------
    # Earnings calendar blackout
    # -------------------------------------------------------------------------
    earnings_blackout_days: int = Field(
        default=2,
        description="Skip tickers with earnings within this many calendar days",
    )

    # -------------------------------------------------------------------------
    # Exit improvements
    # -------------------------------------------------------------------------
    take_profit_pct: float = Field(
        default=0.06,
        description="Take-profit target as fraction of entry price (6% = close at +6%)",
    )
    partial_exit_enabled: bool = Field(
        default=True,
        description="Scale out a fraction of the position at the partial-exit trigger",
    )
    partial_exit_trigger_pct: float = Field(
        default=0.03,
        description="Unrealized gain at which to take partial profits (3%)",
    )
    partial_exit_fraction: float = Field(
        default=0.5,
        description="Fraction of position to close at the partial-exit trigger (0.5 = half)",
    )
    max_hold_days: int = Field(
        default=10,
        description="Close stale positions held longer than this with <1% gain (calendar days)",
    )

    # -------------------------------------------------------------------------
    # Signal quality improvements
    # -------------------------------------------------------------------------
    require_relative_strength: bool = Field(
        default=True,
        description="Long entries require ticker to outperform SPY over 20 days",
    )
    near_high_lookback: int = Field(
        default=63,
        description="Rolling window (bars) for the near-high proximity filter",
    )
    near_high_max_drawdown: float = Field(
        default=0.10,
        description="Max allowed distance below the rolling high (10% = within 10% of high)",
    )

    # -------------------------------------------------------------------------
    # Macro event blackout
    # -------------------------------------------------------------------------
    macro_events_path: str = Field(
        default="config/macro_events.yaml",
        description="Path to YAML file listing high-impact macro event dates",
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
