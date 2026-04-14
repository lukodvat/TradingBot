"""
TradingBot entrypoint — APScheduler process with three job types.

Job A — LLM Sentiment (10:00 ET, 13:00 ET):
    Fetches headlines, runs tiered triage, writes per-ticker bias to SQLite.

Job B — Quant Scanner (10:30, 11:30, 12:30, 13:30, 14:30, 15:30 ET):
    Reads bias from SQLite, applies technical criteria, executes bracket orders.

Job C — Daily Email (16:30 ET):
    Sends an LLM-written summary email with P&L, trades, and performance.

Pre-flight checks at startup:
    1. Paper-only guard (Settings model_validator)
    2. Backtest gate (positive-expectancy report must exist in reports/)
    3. DB schema init
    4. Logging setup

The process runs indefinitely until SIGINT/SIGTERM.
"""

import logging
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from alpaca.trading.enums import OrderSide

from config.settings import Settings
from core.broker import BrokerClient
from core.portfolio import PortfolioManager, _minutes_to_market_close
from core.risk import RiskManager
from data.market import MarketDataClient, load_watchlist, load_sector_map
from data.news import FinnhubProvider, fetch_all_headlines
from db import store, schema
from analysis.sentiment import SentimentAnalyzer
from analysis.signals import SignalScanner
from analysis.volatility import filter_watchlist, passing_tickers
from backtest.harness import check_backtest_gate
from notifications.email import send_daily_email
from llm.budget import assert_budget_ok
from llm.client import LLMClient

log = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# NYSE calendar helpers
# ---------------------------------------------------------------------------

def _is_market_day(dt: datetime) -> bool:
    """True if dt falls on a Mon-Fri (no holiday detection in v1)."""
    return dt.astimezone(_ET).weekday() < 5  # 0=Mon … 4=Fri


def _is_within_trading_hours(dt: datetime, settings: Settings) -> bool:
    """
    True if dt is between market open+no_trade_after_open_minutes and
    market close-no_trade_before_close_minutes (ET).
    """
    et = dt.astimezone(_ET)
    # Market open: 9:30; close: 16:00
    open_cutoff_min = 9 * 60 + 30 + settings.no_trade_after_open_minutes
    close_cutoff_min = 16 * 60 - settings.no_trade_before_close_minutes
    now_min = et.hour * 60 + et.minute
    return open_cutoff_min <= now_min < close_cutoff_min


def _is_friday(dt: datetime) -> bool:
    return dt.astimezone(_ET).weekday() == 4


# ---------------------------------------------------------------------------
# Job A — LLM Sentiment Run
# ---------------------------------------------------------------------------

def run_llm_job(
    llm_run: str,           # 'morning' | 'midday'
    settings: Settings,
    llm_client: LLMClient,
    news_provider: FinnhubProvider,
    conn,
) -> None:
    """
    Job A: fetch headlines → tiered triage → write sentiment_bias to SQLite.
    Runs at 10:00 ET (morning) and 13:00 ET (midday).
    """
    run_dt = datetime.now(timezone.utc)
    run_timestamp = run_dt.isoformat()
    session = f"llm_{llm_run}"

    log.info("=== Job A START [%s] ===", llm_run)
    store.start_session_log(conn, run_timestamp=run_timestamp, session=session)

    # Calendar gate
    if not _is_market_day(run_dt):
        log.info("Job A skipped — not a market day")
        return

    # LLM budget check
    try:
        assert_budget_ok(conn, settings)
    except RuntimeError as exc:
        log.critical("Job A aborted — %s", exc)
        return

    # News fetch
    lookback = (
        settings.news_lookback_morning_hours
        if llm_run == "morning"
        else settings.news_lookback_afternoon_hours
    )
    symbols = load_watchlist(settings.watchlist_path)
    raw_headlines = fetch_all_headlines(
        news_provider, symbols, lookback_hours=lookback
    )

    # De-duplicate and persist
    new_headlines = []
    for h in raw_headlines:
        if store.is_headline_seen(conn, h.id):
            continue
        store.insert_headline(
            conn,
            headline_id=h.id,
            symbol=h.symbol,
            headline=h.headline,
            source=h.source,
            published_at=h.published_at.isoformat() if h.published_at else None,
            fetched_at=run_timestamp,
            run_timestamp=run_timestamp,
        )
        new_headlines.append(h)

    log.info("Headlines: %d raw, %d new", len(raw_headlines), len(new_headlines))
    store.update_session_log(conn, run_timestamp=run_timestamp,
                             tickers_evaluated=len(symbols))

    if not new_headlines:
        log.info("Job A done — no new headlines")
        return

    # Tiered triage
    analyzer = SentimentAnalyzer(settings, llm_client, conn)
    biases = analyzer.analyze(new_headlines, run_timestamp=run_timestamp, llm_run=llm_run)

    # Write sentiment_bias table
    for symbol, bias_obj in biases.items():
        store.upsert_sentiment_bias(
            conn,
            ticker=symbol,
            date=run_dt.astimezone(_ET).strftime("%Y-%m-%d"),
            bias=bias_obj.bias,
            aggregated_score=bias_obj.aggregated_score,
            headline_count=bias_obj.headline_count,
            llm_run=llm_run,
        )

    log.info("Job A DONE [%s] — biases written for %d tickers", llm_run, len(biases))


# ---------------------------------------------------------------------------
# Job B — Quant Scanner
# ---------------------------------------------------------------------------

def run_quant_job(
    session: str,
    settings: Settings,
    broker: BrokerClient,
    risk: RiskManager,
    market_data: MarketDataClient,
    conn,
) -> None:
    """
    Job B: read bias → vol filter → signal scan → risk checks → bracket orders.
    Runs 6× per day at :30 past each hour from 10:30 to 15:30 ET.
    """
    run_dt = datetime.now(timezone.utc)
    run_timestamp = run_dt.isoformat()
    today_str = run_dt.astimezone(_ET).strftime("%Y-%m-%d")
    is_fr = _is_friday(run_dt)

    log.info("=== Job B START [%s] ===", session)
    store.start_session_log(conn, run_timestamp=run_timestamp, session=session)

    # Calendar + time gate
    if not _is_market_day(run_dt):
        log.info("Job B skipped — not a market day")
        return
    if not _is_within_trading_hours(run_dt, settings):
        log.info("Job B skipped — outside trading hours")
        return

    # Account snapshot
    snap = broker.snapshot()

    # Circuit breaker
    cb = risk.check_circuit_breaker(snap.daily_pnl, snap.equity)
    if cb.triggered:
        log.critical(
            "CIRCUIT BREAKER — daily_pnl=%.2f%% — liquidating all",
            cb.daily_pnl_pct * 100,
        )
        broker.close_all_positions()
        store.update_session_log(conn, run_timestamp=run_timestamp,
                                 circuit_breaker_triggered=1)
        return

    # Position management — trailing stops and flattens
    pm = PortfolioManager(settings, broker, risk, conn)
    positions = snap.positions

    pm.manage_trailing_stops(positions, run_timestamp)

    # Load today's biases for flatten decisions
    biases = store.get_all_sentiment_biases_for_date(conn, today_str)
    minutes_to_close = _minutes_to_market_close(run_dt)
    is_flatten_window = minutes_to_close <= settings.flatten_before_close_minutes

    if is_flatten_window or is_fr:
        pm.manage_flattens(
            positions, biases=biases,
            run_dt=run_dt, run_timestamp=run_timestamp,
            is_friday=is_fr,
        )

    # Same-ticker cooldown
    held_today = pm.get_held_today(today_str)

    # Fetch OHLCV for all watchlist tickers
    symbols = load_watchlist(settings.watchlist_path)
    sector_map = load_sector_map(settings.watchlist_path)
    bars = market_data.get_daily_bars(symbols, lookback_days=25)

    # Volatility filter
    vol_results = filter_watchlist(bars, settings)
    for sym, vr in vol_results.items():
        store.record_vol_filter(
            conn,
            run_timestamp=run_timestamp,
            symbol=sym,
            atr_price_ratio=vr.atr_price_ratio,
            realized_vol=vr.realized_vol,
            atr_threshold=settings.vol_atr_threshold,
            vol_threshold=settings.vol_realized_threshold,
            passed=vr.passed,
            fail_reason=vr.fail_reason,
        )

    passing = passing_tickers(vol_results)
    passing_bars = {sym: bars[sym] for sym in passing if sym in bars}
    log.info("Vol filter: %d/%d tickers pass", len(passing), len(symbols))

    # Signal scan
    scanner = SignalScanner(settings, conn)
    candidates = scanner.scan(passing_bars, date=today_str, held_today=held_today)
    log.info("Signals: %d candidates", len(candidates))

    # Refresh snapshot after position management ops
    snap = broker.snapshot()

    # Risk pre-checks + order execution
    orders_submitted = 0
    for candidate in candidates:
        sector = sector_map.get(candidate.symbol, "Unknown")
        side = OrderSide.BUY if candidate.direction == "LONG" else OrderSide.SELL

        sizing = risk.size_position(
            symbol=candidate.symbol,
            sector=sector,
            side=side,
            limit_price=candidate.current_price,
            equity=snap.equity,
            open_positions=snap.positions,
            sector_map=sector_map,
            buying_power=snap.buying_power,
        )

        if not sizing.approved:
            log.info("REJECT %s: %s", candidate.symbol, sizing.rejection_reason)
            continue

        try:
            order = broker.submit_bracket_order(
                symbol=candidate.symbol,
                qty=sizing.qty,
                side=side,
                limit_price=sizing.limit_price,
                stop_price=sizing.stop_price,
            )
            store.record_order(
                conn,
                order_id=str(order.id),
                symbol=candidate.symbol,
                side=side.value,
                order_type="limit",
                qty=sizing.qty,
                limit_price=sizing.limit_price,
                stop_price=sizing.stop_price,
                status=str(order.status),
                run_timestamp=run_timestamp,
            )
            log.info(
                "ORDER %s %s qty=%d @ %.2f stop=%.2f",
                candidate.direction, candidate.symbol,
                sizing.qty, sizing.limit_price, sizing.stop_price,
            )
            orders_submitted += 1

            # Update snapshot so next candidate sees current positions
            snap = broker.snapshot()
        except Exception as exc:
            log.error("Order failed for %s: %s", candidate.symbol, exc)

    # Record equity snapshot
    pm.record_snapshot(snap, run_timestamp=run_timestamp, session=session, date=today_str)
    store.update_session_log(
        conn, run_timestamp=run_timestamp,
        tickers_evaluated=len(passing),
        orders_submitted=orders_submitted,
    )

    log.info(
        "Job B DONE [%s] — candidates=%d orders=%d equity=%.2f",
        session, len(candidates), orders_submitted, snap.equity,
    )


# ---------------------------------------------------------------------------
# Job C — Daily Email
# ---------------------------------------------------------------------------

def run_email_job(settings: Settings, conn) -> None:
    """
    Job C: generate and send the daily summary email at 16:30 ET.
    Skips weekends and when email is disabled in settings.
    """
    run_dt = datetime.now(timezone.utc)
    if not _is_market_day(run_dt):
        log.info("Email job skipped — not a market day")
        return

    date_str = run_dt.astimezone(_ET).strftime("%Y-%m-%d")
    log.info("=== Job C START — daily email for %s ===", date_str)

    ok = send_daily_email(conn, settings, date=date_str)
    if ok:
        log.info("Job C DONE — email sent")
    else:
        log.info("Job C DONE — email not sent (disabled or failed)")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Logging setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log.info("TradingBot starting up...")

    # --- Settings (validates paper-only URL at construction time) ---
    settings = Settings()
    log.info("Settings loaded. DB: %s | Reports: %s", settings.db_path, settings.reports_dir)

    # --- Backtest gate ---
    gate_ok, gate_path = check_backtest_gate(settings.reports_dir)
    if not gate_ok:
        log.critical(
            "STARTUP BLOCKED: No passing backtest report found in '%s'.\n"
            "Run `python run_backtest.py` first and ensure it passes the gate.",
            settings.reports_dir,
        )
        sys.exit(1)
    log.info("Backtest gate passed: %s", gate_path)

    # --- DB init ---
    conn = store.get_connection(settings.db_path)
    schema.init_db(conn)
    log.info("Database ready at %s", settings.db_path)

    # --- Shared clients (created once, reused across runs) ---
    broker = BrokerClient(settings)
    risk = RiskManager(settings)
    llm_client = LLMClient(settings)
    market_data = MarketDataClient(settings)
    news_provider = FinnhubProvider(settings)

    log.info("All clients initialised. Starting scheduler...")

    # --- APScheduler ---
    scheduler = BlockingScheduler(timezone=str(_ET))

    # Job A — LLM runs
    scheduler.add_job(
        run_llm_job,
        CronTrigger(hour=10, minute=0, timezone=str(_ET)),
        args=["morning", settings, llm_client, news_provider, conn],
        id="llm_morning",
        name="LLM Sentiment — Morning",
        misfire_grace_time=300,
    )
    scheduler.add_job(
        run_llm_job,
        CronTrigger(hour=13, minute=0, timezone=str(_ET)),
        args=["midday", settings, llm_client, news_provider, conn],
        id="llm_midday",
        name="LLM Sentiment — Midday",
        misfire_grace_time=300,
    )

    # Job B — Quant scanner (6 windows)
    quant_args = [settings, broker, risk, market_data, conn]
    for hour, minute, session_label in [
        (10, 30, "quant_1030"),
        (11, 30, "quant_1130"),
        (12, 30, "quant_1230"),
        (13, 30, "quant_1330"),
        (14, 30, "quant_1430"),
        (15, 30, "quant_1530"),
    ]:
        scheduler.add_job(
            run_quant_job,
            CronTrigger(hour=hour, minute=minute, timezone=str(_ET)),
            args=[session_label, *quant_args],
            id=session_label,
            name=f"Quant Scanner — {session_label}",
            misfire_grace_time=120,
        )

    # Job C — daily email at 16:30 ET
    scheduler.add_job(
        run_email_job,
        CronTrigger(hour=16, minute=30, timezone=str(_ET)),
        args=[settings, conn],
        id="email_daily",
        name="Daily Summary Email",
        misfire_grace_time=600,
    )

    log.info(
        "Scheduled 2 LLM jobs, 6 quant jobs, and 1 email job. "
        "Waiting for market hours... (Ctrl-C to stop)"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutdown requested — stopping scheduler.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
