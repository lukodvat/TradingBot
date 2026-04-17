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
from analysis.regime import MarketRegimeFilter
from analysis.sentiment import SentimentAnalyzer
from analysis.signals import SignalScanner
from analysis.volatility import filter_watchlist, passing_tickers
from backtest.harness import check_backtest_gate
from notifications.email import send_daily_email, send_circuit_breaker_alert
from llm.budget import assert_budget_ok
from llm.client import LLMClient

log = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# NYSE calendar helpers
# ---------------------------------------------------------------------------

# NYSE-observed holidays for 2025 and 2026.
# Dates are YYYY-MM-DD strings in ET.
_NYSE_HOLIDAYS: frozenset[str] = frozenset({
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
    # 2026
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # Martin Luther King Jr. Day
    "2026-02-16",  # Presidents' Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed, falls on Friday)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
})


def _is_market_day(dt: datetime) -> bool:
    """True if dt falls on a Mon-Fri that is not an NYSE holiday."""
    et = dt.astimezone(_ET)
    if et.weekday() >= 5:  # Saturday or Sunday
        return False
    return et.strftime("%Y-%m-%d") not in _NYSE_HOLIDAYS


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


def _load_macro_blackout_dates(path: str) -> set[str]:
    """Load YYYY-MM-DD blackout dates from macro_events.yaml."""
    import yaml
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
        return {str(e["date"]) for e in data.get("events", []) if "date" in e}
    except Exception as exc:
        log.warning("Could not load macro events from %s: %s", path, exc)
        return set()


def _is_macro_blackout(date_str: str, path: str) -> bool:
    """Return True if date_str (YYYY-MM-DD) is a macro event blackout day."""
    blackout_dates = _load_macro_blackout_dates(path)
    return date_str in blackout_dates


def _compute_spy_return_20d(spy_bars) -> float | None:
    """Return SPY's 20-day price return as a fraction, or None if not enough data."""
    if spy_bars is None or len(spy_bars) < 21:
        return None
    close = spy_bars["close"]
    try:
        ret = (float(close.iloc[-1]) - float(close.iloc[-21])) / float(close.iloc[-21])
        return ret
    except (ZeroDivisionError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Fill reconciliation (populates trades table from Alpaca filled orders)
# ---------------------------------------------------------------------------

def reconcile_fills(broker: BrokerClient, conn, run_timestamp: str) -> int:
    """
    Poll Alpaca for orders filled in the last 24 hours and upsert them into
    the trades table.  This is the canonical way the trades table gets
    populated — main.py only inserts into orders at submission time.

    Returns the number of new trade rows inserted.
    """
    from datetime import timedelta

    since = datetime.now(timezone.utc) - timedelta(hours=24)
    filled_orders = broker.get_filled_orders_since(since)
    inserted = 0

    for order in filled_orders:
        order_id = str(order.id)
        if store.trade_exists(conn, order_id):
            continue

        symbol = str(order.symbol)
        side_raw = str(order.side).lower()  # 'buy' or 'sell'
        qty = float(order.filled_qty or 0)
        fill_price = float(order.filled_avg_price or 0)
        limit_price = float(order.limit_price) if order.limit_price else None
        stop_price = float(order.stop_price) if order.stop_price else None
        notional = qty * fill_price
        filled_at = (
            order.filled_at.isoformat()
            if order.filled_at else None
        )

        store.record_trade(
            conn,
            order_id=order_id,
            symbol=symbol,
            side=side_raw,
            qty=qty,
            fill_price=fill_price,
            limit_price=limit_price,
            stop_price=stop_price,
            notional=notional,
            session=run_timestamp[:13],   # 'YYYY-MM-DDTHH' is sufficient
            run_timestamp=run_timestamp,
            filled_at=filled_at,
        )

        # Keep orders table in sync
        store.update_order_status(
            conn,
            order_id=order_id,
            status="filled",
            fill_price=fill_price,
            filled_at=filled_at,
        )

        log.info("Reconciled fill: %s %s x%.2f @ %.2f", side_raw, symbol, qty, fill_price)
        inserted += 1

    if inserted:
        log.info("reconcile_fills: inserted %d new trade rows", inserted)
    return inserted


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
    lookback_map = {
        "premarket": settings.news_lookback_premarket_hours,  # 9:00 ET — overnight news
        "morning":   settings.news_lookback_morning_hours,    # 10:00 ET — full prior-day narrative
        "midday":    settings.news_lookback_afternoon_hours,  # 13:00 ET — intraday update
    }
    lookback = lookback_map.get(llm_run, settings.news_lookback_morning_hours)
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
    news_provider,
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

    # Fill reconciliation — populate trades table from Alpaca filled orders
    reconcile_fills(broker, conn, run_timestamp)

    # Account snapshot
    snap = broker.snapshot()

    # Capture today's open equity on the first run of the day so the circuit
    # breaker reacts to intraday losses the system causes, not overnight gaps.
    open_equity = store.get_open_equity_for_date(conn, today_str)
    if open_equity is None:
        open_equity = snap.equity
        store.upsert_daily_pnl(conn, date=today_str, open_equity=open_equity)

    # Circuit breaker — intraday drawdown from today's open, not last close.
    intraday_pnl = snap.equity - open_equity
    cb = risk.check_circuit_breaker(intraday_pnl, open_equity)
    if cb.triggered:
        log.critical(
            "CIRCUIT BREAKER — daily_pnl=%.2f%% — liquidating all",
            cb.daily_pnl_pct * 100,
        )
        broker.close_all_positions()
        store.update_session_log(conn, run_timestamp=run_timestamp,
                                 circuit_breaker_triggered=1)
        send_circuit_breaker_alert(settings, cb.daily_pnl_pct, snap.equity)
        return

    # Position management — trailing stops, time-based exits, and flattens
    pm = PortfolioManager(settings, broker, risk, conn)
    positions = snap.positions

    pm.manage_trailing_stops(positions, run_timestamp)
    pm.manage_time_based_exits(positions, run_timestamp)

    # Load today's biases for flatten decisions
    biases = store.get_all_sentiment_biases_for_date(conn, today_str)
    if not biases:
        log.warning(
            "Job B [%s]: no sentiment biases found for %s — "
            "Job A may not have run yet. All tickers will be treated as NEUTRAL.",
            session, today_str,
        )
    minutes_to_close = _minutes_to_market_close(run_dt)
    is_flatten_window = minutes_to_close <= settings.flatten_before_close_minutes

    if is_flatten_window:
        pm.manage_flattens(
            positions, biases=biases,
            run_dt=run_dt, run_timestamp=run_timestamp,
            is_friday=is_fr,
        )

    # Same-ticker cooldown
    held_today = pm.get_held_today(today_str)

    # Fetch OHLCV for all watchlist tickers + SPY (260 days for EMA(200) regime + 63-bar near-high)
    symbols = load_watchlist(settings.watchlist_path)
    sector_map = load_sector_map(settings.watchlist_path)
    all_symbols = list(set(symbols + ["SPY"]))
    all_bars = market_data.get_daily_bars(all_symbols, lookback_days=260)

    spy_bars = all_bars.pop("SPY", None)
    bars = {sym: all_bars[sym] for sym in symbols if sym in all_bars}

    # Macro event blackout — block new entries but let position management continue
    if _is_macro_blackout(today_str, settings.macro_events_path):
        log.info("Job B: macro blackout day (%s) — skipping signal scan", today_str)
        snap = broker.snapshot()
        pm.record_snapshot(snap, run_timestamp=run_timestamp, session=session, date=today_str)
        store.update_session_log(conn, run_timestamp=run_timestamp,
                                 tickers_evaluated=0, orders_submitted=0)
        return

    # Market regime filter
    regime = MarketRegimeFilter(settings).evaluate(spy_bars)
    spy_return_20d = _compute_spy_return_20d(spy_bars)

    if not regime.allow_any_entries:
        log.info("Job B: regime=%s — no new entries this run", regime.label)
        snap = broker.snapshot()
        pm.record_snapshot(snap, run_timestamp=run_timestamp, session=session, date=today_str)
        store.update_session_log(conn, run_timestamp=run_timestamp,
                                 tickers_evaluated=0, orders_submitted=0)
        return

    # Earnings blackout — filter out tickers with upcoming earnings
    earnings_blackout = news_provider.get_upcoming_earnings(
        symbols, days_ahead=settings.earnings_blackout_days
    )
    if earnings_blackout:
        symbols = [s for s in symbols if s not in earnings_blackout]
        bars = {s: v for s, v in bars.items() if s not in earnings_blackout}

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

    # Signal scan (with relative strength and near-high filters active)
    scanner = SignalScanner(settings, conn, spy_return_20d=spy_return_20d)
    candidates = scanner.scan(passing_bars, date=today_str, held_today=held_today)

    # Apply regime direction filter
    if not regime.allow_long_entries:
        candidates = [c for c in candidates if c.direction != "LONG"]
        log.info("Regime %s: long entries suppressed", regime.label)
    if not regime.allow_short_entries:
        candidates = [c for c in candidates if c.direction != "SHORT"]

    log.info("Signals: %d candidates (regime=%s)", len(candidates), regime.label)

    # Refresh snapshot after position management ops
    snap = broker.snapshot()

    # Regime-adjusted position cap
    effective_max_positions = regime.max_positions_override

    # Risk pre-checks + order execution
    orders_submitted = 0
    for candidate in candidates:
        # Respect regime position cap before calling risk manager
        if len(snap.positions) >= effective_max_positions:
            log.info(
                "Regime cap: %d positions at regime limit %d — stopping",
                len(snap.positions), effective_max_positions,
            )
            break

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
                take_profit_price=sizing.take_profit_price,
            )
            store.record_order(
                conn,
                order_id=str(order.id),
                symbol=candidate.symbol,
                side=side.value,
                order_type="bracket",
                qty=sizing.qty,
                limit_price=sizing.limit_price,
                stop_price=sizing.stop_price,
                status=str(order.status),
                submitted_at=run_timestamp,
                run_timestamp=run_timestamp,
            )
            log.info(
                "ORDER %s %s qty=%d @ %.2f stop=%.2f tp=%.2f",
                candidate.direction, candidate.symbol,
                sizing.qty, sizing.limit_price, sizing.stop_price,
                sizing.take_profit_price or 0,
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
    gate_ok, gate_path = check_backtest_gate(settings.reports_dir, settings)
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

    # Job A — LLM runs (3× daily: pre-market overnight, morning narrative, midday update)
    scheduler.add_job(
        run_llm_job,
        CronTrigger(hour=9, minute=0, timezone=str(_ET)),
        args=["premarket", settings, llm_client, news_provider, conn],
        id="llm_premarket",
        name="LLM Sentiment — Pre-Market",
        misfire_grace_time=300,
    )
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
    quant_args = [settings, broker, risk, market_data, news_provider, conn]
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
        "Scheduled 3 LLM jobs, 6 quant jobs, and 1 email job. "
        "Waiting for market hours... (Ctrl-C to stop)"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Shutdown requested — stopping scheduler.")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
