"""
Daily summary email notification — sent at 16:30 ET after market close.

Structure:
  1. LLM-generated paragraph (Haiku): plain-English summary + grade 1-10.
  2. Performance table: today / week / month / 6M / YTD / 1Y.
  3. Today's trades with P&L per trade.
  4. Open positions.
  5. LLM spend tracker (month-to-date vs $10 cap).

Delivery: Gmail SMTP with app password.
Set EMAIL_ENABLED=true, EMAIL_RECIPIENT, EMAIL_SENDER, EMAIL_APP_PASSWORD in .env.
"""

import logging
import smtplib
import sqlite3
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from zoneinfo import ZoneInfo

import anthropic

from config.settings import Settings
from db.store import get_monthly_llm_spend

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Haiku is cheap enough to use for email summaries (~$0.001/email)
_SUMMARY_MODEL = "claude-haiku-4-5-20251001"
_SUMMARY_MAX_TOKENS = 300


# ---------------------------------------------------------------------------
# Data collection from SQLite
# ---------------------------------------------------------------------------

def collect_daily_data(conn: sqlite3.Connection, date: str) -> dict:
    """
    Pull everything needed for today's email from SQLite.
    date: 'YYYY-MM-DD'
    """
    # Today's trades (entries only — have realized_pnl when closed)
    trades_today = conn.execute(
        """
        SELECT symbol, side, qty, fill_price, realized_pnl, notional
        FROM trades
        WHERE DATE(run_timestamp) = ?
        ORDER BY created_at
        """,
        (date,),
    ).fetchall()

    # Open positions proxy: bought today with no matching sell
    open_pos = conn.execute(
        """
        SELECT symbol, side, qty, fill_price, notional
        FROM trades
        WHERE DATE(run_timestamp) = ?
          AND side IN ('buy', 'sell_short')
          AND symbol NOT IN (
              SELECT symbol FROM trades
              WHERE DATE(run_timestamp) = ? AND side IN ('sell', 'buy_to_cover')
          )
        """,
        (date, date),
    ).fetchall()

    # Daily P&L
    daily_row = conn.execute(
        "SELECT realized_pnl, unrealized_pnl, total_pnl, open_equity FROM daily_pnl WHERE date = ?",
        (date,),
    ).fetchone()

    # Equity snapshots for period performance
    snapshots = conn.execute(
        "SELECT run_timestamp, equity FROM equity_snapshots ORDER BY run_timestamp",
    ).fetchall()

    # Sentiment biases used today
    biases = conn.execute(
        """
        SELECT ticker, bias, aggregated_score
        FROM sentiment_bias WHERE date = ?
        ORDER BY ticker
        """,
        (date,),
    ).fetchall()

    return {
        "date": date,
        "trades_today": [dict(r) for r in trades_today],
        "open_positions": [dict(r) for r in open_pos],
        "daily_pnl": dict(daily_row) if daily_row else {},
        "snapshots": [dict(r) for r in snapshots],
        "biases": [dict(r) for r in biases],
    }


def compute_period_returns(snapshots: list[dict], today_str: str) -> dict[str, Optional[float]]:
    """
    Compute returns for standard look-back periods.
    Returns {period_label: pct_return} where pct_return is None if insufficient data.
    """
    if not snapshots:
        return {}

    today_eq = snapshots[-1]["equity"]
    today_dt = datetime.strptime(today_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    def _equity_at_or_before(target_dt: datetime) -> Optional[float]:
        """Find the equity snapshot closest to (but not after) target_dt."""
        result = None
        for s in snapshots:
            ts = datetime.fromisoformat(s["run_timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts <= target_dt:
                result = s["equity"]
        return result

    periods = {
        "Today":   today_dt - timedelta(days=1),
        "Week":    today_dt - timedelta(weeks=1),
        "Month":   today_dt - timedelta(days=30),
        "6 Month": today_dt - timedelta(days=182),
        "YTD":     today_dt.replace(month=1, day=1),
        "1 Year":  today_dt - timedelta(days=365),
    }

    returns = {}
    for label, start_dt in periods.items():
        base_eq = _equity_at_or_before(start_dt)
        if base_eq and base_eq > 0:
            returns[label] = (today_eq - base_eq) / base_eq
        else:
            returns[label] = None

    return returns


# ---------------------------------------------------------------------------
# LLM summary generation
# ---------------------------------------------------------------------------

def generate_llm_summary(data: dict, settings: Settings) -> tuple[str, int]:
    """
    Use Haiku to write a plain-English day summary and assign a grade 1-10.

    Returns (summary_text, grade).
    Falls back to a generic message if the API call fails.
    """
    trades = data["trades_today"]
    daily = data["daily_pnl"]

    total_pnl = daily.get("total_pnl") or 0.0
    realized = daily.get("realized_pnl") or 0.0
    open_eq = daily.get("open_equity") or 10_000.0

    winning = [t for t in trades if (t.get("realized_pnl") or 0) > 0]
    losing  = [t for t in trades if (t.get("realized_pnl") or 0) < 0]

    trade_lines = "\n".join(
        f"  - {t['symbol']} ({t['side'].upper()}) realized P&L: "
        f"${t.get('realized_pnl') or 0:.2f}"
        for t in trades
    ) or "  No closed trades today."

    bias_lines = "\n".join(
        f"  - {b['ticker']}: {b['bias']} (score={b['aggregated_score']:.2f})"
        for b in data.get("biases", [])
    ) or "  No sentiment data."

    prompt = (
        f"You are summarizing a paper trading bot's performance for the day.\n\n"
        f"Date: {data['date']}\n"
        f"Total P&L: ${total_pnl:+.2f} ({total_pnl/open_eq*100:+.2f}% of equity)\n"
        f"Realized P&L: ${realized:+.2f}\n"
        f"Trades executed: {len(trades)} ({len(winning)} winners, {len(losing)} losers)\n\n"
        f"Trades:\n{trade_lines}\n\n"
        f"Sentiment biases used:\n{bias_lines}\n\n"
        f"Write 2-3 sentences summarizing the day in plain English. "
        f"Be concise and direct. Then on a new line write exactly: "
        f'GRADE: X/10 where X is your assessment (1=terrible, 10=perfect). '
        f"Consider both P&L and trade quality."
    )

    try:
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        resp = client.messages.create(
            model=_SUMMARY_MODEL,
            max_tokens=_SUMMARY_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()

        # Parse grade from "GRADE: X/10"
        grade = 5  # default
        for line in raw.splitlines():
            if line.startswith("GRADE:"):
                try:
                    grade = int(line.split(":")[1].strip().split("/")[0])
                    grade = max(1, min(10, grade))
                except (ValueError, IndexError):
                    pass

        # Remove the grade line from the summary text
        summary = "\n".join(
            l for l in raw.splitlines() if not l.startswith("GRADE:")
        ).strip()

        return summary, grade

    except Exception as exc:
        log.error("LLM summary generation failed: %s", exc)
        pnl_word = "profitable" if total_pnl >= 0 else "challenging"
        return (
            f"Today was a {pnl_word} session with {len(trades)} trades executed "
            f"and a total P&L of ${total_pnl:+.2f}.",
            5,
        )


# ---------------------------------------------------------------------------
# HTML email builder
# ---------------------------------------------------------------------------

def _colour(value: float) -> str:
    return "#27ae60" if value >= 0 else "#e74c3c"


def _grade_stars(grade: int) -> str:
    filled = "★" * grade
    empty  = "☆" * (10 - grade)
    return filled + empty


def build_html_email(
    data: dict,
    period_returns: dict,
    summary: str,
    grade: int,
    mtd_spend: float,
    budget: float,
) -> str:
    date_str = data["date"]
    daily = data["daily_pnl"]
    total_pnl = daily.get("total_pnl") or 0.0
    realized  = daily.get("realized_pnl") or 0.0
    open_eq   = daily.get("open_equity") or 10_000.0
    total_pct = total_pnl / open_eq * 100 if open_eq else 0.0

    subject_line = (
        f"TradingBot — {date_str} | {total_pnl:+.2f} ({total_pct:+.1f}%) | "
        f"Grade {grade}/10"
    )

    # --- Performance table rows ---
    perf_rows = ""
    for label, ret in period_returns.items():
        if ret is None:
            val_html = '<td style="color:#888;text-align:right">—</td>'
        else:
            colour = _colour(ret)
            val_html = (
                f'<td style="color:{colour};text-align:right;font-weight:bold">'
                f'{ret*100:+.2f}%</td>'
            )
        perf_rows += f"<tr><td>{label}</td>{val_html}</tr>\n"

    # --- Trades table rows ---
    trade_rows = ""
    for t in data["trades_today"]:
        pnl = t.get("realized_pnl") or 0.0
        pnl_pct = pnl / t["notional"] * 100 if t["notional"] else 0.0
        colour = _colour(pnl)
        side_badge = (
            '<span style="background:#27ae60;color:white;padding:1px 5px;border-radius:3px">LONG</span>'
            if t["side"] == "buy" else
            '<span style="background:#e74c3c;color:white;padding:1px 5px;border-radius:3px">SHORT</span>'
        )
        trade_rows += (
            f"<tr>"
            f"<td><b>{t['symbol']}</b></td>"
            f"<td>{side_badge}</td>"
            f"<td style='text-align:right'>{t['qty']:.0f}</td>"
            f"<td style='text-align:right'>${t['fill_price']:.2f}</td>"
            f"<td style='color:{colour};text-align:right'>${pnl:+.2f} ({pnl_pct:+.1f}%)</td>"
            f"</tr>\n"
        )
    if not trade_rows:
        trade_rows = '<tr><td colspan="5" style="color:#888;text-align:center">No closed trades today</td></tr>'

    # --- Open positions ---
    pos_rows = ""
    for p in data["open_positions"]:
        pos_rows += (
            f"<tr><td><b>{p['symbol']}</b></td>"
            f"<td>{p['side'].upper()}</td>"
            f"<td style='text-align:right'>{p['qty']:.0f} @ ${p['fill_price']:.2f}</td>"
            f"</tr>\n"
        )
    if not pos_rows:
        pos_rows = '<tr><td colspan="3" style="color:#888;text-align:center">No open positions</td></tr>'

    # --- LLM budget bar ---
    spend_pct = min(mtd_spend / budget * 100, 100)
    bar_colour = "#e74c3c" if spend_pct > 83 else "#f39c12" if spend_pct > 60 else "#27ae60"
    budget_bar = (
        f'<div style="background:#eee;border-radius:4px;height:8px;width:200px;display:inline-block">'
        f'<div style="background:{bar_colour};width:{spend_pct:.0f}%;height:8px;border-radius:4px"></div>'
        f'</div>'
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;color:#222;padding:20px">

  <!-- Header -->
  <div style="border-bottom:3px solid #2c3e50;padding-bottom:12px;margin-bottom:20px">
    <h2 style="margin:0;color:#2c3e50">TradingBot Daily Summary</h2>
    <p style="margin:4px 0;color:#666;font-size:13px">{date_str}</p>
    <div style="font-size:28px;font-weight:bold;color:{_colour(total_pnl)}">
      ${total_pnl:+,.2f}
      <span style="font-size:16px;font-weight:normal">({total_pct:+.2f}%)</span>
    </div>
  </div>

  <!-- LLM Summary -->
  <div style="background:#f8f9fa;border-left:4px solid #2c3e50;padding:12px 16px;margin-bottom:20px;border-radius:0 4px 4px 0">
    <p style="margin:0 0 8px 0;font-size:15px;line-height:1.6">{summary}</p>
    <div style="font-size:18px;color:#f39c12">
      {_grade_stars(grade)}
      <span style="font-size:14px;color:#666;margin-left:8px">{grade}/10</span>
    </div>
  </div>

  <!-- Performance Table -->
  <h3 style="color:#2c3e50;margin-bottom:8px">Performance</h3>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:14px">
    <thead>
      <tr style="background:#2c3e50;color:white">
        <th style="padding:8px;text-align:left">Period</th>
        <th style="padding:8px;text-align:right">Return</th>
      </tr>
    </thead>
    <tbody>
{perf_rows}
    </tbody>
  </table>

  <!-- Trades Today -->
  <h3 style="color:#2c3e50;margin-bottom:8px">Today's Trades ({len(data['trades_today'])})</h3>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:13px">
    <thead>
      <tr style="background:#ecf0f1">
        <th style="padding:6px;text-align:left">Ticker</th>
        <th style="padding:6px;text-align:left">Side</th>
        <th style="padding:6px;text-align:right">Qty</th>
        <th style="padding:6px;text-align:right">Fill</th>
        <th style="padding:6px;text-align:right">P&amp;L</th>
      </tr>
    </thead>
    <tbody>
{trade_rows}
    </tbody>
  </table>

  <!-- Open Positions -->
  <h3 style="color:#2c3e50;margin-bottom:8px">Open Positions ({len(data['open_positions'])})</h3>
  <table style="width:100%;border-collapse:collapse;margin-bottom:24px;font-size:13px">
    <thead>
      <tr style="background:#ecf0f1">
        <th style="padding:6px;text-align:left">Ticker</th>
        <th style="padding:6px;text-align:left">Side</th>
        <th style="padding:6px;text-align:right">Position</th>
      </tr>
    </thead>
    <tbody>
{pos_rows}
    </tbody>
  </table>

  <!-- LLM Spend -->
  <div style="border-top:1px solid #eee;padding-top:12px;font-size:12px;color:#888">
    <span>LLM spend MTD: <b style="color:#333">${mtd_spend:.2f}</b> / ${budget:.2f}</span>
    &nbsp;&nbsp;{budget_bar}
  </div>

  <p style="font-size:11px;color:#bbb;margin-top:16px">
    Sent by TradingBot — paper trading only.
    All figures are simulated and do not represent real money.
  </p>

</body>
</html>"""

    return subject_line, html


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def send_daily_email(
    conn: sqlite3.Connection,
    settings: Settings,
    date: str,
) -> bool:
    """
    Collect data, generate LLM summary, build and send the daily email.

    Returns True on success, False on failure.
    date: 'YYYY-MM-DD' string for the trading day being summarised.
    """
    if not settings.email_enabled:
        log.info("Email disabled — skipping daily notification")
        return False

    if not all([settings.email_recipient, settings.email_sender, settings.email_app_password]):
        log.warning("Email settings incomplete — set EMAIL_RECIPIENT, EMAIL_SENDER, EMAIL_APP_PASSWORD")
        return False

    log.info("Building daily summary email for %s...", date)

    data = collect_daily_data(conn, date)
    period_returns = compute_period_returns(data["snapshots"], date)
    summary, grade = generate_llm_summary(data, settings)

    year_month = date[:7]  # 'YYYY-MM'
    mtd_spend = get_monthly_llm_spend(conn, year_month)

    subject, html_body = build_html_email(
        data=data,
        period_returns=period_returns,
        summary=summary,
        grade=grade,
        mtd_spend=mtd_spend,
        budget=settings.llm_budget_monthly_usd,
    )

    return _send_via_gmail(
        sender=settings.email_sender,
        password=settings.email_app_password,
        recipient=settings.email_recipient,
        subject=subject,
        html_body=html_body,
        conn=conn,
        kind="daily_summary",
    )


def send_circuit_breaker_alert(
    settings: Settings,
    daily_pnl_pct: float,
    equity: float,
    conn: Optional[sqlite3.Connection] = None,
) -> bool:
    """
    Send an immediate email when the daily circuit breaker fires.

    This is separate from the end-of-day summary — it fires at the moment
    of liquidation so the operator knows immediately, not 6 hours later.
    """
    if not settings.email_enabled:
        return False
    if not all([settings.email_recipient, settings.email_sender, settings.email_app_password]):
        return False

    now_et = datetime.now(timezone.utc).astimezone(_ET)
    time_str = now_et.strftime("%Y-%m-%d %H:%M ET")

    subject = f"[TradingBot] CIRCUIT BREAKER FIRED — {time_str}"

    html_body = f"""
<html>
<body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
  <h2 style="color: #cc0000; border-bottom: 2px solid #cc0000; padding-bottom: 8px;">
    Daily Circuit Breaker Triggered
  </h2>
  <p>The daily loss limit was breached at <strong>{time_str}</strong>.</p>
  <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
    <tr style="background: #fff0f0;">
      <td style="padding: 8px 12px; border: 1px solid #eee;"><strong>Daily P&amp;L</strong></td>
      <td style="padding: 8px 12px; border: 1px solid #eee; color: #cc0000; font-weight: bold;">
        {daily_pnl_pct*100:+.2f}%
      </td>
    </tr>
    <tr>
      <td style="padding: 8px 12px; border: 1px solid #eee;"><strong>Threshold</strong></td>
      <td style="padding: 8px 12px; border: 1px solid #eee;">-3.00%</td>
    </tr>
    <tr style="background: #f9f9f9;">
      <td style="padding: 8px 12px; border: 1px solid #eee;"><strong>Current Equity</strong></td>
      <td style="padding: 8px 12px; border: 1px solid #eee;">${equity:,.2f}</td>
    </tr>
  </table>
  <p>All positions have been liquidated. No new entries will be taken today.</p>
  <p style="color: #666; font-size: 13px;">
    The bot will resume normal operation at the next trading day's 10:30 ET scan.
  </p>
</body>
</html>"""

    ok = _send_via_gmail(
        settings.email_sender,
        settings.email_app_password,
        settings.email_recipient,
        subject,
        html_body,
        conn=conn,
        kind="circuit_breaker",
    )
    if ok:
        log.info("Circuit breaker alert sent to %s", settings.email_recipient)
    return ok


def _log_email(
    conn: Optional[sqlite3.Connection],
    kind: str,
    recipient: str,
    subject: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    if conn is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        conn.execute(
            """
            INSERT INTO email_log (sent_at, kind, recipient, subject, status, error)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (now, kind, recipient, subject, status, error),
        )
        conn.commit()
    except Exception as exc:
        log.warning("Failed to write email_log: %s", exc)


def _send_via_gmail(
    sender: str,
    password: str,
    recipient: str,
    subject: str,
    html_body: str,
    conn: Optional[sqlite3.Connection] = None,
    kind: str = "daily_summary",
) -> bool:
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        log.info("Daily summary email sent to %s", recipient)
        _log_email(conn, kind, recipient, subject, "sent")
        return True
    except Exception as exc:
        log.error("Failed to send email: %s", exc)
        _log_email(conn, kind, recipient, subject, "failed", str(exc))
        return False
