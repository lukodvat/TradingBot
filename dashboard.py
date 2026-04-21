"""
TradingBot Streamlit Dashboard — read-only view of the SQLite database.

Run with:
    streamlit run dashboard.py --server.port 8501

Sections:
  1. Live Stats — equity, cash, daily P&L, open positions
  2. Equity Curve — interactive Plotly line chart
  3. Drawdown — computed from equity snapshots
  4. Performance Table — daily / week / month / 6M / YTD / 1Y
  5. Today's Trades
  6. All Trades (filterable by ticker, date, side)
  7. Win / Loss Distribution — P&L histogram
  8. Open Positions (inferred from SQLite)
  9. Sentiment Biases (today + historical heatmap)
  10. LLM Cost Tracker — spend by day, by model, monthly total vs cap
  11. Volatility Filter Log
  12. Session Log
  13. Headlines + Triage Scores
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import Settings

_ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TradingBot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# DB connection (cached, shared across reruns)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_db() -> sqlite3.Connection:
    settings = Settings()
    conn = sqlite3.connect(settings.db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_resource
def get_settings() -> Settings:
    return Settings()


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = get_db()
    return pd.read_sql_query(sql, conn, params=params)


# ---------------------------------------------------------------------------
# Sidebar — controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📈 TradingBot")
    st.caption("Paper trading dashboard — read only")
    st.divider()

    auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time
        st.caption(f"Last refresh: {datetime.now(_ET).strftime('%H:%M:%S ET')}")

    st.divider()
    date_filter = st.date_input(
        "Filter trades from",
        value=datetime.now(_ET).date() - timedelta(days=30),
    )
    st.divider()
    st.caption("Data source: SQLite")

# ---------------------------------------------------------------------------
# 1. Live Stats
# ---------------------------------------------------------------------------

st.title("TradingBot Dashboard")

today_str = datetime.now(_ET).strftime("%Y-%m-%d")

# Latest equity snapshot
snap_df = query(
    "SELECT equity, cash, portfolio_value, open_positions, daily_pnl, run_timestamp "
    "FROM equity_snapshots ORDER BY run_timestamp DESC LIMIT 1"
)

if not snap_df.empty:
    snap = snap_df.iloc[0]
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Equity", f"${snap['equity']:,.2f}")
    col2.metric("Cash", f"${snap['cash']:,.2f}")
    col3.metric(
        "Daily P&L",
        f"${snap['daily_pnl']:+,.2f}",
        delta=f"{snap['daily_pnl'] / snap['equity'] * 100:+.2f}%" if snap['equity'] else None,
    )
    col4.metric("Open Positions", int(snap["open_positions"]))
    col5.metric(
        "Last Update",
        pd.to_datetime(snap["run_timestamp"]).strftime("%H:%M ET")
        if snap["run_timestamp"] else "—",
    )
else:
    st.info("No equity snapshots yet. Run the bot during market hours to populate data.")

st.divider()

# ---------------------------------------------------------------------------
# 2 & 3. Equity Curve + Drawdown
# ---------------------------------------------------------------------------

col_eq, col_dd = st.columns(2)

equity_df = query(
    "SELECT run_timestamp, equity, session FROM equity_snapshots ORDER BY run_timestamp"
)

with col_eq:
    st.subheader("Equity Curve")
    if not equity_df.empty:
        equity_df["run_timestamp"] = pd.to_datetime(equity_df["run_timestamp"])
        fig = px.line(
            equity_df, x="run_timestamp", y="equity",
            hover_data=["session"],
            labels={"run_timestamp": "Time", "equity": "Equity ($)"},
            color_discrete_sequence=["#2c3e50"],
        )
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No equity data yet.")

with col_dd:
    st.subheader("Drawdown from Peak")
    if not equity_df.empty:
        eq = equity_df["equity"]
        peak = eq.cummax()
        drawdown = (eq - peak) / peak * 100
        dd_df = equity_df.copy()
        dd_df["drawdown_pct"] = drawdown
        fig2 = px.area(
            dd_df, x="run_timestamp", y="drawdown_pct",
            labels={"run_timestamp": "Time", "drawdown_pct": "Drawdown (%)"},
            color_discrete_sequence=["#e74c3c"],
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No equity data yet.")

# ---------------------------------------------------------------------------
# 4. Performance Table
# ---------------------------------------------------------------------------

st.subheader("Performance")

if not equity_df.empty:
    latest_eq = float(equity_df["equity"].iloc[-1])

    def _ret(days: Optional[int] = None, year_start: bool = False) -> Optional[str]:
        now = equity_df["run_timestamp"].iloc[-1]
        if year_start:
            cutoff = now.replace(month=1, day=1, hour=0, minute=0)
        elif days:
            cutoff = now - pd.Timedelta(days=days)
        else:
            return None
        subset = equity_df[equity_df["run_timestamp"] <= cutoff]
        if subset.empty:
            return "—"
        base = float(subset["equity"].iloc[-1])
        ret = (latest_eq - base) / base * 100
        colour = "green" if ret >= 0 else "red"
        return f'<span style="color:{colour}">{ret:+.2f}%</span>'

    periods = {
        "Today": _ret(1),
        "1 Week": _ret(7),
        "1 Month": _ret(30),
        "3 Month": _ret(90),
        "6 Month": _ret(182),
        "YTD": _ret(year_start=True),
        "1 Year": _ret(365),
    }
    perf_html = "<table style='width:100%;border-collapse:collapse'>"
    perf_html += "<tr style='background:#ecf0f1'>"
    for label in periods:
        perf_html += f"<th style='padding:8px;text-align:center'>{label}</th>"
    perf_html += "</tr><tr>"
    for val in periods.values():
        perf_html += f"<td style='padding:8px;text-align:center;font-weight:bold'>{val or '—'}</td>"
    perf_html += "</tr></table>"
    st.markdown(perf_html, unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# 5 & 6. Trades
# ---------------------------------------------------------------------------

st.subheader("Trades")

tab_today, tab_all, tab_hist = st.tabs(["Today", "All Trades", "Win/Loss Distribution"])

trades_all = query(
    f"""
    SELECT symbol, side, qty, fill_price, stop_price, notional,
           realized_pnl, slippage_bps, session, run_timestamp, filled_at
    FROM trades
    WHERE DATE(run_timestamp) >= ?
    ORDER BY run_timestamp DESC
    """,
    (str(date_filter),),
)

with tab_today:
    trades_today = trades_all[
        pd.to_datetime(trades_all["run_timestamp"]).dt.strftime("%Y-%m-%d") == today_str
    ] if not trades_all.empty else pd.DataFrame()

    if not trades_today.empty:
        display = trades_today[["symbol", "side", "qty", "fill_price", "realized_pnl", "slippage_bps", "session"]].copy()
        display.columns = ["Ticker", "Side", "Qty", "Fill $", "Realized P&L", "Slippage bps", "Session"]
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No trades today.")

with tab_all:
    if not trades_all.empty:
        # Filters
        col_f1, col_f2 = st.columns(2)
        tickers = ["All"] + sorted(trades_all["symbol"].unique().tolist())
        sides = ["All", "buy", "sell", "sell_short", "buy_to_cover"]
        sel_ticker = col_f1.selectbox("Ticker", tickers)
        sel_side = col_f2.selectbox("Side", sides)

        filtered = trades_all.copy()
        if sel_ticker != "All":
            filtered = filtered[filtered["symbol"] == sel_ticker]
        if sel_side != "All":
            filtered = filtered[filtered["side"] == sel_side]

        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.caption(f"{len(filtered)} rows")
    else:
        st.info("No trades in selected date range.")

with tab_hist:
    closed = trades_all[trades_all["realized_pnl"].notna()] if not trades_all.empty else pd.DataFrame()
    if not closed.empty:
        fig3 = px.histogram(
            closed,
            x="realized_pnl",
            nbins=30,
            color_discrete_sequence=["#2c3e50"],
            labels={"realized_pnl": "Realized P&L ($)"},
            title="Trade P&L Distribution",
        )
        fig3.add_vline(x=0, line_dash="dash", line_color="red")
        fig3.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)

        win_rate = (closed["realized_pnl"] > 0).mean() * 100
        avg_win = closed[closed["realized_pnl"] > 0]["realized_pnl"].mean()
        avg_loss = closed[closed["realized_pnl"] < 0]["realized_pnl"].mean()
        m1, m2, m3 = st.columns(3)
        m1.metric("Win Rate", f"{win_rate:.1f}%")
        m2.metric("Avg Win", f"${avg_win:+.2f}" if pd.notna(avg_win) else "—")
        m3.metric("Avg Loss", f"${avg_loss:+.2f}" if pd.notna(avg_loss) else "—")
    else:
        st.info("No closed trades yet.")

st.divider()

# ---------------------------------------------------------------------------
# 7. Open Positions (inferred from SQLite)
# ---------------------------------------------------------------------------

st.subheader("Open Positions (inferred)")

open_pos = query(
    """
    SELECT t.symbol, t.side, t.qty, t.fill_price, t.notional, t.run_timestamp
    FROM trades t
    WHERE t.side IN ('buy', 'sell_short')
      AND t.symbol NOT IN (
          SELECT symbol FROM trades
          WHERE side IN ('sell', 'buy_to_cover')
            AND DATE(run_timestamp) = DATE(t.run_timestamp)
      )
    ORDER BY t.run_timestamp DESC
    """
)
if not open_pos.empty:
    st.dataframe(open_pos, use_container_width=True, hide_index=True)
else:
    st.info("No open positions found in trade history.")

st.divider()

# ---------------------------------------------------------------------------
# 8. Sentiment Biases
# ---------------------------------------------------------------------------

st.subheader("Sentiment Biases")

tab_bias_today, tab_bias_hist = st.tabs(["Today", "Historical Heatmap"])

with tab_bias_today:
    biases_today = query(
        "SELECT ticker, bias, aggregated_score, llm_run, updated_at "
        "FROM sentiment_bias WHERE date = ? ORDER BY ticker",
        (today_str,),
    )
    if not biases_today.empty:
        def _bias_color(b):
            return "background-color: #d5f5e3" if b == "BULLISH" else \
                   "background-color: #fadbd8" if b == "BEARISH" else ""
        st.dataframe(
            biases_today.style.map(_bias_color, subset=["bias"]),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No sentiment biases for today yet.")

with tab_bias_hist:
    biases_hist = query(
        """
        SELECT date, ticker, bias FROM sentiment_bias
        WHERE date >= ?
        ORDER BY date, ticker
        """,
        (str(date_filter),),
    )
    if not biases_hist.empty:
        pivot = biases_hist.pivot_table(
            index="date", columns="ticker", values="bias", aggfunc="last"
        ).fillna("—")
        bias_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1, "—": None}
        numeric = pivot.map(lambda x: bias_map.get(x, 0))
        fig4 = px.imshow(
            numeric.T,
            color_continuous_scale=["#e74c3c", "#f5f5f5", "#27ae60"],
            zmin=-1, zmax=1,
            aspect="auto",
            labels={"color": "Bias"},
        )
        fig4.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No historical bias data.")

st.divider()

# ---------------------------------------------------------------------------
# 9. LLM Cost Tracker
# ---------------------------------------------------------------------------

st.subheader("LLM Cost Tracker")

settings = get_settings()
year_month = datetime.now(_ET).strftime("%Y-%m")

llm_df = query(
    """
    SELECT model, tier, symbol, cost_usd, prompt_tokens, completion_tokens,
           run_timestamp, sentiment, confidence
    FROM llm_calls
    WHERE strftime('%Y-%m', run_timestamp) = ?
    ORDER BY run_timestamp DESC
    """,
    (year_month,),
)

if not llm_df.empty:
    mtd_spend = llm_df["cost_usd"].sum()
    budget = settings.llm_budget_monthly_usd
    pct = mtd_spend / budget * 100

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("MTD Spend", f"${mtd_spend:.4f}")
    col_s2.metric("Budget", f"${budget:.2f}")
    col_s3.metric("Used", f"{pct:.1f}%")

    st.progress(min(pct / 100, 1.0))

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        by_model = llm_df.groupby("model")["cost_usd"].sum().reset_index()
        fig5 = px.bar(by_model, x="model", y="cost_usd",
                      labels={"cost_usd": "Cost ($)", "model": "Model"},
                      color_discrete_sequence=["#2c3e50", "#7f8c8d"])
        fig5.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig5, use_container_width=True)

    with col_c2:
        llm_df["date"] = pd.to_datetime(llm_df["run_timestamp"]).dt.date
        by_day = llm_df.groupby("date")["cost_usd"].sum().reset_index()
        fig6 = px.bar(by_day, x="date", y="cost_usd",
                      labels={"cost_usd": "Cost ($)", "date": "Date"},
                      color_discrete_sequence=["#3498db"])
        fig6.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig6, use_container_width=True)

    with st.expander("All LLM calls this month"):
        st.dataframe(llm_df, use_container_width=True, hide_index=True)
else:
    st.info("No LLM calls recorded this month.")

st.divider()

# ---------------------------------------------------------------------------
# 10. Volatility Filter Log
# ---------------------------------------------------------------------------

st.subheader("Volatility Filter Log")

vol_df = query(
    """
    SELECT symbol, passed, atr_price_ratio, realized_vol,
           atr_threshold, vol_threshold, fail_reason, run_timestamp
    FROM volatility_filter_log
    ORDER BY run_timestamp DESC
    LIMIT 200
    """,
)
if not vol_df.empty:
    passed_count = vol_df["passed"].sum()
    st.caption(f"Last 200 entries — {passed_count} passed, {len(vol_df) - passed_count} filtered out")

    def _pass_color(v):
        return "background-color: #d5f5e3" if v == 1 else "background-color: #fadbd8"

    st.dataframe(
        vol_df.style.map(_pass_color, subset=["passed"]),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No volatility filter records yet.")

st.divider()

# ---------------------------------------------------------------------------
# 11. Session Log
# ---------------------------------------------------------------------------

st.subheader("Session Log")

session_df = query(
    """
    SELECT session, run_timestamp, tickers_evaluated, tier1_passes AS tier1_calls, tier2_calls,
           tier3_calls, orders_submitted, circuit_breaker_triggered
    FROM session_log
    ORDER BY run_timestamp DESC
    LIMIT 50
    """,
)
if not session_df.empty:
    st.dataframe(session_df, use_container_width=True, hide_index=True)
else:
    st.info("No session records yet.")

st.divider()

# ---------------------------------------------------------------------------
# 12. Headlines + Triage Scores
# ---------------------------------------------------------------------------

st.subheader("Headlines & Triage Scores")

headlines_df = query(
    """
    SELECT symbol, headline, source, published_at,
           tier1_pass, tier1_reason,
           tier2_sentiment, tier2_confidence,
           tier3_assessment, fetched_at
    FROM headlines_seen
    ORDER BY fetched_at DESC
    LIMIT 200
    """,
)
if not headlines_df.empty:
    col_h1, col_h2 = st.columns(2)
    ticker_filter = col_h1.selectbox(
        "Filter by ticker",
        ["All"] + sorted(headlines_df["symbol"].unique().tolist()),
        key="headlines_ticker",
    )
    tier_filter = col_h2.selectbox(
        "Filter by tier reached",
        ["All", "Tier 1 pass", "Tier 2 scored", "Tier 3 assessed"],
        key="headlines_tier",
    )

    filtered_h = headlines_df.copy()
    if ticker_filter != "All":
        filtered_h = filtered_h[filtered_h["symbol"] == ticker_filter]
    if tier_filter == "Tier 1 pass":
        filtered_h = filtered_h[filtered_h["tier1_pass"] == 1]
    elif tier_filter == "Tier 2 scored":
        filtered_h = filtered_h[filtered_h["tier2_sentiment"].notna()]
    elif tier_filter == "Tier 3 assessed":
        filtered_h = filtered_h[filtered_h["tier3_assessment"].notna()]

    st.dataframe(
        filtered_h[["symbol", "headline", "source", "tier1_pass",
                     "tier2_sentiment", "tier2_confidence", "fetched_at"]],
        use_container_width=True, hide_index=True,
    )
    st.caption(f"{len(filtered_h)} headlines shown")
else:
    st.info("No headlines fetched yet.")

st.divider()

# ---------------------------------------------------------------------------
# 13. Email Log
# ---------------------------------------------------------------------------

st.subheader("Email Log")

email_log_df = query(
    """
    SELECT sent_at, kind, recipient, subject, status, error
    FROM email_log
    ORDER BY sent_at DESC
    LIMIT 100
    """,
)
if not email_log_df.empty:
    sent_count = (email_log_df["status"] == "sent").sum()
    failed_count = (email_log_df["status"] == "failed").sum()
    st.caption(f"Last 100 attempts — {sent_count} sent, {failed_count} failed")

    def _status_color(v):
        return "background-color: #d5f5e3" if v == "sent" else "background-color: #fadbd8"

    st.dataframe(
        email_log_df.style.map(_status_color, subset=["status"]),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("No emails sent yet.")

# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    import time
    time.sleep(30)
    st.rerun()
