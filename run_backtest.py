"""
Standalone backtest runner — Phase 1 quant-only strategy.

Usage:
    python run_backtest.py
    python run_backtest.py --slippage-bps 15
    python run_backtest.py --start 2024-01-01 --end 2024-12-31
    python run_backtest.py --tickers AAPL MSFT NVDA

The script fetches historical bars, runs the backtest harness, saves a JSON
report to reports/, and prints a human-readable summary. The report must show
passed_gate=True before main.py will start paper trading.
"""

import argparse
import logging
import sys
from datetime import datetime, timezone

from config.settings import Settings
from backtest.harness import BacktestHarness, check_backtest_gate
from backtest.loader import load_bars_for_backtest
from data.market import load_watchlist

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 1 quant-only backtest and save a report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--slippage-bps",
        type=int,
        default=None,
        metavar="N",
        help="Slippage in basis points per fill (default: from settings)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Backtest start date (default: earliest available data)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Backtest end date (default: today)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        metavar="SYM",
        help="Override watchlist tickers (default: all from config/watchlist.yaml)",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Starting equity for backtest (default: 10000)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()
    settings = Settings()

    # Resolve tickers
    if args.tickers:
        symbols = args.tickers
        log.info("Using custom tickers: %s", symbols)
    else:
        symbols = load_watchlist(settings.watchlist_path)
        log.info("Using watchlist: %d tickers", len(symbols))

    # Parse date range
    start_date = (
        datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.start else None
    )
    end_date = (
        datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end else None
    )

    # Load historical bars
    log.info("Loading historical bars...")
    bars = load_bars_for_backtest(
        settings=settings,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )
    loaded = [s for s, df in bars.items() if not df.empty]
    log.info("Loaded bars for %d/%d tickers", len(loaded), len(symbols))

    if not loaded:
        log.error("No bars loaded — check API credentials and ticker list")
        sys.exit(1)

    # Run backtest
    log.info("Running backtest (slippage=%s bps)...",
             args.slippage_bps if args.slippage_bps is not None else f"{settings.backtest_slippage_bps} (default)")

    harness = BacktestHarness(
        settings=settings,
        bars=bars,
        slippage_bps=args.slippage_bps,
        initial_equity=args.initial_equity,
    )
    trades, equity_curve, metrics = harness.run()

    # Save report
    report_path = harness.save_report(trades, metrics)

    # Print summary
    _print_summary(metrics, report_path, len(trades))

    # Exit code: 0 if gate passed, 1 if not
    if not metrics.passed_gate:
        log.warning(
            "Gate FAILED — strategy does not meet positive-expectancy threshold.\n"
            "main.py will not start paper trading until a passing report exists."
        )
        sys.exit(1)

    log.info("Gate PASSED — ready to start paper trading with main.py")


def _print_summary(metrics, report_path: str, trade_count: int) -> None:
    gate_str = "PASS ✓" if metrics.passed_gate else "FAIL ✗"
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS — {gate_str}")
    print("=" * 60)
    print(f"  Trades:           {trade_count}")
    print(f"  Win rate:         {metrics.win_rate * 100:.1f}%")
    print(f"  Expectancy:       {metrics.expectancy_pct * 100:.2f}% per trade")
    print(f"  Sharpe ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown:     {metrics.max_drawdown_pct * 100:.1f}%")
    print(f"  Annualized return:{metrics.annualized_return_pct * 100:.1f}%")
    print(f"  Gate:             {gate_str}")
    print(f"  Report saved to:  {report_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
