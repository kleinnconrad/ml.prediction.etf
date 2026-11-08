"""
Entry point of the ETF predictor: one pooled cross-sectional model for all target ETFs.

Downloads the prices of the target ETFs (config/batch_targets.json), the market index and
the macro factors, runs the pooled model (src/modeling.py) and writes the batch report
artifacts/latest_batch_results.json. The workflow `Pipeline: 2. Run Batch` runs this
module and commits the report.

Usage (from the repository root):
    uv run python src/main.py [--ticker TICKER ...] [--price-cache DIR] [--output FILE]
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from config import BATCH_RESULTS_PATH, load_settings, load_universe
from modeling import ModelResult, etf_prediction, run_pooled_model
from prices import PricePanel, fetch_price_panel, load_price_panel, save_price_panel
from reporting import build_batch_report, configure_logging, write_json_atomic

logger = logging.getLogger(__name__)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parses the command line.

    Args:
        argv (Optional[Sequence[str]]): Arguments (default: sys.argv[1:]).

    Returns:
        argparse.Namespace: ticker, price_cache and output.
    """
    parser = argparse.ArgumentParser(description="Pooled cross-sectional ranking of the target ETFs.")
    parser.add_argument('--ticker', nargs='+', default=[],
                        help="tickers to log in detail; tickers outside the target list are added to the ranking")
    parser.add_argument('--price-cache', type=Path, default=None,
                        help="directory of a CSV price cache: read if it exists, otherwise downloaded and written")
    parser.add_argument('--output', type=Path, default=BATCH_RESULTS_PATH,
                        help=f"batch report file (default: {BATCH_RESULTS_PATH.relative_to(BATCH_RESULTS_PATH.parents[1])})")
    return parser.parse_args(argv)


def load_prices(tickers: Sequence[str], history_years: int, settings: dict,
                cache: Optional[Path] = None) -> PricePanel:
    """
    Reads the prices from a cache, or downloads them (and fills the cache if one is given).

    Args:
        tickers (Sequence[str]): ETF tickers to rank.
        history_years (int): Years of history to download.
        settings (dict): Loaded settings.
        cache (Optional[Path]): Price cache directory.

    Returns:
        PricePanel: Prices of the ETFs, the market index and the macro factors.
    """
    if cache is not None and (cache / 'close.csv').exists():
        panel = load_price_panel(cache)
        unknown = sorted(set(tickers) - set(panel.close.columns) - set(panel.missing))
        if unknown:
            logger.warning(f"The price cache {cache} has no entry for {len(unknown)} tickers "
                           f"(delete it to download again): {unknown}")
        return panel
    panel = fetch_price_panel(tickers, history_years, settings)
    if cache is not None:
        save_price_panel(panel, cache)
    return panel


def log_ticker_details(result: ModelResult, tickers: Sequence[str]) -> None:
    """
    Logs the rank, probability and feature contributions of selected ETFs.

    Args:
        result (ModelResult): The pooled model result.
        tickers (Sequence[str]): Tickers to describe.
    """
    for ticker in tickers:
        if ticker not in result.results.index:
            logger.warning(f"{ticker}: no price data; not ranked.")
            continue
        entry = etf_prediction(result, ticker)
        logger.info(f"{ticker}: {entry['prediction_label']} (status {entry['status']}), rank {entry['rank']} of "
                    f"{entry['n_ranked']}, probability {entry['probability']}, top {entry['top_percent']}%")
        for name, contribution in (entry['logit_contributions'] or {}).items():
            logger.info(f"  {name}: percentile {entry['feature_percentiles'][name]}, "
                        f"log-odds contribution {contribution}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Runs the pooled model on the target ETFs and writes the batch report.

    Args:
        argv (Optional[Sequence[str]]): Command line arguments.

    Returns:
        int: Exit code 0.
    """
    configure_logging()
    args = parse_args(argv)
    settings = load_settings()
    universe: List[str] = list(dict.fromkeys(load_universe() + list(args.ticker)))
    history_years = int(settings['model_history_years']) + int(settings['feature_warmup_years'])
    logger.info(f"Pooled model run: {len(universe)} ETFs, {history_years} years of prices.")
    prices = load_prices(universe, history_years, settings, args.price_cache)
    result = run_pooled_model(prices, settings)
    report = build_batch_report(result, settings)
    write_json_atomic(args.output, report)
    logger.info(f"Wrote {args.output}: {len(report['results'])} ETFs, {len(result.up_tickers)} Up "
                f"(gate {'passed' if result.summary['gate']['passed'] else 'failed'}).")
    log_ticker_details(result, args.ticker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
