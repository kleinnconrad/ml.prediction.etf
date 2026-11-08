"""
Uplift of the published predictions: the realized returns of archived batch reports.

Every batch run archives a compact copy of its report as history/YYYY-MM/report_YYYY-MM-DD.json
next to the batch report (archive_batch_report()). For the archived reports closest to the
lookbacks of `uplift_lookback_days` (by default 1, 3 and 6 months ago, within
`uplift_match_tolerance_days`), evaluate_uplift() measures the return of every ranked ETF
from its first close on or after the report date to the latest close and averages it per
cohort: `Up`, `Not Up` and all ranked ETFs (`baseline`). ETFs that were not ranked in the
report are left out. The result, uplift_report.json, feeds the uplift charts of the dashboard.

Run from the repository root to refresh the uplift report without a batch run:
    uv run python src/uplift.py
"""
import argparse
import json
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, load_settings
from modeling import CLASS_NOT_UP, CLASS_UP

logger = logging.getLogger(__name__)

HISTORY_DIR = ARTIFACTS_DIR / 'history'
UPLIFT_PATH = ARTIFACTS_DIR / 'uplift_report.json'
BASELINE = 'baseline'
COHORTS = (BASELINE, CLASS_UP, CLASS_NOT_UP)
REPORT_NAME = re.compile(r'^report_(\d{4}-\d{2}-\d{2})\.json$')
# Per-ETF fields kept in the archive; the feature values and contributions stay in the batch report
ARCHIVED_RESULT_KEYS = ('ticker', 'status', 'prediction_class', 'prediction_label', 'probability', 'rank',
                        'n_ranked', 'top_percent', 'latest_price')


def archive_path(history_dir: Path, report_date: date) -> Path:
    """
    The archive file of a batch run.

    Args:
        history_dir (Path): The history directory.
        report_date (date): Date of the batch run (Europe/Berlin).

    Returns:
        Path: history_dir/YYYY-MM/report_YYYY-MM-DD.json.
    """
    return Path(history_dir) / report_date.strftime('%Y-%m') / f"report_{report_date.isoformat()}.json"


def report_date_from_path(path: Path) -> Optional[date]:
    """
    The run date encoded in an archive file name.

    Args:
        path (Path): An archive file.

    Returns:
        Optional[date]: The date, or None if the name is not report_YYYY-MM-DD.json.
    """
    match = REPORT_NAME.match(Path(path).name)
    if not match:
        return None
    try:
        return date.fromisoformat(match.group(1))
    except ValueError:
        return None


def load_report(path: Path) -> Dict[str, Any]:
    """
    Reads an archived report.

    Args:
        path (Path): An archive file.

    Returns:
        Dict[str, Any]: The report.

    Raises:
        ValueError: If the file is not a JSON object.
    """
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(report, dict):
        raise ValueError(f"The archived report {path} is not a JSON object.")
    return report


def compact_report(report: Dict[str, Any]) -> Dict[str, Any]:
    """
    The part of a batch report that the archive keeps.

    Args:
        report (Dict[str, Any]): A batch report (reporting.build_batch_report()).

    Returns:
        Dict[str, Any]: execution_time, scoring_date, target, the model essentials (features,
        adoption date, gate, overall validation) and the per-ETF predictions without features.
    """
    model = report.get('model', {})
    validation = model.get('validation', {})
    return {
        'execution_time': report.get('execution_time'),
        'scoring_date': report.get('scoring_date'),
        'target': report.get('target'),
        'model': {
            'features': model.get('features'),
            'features_adopted': model.get('features_adopted'),
            'gate': model.get('gate'),
            'validation': {key: validation.get(key) for key in ('auc_mean', 'auc_ci', 'n_dates', 'top_hit_rate',
                                                                 'base_rate', 'top_minus_universe_fwd')},
            'cross_section': model.get('cross_section'),
        },
        'results': [{key: entry.get(key) for key in ARCHIVED_RESULT_KEYS} for entry in report.get('results', [])],
    }


def prune_history(history_dir: Path, today: date, retention_days: int) -> List[Path]:
    """
    Deletes archived reports older than the retention window (by the date in the file name).

    Args:
        history_dir (Path): The history directory.
        today (date): The reference date.
        retention_days (int): Reports dated more than this many days before `today` are deleted.

    Returns:
        List[Path]: The deleted files.
    """
    removed: List[Path] = []
    for path in sorted(Path(history_dir).glob('*/report_*.json')):
        report_date = report_date_from_path(path)
        if report_date is not None and (today - report_date).days > retention_days:
            path.unlink()
            removed.append(path)
    for month_dir in Path(history_dir).glob('*'):
        if month_dir.is_dir() and not any(month_dir.iterdir()):
            month_dir.rmdir()
    if removed:
        logger.info(f"Pruned {len(removed)} archived reports older than {retention_days} days.")
    return removed


def find_closest_report(history_dir: Path, target: date, tolerance_days: int) -> Optional[Tuple[Path, date]]:
    """
    The archived report dated closest to `target` within the tolerance.

    Ties go to the earlier report, which has the longer realized period.

    Args:
        history_dir (Path): The history directory.
        target (date): The date to match.
        tolerance_days (int): Maximum distance in days.

    Returns:
        Optional[Tuple[Path, date]]: (path, report date), or None if no report is close enough.
    """
    best: Optional[Tuple[int, date, Path]] = None
    for path in Path(history_dir).glob('*/report_*.json'):
        report_date = report_date_from_path(path)
        if report_date is None:
            continue
        distance = abs((report_date - target).days)
        if distance <= tolerance_days and (best is None or (distance, report_date) < (best[0], best[1])):
            best = (distance, report_date, path)
    return (best[2], best[1]) if best else None


def realized_return(closes: pd.Series, start: date) -> Optional[Tuple[float, pd.Timestamp, pd.Timestamp]]:
    """
    The return from the first close on or after `start` to the latest close.

    Args:
        closes (pd.Series): Adjusted closes of one ETF.
        start (date): The report date.

    Returns:
        Optional[Tuple[float, pd.Timestamp, pd.Timestamp]]: (return, date of the first close,
        date of the latest close), or None without a close after `start` or with a single one.
    """
    series = closes.dropna()
    series = series[series.index >= pd.Timestamp(start)]
    if len(series) < 2 or series.iloc[0] <= 0:
        return None
    return float(series.iloc[-1] / series.iloc[0] - 1.0), series.index[0], series.index[-1]


def cohort_returns(results: Sequence[Dict[str, Any]],
                   returns: Dict[str, float]) -> Tuple[Dict[str, Optional[float]], Dict[str, int]]:
    """
    Averages the realized returns of the ranked ETFs per cohort.

    Args:
        results (Sequence[Dict[str, Any]]): The per-ETF entries of an archived report.
        returns (Dict[str, float]): Realized return per ticker.

    Returns:
        Tuple[Dict[str, Optional[float]], Dict[str, int]]: Mean return per cohort (None for an
        empty cohort, which is not a 0% return) and the cohort sizes.
    """
    groups: Dict[str, List[float]] = {cohort: [] for cohort in COHORTS}
    for entry in results:
        ticker = entry.get('ticker')
        if entry.get('rank') is None or ticker not in returns:
            continue
        groups[BASELINE].append(returns[ticker])
        cohort = CLASS_UP if entry.get('prediction_label') == CLASS_UP else CLASS_NOT_UP
        groups[cohort].append(returns[ticker])
    metrics = {cohort: (float(np.mean(values)) if values else None) for cohort, values in groups.items()}
    counts = {cohort: len(values) for cohort, values in groups.items()}
    return metrics, counts


def evaluate_uplift(history_dir: Path, closes: pd.DataFrame, today: date, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates the archived reports closest to each lookback of `uplift_lookback_days`.

    Args:
        history_dir (Path): The history directory with the archived reports.
        closes (pd.DataFrame): Adjusted daily closes, dates x tickers, up to the latest close.
        today (date): The reference date of the lookbacks.
        settings (Dict[str, Any]): Loaded settings (uplift_lookback_days, uplift_match_tolerance_days).

    Returns:
        Dict[str, Any]: generated_for (today), lookback_days, tolerance_days and horizons: per
        lookback label the matched report (report_date, scoring_date, gate_passed), the
        realized period (start, end, calendar days), metrics and counts per cohort and the
        number of ranked ETFs without prices; None for a lookback without a matching report.
    """
    lookbacks = {str(label): int(days) for label, days in settings['uplift_lookback_days'].items()}
    tolerance = int(settings['uplift_match_tolerance_days'])
    horizons: Dict[str, Optional[Dict[str, Any]]] = {}
    for label, days in lookbacks.items():
        match = find_closest_report(history_dir, today - timedelta(days=days), tolerance)
        if match is None:
            logger.info(f"Uplift {label}: no archived report within {tolerance} days of {days} days ago.")
            horizons[label] = None
            continue
        path, report_date = match
        report = load_report(path)
        results = list(report.get('results') or [])
        returns: Dict[str, float] = {}
        starts, ends = [], []
        for entry in results:
            ticker = entry.get('ticker')
            if entry.get('rank') is None or ticker not in closes.columns:
                continue
            realized = realized_return(closes[ticker], report_date)
            if realized is None:
                continue
            returns[ticker], start, end = realized
            starts.append(start)
            ends.append(end)
        metrics, counts = cohort_returns(results, returns)
        ranked = sum(1 for entry in results if entry.get('rank') is not None)
        gate = (report.get('model') or {}).get('gate') or {}
        horizons[label] = {
            'report_date': report_date.isoformat(),
            'scoring_date': report.get('scoring_date'),
            'gate_passed': gate.get('passed'),
            'start': min(starts).date().isoformat() if starts else None,
            'end': max(ends).date().isoformat() if ends else None,
            'days': int((max(ends) - min(starts)).days) if starts else None,
            'metrics': metrics,
            'counts': counts,
            'n_ranked_without_prices': ranked - counts[BASELINE],
        }
        logger.info(f"Uplift {label} (report {report_date}): Up {metrics[CLASS_UP]}, all ranked "
                    f"{metrics[BASELINE]} (n = {counts[BASELINE]}).")
    return {'generated_for': today.isoformat(), 'lookback_days': lookbacks, 'tolerance_days': tolerance,
            'horizons': horizons}


def _archived_tickers(history_dir: Path, today: date, settings: Dict[str, Any]) -> List[str]:
    """The tickers of the archived reports that the lookbacks would match."""
    tickers: set = set()
    tolerance = int(settings['uplift_match_tolerance_days'])
    for days in settings['uplift_lookback_days'].values():
        match = find_closest_report(history_dir, today - timedelta(days=int(days)), tolerance)
        if match:
            report = load_report(match[0])
            tickers.update(entry['ticker'] for entry in report.get('results') or [] if entry.get('ticker'))
    return sorted(tickers)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Refreshes the uplift report from the archive, downloading the prices it needs.

    Args:
        argv (Optional[Sequence[str]]): Command line arguments.

    Returns:
        int: Exit code 0.
    """
    from prices import fetch_price_panel
    from reporting import configure_logging, write_json_atomic

    configure_logging()
    parser = argparse.ArgumentParser(description="Realized returns of the archived batch reports.")
    parser.add_argument('--history-dir', type=Path, default=HISTORY_DIR, help="directory of the archived reports")
    parser.add_argument('--output', type=Path, default=UPLIFT_PATH, help="uplift report file")
    args = parser.parse_args(argv)
    settings = load_settings()
    today = pd.Timestamp.now(tz='Europe/Berlin').date()
    tickers = _archived_tickers(args.history_dir, today, settings)
    closes = pd.DataFrame()
    if tickers:
        longest = max(int(days) for days in settings['uplift_lookback_days'].values())
        years = int(np.ceil((longest + int(settings['uplift_match_tolerance_days']) + 10) / 365.0))
        closes = fetch_price_panel(tickers, years, {**settings, 'market_index_ticker': None,
                                                    'macro_factor_tickers': {}}).close
    write_json_atomic(args.output, evaluate_uplift(args.history_dir, closes, today, settings))
    logger.info(f"Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
