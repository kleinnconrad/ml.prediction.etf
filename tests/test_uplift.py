"""
Offline tests of the uplift of the published predictions (src/uplift.py): the archive file
names, the compact archive, the retention, the matching of archived reports to the lookbacks,
the realized returns from the first close on or after the report date, the cohort averages
and the batch entry point writing the archive and the uplift report.
"""
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import main
import uplift as U
from conftest import ohlc_panel
from prices import save_price_panel
from reporting import write_json_atomic

SETTINGS = {'uplift_lookback_days': {'1m': 30, '3m': 90, '6m': 180}, 'uplift_match_tolerance_days': 15,
            'history_retention_days': 400}


def _entry(ticker: str, label: Optional[str], rank: Optional[int]) -> Dict[str, Any]:
    """One per-ETF entry of a batch report (label None: not ranked)."""
    return {'ticker': ticker, 'status': 'ok' if rank else 'not_eligible: stale_price',
            'prediction_class': 1 if label == 'Up' else 0, 'prediction_label': label or 'Not Up',
            'probability': 0.5 if rank else None, 'rank': rank, 'n_ranked': 3, 'top_percent': None,
            'latest_price': 10.0, 'feature_values': {'ret_21': 0.1}, 'feature_percentiles': None,
            'logit_contributions': None}


def _report(results: List[Dict[str, Any]], passed: bool = True) -> Dict[str, Any]:
    """A minimal batch report."""
    return {'execution_time': '2026-11-08 10:00:00', 'scoring_date': '2026-11-06',
            'target': {'horizon_days': 126, 'threshold': 0.1},
            'model': {'features': ['ret_21'], 'features_adopted': '2026-11-08', 'coefficients': {'ret_21': 0.5},
                      'gate': {'passed': passed}, 'validation': {'auc_mean': 0.6, 'years': [{'year': 2025}]},
                      'cross_section': {'n_ranked': 3}},
            'validation_per_date': [{'date': '2025-01-31'}], 'results': results}


def test_archive_file_names_and_dates(tmp_path: Path) -> None:
    path = U.archive_path(tmp_path, date(2026, 11, 8))
    assert path == tmp_path / '2026-11' / 'report_2026-11-08.json'
    assert U.report_date_from_path(path) == date(2026, 11, 8)
    for name in ('report_2026-13-01.json', 'summary_2026-11-08.json', 'report_2026-11-08.md'):
        assert U.report_date_from_path(tmp_path / name) is None


def test_the_archive_keeps_the_predictions_without_the_features() -> None:
    report = _report([_entry('A', 'Up', 1), _entry('B', 'Not Up', 2), _entry('C', None, None)])
    compact = U.compact_report(report)
    assert set(compact) == {'execution_time', 'scoring_date', 'target', 'model', 'results'}
    assert [entry['ticker'] for entry in compact['results']] == ['A', 'B', 'C']
    assert set(compact['results'][0]) == set(U.ARCHIVED_RESULT_KEYS)
    assert compact['model']['gate'] == {'passed': True} and 'years' not in compact['model']['validation']
    json.dumps(compact, allow_nan=False)


def test_old_reports_are_pruned_and_empty_month_folders_removed(tmp_path: Path) -> None:
    for day in (date(2025, 9, 1), date(2025, 10, 10), date(2026, 11, 1)):
        write_json_atomic(U.archive_path(tmp_path, day), {'results': []})
    removed = U.prune_history(tmp_path, date(2026, 11, 8), retention_days=400)
    assert [path.name for path in removed] == ['report_2025-09-01.json']
    assert not (tmp_path / '2025-09').exists()
    assert sorted(path.name for path in tmp_path.glob('*/report_*.json')) == ['report_2025-10-10.json',
                                                                          'report_2026-11-01.json']


def test_the_closest_report_within_the_tolerance_is_matched(tmp_path: Path) -> None:
    for day in (date(2026, 10, 1), date(2026, 10, 15), date(2026, 8, 1)):
        write_json_atomic(U.archive_path(tmp_path, day), {'results': []})
    (tmp_path / '2026-10' / 'notes.json').write_text('{}')
    match = U.find_closest_report(tmp_path, date(2026, 10, 8), tolerance_days=15)
    assert match is not None and match[1] == date(2026, 10, 1)             # tie: the earlier report
    assert U.find_closest_report(tmp_path, date(2026, 10, 13), 15)[1] == date(2026, 10, 15)
    assert U.find_closest_report(tmp_path, date(2026, 9, 8), 15) is None   # 24 days from both neighbors
    assert U.find_closest_report(tmp_path / 'missing', date(2026, 10, 8), 15) is None


def test_the_realized_return_starts_at_the_first_close_on_or_after_the_report_date() -> None:
    closes = pd.Series([100.0, np.nan, 110.0, 121.0],
                       index=pd.to_datetime(['2026-10-30', '2026-11-09', '2026-11-10', '2026-12-04']))
    value, start, end = U.realized_return(closes, date(2026, 11, 8))
    assert value == pytest.approx(0.1) and start == pd.Timestamp('2026-11-10') and end == pd.Timestamp('2026-12-04')
    assert U.realized_return(closes, date(2026, 12, 1)) is None            # a single close after the date


def test_cohorts_average_the_ranked_etfs_only() -> None:
    results = [_entry('A', 'Up', 1), _entry('B', 'Not Up', 2), _entry('C', 'Not Up', 3), _entry('D', None, None),
               _entry('E', 'Not Up', 4)]
    metrics, counts = U.cohort_returns(results, {'A': 0.10, 'B': 0.02, 'C': -0.04, 'D': 0.50})
    assert counts == {'baseline': 3, 'Up': 1, 'Not Up': 2}                 # D not ranked, E without a return
    assert metrics['Up'] == pytest.approx(0.10) and metrics['Not Up'] == pytest.approx(-0.01)
    assert metrics['baseline'] == pytest.approx(0.08 / 3)
    no_up, counts = U.cohort_returns([_entry('B', 'Not Up', 1)], {'B': 0.02})
    assert no_up['Up'] is None and counts['Up'] == 0                      # an empty cohort is not a 0% return


def test_the_uplift_report_evaluates_each_lookback(tmp_path: Path) -> None:
    results = [_entry('A', 'Up', 1), _entry('B', 'Not Up', 2), _entry('C', 'Not Up', 3), _entry('GONE', 'Not Up', 4)]
    write_json_atomic(U.archive_path(tmp_path, date(2026, 10, 9)), U.compact_report(_report(results)))
    write_json_atomic(U.archive_path(tmp_path, date(2026, 8, 10)), U.compact_report(_report(results, passed=False)))
    days = pd.bdate_range('2026-07-01', '2026-11-06')
    closes = pd.DataFrame({'A': np.linspace(100, 130, len(days)), 'B': 50.0, 'C': np.linspace(20, 18, len(days))},
                          index=days)
    report = U.evaluate_uplift(tmp_path, closes, date(2026, 11, 8), SETTINGS)

    assert set(report['horizons']) == {'1m', '3m', '6m'} and report['horizons']['6m'] is None
    one = report['horizons']['1m']
    assert one['report_date'] == '2026-10-09' and one['start'] == '2026-10-09' and one['end'] == '2026-11-06'
    assert one['days'] == 28 and one['gate_passed'] is True and one['n_ranked_without_prices'] == 1
    a = closes['A']
    assert one['metrics']['Up'] == pytest.approx(a.iloc[-1] / a.loc['2026-10-09'] - 1.0)
    assert one['counts'] == {'baseline': 3, 'Up': 1, 'Not Up': 2}
    assert report['horizons']['3m']['report_date'] == '2026-08-10' and report['horizons']['3m']['gate_passed'] is False
    assert report['generated_for'] == '2026-11-08' and report['tolerance_days'] == 15
    json.dumps(report, allow_nan=False)


def test_the_batch_run_writes_the_archive_and_the_uplift_report(tmp_path: Path,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    prices = ohlc_panel(n_etfs=40, n_days=1300, seed=3)
    save_price_panel(prices, tmp_path / 'cache')
    monkeypatch.setattr(main, 'load_universe', lambda: list(prices.close.columns))
    output = tmp_path / 'out' / 'latest_batch_results.json'

    assert main.main(['--price-cache', str(tmp_path / 'cache'), '--output', str(output)]) == 0

    report = json.loads(output.read_text())
    archived = list((tmp_path / 'out' / 'history').glob('*/report_*.json'))
    assert len(archived) == 1
    assert U.report_date_from_path(archived[0]) == pd.Timestamp(report['execution_time']).date()
    assert json.loads(archived[0].read_text()) == U.compact_report(report)
    uplift = json.loads((tmp_path / 'out' / 'uplift_report.json').read_text())
    assert set(uplift['horizons']) == {'1m', '3m', '6m'}
    assert uplift['generated_for'] == U.report_date_from_path(archived[0]).isoformat()
