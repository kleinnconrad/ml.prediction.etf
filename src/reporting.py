"""
Logging setup and the JSON batch report of the pooled model.

The batch report (artifacts/latest_batch_results.json) holds the model summary (features,
coefficients, walk-forward validation and gate), the per-date validation table and one
entry per ETF. Every JSON file is written atomically, with non-finite numbers as null.
"""
import datetime
import json
import logging
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from modeling import ModelResult, etf_prediction

logger = logging.getLogger(__name__)


def configure_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Configures root logging for command line entry points.

    Args:
        log_file (Optional[str]): Additional file to write the log to.
        level (int): Minimum log level.
    """
    handlers: list = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=handlers, force=True)


def to_json_safe(value: Any) -> Any:
    """
    Converts numpy types to native Python types, non-finite floats to None and dates to ISO text.

    Python's json module writes NaN and Infinity as bare tokens, which are not valid JSON
    and break JSON.parse in the dashboard.

    Args:
        value (Any): A JSON-like structure.

    Returns:
        Any: The same structure containing only JSON-serializable values.
    """
    if isinstance(value, dict):
        return {str(k): to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NaT:
        return None
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json_atomic(path: Path, data: Any) -> None:
    """
    Serializes `data` first and then replaces `path` atomically.

    A serialization error therefore never truncates an existing file.

    Args:
        path (Path): Target file.
        data (Any): JSON-like structure.
    """
    text = json.dumps(to_json_safe(data), indent=2, allow_nan=False)
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
        # mkstemp creates the file readable by its owner only; the reports are shared files
        os.chmod(tmp_path, 0o644)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def build_batch_report(result: ModelResult, settings: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds the batch report of a pooled model run.

    Args:
        result (ModelResult): The pooled model result.
        settings (Dict[str, Any]): Loaded settings (horizon_days, threshold).

    Returns:
        Dict[str, Any]: execution_time, scoring_date, target, model (the summary),
        validation_per_date and results (one entry per ETF, ranked ETFs first by rank,
        then the others by ticker).
    """
    entries = [etf_prediction(result, ticker) for ticker in result.results.index]
    entries.sort(key=lambda entry: (entry['rank'] is None, entry['rank'] or 0, entry['ticker']))
    per_date = [{key: (value.date().isoformat() if isinstance(value, pd.Timestamp) else value)
                 for key, value in row.items()}
                for row in result.validation.per_date.to_dict(orient='records')]
    return {
        'execution_time': pd.Timestamp.now(tz='Europe/Berlin').strftime('%Y-%m-%d %H:%M:%S'),
        'scoring_date': result.summary['scoring_date'],
        'target': {'horizon_days': int(settings['horizon_days']), 'threshold': float(settings['threshold'])},
        'model': result.summary,
        'validation_per_date': per_date,
        'results': entries,
    }
