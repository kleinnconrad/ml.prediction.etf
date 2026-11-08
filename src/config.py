"""
Paths and settings of the ETF predictor.

The user-defined parameters live in config/settings.yaml; load_settings() reads them. This
module also defines the repository paths shared by the pipeline (src/main.py), the
Streamlit app, the variable review (scripts/review_variables.py) and the universe scripts.
"""
import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / 'config'
SETTINGS_PATH = CONFIG_DIR / 'settings.yaml'
BATCH_TARGETS_PATH = CONFIG_DIR / 'batch_targets.json'
ARTIFACTS_DIR = REPO_ROOT / 'artifacts'
BATCH_RESULTS_PATH = ARTIFACTS_DIR / 'latest_batch_results.json'
REVIEW_DIR = ARTIFACTS_DIR / 'variable_reviews'


@lru_cache(maxsize=1)
def _read_settings() -> Dict[str, Any]:
    """Parses config/settings.yaml once per process."""
    with SETTINGS_PATH.open('r', encoding='utf-8') as handle:
        return yaml.safe_load(handle) or {}


def load_settings() -> Dict[str, Any]:
    """
    Loads config/settings.yaml independent of the current working directory.

    Returns:
        Dict[str, Any]: A copy of the parsed settings, safe for the caller to modify.
    """
    return copy.deepcopy(_read_settings())


def load_universe(path: Path = BATCH_TARGETS_PATH) -> List[str]:
    """
    Reads the target ETFs written by scripts/build_etf_batch.py.

    Args:
        path (Path): The batch target file (default: config/batch_targets.json).

    Returns:
        List[str]: The Yahoo Finance tickers of the target ETFs, without duplicates.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file holds no ticker list.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run scripts/build_etf_batch.py first.")
    tickers = json.loads(path.read_text(encoding='utf-8')).get('tickers')
    if not isinstance(tickers, list) or not tickers:
        raise ValueError(f"{path} holds no ticker list.")
    return list(dict.fromkeys(str(ticker) for ticker in tickers))


# Minimum price history of a target ETF in years (scripts/build_etf_batch.py)
MIN_YEARS_HISTORY = int(load_settings()['min_years_history'])
