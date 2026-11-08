"""
Shared fixtures: synthetic price panels (closes and volumes, or full OHLCV with a market
index and macro factors), fast model settings and small variable review settings (offline,
no network).
"""
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from config import load_settings
from prices import PricePanel

PanelFactory = Callable[..., PricePanel]

# The default champion features
DEFAULT_CHAMPION_FEATURES: List[str] = ['ret_21', 'ret_63', 'ret_126', 'mom_12_1', 'dist_sma50', 'dist_sma200',
                                        'vol_21', 'vol_126', 'drawdown_252']

# Macro factor tickers of the tests (the default setting); the synthetic panels provide them
TEST_FACTOR_TICKERS: Dict[str, Any] = {
    'eurusd': 'EURUSD=X', 'us_10y': '^TNX', 'oil': 'BZ=F', 'gold': 'GC=F', 'copper': 'HG=F',
    'credit': ['HYG', 'LQD'], 'vix': '^VIX', 'em_equity': 'EEM',
}
# Starting level and kind of every synthetic factor series ('level' factors move additively)
_FACTOR_SERIES: Dict[str, tuple] = {
    'EURUSD=X': (1.10, 'price'), '^TNX': (2.5, 'level'), 'BZ=F': (70.0, 'price'), 'GC=F': (1500.0, 'price'),
    'HG=F': (3.0, 'price'), 'HYG': (80.0, 'price'), 'LQD': (120.0, 'price'), '^VIX': (18.0, 'level'),
    'EEM': (40.0, 'price'),
}

# Parameters whose values the tests rely on. They are pinned here, so that changing them in
# config/settings.yaml does not break the tests.
PINNED_MODEL_SETTINGS: Dict[str, Any] = {
    'horizon_days': 126,
    'threshold': 0.10,
    'champion_features': DEFAULT_CHAMPION_FEATURES,
    'model_l2_c': 0.1,
    'model_solver_tol': 1.0e-8,
    'model_max_iter': 1000,
    'up_share': 0.10,
    'min_price_eur': 1.0,
    'min_history_days': 253,
    'ffill_limit_days': 5,
    'bootstrap_block_months': 6,
    'bootstrap_seed': 0,
    'min_validation_auc_lower': 0.5,
    'market_index_ticker': 'MKT.DE',
    'macro_factor_tickers': TEST_FACTOR_TICKERS,
}


# Small variable review parameters for synthetic data (src/variable_review.py)
REVIEW_TEST_SETTINGS: Dict[str, Any] = {
    'model_history_years': 4,
    'min_etfs_per_date': 20,
    'bootstrap_replications': 300,
    'review_validation_years': 3,
    'review_inner_splits': 2,
    'review_inner_validation_months': 12,
    'review_inner_min_train_months': 12,
    'review_anova_k': 3,
    'review_stepwise_min_gain': 0.002,
    'review_max_features': 4,
    'review_lasso_c_grid': [0.001, 0.01, 0.1, 1.0],
    'review_lasso_max_features': 10,
    'review_lasso_seed': 0,
    'review_fm_t_hurdle': 3.0,
    'review_fm_nw_lag': 6,
    'review_auc_hurdle': 0.015,
    'review_family_alpha': 0.05,
    'review_min_winning_years': 6,
    'review_fresh_check_min_dates': 6,
    'review_manual_sets': {},
    'review_candidate_features': 'all',
}


def pinned(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Overwrites the pinned keys of `settings` with copies of their test values."""
    settings.update({key: (list(value) if isinstance(value, list) else dict(value) if isinstance(value, dict)
                           else value)
                     for key, value in PINNED_MODEL_SETTINGS.items()})
    return settings


@pytest.fixture
def model_settings() -> Dict[str, Any]:
    """Production settings with the pinned model parameters, a short history and few bootstrap replications."""
    settings = pinned(load_settings())
    settings.update({
        'model_history_years': 3,
        'validation_years': 4,
        'min_etfs_per_date': 10,
        'bootstrap_replications': 300,
    })
    return settings


def synthetic_factors(dates: pd.DatetimeIndex, seed: int = 99) -> Dict[str, pd.Series]:
    """
    Random-walk closes of every macro factor ticker of TEST_FACTOR_TICKERS.

    Args:
        dates (pd.DatetimeIndex): Trading days of the factors.
        seed (int): Random seed.

    Returns:
        Dict[str, pd.Series]: Closes keyed by ticker.
    """
    rng = np.random.default_rng(seed)
    factors: Dict[str, pd.Series] = {}
    for ticker, (start, kind) in _FACTOR_SERIES.items():
        steps = rng.standard_normal(len(dates))
        values = start + np.cumsum(0.03 * steps) if kind == 'level' else start * np.exp(np.cumsum(0.01 * steps))
        factors[ticker] = pd.Series(np.maximum(values, 0.05), index=dates, name=ticker)
    return factors


@pytest.fixture
def simulate_panel() -> PanelFactory:
    """
    Returns a factory for synthetic universes of geometric random walks.

    Every ETF has its own constant daily drift (normal with sd `drift_sd`). With a positive
    `drift_sd`, past returns predict future returns across ETFs; with 0, the returns are
    pure noise.
    """
    def factory(n_etfs: int = 40, n_days: int = 2300, drift_sd: float = 0.0, vol: float = 0.015,
                seed: int = 0, start: str = '2014-01-01', volume: Optional[float] = 1000.0) -> PricePanel:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, periods=n_days)
        drifts = rng.normal(0.0, drift_sd, n_etfs) if drift_sd > 0 else np.zeros(n_etfs)
        returns = drifts[None, :] + vol * rng.standard_normal((n_days, n_etfs))
        tickers = [f"S{i:03d}.DE" for i in range(n_etfs)]
        close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
        volumes = pd.DataFrame(volume, index=dates, columns=tickers)
        return PricePanel(close=close, volume=volumes, missing=[], factors=synthetic_factors(dates))
    return factory


def synthetic_etf(n_days: int = 900, seed: int = 0, gap_share: float = 0.03) -> Dict[str, Any]:
    """
    One synthetic ETF with consistent OHLCV, a market index and macro factors on a slightly different calendar.

    The ETF's returns load on the market returns plus noise; a share `gap_share` of the
    business days has no trade (no row), and some days have no volume.

    Args:
        n_days (int): Business days of the market calendar.
        seed (int): Random seed.
        gap_share (float): Share of market days without a trade of the ETF.

    Returns:
        Dict[str, Any]: close, volume, open, high and low on the ETF's own days, market (the
        index closes on all market days) and factors (closes keyed by ticker).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2018-01-01', periods=n_days)
    market_returns = 0.0002 + 0.01 * rng.standard_normal(n_days)
    returns = 0.0003 + 1.2 * market_returns + 0.015 * rng.standard_normal(n_days)
    close = pd.Series(40.0 * np.exp(np.cumsum(returns)), index=dates)
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(0.004 * rng.standard_normal(n_days))
    high = np.maximum(close, open_) * np.exp(np.abs(0.006 * rng.standard_normal(n_days)))
    low = np.minimum(close, open_) * np.exp(-np.abs(0.006 * rng.standard_normal(n_days)))
    volume = pd.Series(rng.integers(0, 3000, n_days).astype(float), index=dates)
    volume[rng.random(n_days) < 0.05] = 0.0
    own = rng.random(n_days) >= gap_share
    own[:2] = True
    market = pd.Series(12000.0 * np.exp(np.cumsum(market_returns)), index=dates)
    return {'close': close[own], 'volume': volume[own], 'open': open_[own], 'high': high[own], 'low': low[own],
            'market': market, 'factors': synthetic_factors(dates, seed + 1)}


def ohlc_panel(n_etfs: int = 40, n_days: int = 1500, drift_sd: float = 0.0008, seed: int = 0,
               start: str = '2016-01-04', missing_days: int = 10) -> PricePanel:
    """
    A synthetic universe with opens, highs, lows, volumes, a market index and macro factors.

    Every ETF has its own constant daily drift (normal with sd `drift_sd`) and loads on a
    common market return; `missing_days` days without a trade are scattered per ETF.

    Args:
        n_etfs (int): Number of ETFs.
        n_days (int): Business days.
        drift_sd (float): Standard deviation of the daily drifts (0: pure noise).
        seed (int): Random seed.
        start (str): First business day.
        missing_days (int): Days without a trade per ETF.

    Returns:
        PricePanel: The universe with the market index 'MKT.DE' and the factors of TEST_FACTOR_TICKERS.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    market = 0.0002 + 0.01 * rng.standard_normal(n_days)
    drifts = rng.normal(0.0, drift_sd, n_etfs)
    betas = rng.uniform(0.5, 1.5, n_etfs)
    returns = drifts[None, :] + market[:, None] * betas[None, :] + 0.015 * rng.standard_normal((n_days, n_etfs))
    tickers = [f"S{i:03d}.DE" for i in range(n_etfs)]
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=tickers)
    open_ = close.shift(1).bfill() * np.exp(0.003 * rng.standard_normal(close.shape))
    high = np.maximum(close, open_) * np.exp(np.abs(0.005 * rng.standard_normal(close.shape)))
    low = np.minimum(close, open_) * np.exp(-np.abs(0.005 * rng.standard_normal(close.shape)))
    volume = pd.DataFrame(rng.integers(500, 5000, close.shape).astype(float), index=dates, columns=tickers)
    for j in range(n_etfs):
        gaps = rng.choice(np.arange(1, n_days - 1), size=missing_days, replace=False)
        for frame in (close, open_, high, low, volume):
            frame.iloc[gaps, j] = np.nan
    index = pd.Series(10000.0 * np.exp(np.cumsum(market)), index=dates, name='MKT.DE')
    return PricePanel(close=close, volume=volume, missing=[], open=open_, high=high, low=low, market=index,
                      factors=synthetic_factors(dates, seed + 7))


@pytest.fixture
def simulate_ohlc_panel() -> PanelFactory:
    """Returns ohlc_panel(), the factory for synthetic universes with OHLCV, a market index and macro factors."""
    return ohlc_panel
