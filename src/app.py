"""
Streamlit app of the ETF predictor.

Runs the pooled cross-sectional model (src/modeling.py) on the target ETFs of
config/batch_targets.json, optionally with one additional ticker, and shows the model gate,
the walk-forward validation, the ranking and the details of one ETF. A run downloads about
12 years of prices for the whole universe; its result is cached for 12 hours.

Usage (from the repository root):
    uv run streamlit run src/app.py
"""
from typing import Tuple

import pandas as pd
import streamlit as st

from config import load_settings, load_universe
from modeling import CLASS_UP, ModelResult, etf_prediction, run_pooled_model
from prices import fetch_price_panel

st.set_page_config(page_title="ETF Ranking", layout="wide")


@st.cache_resource(ttl=43200, show_spinner=False)
def compute_ranking(tickers: Tuple[str, ...]) -> ModelResult:
    """
    Downloads the prices and runs the pooled model (cached for 12 hours per ticker set).

    Args:
        tickers (Tuple[str, ...]): The ETFs to rank.

    Returns:
        ModelResult: The pooled model result.
    """
    settings = load_settings()
    history_years = int(settings['model_history_years']) + int(settings['feature_warmup_years'])
    prices = fetch_price_panel(list(tickers), history_years, settings)
    return run_pooled_model(prices, settings)


def ranking_table(result: ModelResult) -> pd.DataFrame:
    """
    The ranking of all ETFs as a display table.

    Args:
        result (ModelResult): The pooled model result.

    Returns:
        pd.DataFrame: One row per ETF, ranked ETFs first.
    """
    rows = [etf_prediction(result, ticker) for ticker in result.results.index]
    table = pd.DataFrame(rows)[['rank', 'ticker', 'prediction_label', 'probability', 'top_percent',
                                'latest_price', 'status']]
    return table.sort_values(['rank', 'ticker'], na_position='last').reset_index(drop=True)


def show_etf_details(result: ModelResult, ticker: str) -> None:
    """
    Shows the rank, probability and feature contributions of one ETF.

    Args:
        result (ModelResult): The pooled model result.
        ticker (str): A ticker of the result.
    """
    entry = etf_prediction(result, ticker)
    col_class, col_rank, col_prob = st.columns(3)
    col_class.metric("Forecast", entry['prediction_label'])
    col_rank.metric("Rank", f"{entry['rank']} of {entry['n_ranked']}" if entry['rank'] else "not ranked")
    col_prob.metric("Probability", f"{entry['probability']:.1%}" if entry['probability'] is not None else "-")
    st.caption(f"Status: {entry['status']}")
    if entry['feature_percentiles'] is None:
        st.info("This ETF is not ranked on the scoring date, so it has no percentiles or contributions.")
        return
    features = pd.DataFrame({
        'value': entry['feature_values'],
        'percentile': entry['feature_percentiles'],
        'log-odds contribution': entry['logit_contributions'],
    })
    features.index.name = 'feature'
    st.markdown("Feature percentiles within the ranked ETFs and their contributions to the log-odds "
                "relative to a median ETF (coefficient x centered rank):")
    st.dataframe(features, use_container_width=True)
    st.bar_chart(features['log-odds contribution'])


settings = load_settings()
universe = load_universe()

with st.sidebar:
    st.title("Controls")
    extra_ticker = st.text_input("Additional ticker (optional)", value="").strip().upper()
    st.divider()
    st.caption("Model Parameters")
    st.write(f"**Target:** gain of at least {settings['threshold']:.0%} over {settings['horizon_days']} trading days")
    st.write(f"**Up:** top {settings['up_share']:.0%} of the ranking")
    st.write(f"**Features:** {', '.join(settings['champion_features'])}")
    st.write(f"**Adopted:** {settings['champion_adopted']}")
    st.divider()
    run_button = st.button("Start Analysis", use_container_width=True)

st.title("ETF Ranking")
st.caption("Pooled cross-sectional logistic regression over the target ETFs")

if run_button:
    tickers = tuple(sorted(set(universe + ([extra_ticker] if extra_ticker else []))))
    with st.spinner(f"Downloading prices and fitting the model for {len(tickers)} ETFs..."):
        st.session_state.result = compute_ranking(tickers)
    st.session_state.extra_ticker = extra_ticker

if 'result' not in st.session_state:
    st.info("Start the analysis in the sidebar. A run downloads the prices of all target ETFs and fits the pooled "
            "model with its walk-forward validation.")
    st.stop()

result: ModelResult = st.session_state.result
summary = result.summary
validation, gate = summary['validation'], summary['gate']

if gate['passed']:
    st.success(f"Model gate passed: the lower bound of the 95% interval of the validation AUC "
               f"({gate['ci_lower']:.3f}) exceeds {gate['threshold']}.")
else:
    st.error(f"Model gate failed ({gate['reason']}); no ETF is Up.")

col_ranked, col_up, col_auc, col_hit = st.columns(4)
col_ranked.metric("Ranked ETFs", summary['cross_section']['n_ranked'])
col_up.metric("Up", summary['cross_section']['n_predicted_up'])
auc_ci = validation.get('auc_ci')
col_auc.metric("Validation AUC", f"{validation['auc_mean']:.3f}" if validation['auc_mean'] is not None else "-",
               help=f"95% CI {auc_ci}" if auc_ci else None)
col_hit.metric("Top-share hit rate", f"{validation['top_hit_rate']:.1%}" if validation['top_hit_rate'] else "-",
               help=f"Base rate {validation['base_rate']:.1%}" if validation['base_rate'] else None)

tab_ranking, tab_etf, tab_validation, tab_model = st.tabs(["Ranking", "ETF Details", "Validation", "Model"])

with tab_ranking:
    st.subheader(f"Scoring date {summary['scoring_date']}")
    st.dataframe(ranking_table(result), hide_index=True, use_container_width=True)

with tab_etf:
    up_first = sorted(result.results.index, key=lambda t: (result.results.loc[t, 'predicted_class'] != CLASS_UP, t))
    default = st.session_state.get('extra_ticker') or up_first[0]
    options = list(result.results.index)
    selected = st.selectbox("ETF", options, index=options.index(default) if default in options else 0)
    show_etf_details(result, selected)

with tab_validation:
    st.markdown("Cross-sectional AUC of every test month-end of the walk-forward validation (0.5 is chance):")
    per_date = result.validation.per_date
    if not per_date.empty:
        st.line_chart(per_date.set_index('date')[['auc']])
    years = pd.DataFrame([year for year in validation['years'] if year.get('status') == 'ok'])
    if not years.empty:
        st.dataframe(years[['year', 'n_train', 'n_test_dates', 'auc_mean', 'top_hit_rate', 'base_rate',
                            'top_minus_universe_fwd']], hide_index=True, use_container_width=True)

with tab_model:
    coefficients = pd.Series(summary['coefficients'], name='coefficient')
    coefficients.index.name = 'feature'
    st.markdown(f"Intercept: {summary['intercept']}; training rows: {summary['training']['n_rows']} on "
                f"{summary['training']['n_dates']} month-ends ({summary['training']['first_date']} to "
                f"{summary['training']['last_date']}).")
    st.dataframe(coefficients, use_container_width=True)
