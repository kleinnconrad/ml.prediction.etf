# Feature Engineering

## Table of Contents
- [1. ETF Series on Own Trading Days](#1-etf-series-on-own-trading-days)
- [2. Point in Time](#2-point-in-time)
- [3. Union Calendar and Eligibility](#3-union-calendar-and-eligibility)
- [4. Within-Date Ranks](#4-within-date-ranks)
- [5. Month-End Panel and Label](#5-month-end-panel-and-label)
- [6. Macro Exposure Candidates](#6-macro-exposure-candidates)
- [Other Documents](#other-documents)

The feature engineering is implemented in `src/features.py`. It turns the daily prices of the target ETFs, the market index and the macro factors into a long panel with one row per eligible ETF and month-end, which the pooled model (`src/modeling.py`) and the variable review (`src/variable_review.py`) use.

### 1. ETF Series on Own Trading Days

Every feature is computed per ETF on the days on which the ETF has a close (its own trading days). A feature function receives the ETF's adjusted open, high, low, close and volume on these days, the market index carried to them and the macro factor series. Intermediate series that several features share (returns, moving averages, rolling extremes, factor changes) are computed once per ETF. The registry `FEATURE_FUNCTIONS` holds 277 named features in 16 families; the pooled model computes only the features in `champion_features`.

### 2. Point in Time

* Features use trailing windows only: a feature on day t depends on prices up to t. A test truncates the inputs after a date and checks that no feature on that date changes.
* The market index (EUNL.DE) trades on Xetra and closes with the ETFs; it is carried with its close on or before each own trading day.
* The macro factors (US and global markets, currencies and futures) close after Xetra. Each own trading day gets the last factor close dated before that day, so a factor move after the Xetra close of day t enters the features of the next trading day.
* No price level is a feature: a past adjusted close depends on splits and distributions that happened later.

### 3. Union Calendar and Eligibility

The own-day rows of all ETFs are placed on the union calendar of their trading days. A day without a trade carries the ETF's last trading day for at most `ffill_limit_days` (5) days; after that the ETF has no price (`stale_price`). An ETF is eligible on a date if it has at least `min_history_days` (253) own trading days up to the date, a positive 126-day traded value, a price above `min_price_eur` and a last trade within the carry limit. The rule does not depend on the feature list.

### 4. Within-Date Ranks

On every date, each feature of the eligible ETFs is converted to its percentile rank among the eligible ETFs that have a value (ties share their average rank), minus 0.5. The inputs therefore lie between -0.5 and 0.5, and a median ETF has 0. An eligible ETF without a value gets the neutral rank 0. Ranks remove the scale of a feature, bound extreme values and do not need a scaler fitted on training data.

### 5. Month-End Panel and Label

The model uses the last trading day of every calendar month. The forward return of a row is `close[t + 126] / close[t] - 1`, with 126 counted in the ETF's own trading days, and `label_end` is the date of that later close. The label is 1 if the forward return reaches `threshold` (10%) and unknown while the later close does not exist. Training rows always have a `label_end` before the first date that the model is scored on, so no training outcome overlaps a test or scoring period. The layout of the long panel (illustrative values):

| date | ticker | r_ret_21 | r_mom_12_1 | r_vol_126 | ... | fwd | label_end | y |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2025-03-31 | EXS1.DE | -0.21 | 0.12 | 0.18 | ... | 0.08 | 2025-09-26 | 0 |
| 2025-03-31 | SXR8.DE | -0.35 | 0.27 | 0.22 | ... | 0.11 | 2025-09-29 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### 6. Macro Exposure Candidates

A macro factor has the same value for every ETF on a date, so it cannot rank the ETFs directly. The family `macro_exposure` of the review pool measures each ETF's sensitivity instead:
* `beta_<factor>_<window>`: the slope of the ETF's 5-day returns on the factor's 5-day changes over 126 or 252 days. 5-day changes reduce the effect of the different closing times.
* `expo_<factor>_<k>`: the 252-day beta times the factor's change over the last 21, 63 or 126 days. On one date it orders the ETFs by their beta when the factor has risen and in reverse order when it has fallen.

The factors are fixed in `src/features.py` (`MACRO_FACTOR_KINDS`); their Yahoo Finance tickers are settings (`macro_factor_tickers`). A price factor changes relatively, a yield or volatility index by differences, and the credit factor is the relative change of HYG minus that of LQD.

## Other Documents

* `glm_checklist.md`: The GLM modeling checklist applied to the pooled model.
