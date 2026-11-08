[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/kleinnconrad/analytics.demo.glmchecklist)

# GLM Checklist

## Table of Contents
- [1. Data Preparation](#1-data-preparation)
- [2. Feature Engineering & Selection](#2-feature-engineering--selection)
- [3. Model Setup](#3-model-setup)
- [4. Training & Validation (Validation Strategy)](#4-training--validation-validation-strategy)
- [5. Model Evaluation & Business Metrics](#5-model-evaluation--business-metrics)

The checklist applied to the pooled cross-sectional logistic regression of release 2.0.0 (`src/modeling.py`).

### 1. Data Preparation
* **Imputation [Pass]:** Missing values are handled explicitly. *Implementation: A day without a trade carries the ETF's last trading day for at most 5 days; an eligible ETF without a feature value gets the neutral rank 0 (the median).*
* **Outlier Handling [Pass]:** Extreme values cannot dominate the fit. *Implementation: Every input is a within-date percentile rank between -0.5 and 0.5.*
* **Skewness [Pass]:** Skewed raw distributions are transformed. *Implementation: Within-date ranks are uniformly distributed by construction.*
* **Scaling & Normalization [Pass]:** All inputs share one scale. *Implementation: Ranks need no scaler fitted on training data, so no scaling parameter is estimated on test data.*
* **Categorization (Binning) [Partial Pass / Deliberate Choice]:** The continuous forward return is binned into a binary target. *Implementation: A gain of at least 10% over 126 trading days. With an absolute threshold, volatile ETFs reach the target more often; the README section Expected Performance and Limitations quantifies this.*

### 2. Feature Engineering & Selection
* **Pattern Features [Pass]:** Behavioral patterns are extracted from trailing windows. *Implementation: Returns, momentum, moving-average distances and volatilities on each ETF's own trading days; tests check that no feature uses data after its date.*
* **Interaction Effects [Pass]:** Relevant interactions are available. *Implementation: The review pool contains each ETF's beta to eight macro factors and the beta times the recent factor change.*
* **Multicollinearity [Pass]:** Correlated predictors are controlled. *Implementation: L2 regularization (C = 0.1) and a small fixed champion set of 9 features.*
* **Variable Selection (Stepwise Selection) [Pass]:** Predictors are selected without grading the selection on its own rows. *Implementation: A fixed champion set in production. The quarterly review re-runs ANOVA-prefiltered stepwise selection, lasso and a Fama-MacBeth screen inside every walk-forward training window and switches only under a Bonferroni-corrected switch rule.*

### 3. Model Setup
* **Link Function & Distribution Family [Pass]:** The response is binary. *Implementation: Binomial logistic regression (logit link), pooled across all ETFs with one intercept and shared coefficients.*

### 4. Training & Validation (Validation Strategy)
* **Out-of-Time Validation [Pass]:** Training and test periods do not overlap. *Implementation: Yearly walk-forward refits over 8 years; every training label ends before the first test date of its year.*
* **Overfitting / Underfitting Check [Pass]:** In-sample and out-of-sample performance are compared. *Implementation: The batch report lists the in-sample and out-of-sample AUC of every validation year.*

### 5. Model Evaluation & Business Metrics
* **Optimal Cutoff Point [Deliberate Choice]:** No probability cutoff is tuned. *Implementation: The top 10% of the daily ranking are Up (`up_share`), if the lower bound of the 95% interval of the validation AUC is above 0.5.*
* **Alpha & Beta Error Analysis [Pass]:** Hits and misses of the decision are measured. *Implementation: The top-share hit rate against the base rate on every test date.*
* **Uplift Modeling [Pass]:** The value of the decision against a baseline is quantified. *Implementation: The mean forward return of the top share minus that of all ranked ETFs, per test date and overall.*
