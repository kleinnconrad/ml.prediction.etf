### Feature Engineering Architecture

**Table of Contents**
- [1. Data Merging & Imputation](#1-data-merging--imputation)
- [2. Interaction Effects (Macro Ratios)](#2-interaction-effects-macro-ratios)
- [3. Advanced Quantitative Feature Engineering](#3-advanced-quantitative-feature-engineering)
- [4. Temporal Boundaries & Target Isolation](#4-temporal-boundaries--target-isolation)
- [5. Outlier Treatment (Winsorization)](#5-outlier-treatment-winsorization)
- [6. Z-Score Normalization](#6-z-score-normalization)

The feature engineering pipeline processes a 10-year historical dataset using a row-by-row sliding window. This continuous timeframe captures multiple macroeconomic regimes and establishes a statistical baseline for feature scaling. 

For every trading day in the dataset, the algorithm executes a backward-looking function to engineer predictor variables and a forward-looking function to establish the target classification. 

#### 1. Data Merging & Imputation
* Macroeconomic indicators sourced from FRED are published at lower frequencies (typically monthly) than daily equity pricing. 
* To reconcile these datasets, the pipeline applies a forward-fill imputation strategy upon merging. Each daily observation for a macroeconomic variable inherits the most recent published value, ensuring the model exclusively consumes "point-in-time" verified data. 
* This logic forces the algorithm to recognize macro regime shifts only as they occur, preventing look-ahead bias and aligning physical economic reality with high-frequency market dynamics.
* **Hard History Validation:** After merging, the pipeline evaluates the historical length of each individual macro predictor against a centralized minimum history parameter (e.g., 10 years). Predictors that violate this threshold (allowing for a 365-day grace period for publication lags) are explicitly dropped. This prevents a single short-history variable from silently truncating the entire multi-year dataset during NA-dropping.

#### 2. Interaction Effects (Macro Ratios)
* Instead of relying on raw asset prices, the pipeline deterministically engineers specific economic ratios (Interaction Effects) directly into the historical matrix.
* These ratios represent tangible market spreads (e.g., Copper/Gold for growth vs. inflation, HYG/LQD for credit stress, SPY/TLT for risk appetite).
* By structuring features as spreads, the model reads relative outperformance rather than single-asset drift.

#### 3. Advanced Quantitative Feature Engineering
The pipeline applies distinct transformations depending on whether a variable is an absolute asset price (structurally drifting) or a rate/spread (naturally stationary):
* **Preserving Stationary Levels:** For interest rates, spreads, and volatility indices (e.g., VIX, Yield Curve), the absolute levels are preserved as the model must recognize specific thresholds (e.g., yield curve inversion below 0).
* **Multi-Timeframe Momentum:** The backward window calculates 1-month, 3-month, 6-month, and 1-year momentum. For prices, this is computed as percentage returns (`pct_change`). For rates/spreads, it is calculated as absolute point differences (`diff`).
* **Distance to Trend (Mean Reversion):** Calculates the distance of an asset to its 200-day Simple Moving Average (SMA) to identify overextensions.
* **Macro Acceleration (2nd Derivative):** Evaluates whether the 1-year macroeconomic trend is accelerating or decelerating compared to 3 months ago.
* **Rolling Z-Scores (Regime Normalization):** Normalizes stress indicators (like VIX and Credit Spreads) over a rolling 2-year window to determine if current volatility is statistically abnormal for the specific market regime.

<img src="https://github.com/kleinnconrad/ml.prediction.etf/blob/main/docs/sliding_window.png?raw=true" width="50%">

#### 4. Temporal Boundaries & Target Isolation
* **Initialization Phase:** The initial 126 days of the dataset lack sufficient historical data to calculate the 6-month momentum features. Rows falling within this phase are discarded. 
* **Terminal Phase:** The forward window evaluates the price of the target asset exactly 126 trading days in the future. The final 126 days of the historical dataset lack this future price data required to assign a target class. 
* **Live Prediction:** The absolute latest row in this terminal phase is isolated and passed to the inference engine for live prediction. All other rows lacking a target class are excluded from the training matrix.

#### 5. Outlier Treatment (Winsorization)
* Financial markets exhibit non-normal distributions with "fat tails" (e.g., flash crashes, extreme macroeconomic shocks).
* To prevent extreme volatility spikes from artificially expanding the standard deviation and "squashing" normal cyclic signals during normalization, the dataset is subjected to deterministic clipping.
* All feature variables are clipped at the 1st and 99th percentiles. Crucially, these thresholds are calculated *exclusively* on the historical training rows and identically applied to the live prediction row to prevent look-ahead bias.

#### 6. Z-Score Normalization
* The continuous, clipped return variables are transformed into Z-scores. 
* This standardizes the data distribution, neutralizing the magnitude differences between highly volatile assets (e.g., Crypto) and slow-moving spreads (e.g., Copper/Gold).

The resulting output is a two-dimensional matrix. The primary key is the valid trading date. The columns consist of a single discrete target variable and approximately 150 continuous, scaled features representing the multi-timeframe momentum of the macro universe and its deterministic interactions.

| date | target_class | spy_1M_scaled | spy_3M_scaled | spy_6M_scaled | ratio_copper_gold_1M_scaled | ... up to X150 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2016-11-20 | 1 | 0.85 | 1.12 | 0.45 | -0.30 | ... |
| 2016-11-21 | 1 | 0.91 | 1.15 | 0.48 | -0.25 | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 2020-03-15 | -1 | -3.50 | -4.10 | -2.80 | -2.10 | ... |
| ... | ... | ... | ... | ... | ... | ... |
| 2025-11-23 | 0 | 0.15 | -0.10 | 0.22 | 1.05 | ... |