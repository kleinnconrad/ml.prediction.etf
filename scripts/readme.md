# Scripts

This directory contains the entry point of the quarterly variable review and the scripts that curate the target ETF list.

* `review_variables.py`: Runs the quarterly champion-challenger review of the model features (`src/variable_review.py`) and writes `artifacts/variable_reviews/review_YYYY-MM-DD.json` and `.md`. On a switch it updates `champion_features` and `champion_adopted` in `config/settings.yaml` and writes the pull request and commit texts used by the workflow `Pipeline: 3. Quarterly Variable Review`.
* `build_etf_batch.py`: Filters the ETF seed list based on historical data depth and trading liquidity. Keeps instruments with at least `min_years_history` years of trading history and 1 million EUR average daily turnover, and exports them as `config/batch_targets.json`.
* `download_t7_dump.py`: Retrieves the Xetra T7 all tradable instruments dump from the exchange portal via a static URL download.
* `generate_seed.py`: Parses raw Xetra T7 instrument data to extract tradeable EUR-denominated ETFs and ETCs. Formats mnemonics for the Yahoo Finance API and outputs a deduplicated text seed list.
