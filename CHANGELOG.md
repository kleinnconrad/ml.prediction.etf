# Changelog

## [2.0.0](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.6.1...v2.0.0) (2026-11-08)


### ⚠ BREAKING CHANGES

* The per-ETF models (one multinomial regression per ETF on macro variables, ANOVA pre-filter and stepwise selection in every run, Kolmogorov-Smirnov cutoff for Down) are replaced by one pooled logistic regression for all target ETFs with a fixed champion set of 9 price features ranked within each date. Reason: the per-ETF validation graded the feature selection on its own folds and reported about 70% accuracy even for targets without any predictable component, while a walk-forward re-run was below a constant forecast. The feature set now changes only through the quarterly variable review and a merged pull request.
* The forecast is a ranking instead of three return classes. Down, Flat and Up from absolute thresholds (Down meant a 6-month return below +1.2%) are replaced by Up for the top 10% of the daily ranking by the probability of a gain of at least 10% over 126 trading days, and Not Up for all other ETFs. No ETF is Up if the walk-forward validation gate fails.
* artifacts/latest_batch_results.json has a new schema. The fields model, validation_per_date and target replace baseline_parameters, and every entry of results has prediction_class (1 Up, 0 Not Up), prediction_label, probability, rank, n_ranked, top_percent, latest_price, status, feature_values, feature_percentiles and logit_contributions instead of prob_up, prob_flat, prob_down, cv_accuracy, quality_gate_passed, ks_cutoff, selected_features and feature_weights. Consumers of the file, such as the hedging repository, need to read the new fields.
* FRED data and the Gemini interpretation are removed, so FRED_API_KEY and GEMINI_API_KEY are no longer used. The per-ETF variable audit, the catalog workflow with scripts/update_catalog.py, docs/var_catalog.md, docs/features.sql and the output folder are gone. Macro information enters only as each ETF's exposure to eight daily Yahoo Finance factors in the candidate pool of the variable review.
* The parameters moved from src/config.py to config/settings.yaml (TARGET_ETF, ANNUAL_INFLATION_RATE, ANNUAL_MARGIN_UP and ANNUAL_MARGIN_DOWN no longer exist), and src/main.py no longer accepts --batch, --runner-id and --total-runners: one run ranks all target ETFs, and the workflow Pipeline: 2. Run Batch runs as a single job instead of three runners.

### Features

* add uplift charts of the predictions to the dashboard ([5a354a9](https://github.com/kleinnconrad/ml.prediction.etf/commit/5a354a96438a878641055f0d879b9a77808a1b85))
* replace the per-ETF models with one pooled ETF ranking model ([dc7531b](https://github.com/kleinnconrad/ml.prediction.etf/commit/dc7531baa84a1865687af0cb0ed9a8c3ba8262c3))
* run the batch pipeline every week ([752ace8](https://github.com/kleinnconrad/ml.prediction.etf/commit/752ace86224521c1bde61e75ebf2d90bcedd4e9a))

## [1.6.1](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.6.0...v1.6.1) (2026-11-03)


### Bug Fixes

* switch the Gemini calls to gemini-3.5-flash-lite ([5c32902](https://github.com/kleinnconrad/ml.prediction.etf/commit/5c329027a6377dce8760c58517fe115ff662736b))
* switch the Gemini calls to gemini-3.5-flash-lite ([1c2f056](https://github.com/kleinnconrad/ml.prediction.etf/commit/1c2f056b62239d1d6e1ab947fccaefe4fc020af7))

## [1.6.0](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.5.0...v1.6.0) (2026-10-09)


### Features

* pre merge pipeline ([da90409](https://github.com/kleinnconrad/ml.prediction.etf/commit/da90409e7f5d60ed74a41f08a37c6db4c6a46545))
* pre merge pipeline ([be01440](https://github.com/kleinnconrad/ml.prediction.etf/commit/be014404c27577e8aebec6aa14a42d35d66657b6))

## [1.5.0](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.4.1...v1.5.0) (2026-09-16)


### Features

* switching to pyproject.toml for this project ([6b3f6e5](https://github.com/kleinnconrad/ml.prediction.etf/commit/6b3f6e5b0558c92ba3b1f2c599fe6896604f210a))


### Bug Fixes

* locking transitive dependencies with uv.lock ([e9b7882](https://github.com/kleinnconrad/ml.prediction.etf/commit/e9b7882c58d534a1ecd94d6b059003445edf366f))

## [1.4.1](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.4.0...v1.4.1) (2026-07-22)


### Bug Fixes

* prevent depandabot from scanning seed list by changing file extension ([a2fcecf](https://github.com/kleinnconrad/ml.prediction.etf/commit/a2fcecf6747ff9575ff15ee43f8981bdbbcd3504))

## [1.4.0](https://github.com/kleinnconrad/ml.prediction.etf/compare/v1.3.0...v1.4.0) (2026-07-16)


### Features

* add baseline parameter display and update top ETF accuracy threshold to 65% ([4c8f014](https://github.com/kleinnconrad/ml.prediction.etf/commit/4c8f0140435497d0b2ff7434e3f5388b93aa69bc))
* add ETF filtering, sorting, and recommendation features to the … ([22ba4b8](https://github.com/kleinnconrad/ml.prediction.etf/commit/22ba4b80b41aaf85b49dd25ce197179a4e2eea63))
* add ETF filtering, sorting, and recommendation features to the dashboard and remove emojis from the interface. ([a219493](https://github.com/kleinnconrad/ml.prediction.etf/commit/a2194936f2d1c08f88d959bf638e186b60bffeeb))
* add global configuration file and update economic return threshold to 12.5% ([a9564b6](https://github.com/kleinnconrad/ml.prediction.etf/commit/a9564b681d17ae3b79f47aecadd7b30d128fc517))
* add SQL feature engineering script for training matrix generation and include project configuration file ([61de43d](https://github.com/kleinnconrad/ml.prediction.etf/commit/61de43d66c9057e514f6810b8693a07bdfa93ea4))
* enhance feature engineering logic ([7d4c411](https://github.com/kleinnconrad/ml.prediction.etf/commit/7d4c41109b37a1f0fcd7b407c08a870b39cd5b54))
* implement automated CI/CD workflows for ETF batch processing, variable catalog updates, and distributed execution. ([c9202d5](https://github.com/kleinnconrad/ml.prediction.etf/commit/c9202d551250fc57664faa470aca29d78c76a268))
* initialize project structure with comprehensive market indicator configuration and main pipeline entry point ([f11af9b](https://github.com/kleinnconrad/ml.prediction.etf/commit/f11af9b882232a24cef8d9e524dfcf18893020af))
* paremeterize min history constraints & chore: add new feature e… ([ef0b91a](https://github.com/kleinnconrad/ml.prediction.etf/commit/ef0b91afe45662eb3ab12bac021f7b5cd257374c))
* paremeterize min history constraints & chore: add new feature engineering logic to documents ([549acad](https://github.com/kleinnconrad/ml.prediction.etf/commit/549acada61302b49b6a77339150e34793920dc00))


### Bug Fixes

* rm flawed var ([25adde7](https://github.com/kleinnconrad/ml.prediction.etf/commit/25adde7b48c97270ad91317f7326ab4c047f6252))
