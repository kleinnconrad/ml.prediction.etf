# GitHub Workflows

This directory contains the automated pipelines orchestrated by GitHub Actions.

* `update_etf_seed.yml` (`Pipeline: 1. Update Seed`): Runs every Sunday at 04:41 UTC and manually. Downloads the latest Xetra T7 tradable instruments dump, regenerates the seed list and screens all instruments for age and liquidity. The updated target list `config/batch_targets.json` is committed to `main`.
* `manual_batch.yml` (`Pipeline: 2. Run Batch`): Runs every Saturday at 05:23 UTC and manually. Runs the pooled model on all target ETFs (`src/main.py`) and commits `artifacts/latest_batch_results.json`, the report archive `artifacts/history/` and the uplift report `artifacts/uplift_report.json` to `main`.
* `variable_review.yml` (`Pipeline: 3. Quarterly Variable Review`): Runs on the 2nd of February, May, August and November and manually. Runs `scripts/review_variables.py`, commits the review report to `main` and, if the switch rule holds, opens a pull request that changes `champion_features` and `champion_adopted`.
* `ci.yml` (`CI: Pre-Merge Checks`): Runs on pull requests and pushes to `main`: lock file check, `ruff`, `actionlint`, the tests on Python 3.12 and 3.14 and the Conventional Commits check.
* `release-please.yml`: Opens and updates the release pull request and the changelog from the Conventional Commits on `main`.
