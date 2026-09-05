# Contributing

Contributions that improve reproducibility, test coverage, documentation, or
responsible model evaluation are welcome.

1. Create a focused branch from `main`.
2. Install the development environment with `pip install -e ".[dev]"`.
3. Run `python -m ruff check src tests scripts` and `python -m pytest`.
4. Keep raw benchmark data, exported predictions, credentials, and local paths
   out of commits.
5. Open a pull request that explains the problem, the change, and the evidence
   used to verify it.

Changes to feature semantics or evaluation logic should include a regression
test and an explanation of their effect on the recorded model evidence.

