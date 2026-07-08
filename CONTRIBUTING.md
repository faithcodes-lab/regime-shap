# Contributing to regime-shap

Thanks for your interest in contributing.

## Reporting Issues

Use the [GitHub Issues page](https://github.com/faithcodes-lab/regime-shap/issues) to report bugs or suggest features.

## Development Setup

```bash
git clone https://github.com/faithcodes-lab/regime-shap.git
cd regime-shap
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs,examples]"
```

## Running Tests

```bash
pytest
```

Coverage report:

```bash
pytest --cov=regime_shap --cov-report=term-missing
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and [black](https://black.readthedocs.io/) for formatting:

```bash
ruff check regime_shap/ tests/
black regime_shap/ tests/
```

## Pull Requests

- Fork the repository
- Create a feature branch (`git checkout -b feature/your-feature`)
- Make your changes with tests
- Ensure tests pass and lint is clean
- Submit a PR with a clear description

## Code of Conduct

Be respectful. Constructive criticism welcome; personal attacks not.