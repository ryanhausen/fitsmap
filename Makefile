## Runs all of the commands (excluding setup) (runs by default)
all: check-env check-security format check-style dist test docs

## Runs bandit on the project to check for security issues
check-security:
	uvx "bandit[toml]" -c pyproject.toml --severity-level=medium -r fitsmap

## Runs ruff to check for style issues
check-style:
	uvx ruff format --check .
	uvx ruff check .

## Checks the environment for consistency
check-env:
	uv lock --check-exists
	uv lock --check

# Builds the package distribution using build
dist:
	uv build

# Build the documentation using Sphinx
docs:
	uv run --directory docs --dev --locked make clean html epub

# Formats the code using ruff according the the `black` style
format:
	uvx ruff format .
	uvx ruff check --select I --fix .

# Runs the tests using pytest and generates coverage reports
test:
	uv run --group prd --locked pytest -v --cov=fitsmap --cov-report=xml --cov-report=html

# https://stackoverflow.com/a/77245502/2691018
## Print this help
help:
	@awk '/^## / \
        { if (c) {print c}; c=substr($$0, 4); next } \
         c && /(^[[:alpha:]][[:alnum:]_-]+:)/ \
        {print $$1, "\t", c; c=0} \
         END { print c }' $(MAKEFILE_LIST)

.PHONY: all check-env check-security check-style dist docs format help test
