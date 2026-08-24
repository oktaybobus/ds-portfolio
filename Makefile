.PHONY: install lint format typecheck test test-fast cov fetch train-all results check clean api docker spark rl

install:
	uv sync --all-extras

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run mypy

test-fast:
	uv run pytest -m "not slow and not needs_data and not needs_dl"

test:
	uv run pytest

cov:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

fetch:
	uv run python scripts/fetch_assets.py --all

train-all:
	uv run dsj train laptop_price
	uv run dsj train loan_default
	uv run python projects/customer_segments/train.py
	uv run python projects/review_sentiment/train.py
	$(MAKE) results

# Needs a JVM: `brew install openjdk@17`, or apt install openjdk-17-jdk.
spark:
	uv run python projects/marvel_network/train.py --benchmark
	uv run python projects/diabetes_screening/train.py
	$(MAKE) results

# Needs `uv sync --extra deeprl`; the tabular half runs on --extra rl alone.
rl:
	uv run python projects/frozenlake_control/train.py
	uv run python projects/cartpole_balance/train.py
	$(MAKE) results

results:
	uv run python scripts/update_results.py

check: lint typecheck test-fast
	uv run ruff format --check .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

api:
	uv run dsj api --reload

docker:
	docker build -t ds-portfolio:latest .
