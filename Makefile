.PHONY: setup format format-check lint typecheck test check

setup:
	uv sync --locked --group dev

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

test:
	uv run pytest

check: format-check lint typecheck test
