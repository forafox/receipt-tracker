.PHONY: setup format format-check lint typecheck test test-slow check

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

test-slow:
	uv run pytest -m slow

check: format-check lint typecheck test
