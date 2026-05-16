.PHONY: marimo test lint typecheck

marimo:
	uv run marimo edit book/marimo/notebooks/Experiment1.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run pyright src/
