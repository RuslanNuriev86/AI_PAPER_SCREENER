# Development entry points.
#
# UV_PYTHON_INSTALL_DIR is exported by every target because uv 0.8.16 cannot read it from
# uv.toml and the default location is read-only on this host (DESIGN.md §16). Running
# `uv` directly instead of `make` will fail here — that is an environment property, not a
# project one.

export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python

.PHONY: venv install lint fmt types test check run dry doctor clean migrations

venv: ## create the virtualenv against Python 3.12
	uv venv --python 3.12

install: venv ## sync all dependencies
	uv sync --all-groups

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

types:
	uv run mypy

test:
	uv run pytest

check: lint types test ## what CI runs

run: ## real run (spends money, sends to Telegram)
	uv run screener run

dry: ## full pipeline, no send, no spend on the digest path
	uv run screener dry-run

doctor:
	uv run screener doctor

clean:
	rm -rf .venv .uv-cache .uv-python screener.db* outbox dist
	find . -name __pycache__ -prune -exec rm -rf {} +
