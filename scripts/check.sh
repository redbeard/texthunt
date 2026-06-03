#!/usr/bin/env bash
# Run the full quality gate. Use before every commit.
set -euo pipefail

uv run ruff format --check
uv run ruff check
uv run ty check
uv run pytest
