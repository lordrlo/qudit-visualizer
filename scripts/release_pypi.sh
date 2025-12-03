#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "[pypi] Building local frontend and copying to static/..."
./scripts/build_local_frontend.sh

echo "[pypi] Cleaning old Python build artifacts..."
rm -rf dist build

echo "[pypi] Building sdist and wheel..."
python -m build

echo "[pypi] Uploading to PyPI via twine..."
twine upload dist/*

echo "[pypi] Release complete."
