#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[frontend] Building local app bundle..."
cd "$ROOT/frontend"
npm install
npm run build:local

echo "[frontend] Copying dist/ to qudit_visualizer/static/..."
cd "$ROOT"
rm -rf qudit_visualizer/static/*
cp -r frontend/dist/* qudit_visualizer/static/

echo "[frontend] Done."
