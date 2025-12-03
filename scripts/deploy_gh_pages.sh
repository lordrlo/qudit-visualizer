#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[gh-pages] Building production bundle..."
cd "$ROOT/frontend"
npm install
npm run build:gh    # uses .env.production -> duckdns API

echo "[gh-pages] Deploying to GitHub Pages..."
npm run deploy  

echo "[gh-pages] Done."
