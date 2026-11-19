#!/usr/bin/env bash
# Serve the exhibit locally at http://localhost:8000
cd "$(dirname "$0")/docs" || exit 1

if command -v py >/dev/null 2>&1; then PY=py
elif command -v python3 >/dev/null 2>&1; then PY=python3
else PY=python; fi

echo "Serving at http://localhost:8000  (Ctrl+C to stop)"
# Open the browser shortly after the server starts (Windows / macOS / Linux)
( sleep 1; start http://localhost:8000 2>/dev/null || open http://localhost:8000 2>/dev/null || xdg-open http://localhost:8000 2>/dev/null ) &
exec $PY -m http.server 8000
