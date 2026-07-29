#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
SERVER_URL="${KUNPENG_SERVER_URL:-http://127.0.0.1:${WEB_PORT:-8765}}"
SERVER_PID=""

health_url="${SERVER_URL%/}/health"
if ! "$PYTHON" - "$health_url" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
then
  echo "Starting local KunPeng server at ${SERVER_URL}"
  # Keep credentials in the server process only; the desktop host receives none.
  WEB_HOST="${WEB_HOST:-127.0.0.1}" WEB_PORT="${WEB_PORT:-8765}" \
    PYTHONPATH="$SCRIPT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$SCRIPT_DIR/src/web/server.py" &
  SERVER_PID=$!
  trap 'if [[ -n "${SERVER_PID}" ]]; then kill "${SERVER_PID}" 2>/dev/null || true; fi' EXIT
fi

for _ in {1..40}; do
  if "$PYTHON" - "$health_url" <<'PY'
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=1) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
  then break; fi
  sleep 0.25
done

exec "$PYTHON" "$SCRIPT_DIR/src/web/desktop_companion.py" --url "$SERVER_URL"
