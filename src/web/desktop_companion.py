#!/usr/bin/env python3
"""Small optional desktop host for the shared companion surface."""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request


def wait_for_server(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    health_url = base_url.rstrip("/") + "/health"
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.25)
    raise RuntimeError(f"Companion server unavailable at {health_url}: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Open the KunPeng companion as a desktop window")
    parser.add_argument("--url", default=os.environ.get("KUNPENG_SERVER_URL", "http://127.0.0.1:8765"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--height", type=int, default=500)
    args = parser.parse_args()

    # The desktop UI must never receive model credentials. The FastAPI process owns them.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    wait_for_server(args.url, args.timeout)

    try:
        import webview
    except ImportError:
        print("pywebview is not installed; install requirements-desktop.txt", file=sys.stderr)
        return 2

    webview.create_window(
        "小鲲桌宠",
        args.url.rstrip("/") + "/companion",
        width=args.width,
        height=args.height,
        resizable=True,
        frameless=True,
        on_top=True,
        transparent=True,
    )
    webview.start(gui="gtk", debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
