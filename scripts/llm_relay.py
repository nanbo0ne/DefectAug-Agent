"""Retrying HTTP relay for the LAN LLM endpoint (execution plumbing, no experiment logic).

The Wi-Fi link to the llama.cpp host drops intermittently; the frozen agents raise on
any connection error.  This relay forwards every request byte-for-byte to UPSTREAM and,
when the upstream connection fails or is reset before a full response arrives, waits
and re-sends the same request (up to RETRIES times).  Responses are passed through
unchanged, so the served_model check in the agents still applies.

    python scripts/llm_relay.py --listen 127.0.0.1:18184 --upstream http://192.168.31.47:18182
"""

from __future__ import annotations

import argparse
import http.client
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RETRIES, WAIT, TIMEOUT = 40, 15.0, 2400.0
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


class Relay(BaseHTTPRequestHandler):
    upstreams: list[str] = []

    def _forward(self, method: str) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        headers = {k: v for k, v in self.headers.items() if k.lower() in ("content-type", "authorization", "accept")}
        for attempt in range(1, RETRIES + 1):
            up = self.upstreams[(attempt - 1) % len(self.upstreams)]   # rotate upstreams on failure
            req = urllib.request.Request(up + self.path, data=body, headers=headers, method=method)
            try:
                with OPENER.open(req, timeout=TIMEOUT) as resp:
                    status, rh, data = resp.status, resp.headers, resp.read()
                break
            except urllib.error.HTTPError as exc:          # upstream answered: pass it through
                status, rh, data = exc.code, exc.headers, exc.read()
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError, http.client.HTTPException, OSError) as exc:
                log(f"{method} {self.path} via {up} attempt {attempt}/{RETRIES} failed: {exc!r}; retry in {WAIT}s")
                time.sleep(WAIT)
        else:
            status, rh, data = 502, {}, b'{"error": "relay: upstream unreachable"}'
            log(f"{method} {self.path} GAVE UP")
        self.send_response(status)
        self.send_header("Content-Type", rh.get("Content-Type", "application/json") if rh else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        self._forward("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._forward("POST")

    def log_message(self, fmt: str, *args) -> None:
        log(fmt % args)


def main() -> int:
    global RETRIES
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", default="127.0.0.1:18184")
    ap.add_argument("--upstream", default="http://192.168.31.47:18182", help="comma-separated, tried in turn")
    ap.add_argument("--retries", type=int, default=RETRIES)
    a = ap.parse_args()
    RETRIES = a.retries
    host, port = a.listen.rsplit(":", 1)
    Relay.upstreams = [u.strip().rstrip("/") for u in a.upstream.split(",") if u.strip()]
    log(f"relay {a.listen} -> {Relay.upstreams} (retries {RETRIES} x {WAIT}s)")
    ThreadingHTTPServer((host, int(port)), Relay).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
