#!/usr/bin/env python3
"""Send a Slack notification via incoming webhook.

Usage:
    python notify_slack.py "Your message here"

Reads SLACK_WEBHOOK_URL from .env in the current directory.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

RETRIES = 3        # total attempts, not extra attempts
BACKOFF = 1.0      # seconds; grows with each retry
TIMEOUT = 10


def load_webhook() -> str:
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "")
    if webhook:
        return webhook
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("SLACK_WEBHOOK_URL="):
                return line.split("=", 1)[1].strip()
    return ""


def _is_retryable(exc: Exception) -> bool:
    """Transient faults are worth another attempt; a bad webhook is not.

    A 4xx means Slack understood us and said no — the URL is wrong or revoked,
    and three more identical POSTs only delay the log line. Connection resets,
    timeouts and 5xx are the ones that succeed on a second try.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code >= 500
    return isinstance(exc, (urllib.error.URLError, OSError, TimeoutError))


def send(message: str, *, retries: int = RETRIES, backoff: float = BACKOFF,
         timeout: int = TIMEOUT, _sleep=time.sleep) -> bool:
    """POST `message` to the webhook, retrying transient failures.

    Raises on permanent failure rather than calling sys.exit: this is library
    code reached from the long-running overseer, and SystemExit derives from
    BaseException, so `except Exception` around the call would NOT catch it and
    the process would die on its first notification.

    The retry exists because a single dropped POST is invisible in the worst
    way. On 2026-10-02 the END OF DAY summary was lost to one
    "[Errno 54] Connection reset by peer" — the only Slack failure in the whole
    log. Slack is how this system is observed, so silence reads as a dead
    overseer; the next status check was spent believing it had crashed.
    """
    webhook = load_webhook()
    if not webhook:
        raise RuntimeError(
            "SLACK_WEBHOOK_URL not set in .env or environment — cannot notify")

    payload = json.dumps({"text": message}).encode()
    last_exc: Exception | None = None

    for attempt in range(1, max(1, retries) + 1):
        req = urllib.request.Request(
            webhook, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(req, timeout=timeout)
            return True
        except Exception as exc:          # noqa: BLE001 - re-raised below
            last_exc = exc
            if not _is_retryable(exc) or attempt >= retries:
                break
            # No sleep after the final attempt; it would only delay the caller.
            _sleep(backoff * attempt)

    raise last_exc if last_exc else RuntimeError("Slack send failed")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <message>", file=sys.stderr)
        sys.exit(1)
    try:
        send(" ".join(sys.argv[1:]))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Sent.")
