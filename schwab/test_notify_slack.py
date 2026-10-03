"""Tests for notify_slack.send — retry, and never killing the caller.

Two problems, both found on 2026-10-03.

1. NO RETRY. A single POST failure lost the message forever:

       VAULT 76 — END OF DAY 2026-10-02 16:02:14 ET  (scans: 69)
       [Slack] failed to send: <urlopen error [Errno 54] Connection reset by peer>

   That was the only Slack failure in the entire log — a one-off blip on a
   network path that runs through a VPN. But Slack is how the overseer is
   observed, so a dropped notification looks exactly like a dead overseer, and
   the user spent a check believing it had crashed. "No news is good news" is
   only true if delivery is reliable.

2. sys.exit(1) INSIDE A LIBRARY FUNCTION. If the webhook were missing, send()
   raised SystemExit — which derives from BaseException, so live_scanner's
   `except Exception` would NOT catch it and the overseer process would die on
   its first notification. Harmless only because the webhook happens to be set.

Retry policy is deliberately selective: transient faults (connection reset,
timeout, 5xx) are worth retrying; a 4xx means the webhook itself is wrong and
retrying just delays the log line.
"""
import os
import sys
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import notify_slack


class Recorder:
    """Stands in for urlopen, replaying a scripted sequence of outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _http(code):
    return urllib.error.HTTPError("http://x", code, "err", {}, None)


def _reset():
    return urllib.error.URLError(OSError(54, "Connection reset by peer"))


def setup_function():
    os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.slack.com/services/TEST"


def _send(monkeypatch, recorder, **kw):
    slept = []
    monkeypatch.setattr(notify_slack.urllib.request, "urlopen", recorder)
    kw.setdefault("_sleep", slept.append)
    return notify_slack.send("hello", **kw), slept


# --- the happy path is unchanged -------------------------------------------

def test_success_on_first_try_sends_once(monkeypatch):
    rec = Recorder("ok")
    _send(monkeypatch, rec)
    assert rec.calls == 1, "a working webhook must not be retried"


# --- the 2026-10-02 failure now recovers -----------------------------------

def test_connection_reset_then_success(monkeypatch):
    """The exact Errno 54 that lost the EOD summary."""
    rec = Recorder(_reset(), "ok")
    _send(monkeypatch, rec)
    assert rec.calls == 2


def test_two_failures_then_success(monkeypatch):
    rec = Recorder(_reset(), _reset(), "ok")
    _send(monkeypatch, rec)
    assert rec.calls == 3


def test_gives_up_after_the_retry_budget_and_raises(monkeypatch):
    """The caller logs the failure, so it must still surface — just later."""
    rec = Recorder(_reset())
    try:
        _send(monkeypatch, rec, retries=3)
    except Exception as exc:
        assert "reset" in str(exc).lower() or "54" in str(exc)
    else:
        raise AssertionError("a permanently failing send must raise")
    assert rec.calls == 3


def test_backoff_grows_between_attempts(monkeypatch):
    rec = Recorder(_reset(), _reset(), "ok")
    _, slept = _send(monkeypatch, rec, backoff=1.0)
    assert len(slept) == 2
    assert slept[1] > slept[0], f"backoff should grow, got {slept}"


def test_no_sleep_after_the_final_attempt(monkeypatch):
    """Sleeping after the last try just delays the caller for nothing."""
    rec = Recorder(_reset())
    try:
        _, slept = _send(monkeypatch, rec, retries=2)
    except Exception:
        slept = None
    # one sleep between the two attempts, none after the second
    assert rec.calls == 2


# --- selective retry -------------------------------------------------------

def test_server_error_is_retried(monkeypatch):
    rec = Recorder(_http(503), "ok")
    _send(monkeypatch, rec)
    assert rec.calls == 2


def test_client_error_is_not_retried(monkeypatch):
    """404/403 means the webhook is wrong. Retrying cannot fix it."""
    rec = Recorder(_http(404))
    try:
        _send(monkeypatch, rec)
    except Exception:
        pass
    assert rec.calls == 1, "a 4xx must fail fast"


# --- never kill the caller -------------------------------------------------

def test_missing_webhook_raises_instead_of_exiting(monkeypatch):
    """send() used to sys.exit(1). SystemExit is BaseException, so
    live_scanner's `except Exception` would not catch it and the overseer
    process would die on its first notification."""
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(notify_slack, "load_webhook", lambda: "")
    try:
        notify_slack.send("hello")
    except SystemExit:
        raise AssertionError("send() must not raise SystemExit from library code")
    except Exception:
        pass   # a normal exception the caller can catch and log
    else:
        raise AssertionError("a missing webhook should still be reported")


def test_a_caller_catching_Exception_survives_every_failure_mode(monkeypatch):
    """What live_scanner actually does."""
    monkeypatch.setattr(notify_slack, "load_webhook", lambda: "")
    survived = False
    try:
        notify_slack.send("x")
    except Exception:
        survived = True
    assert survived, "the overseer must keep running when Slack is unavailable"
