"""A 429 must say what Schwab actually told us.

Context, 2026-10-13: the GTC cover for a freshly opened NVDA $215 put failed
after all six retries (76.5s of backoff) and the position sat without its exit
for one scan cycle. The log recorded only httpx's one-line summary, so there
was no way to tell a burst limit from a longer window, and no way to know
whether Schwab had named a quota in the response.

`data/overseer.log` holds 51 "429 rate-limited" lines, so this is systemic
rather than a one-off, and every one of them threw away whatever the body and
headers said.

Schwab publishes no quota, so the only way to learn it is to record what comes
back when we trip it.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import real_overseer as ro  # noqa: E402


class _Resp:
    def __init__(self, status=429, headers=None, text=""):
        self.status_code = status
        self.headers = headers or {}
        self.text = text


class _Exc(Exception):
    def __init__(self, resp):
        super().__init__("429 Too Many Requests")
        self.response = resp


def test_names_the_rate_limit_headers_schwab_sent():
    line = ro._rate_limit_detail(_Exc(_Resp(headers={
        "X-RateLimit-Limit": "120",
        "X-RateLimit-Remaining": "0",
        "X-RateLimit-Reset": "1791477051",
    })))
    assert "X-RateLimit-Limit=120" in line
    assert "X-RateLimit-Remaining=0" in line
    assert "X-RateLimit-Reset=1791477051" in line


def test_includes_retry_after_when_present():
    line = ro._rate_limit_detail(_Exc(_Resp(headers={"Retry-After": "30"})))
    assert "Retry-After=30" in line


def test_includes_the_response_body_truncated():
    body = "x" * 500
    line = ro._rate_limit_detail(_Exc(_Resp(text=body)))
    assert "xxx" in line
    assert len(line) < 400, "body must be truncated so one 429 cannot flood the log"


def test_says_so_plainly_when_schwab_sent_nothing():
    """
    The silent case is the informative one: no headers and no body means the
    quota cannot be learned from the response, and the backoff schedule stays
    a judgement call rather than a measurement.
    """
    line = ro._rate_limit_detail(_Exc(_Resp()))
    assert "no rate-limit headers" in line


def test_never_raises_on_a_malformed_exception():
    """Diagnostics must not be able to break the retry path they describe."""
    class Weird(Exception):
        pass
    assert isinstance(ro._rate_limit_detail(Weird("boom")), str)
    assert isinstance(ro._rate_limit_detail(None), str)


def test_ignores_unrelated_headers():
    line = ro._rate_limit_detail(_Exc(_Resp(headers={
        "Content-Type": "application/json",
        "Set-Cookie": "session=secret-value",
    })))
    assert "secret-value" not in line, "must not echo cookies into the log"
    assert "Content-Type" not in line


def test_the_retry_loop_logs_the_detail_once_per_429(monkeypatch, capsys):
    """The detail belongs on the attempt, not only on the final failure."""
    monkeypatch.setattr(ro, "ORDER_RETRY_DELAYS", (0.0,))
    monkeypatch.setattr(ro.time, "sleep", lambda s: None)
    ro._reset_order_throttle()

    calls = []

    def _call():
        calls.append(1)
        raise _Exc(_Resp(headers={"X-RateLimit-Remaining": "0"}))

    with pytest.raises(Exception):
        ro._order_call_with_retry(_call, "placing order", object())

    out = capsys.readouterr().out
    assert "X-RateLimit-Remaining=0" in out
    assert len(calls) == 2, "one initial attempt plus one retry"
