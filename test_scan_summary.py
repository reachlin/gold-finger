"""Tests for the per-scan Slack summary message builder (live_scanner)."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "schwab"))


def test_summary_has_header_counts_and_cash():
    from live_scanner import _scan_summary_message
    msg = _scan_summary_message(
        scan_count=12, global_scan=940, n_scanned=15, n_signals=3,
        events=["⛔ PG SELL_PUT $137 budget block",
                "✅ KO SELL_PUT $78"],
        free=11063.0, committed=19500.0, open_positions=["NVDA $195P"])
    assert "Scan #12" in msg
    assert "15 scanned" in msg
    assert "3 signal" in msg
    assert "11,063" in msg and "19,500" in msg
    assert "PG SELL_PUT" in msg and "KO SELL_PUT" in msg
    assert "NVDA $195P" in msg


def test_summary_no_signals_is_explicit():
    from live_scanner import _scan_summary_message
    msg = _scan_summary_message(
        scan_count=1, global_scan=1, n_scanned=15, n_signals=0,
        events=[], free=11063.0, committed=19500.0, open_positions=[])
    assert "Scan #1" in msg
    assert "0 signal" in msg
    # explicit "no signals / none" wording so the user isn't left guessing
    assert "no" in msg.lower()
    assert "none" in msg.lower()


def test_summary_lists_every_event():
    from live_scanner import _scan_summary_message
    events = [f"✅ SYM{i} SELL_PUT ${i}" for i in range(5)]
    msg = _scan_summary_message(
        scan_count=7, global_scan=100, n_scanned=15, n_signals=5,
        events=events, free=0.0, committed=30000.0, open_positions=[])
    for e in events:
        assert e in msg
