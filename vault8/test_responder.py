"""Tests for vault8 Responder role."""
import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vault8.armory.responder import Responder
from vault76.overseer import Overseer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESPONDER = Responder()


def _load(ticker: str) -> pd.DataFrame:
    return pd.read_csv(
        os.path.join(DATA_DIR, f"{ticker}_history.csv"),
        parse_dates=["datetime"],
    )


def test_codename():
    assert RESPONDER.codename == "responder"
    assert RESPONDER.name == "The Responder"


def test_deploys_in_reclamation():
    assert RESPONDER.should_deploy(Overseer.RECLAMATION)


def test_deploys_in_wasteland():
    assert RESPONDER.should_deploy(Overseer.WASTELAND)


def test_benched_in_nuked_zone():
    df = _load("ko")
    result = RESPONDER.scan("KO", df, regime=Overseer.NUKED_ZONE)
    assert result["signal"] == "NONE"
    assert "NUKED_ZONE" in result["reason"]


def test_scan_ko_reclamation():
    df = _load("ko")
    result = RESPONDER.scan("KO", df, regime=Overseer.RECLAMATION)
    assert result["symbol"] == "KO"
    if result["signal"] == "BUY_WEEK_LOW":
        assert result["entry"] < result["target"]
        assert result["stop"]  < result["entry"]
        assert 0 < result["confidence"] <= 100
        assert result["pred_range_pct"] >= 1.0
        print(f"\n  KO signal: entry=${result['entry']}  "
              f"target=${result['target']}  stop=${result['stop']}  "
              f"range={result['pred_range_pct']}%  conf={result['confidence']}")


def test_scan_nvda_reclamation():
    df = _load("nvda")
    result = RESPONDER.scan("NVDA", df, regime=Overseer.RECLAMATION)
    assert result["symbol"] == "NVDA"
    if result["signal"] == "BUY_WEEK_LOW":
        assert result["entry"] < result["target"]
        print(f"\n  NVDA signal: entry=${result['entry']}  "
              f"target=${result['target']}  range={result['pred_range_pct']}%  "
              f"conf={result['confidence']}")


def test_scan_returns_required_keys():
    df = _load("msft")
    result = RESPONDER.scan("MSFT", df, regime=Overseer.RECLAMATION)
    for key in ("symbol", "signal", "role", "reason"):
        assert key in result
    if result["signal"] != "NONE":
        for key in ("entry", "target", "stop", "confidence", "pred_range_pct"):
            assert key in result


def test_no_crash_on_all_bluechips():
    tickers = [
        "aapl", "msft", "nvda", "amd", "tsla", "ko", "pg", "jnj",
        "xom", "spy", "qqq", "gld", "jpm", "wmt", "mcd",
    ]
    for t in tickers:
        path = os.path.join(DATA_DIR, f"{t}_history.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["datetime"])
        result = RESPONDER.scan(t.upper(), df, regime=Overseer.RECLAMATION)
        assert result["symbol"] == t.upper(), f"symbol mismatch for {t}"
        assert result["signal"] in ("BUY_WEEK_LOW", "NONE"), f"bad signal for {t}"
