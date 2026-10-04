import os
import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def test_env_vars_loaded():
    assert os.environ.get("SCHWAB_CLIENT_ID"), "SCHWAB_CLIENT_ID not set"
    assert os.environ.get("SCHWAB_CLIENT_SECRET"), "SCHWAB_CLIENT_SECRET not set"


def test_get_account_info_returns_data():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {
            "securitiesAccount": {
                "accountNumber": "12345678",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": 10000.0,
                    "cashBalance": 500.0,
                },
            }
        }
    ]
    mock_response.raise_for_status = MagicMock()
    mock_client.get_accounts.return_value = mock_response

    from schwab_account import get_account_info
    result = get_account_info(mock_client)

    assert isinstance(result, list)
    assert result[0]["securitiesAccount"]["accountNumber"] == "12345678"


def test_print_account_summary(capsys):
    accounts = [
        {
            "securitiesAccount": {
                "accountNumber": "12345678",
                "type": "MARGIN",
                "currentBalances": {
                    "liquidationValue": 10000.50,
                    "cashBalance": 500.25,
                },
            }
        }
    ]

    from schwab_account import print_account_summary
    print_account_summary(accounts)

    captured = capsys.readouterr()
    assert "12345678" in captured.out
    assert "10000.5" in captured.out


# --- token file permissions (added 2026-10-04) ------------------------------
#
# schwab-py's manual flow writes schwab_token.json with the default umask, i.e.
# mode 644 — a live brokerage refresh token readable by every local process.
# Observed on the 2026-10-04 reauth: the file came out 644 and had to be
# chmod'ed by hand. Every previous install was 600 only because a human
# remembered. Now the script enforces it.

import sys
import stat as _stat

sys.path.insert(0, os.path.dirname(__file__))


def test_secure_token_file_tightens_permissions(tmp_path, monkeypatch):
    import schwab_account as sa
    p = tmp_path / "schwab_token.json"
    p.write_text('{"token": {"refresh_token": "r"}}')
    p.chmod(0o644)
    monkeypatch.setattr(sa, "TOKEN_PATH", str(p))

    sa._secure_token_file()

    mode = _stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_secure_token_file_is_idempotent(tmp_path, monkeypatch):
    import schwab_account as sa
    p = tmp_path / "schwab_token.json"
    p.write_text("{}")
    p.chmod(0o600)
    monkeypatch.setattr(sa, "TOKEN_PATH", str(p))
    sa._secure_token_file()
    sa._secure_token_file()
    assert _stat.S_IMODE(p.stat().st_mode) == 0o600


def test_secure_token_file_tolerates_a_missing_file(tmp_path, monkeypatch):
    """Called before the OAuth flow has written anything — must not raise and
    kill the reauth."""
    import schwab_account as sa
    monkeypatch.setattr(sa, "TOKEN_PATH", str(tmp_path / "nope.json"))
    sa._secure_token_file()      # must not raise


def test_get_client_secures_the_token_on_both_paths():
    """A refresh rewrites the file, so the existing-token path must tighten it
    too — not just the fresh-OAuth path."""
    import inspect
    import schwab_account as sa
    src = inspect.getsource(sa.get_client)
    assert src.count("_secure_token_file()") >= 2, (
        "both the existing-token and fresh-OAuth branches should secure the file"
    )
