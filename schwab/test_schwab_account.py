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
