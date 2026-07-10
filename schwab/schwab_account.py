import os
import json
import time
import schwab
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI = "https://127.0.0.1"
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "schwab_token.json")


def _stamp_creation_timestamp():
    """Inject creation_timestamp into the token file if missing. Safe to call after any OAuth."""
    try:
        data = json.load(open(TOKEN_PATH))
        if "creation_timestamp" not in data:
            data["creation_timestamp"] = time.time()
            with open(TOKEN_PATH, "w") as f:
                json.dump(data, f)
            print(f"  [auth] creation_timestamp stamped: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as exc:
        print(f"  [auth] could not stamp creation_timestamp: {exc}")


def get_client():
    if os.path.exists(TOKEN_PATH):
        return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)
    client = schwab.auth.client_from_manual_flow(
        CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TOKEN_PATH
    )
    _stamp_creation_timestamp()
    return client


def get_account_info(client):
    resp = client.get_accounts()
    resp.raise_for_status()
    return resp.json()


def print_account_summary(accounts):
    for entry in accounts:
        acct = entry["securitiesAccount"]
        balances = acct.get("currentBalances", {})
        print(f"Account : {acct['accountNumber']}")
        print(f"Type    : {acct['type']}")
        print(f"Value   : {balances.get('liquidationValue', 'N/A')}")
        print(f"Cash    : {balances.get('cashBalance', 'N/A')}")
        print()


if __name__ == "__main__":
    client = get_client()
    accounts = get_account_info(client)
    print_account_summary(accounts)
