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


def _secure_token_file():
    """chmod 0600 the token file.

    schwab-py's OAuth flow writes schwab_token.json with the default umask,
    which on this machine means mode 644 — a live brokerage refresh token
    readable by every local process. Seen on the 2026-10-04 reauth; every
    earlier install was 600 only because a human remembered to chmod it.

    Called on BOTH branches of get_client(), not just after a fresh OAuth: a
    token refresh rewrites the file, and running the script is the natural
    moment to let the permissions converge. Never raises — a failure here must
    not abort a reauth.
    """
    try:
        if os.path.exists(TOKEN_PATH):
            os.chmod(TOKEN_PATH, 0o600)
    except Exception as exc:
        print(f"  [auth] could not chmod 600 the token file: {exc}")


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
        _secure_token_file()
        return schwab.auth.client_from_token_file(TOKEN_PATH, CLIENT_ID, CLIENT_SECRET)
    client = schwab.auth.client_from_manual_flow(
        CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, TOKEN_PATH
    )
    _stamp_creation_timestamp()
    _secure_token_file()
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
