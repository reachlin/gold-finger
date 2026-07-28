"""
One-shot re-auth: pass the redirect URL as a CLI argument.

Usage:
  python schwab/reauth.py --url "https://127.0.0.1/?code=...&state=..."

Step 1 (no args): prints the auth URL to open in browser.
Step 2 (--url):   completes the OAuth flow with the redirect URL.
"""
import os
import sys
import json
import time
import argparse
import schwab
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

CLIENT_ID     = os.environ["SCHWAB_CLIENT_ID"]
CLIENT_SECRET = os.environ["SCHWAB_CLIENT_SECRET"]
REDIRECT_URI  = "https://127.0.0.1"
TOKEN_PATH    = os.path.join(os.path.dirname(__file__), "schwab_token.json")
STATE_PATH    = os.path.join(os.path.dirname(__file__), ".reauth_state")


def step1_print_url():
    from authlib.integrations.httpx_client import OAuth2Client
    oauth = OAuth2Client(CLIENT_ID, redirect_uri=REDIRECT_URI)
    url, state = oauth.create_authorization_url(
        "https://api.schwabapi.com/v1/oauth/authorize"
    )
    # Save the state that authlib actually embedded in the URL
    with open(STATE_PATH, "w") as f:
        f.write(state)
    print("\nOpen this URL in your browser:\n")
    print(f"  {url}\n")
    print("Log in, click Allow, then copy the full redirect URL.")
    print("Run:  python schwab/reauth.py --url \"<paste here>\"")


def step2_complete(redirect_url: str):
    if not os.path.exists(STATE_PATH):
        print("ERROR: no saved state — run without --url first to get the auth URL.")
        sys.exit(1)
    with open(STATE_PATH) as f:
        state = f.read().strip()

    # Reconstruct AuthContext directly with the saved state (avoids new state generation)
    import collections
    AuthContext = collections.namedtuple("AuthContext", ["callback_url", "authorization_url", "state"])
    auth_context = AuthContext(REDIRECT_URI, "", state)

    def token_write_func(token):
        with open(TOKEN_PATH, "w") as f:
            json.dump(token, f)

    client = schwab.auth.client_from_received_url(
        CLIENT_ID, CLIENT_SECRET, auth_context, redirect_url, token_write_func
    )

    os.remove(STATE_PATH)

    # Stamp creation_timestamp
    try:
        data = json.load(open(TOKEN_PATH))
        if "creation_timestamp" not in data:
            data["creation_timestamp"] = time.time()
            with open(TOKEN_PATH, "w") as f:
                json.dump(data, f)
        ct = data.get("creation_timestamp", time.time())
        expires = ct + 7 * 86400
        print(f"Token saved.")
        print(f"Created : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ct))}")
        print(f"Expires : {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(expires))}")
        print(f"TTL     : {(expires - time.time()) / 3600:.1f} hours remaining")
    except Exception as exc:
        print(f"Token saved (could not read TTL: {exc})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None, help="Redirect URL from browser")
    args = parser.parse_args()

    if args.url:
        step2_complete(args.url)
    else:
        step1_print_url()
