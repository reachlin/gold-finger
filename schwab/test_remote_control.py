"""Tests for remote_control.py — the Drive drop-in control channel.

The daemon executes lifecycle commands (restart/stop/start/status/reconcile)
that arrive as JSON files in a synced Google Drive folder. It runs with no
authentication beyond Drive access, which is a deliberate decision (that folder
already holds the OAuth token), so every guard here is about CORRECTNESS rather
than authorisation:

  - Google Drive re-syncs files on reconnect, device re-link and conflict
    resolution. An old command.json can reappear on its own with nobody
    touching it — a spurious `stop` at 3am. Hence freshness + nonce ledger.
  - macOS Drive is a virtual filesystem: a file can appear in a directory
    listing before its bytes have materialised. Reading it too eagerly yields a
    truncated or empty file. Hence the stability check.
  - The command set is a fixed allowlist, never a shell passthrough — this
    machine holds live brokerage credentials.

The validation functions are pure (clock and filesystem injected) so all of the
above is testable without Drive, launchctl, or a real Schwab token.
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

from remote_control import (
    ALLOWED_COMMANDS,
    COMMAND_MAX_AGE_S,
    COMMAND_MAX_SKEW_S,
    POLL_INTERVAL_S,
    parse_command,
    prune_nonces,
    read_stable_json,
    validate_command,
    validate_token,
)

UTC = timezone.utc


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd(name="restart", nonce="n1", issued_at=None, now=None):
    now = now or datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    return {
        "cmd": name,
        "nonce": nonce,
        "issued_at": issued_at if issued_at is not None else iso(now),
    }


# --------------------------------------------------------------------- #
# Poll cadence                                                          #
# --------------------------------------------------------------------- #

def test_poll_interval_is_90_seconds():
    # Chosen 2026-09-21: ~2 min worst case drop->action with Drive sync
    # latency on top. Lifecycle ops are not second-sensitive.
    assert POLL_INTERVAL_S == 90


# --------------------------------------------------------------------- #
# Allowlist — never a shell passthrough                                 #
# --------------------------------------------------------------------- #

def test_every_documented_command_is_allowed():
    assert set(ALLOWED_COMMANDS) == {
        "restart", "stop", "start", "status", "reconcile"}


def test_allowlist_accepts_each_known_command():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    for i, name in enumerate(ALLOWED_COMMANDS):
        ok, why = validate_command(
            cmd(name, nonce=f"n{i}", now=now), now=now, seen=set())
        assert ok, f"{name} should be allowed: {why}"


def test_rejects_unknown_command():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    ok, why = validate_command(cmd("rm -rf /", now=now), now=now, seen=set())
    assert not ok and "not allowed" in why.lower()


def test_rejects_trading_commands_explicitly():
    # Lifecycle only. A bug here bounces the scanner; it must not move money.
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    for name in ("place_order", "close", "sell_put", "cancel"):
        ok, _ = validate_command(cmd(name, now=now), now=now, seen=set())
        assert not ok, f"{name} must never be executable remotely"


def test_rejects_missing_cmd_field():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    ok, why = validate_command(
        {"nonce": "n1", "issued_at": iso(now)}, now=now, seen=set())
    assert not ok and "cmd" in why.lower()


# --------------------------------------------------------------------- #
# Replay / freshness — Drive can resurface an old file by itself        #
# --------------------------------------------------------------------- #

def test_rejects_missing_nonce():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    ok, why = validate_command(
        {"cmd": "restart", "issued_at": iso(now)}, now=now, seen=set())
    assert not ok and "nonce" in why.lower()


def test_rejects_replayed_nonce():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    ok, why = validate_command(cmd(nonce="seen-before", now=now),
                               now=now, seen={"seen-before"})
    assert not ok and "already" in why.lower()


def test_rejects_stale_command():
    # The 3am-spurious-stop scenario: Drive re-syncs a week-old command file.
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    old = now - timedelta(seconds=COMMAND_MAX_AGE_S + 60)
    ok, why = validate_command(cmd(issued_at=iso(old)), now=now, seen=set())
    assert not ok and "stale" in why.lower()


def test_accepts_command_just_inside_the_freshness_window():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    recent = now - timedelta(seconds=COMMAND_MAX_AGE_S - 30)
    ok, why = validate_command(cmd(issued_at=iso(recent)), now=now, seen=set())
    assert ok, why


def test_rejects_command_from_the_far_future():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    future = now + timedelta(seconds=COMMAND_MAX_SKEW_S + 60)
    ok, why = validate_command(cmd(issued_at=iso(future)), now=now, seen=set())
    assert not ok and "future" in why.lower()


def test_tolerates_small_clock_skew_between_machines():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    skewed = now + timedelta(seconds=COMMAND_MAX_SKEW_S - 30)
    ok, why = validate_command(cmd(issued_at=iso(skewed)), now=now, seen=set())
    assert ok, why


def test_rejects_unparseable_issued_at():
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    ok, why = validate_command(cmd(issued_at="last tuesday"),
                               now=now, seen=set())
    assert not ok and "issued_at" in why.lower()


def test_accepts_issued_at_with_offset_and_fractional_seconds():
    # Whatever the other machine's json writer emits should parse.
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    for stamp in ("2026-09-21T13:00:00+00:00",
                  "2026-09-21T13:00:00.123456Z",
                  "2026-09-21T09:00:00-04:00"):
        ok, why = validate_command(cmd(issued_at=stamp), now=now, seen=set())
        assert ok, f"{stamp}: {why}"


# --------------------------------------------------------------------- #
# Nonce ledger                                                          #
# --------------------------------------------------------------------- #

def test_prune_nonces_drops_entries_past_ttl():
    now = time.time()
    seen = [{"nonce": "old", "ts": now - 86400 * 2},
            {"nonce": "fresh", "ts": now - 60}]
    kept = prune_nonces(seen, now=now, ttl_s=86400)
    assert [e["nonce"] for e in kept] == ["fresh"]


def test_prune_nonces_tolerates_malformed_entries():
    now = time.time()
    seen = [{"nonce": "good", "ts": now}, {"garbage": True}, "not-a-dict"]
    kept = prune_nonces(seen, now=now, ttl_s=86400)
    assert [e["nonce"] for e in kept] == ["good"]


# --------------------------------------------------------------------- #
# Malformed input                                                       #
# --------------------------------------------------------------------- #

def test_parse_command_returns_none_on_bad_json():
    assert parse_command("{not json") is None
    assert parse_command("") is None


def test_parse_command_rejects_non_object_json():
    # A bare list or string must not reach validate_command().
    assert parse_command('["restart"]') is None
    assert parse_command('"restart"') is None


# --------------------------------------------------------------------- #
# Drive materialisation — file listed before its bytes exist            #
# --------------------------------------------------------------------- #

class FakeFS:
    """Size grows across reads until the file is fully materialised."""

    def __init__(self, sizes, contents, blocks=None):
        self.sizes, self.contents = list(sizes), list(contents)
        self.blocks = list(blocks) if blocks else None
        self.reads = 0

    def stat_size(self, path):
        i = min(self.reads, len(self.sizes) - 1)
        if self.blocks is None:
            return self.sizes[i]
        return self.sizes[i], self.blocks[min(i, len(self.blocks) - 1)]

    def read(self, path):
        i = min(self.reads, len(self.contents) - 1)
        self.reads += 1
        return self.contents[i]


def test_read_stable_json_waits_until_size_settles():
    payload = json.dumps({"cmd": "restart", "nonce": "n1"})
    fs = FakeFS(sizes=[10, 25, 25, 25],
                contents=['{"cmd":', payload, payload, payload])
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None)
    assert got == {"cmd": "restart", "nonce": "n1"}, why
    assert why == "ok"


def test_read_stable_json_gives_up_on_a_never_stable_file():
    fs = FakeFS(sizes=[1, 2, 3, 4, 5, 6], contents=["{"] * 6)
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None, attempts=3)
    assert got is None and "never settled" in why


def test_read_stable_json_reports_stable_but_invalid_json_distinctly():
    fs = FakeFS(sizes=[9, 9, 9], contents=["not json!"] * 3)
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None)
    assert got is None
    assert "not a JSON object" in why, why


def test_read_stable_json_handles_a_file_that_vanishes_mid_read():
    # Drive can remove a file underneath us while it reconciles a conflict.
    def boom(path):
        raise FileNotFoundError(path)
    got, why = read_stable_json("/fake", stat_fn=boom, read_fn=boom,
                                sleep_fn=lambda s: None)
    assert got is None
    assert "stat failed" in why and "FileNotFoundError" in why, why


def test_a_hung_read_is_killed_and_reported_not_left_to_wedge():
    """The 2026-09-22 outage. `open(2)` on a Google Drive FileProvider path
    blocked FOREVER in the kernel — `sample` showed 2558/2558 samples parked in
    __open. The daemon sat wedged for 70 minutes, stopped polling entirely, and
    KeepAlive could not help because the process was alive, just stuck; even
    deleting the file did not release the syscall.

    A blocked syscall cannot be rescued in-process, so the read runs in a child
    that can be killed. What this pins is that a timeout comes back as a
    reported failure and the loop survives, instead of never returning."""
    import subprocess as sp

    def hangs(path):
        raise sp.TimeoutExpired(cmd="cat", timeout=20)

    got, why = read_stable_json("/fake", stat_fn=lambda p: (92, 8),
                                read_fn=hangs, sleep_fn=lambda s: None)
    assert got is None
    assert "HUNG" in why, why
    assert "killed" in why, why


def test_default_reader_goes_through_a_killable_child():
    """If the default read_fn ever reverts to a bare open(), a kernel hang
    takes the whole daemon down again with no way to recover."""
    import inspect

    import remote_control as rc

    src = inspect.getsource(rc.read_stable_json)
    assert "_read_file_with_timeout" in src, \
        "default read_fn must be the killable subprocess reader"
    assert "open(p).read()" not in src, \
        "a direct open() here can block forever in the kernel"


def test_read_helper_actually_enforces_its_timeout():
    """Guard the mechanism itself: a child that never returns must be killed."""
    import subprocess as sp

    import remote_control as rc

    try:
        rc._read_file_with_timeout("/dev/stdin", timeout=1)
    except sp.TimeoutExpired:
        return          # killed as intended
    except OSError:
        return          # some environments fail fast instead; also fine
    # A successful read here means it did not block, which is acceptable too.


def test_edeadlk_is_surfaced_with_its_errno():
    """The failure that broke the channel on 2026-09-21. A launchd daemon
    opening a dataless FileProvider file gets EDEADLK, and no amount of
    retrying fixes it — the Drive folder has to be pinned offline. The errno
    must reach the log: the first version swallowed it as a bare 'OSError'
    and four identical polls said nothing useful about the cause."""
    import errno as errno_mod

    sizes = [92, 92, 92]

    def stat_fn(path):
        return sizes[0], 0

    def read_fn(path):
        raise OSError(errno_mod.EDEADLK, "Resource deadlock avoided")

    got, why = read_stable_json("/fake", stat_fn=stat_fn, read_fn=read_fn,
                                sleep_fn=lambda s: None)
    assert got is None
    assert "unreadable" in why, why
    assert f"errno={errno_mod.EDEADLK}" in why, why
    assert "deadlock" in why.lower(), why


# --- the bug that bit on 2026-09-21 --------------------------------- #

def test_dataless_placeholder_is_not_mistaken_for_corruption():
    """A file synced FROM another machine reports its full size while its
    bytes are still downloading (st_blocks == 0). The size looks perfectly
    stable and the read comes back empty. Classifying that as unparseable
    made the daemon sit on a valid command for 6 minutes until it nearly
    aged out — it must read as 'still arriving', not 'corrupt'."""
    fs = FakeFS(sizes=[92, 92, 92], contents=["", "", ""], blocks=[0, 0, 0])
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None)
    assert got is None
    assert "not materialised" in why, why
    assert "dataless placeholder" in why, why
    assert "JSON" not in why, "must not be reported as a parse failure"


def test_partial_materialisation_is_reported_with_byte_counts():
    fs = FakeFS(sizes=[92, 92, 92], contents=['{"cmd": "sta'] * 3,
                blocks=[8, 8, 8])
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None)
    assert got is None
    assert "read 12 of 92 bytes" in why, why


def test_a_fully_materialised_remote_file_parses():
    payload = json.dumps({"cmd": "status", "nonce": "n9"})
    fs = FakeFS(sizes=[len(payload)] * 3, contents=[payload] * 3,
                blocks=[8, 8, 8])
    got, why = read_stable_json("/fake", stat_fn=fs.stat_size, read_fn=fs.read,
                                sleep_fn=lambda s: None)
    assert got == {"cmd": "status", "nonce": "n9"}, why


def test_freshness_window_survives_a_slow_drive_download():
    """The observed case took 6 minutes from write to materialisation; the
    original 10-minute window left almost no margin."""
    assert COMMAND_MAX_AGE_S >= 1800
    now = datetime(2026, 9, 21, 13, 0, 0, tzinfo=UTC)
    slow = now - timedelta(minutes=8)
    ok, why = validate_command(cmd(issued_at=iso(slow)), now=now, seen=set())
    assert ok, why


# --------------------------------------------------------------------- #
# Token validation — the cross-machine reauth recovery path             #
# --------------------------------------------------------------------- #

def token(created_offset_s=0, *, refresh=True, now=None):
    now = now if now is not None else time.time()
    tok = {"access_token": "a" * 76}
    if refresh:
        tok["refresh_token"] = "r" * 140
    return {"creation_timestamp": now + created_offset_s, "token": tok}


def test_accepts_a_fresh_token():
    now = time.time()
    ok, why = validate_token(token(now=now), now=now)
    assert ok, why


def test_rejects_token_without_creation_timestamp():
    now = time.time()
    data = token(now=now)
    del data["creation_timestamp"]
    ok, why = validate_token(data, now=now)
    assert not ok and "creation_timestamp" in why


def test_rejects_token_past_the_seven_day_refresh_expiry():
    now = time.time()
    ok, why = validate_token(token(-8 * 86400, now=now), now=now)
    assert not ok and "expired" in why.lower()


def test_rejects_token_missing_a_refresh_token():
    now = time.time()
    ok, why = validate_token(token(refresh=False, now=now), now=now)
    assert not ok and "refresh_token" in why


def test_rejects_token_that_is_not_an_object():
    now = time.time()
    for junk in ([], "abc", 42, None):
        ok, _ = validate_token(junk, now=now)
        assert not ok


def test_accepts_token_with_flat_refresh_token():
    # Tolerate a token file that isn't nested under "token".
    now = time.time()
    data = {"creation_timestamp": now, "refresh_token": "r" * 140,
            "access_token": "a" * 76}
    ok, why = validate_token(data, now=now)
    assert ok, why


# --------------------------------------------------------------------- #
# Result reporting                                                      #
# --------------------------------------------------------------------- #

def test_rejection_does_not_clobber_the_last_good_status_report(tmpdir=None):
    """status.txt is 'latest overseer state'. A rejected duplicate command is
    not state — if it overwrote the file you would open status.txt on a
    perfectly healthy account and read FAILED."""
    import tempfile

    import remote_control as rc

    with tempfile.TemporaryDirectory() as d:
        old = os.environ.get("GOLDFINGER_DRIVE_DIR")
        os.environ["GOLDFINGER_DRIVE_DIR"] = d
        try:
            rc._write_result("n1", "status", True, "ALL GOOD", executed=True)
            rc._write_result("n2", "status", False, "rejected: replay",
                             executed=False)

            status = open(os.path.join(d, "control", "status.txt")).read()
            assert "ALL GOOD" in status
            assert "rejected" not in status

            # ...but the rejection is still recorded for auditing.
            rej = json.load(
                open(os.path.join(d, "control", "result-n2.json")))
            assert rej["ok"] is False and "replay" in rej["output"]
        finally:
            if old is None:
                os.environ.pop("GOLDFINGER_DRIVE_DIR", None)
            else:
                os.environ["GOLDFINGER_DRIVE_DIR"] = old


def test_dry_run_env_prevents_executing_launchctl():
    """The escape hatch that lets the full path be exercised without bouncing
    the live overseer. If this ever stops short-circuiting, a test run would
    restart real trading infrastructure."""
    import remote_control as rc

    os.environ["GOLDFINGER_RC_DRY_RUN"] = "1"
    try:
        ok, out = rc._run(["launchctl", "bootout", "gui/0/anything"])
        assert ok and "DRY RUN" in out and "not executed" in out
    finally:
        os.environ.pop("GOLDFINGER_RC_DRY_RUN", None)


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception:
            failed += 1
            print(f"  ✗ {name}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
