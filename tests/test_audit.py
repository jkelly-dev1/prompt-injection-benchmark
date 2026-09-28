"""The append only hash chained log, and what it can still be trusted to prove.

An audit log's only value is that it cannot be silently revised after the fact,
so the property asserted here is not that records are written but that editing,
reordering or excising one is detectable. `prev_hash` sits inside each record's
hashed payload, which is what turns the file into a chain rather than a list of
independently hashed lines: without it every surviving record would still
verify against its own content while a deleted record went unnoticed. For a
benchmark that publishes numbers, that deletion is the tampering that matters,
because the cheapest way to improve a result is to drop the trials that went
badly.

The determinism boundary is stated rather than worked around. Timestamps are
not frozen anywhere in this module: a faked timestamp in an audit trail is
worse than an honest one that varies, since the log exists to say when a trial
was run. Determinism comes from canonical JSON with sorted keys instead, so the
key order test below holds the timestamp fixed by building records directly,
and then shows that a different timestamp does change the hash.
"""

from __future__ import annotations

import json

from bench.audit import (
    GENESIS_HASH,
    AuditLog,
    compute_record_hash,
    payload_for_hash,
)


def _three_chained_records(log: AuditLog) -> list[dict]:
    """Append three linked records and hand back what was written."""
    return [
        log.append("run_started", "run-000000000001", {"trials": 240}),
        log.append(
            "trial_scored",
            "run-000000000001",
            {"trial_id": "pi-007", "complied": True, "contained": True},
        ),
        log.append("run_finished", "run-000000000001", {"exit_code": 0}),
    ]


def _lines(log: AuditLog) -> list[str]:
    return log.path.read_text(encoding="utf-8").strip().splitlines()


def _rewrite(log: AuditLog, rows: list[dict]) -> None:
    text = "\n".join(json.dumps(row) for row in rows) + "\n"
    log.path.write_text(text, encoding="utf-8")


def test_each_record_links_to_the_hash_of_the_one_before_it(tmp_path):
    """The file is a chain, so position is part of what a record claims.

    The first record links to the genesis hash rather than to nothing, which is
    what lets a verifier tell a truncated log from a fresh one.
    """
    log = AuditLog(tmp_path / "chain" / "audit.jsonl")
    first, second, third = _three_chained_records(log)

    assert first["prev_hash"] == GENESIS_HASH
    assert second["prev_hash"] == first["record_hash"]
    assert third["prev_hash"] == second["record_hash"]
    assert len({r["record_hash"] for r in (first, second, third)}) == 3


def test_verify_chain_returns_true_on_an_untampered_log(tmp_path):
    """The honest case has to pass, or every failure below proves nothing.

    A verifier that returned False for everything would satisfy the tampering
    tests and be useless, so the untampered log is asserted first.
    """
    log = AuditLog(tmp_path / "audit.jsonl")
    assert log.verify_chain() is True, "an absent log is vacuously intact"

    _three_chained_records(log)
    assert log.verify_chain() is True
    assert len(log.read_all()) == 3


def test_editing_a_records_payload_is_detected(tmp_path):
    """Rewriting a recorded outcome breaks that record's own hash.

    This is the tampering that flatters a benchmark most directly: flipping a
    trial that the agent failed into one it passed. The edited line no longer
    hashes to its stored `record_hash`, so the log says so.
    """
    log = AuditLog(tmp_path / "audit.jsonl")
    _three_chained_records(log)

    rows = [json.loads(line) for line in _lines(log)]
    rows[1]["payload"]["contained"] = False
    _rewrite(log, rows)

    assert log.verify_chain() is False


def test_reordering_two_records_is_detected(tmp_path):
    """Order is signed, so a swap cannot be passed off as the original run.

    Reordering leaves both records byte identical, so only the link check can
    catch it. That check works because `prev_hash` is hashed: a record carries
    its position, it does not just sit in one.
    """
    log = AuditLog(tmp_path / "audit.jsonl")
    _three_chained_records(log)

    rows = [json.loads(line) for line in _lines(log)]
    rows[1], rows[2] = rows[2], rows[1]
    _rewrite(log, rows)

    assert log.verify_chain() is False


def test_excising_a_middle_record_is_detected(tmp_path, monkeypatch):
    """Mutation check: drop prev_hash from the hashed payload and an excised record goes undetected."""
    log = AuditLog(tmp_path / "honest.audit.jsonl")
    _three_chained_records(log)
    assert log.verify_chain() is True

    def excise_the_middle_record(target: AuditLog) -> None:
        """Delete record two and relink the survivor, the careful attack.

        Simply deleting a line is caught by the link check alone. Relinking is
        what a tamperer who understands the format would do, and it is caught
        only because the survivor's own hash covers the `prev_hash` that just
        changed.
        """
        rows = [json.loads(line) for line in _lines(target)]
        survivor = dict(rows[2])
        survivor["prev_hash"] = rows[0]["record_hash"]
        _rewrite(target, [rows[0], survivor])

    excise_the_middle_record(log)
    assert log.verify_chain() is False
    assert len(log.read_all()) == 2, "the excision really did remove a record"

    # The mutation, executed rather than described: hash everything except the
    # link. Each record still verifies against its own content and the links
    # still line up, so the same excision now passes verification.
    def payload_without_prev_hash(record: dict) -> dict:
        return {
            "kind": record.get("kind"),
            "run_id": record.get("run_id"),
            "timestamp": record.get("timestamp"),
            "payload": record.get("payload"),
        }

    monkeypatch.setattr(
        "bench.audit.payload_for_hash", payload_without_prev_hash
    )

    mutant = AuditLog(tmp_path / "mutant.audit.jsonl")
    _three_chained_records(mutant)
    assert mutant.verify_chain() is True, (
        "the mutant chain must be internally consistent first"
    )
    excise_the_middle_record(mutant)
    assert mutant.verify_chain() is True, (
        "with prev_hash outside the hashed payload, excision is undetectable"
    )


def test_the_record_hash_is_stable_across_key_insertion_order(tmp_path):
    """Canonical JSON with sorted keys, so two equal payloads agree on a hash.

    Dict insertion order is an accident of how a payload was assembled, and if
    it changed the hash then re-verifying an old log after an unrelated
    refactor would report tampering that never happened. The timestamp is held
    fixed here by building the records directly rather than by freezing the
    clock, because the determinism claim covers key order and explicitly does
    not cover time, as the second half of this test shows.
    """
    stamp = "2026-07-27T22:31:04.118427+00:00"
    payload = {"trial_id": "pi-007", "complied": True, "channel": "document"}
    reordered = {"channel": "document", "complied": True, "trial_id": "pi-007"}
    assert list(payload) != list(reordered), "key order must actually differ"

    def record(body: dict, timestamp: str = stamp) -> dict:
        return {
            "kind": "trial_scored",
            "run_id": "run-000000000001",
            "timestamp": timestamp,
            "payload": body,
            "prev_hash": GENESIS_HASH,
        }

    assert compute_record_hash(record(payload)) == compute_record_hash(
        record(reordered)
    )

    later = record(payload, timestamp="2026-07-27T23:00:00.000000+00:00")
    assert compute_record_hash(later) != compute_record_hash(record(payload))

    # The same rule holds for the record wrapper, not just the inner payload.
    written = AuditLog(tmp_path / "audit.jsonl").append(
        "trial_scored", "run-000000000001", payload
    )
    assert list(payload_for_hash(written)) != sorted(payload_for_hash(written))
    assert compute_record_hash(written) == written["record_hash"]


def test_a_torn_line_is_reported_not_raised_and_the_log_keeps_working(
    tmp_path, monkeypatch
):
    """A process killed mid-write leaves half a record. The log survives it.

    Three good records, then a torn tail with no newline. The verifier answers
    False instead of raising, because a report prints "chain intact yes/no"
    and a traceback is neither. The readers skip the torn line. And the next
    append still works, chaining from the last intact record, so one lost
    record does not turn into a file that can never be written to again.

    The torn line is NOT repaired or removed: the chain over the file stays
    broken, which is the truthful answer about a damaged log.

    The append is also fsynced, asserted on the call the code makes rather
    than on the file, because a buffered write looks identical from here.

    Mutation checks: restore the unguarded json.loads and the first assertion
    raises JSONDecodeError instead of returning; drop the fsync and the spy
    records nothing.
    """
    log = AuditLog(tmp_path / "audit.jsonl")
    _one, _two, third = _three_chained_records(log)
    with log.path.open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "trial_scored", "run_id": "run-0000000')

    assert log.verify_chain() is False
    assert [r["record_hash"] for r in log.read_all()] == [
        _one["record_hash"], _two["record_hash"], third["record_hash"]
    ]

    synced: list[int] = []
    monkeypatch.setattr("bench.audit.os.fsync", lambda fd: synced.append(fd))
    fourth = log.append("run_finished", "run-000000000002", {"exit_code": 0})
    assert synced, "append must fsync the record it just reported as written"
    assert fourth["prev_hash"] == third["record_hash"], (
        "a new record chains from the last INTACT record"
    )
    assert len(log.read_all()) == 4
    assert log.verify_chain() is False, "the torn line is still there; the chain is broken"
    assert "run-0000000" in log.path.read_text(encoding="utf-8"), (
        "the torn line is kept, not silently repaired"
    )


# The excision test above relinks without rehashing. These two pin what the
# chain does NOT stop, so the README cannot claim it does.


def test_a_rewrite_that_rehashes_after_an_excision_verifies_clean(tmp_path):
    """The chain is unkeyed. Excise a record and recompute every later hash,
    and verify_chain accepts the result: this is the documented limit."""
    log = AuditLog(tmp_path / "rewritten.audit.jsonl")
    _three_chained_records(log)
    rows = [json.loads(line) for line in _lines(log)]
    survivor = dict(rows[2], prev_hash=rows[0]["record_hash"])
    survivor["record_hash"] = compute_record_hash(survivor)
    _rewrite(log, [rows[0], survivor])
    assert log.verify_chain() is True


def test_removing_the_last_record_verifies_clean(tmp_path):
    log = AuditLog(tmp_path / "truncated.audit.jsonl")
    _three_chained_records(log)
    rows = [json.loads(line) for line in _lines(log)]
    _rewrite(log, rows[:-1])
    assert log.verify_chain() is True
