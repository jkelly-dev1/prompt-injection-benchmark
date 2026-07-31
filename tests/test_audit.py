"""The append only hash chained log, and what it can still be trusted to prove.

An audit log's only value is that it cannot be quietly revised after the fact,
so the property asserted here is not that records are written but that editing,
reordering or excising one is detectable. `prev_hash` sits inside each record's
hashed payload, which is what turns the file into a chain rather than a list of
independently hashed lines: without it every surviving record would still
verify against its own content while a deleted record went unnoticed. For a
benchmark that publishes numbers, that deletion is the tampering that matters,
because the cheapest way to improve a result is to drop the trials that went
badly.

The determinism boundary is stated rather than worked around. TIMESTAMPS ARE
NOT FROZEN anywhere in this module: a faked timestamp in an audit trail is
worse than an honest one that varies, since the whole point of the log is that
it says when a trial was run. Determinism comes from canonical JSON with sorted
keys instead, so the key order test below holds the timestamp fixed by building
records directly, and then shows that a different timestamp does change the
hash.
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
    its position, it does not merely sit in one.
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
