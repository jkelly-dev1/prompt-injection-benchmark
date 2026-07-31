"""Append only, hash chained audit log.

Each record is one JSON line. Before writing, `prev_hash` is set to the
previous record's `record_hash`, and this record's hash is computed over its
canonical payload, which INCLUDES `prev_hash`. Editing, reordering, or removing
a past record therefore breaks every hash after it, and `verify_chain` reports
the break instead of quietly accepting the file.

Putting `prev_hash` inside the hashed payload is the whole difference between a
chain and a list of independently hashed lines. Without it every surviving
record would still verify against its own content while a deleted record went
unnoticed, which is exactly the tampering an injection benchmark most needs to
survive: a run whose embarrassing trials were excised before publication should
not verify clean.

Timestamps are NOT frozen. They are excluded from the determinism claim rather
than faked, because a faked timestamp in an audit trail is worse than an honest
one that varies: the whole value of the log is that it says when a trial was
run. Reproducibility comes from canonical JSON instead (sorted keys, tight
separators), so two runs that recorded the same decisions produce the same
hashes for everything except the timestamped fields.

This module imports nothing from the rest of the benchmark on purpose. The log
outlives the code that wrote it, and a verifier that needs the project's models
to be importable is a verifier that stops working the moment those models
change.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS_HASH = "0" * 64

#: The fields covered by a record's hash, in no particular order because
#: canonical JSON sorts them anyway. `record_hash` is absent because it is the
#: output of the hash and cannot also be an input to it.
HASHED_FIELDS = ("kind", "run_id", "timestamp", "payload", "prev_hash")


def utc_now() -> str:
    """An honest UTC ISO timestamp, never frozen and never injectable.

    Tests do not monkeypatch this to a constant. Determinism in this module
    comes from canonical JSON, not from pretending that two runs happened at
    the same instant.
    """
    return datetime.now(timezone.utc).isoformat()


def _hash_payload(payload: dict) -> str:
    """SHA-256 over canonical JSON, so key order cannot change the hash.

    `sort_keys=True` is what makes two dicts built in different orders agree,
    and `default=str` keeps a stray datetime or Path in a payload from raising
    at the moment the run is trying to record what it just did.
    """
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def payload_for_hash(record: dict) -> dict:
    """The subset of a record that its hash covers, including `prev_hash`.

    Kept as a module level function rather than folded into
    `compute_record_hash` so that a test can replace it and demonstrate what
    the chain loses when the link is left outside the hash.
    """
    return {field: record.get(field) for field in HASHED_FIELDS}


def compute_record_hash(record: dict) -> str:
    return _hash_payload(payload_for_hash(record))


class AuditLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS_HASH
        last = GENESIS_HASH
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    last = json.loads(line)["record_hash"]
        return last

    def append(self, kind: str, run_id: str, payload: dict) -> dict:
        """Write one record and hand back exactly what was written.

        The hash is computed after `prev_hash` is filled in, never before, so a
        record can never be written with a hash that omits its own link.
        """
        record = {
            "kind": kind,
            "run_id": run_id,
            "timestamp": utc_now(),
            "payload": payload,
            "prev_hash": self._last_hash(),
        }
        record["record_hash"] = compute_record_hash(record)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records: list[dict] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def verify_chain(self) -> bool:
        """True only if every record hashes correctly and links its predecessor.

        Both checks are required. The link check alone would miss an edited
        payload, and the hash check alone would miss a record that was moved,
        so a caller gets one answer that covers editing, reordering and
        excision together.
        """
        previous = GENESIS_HASH
        for record in self.read_all():
            if record.get("prev_hash") != previous:
                return False
            if compute_record_hash(record) != record.get("record_hash"):
                return False
            previous = record["record_hash"]
        return True


def verify_chain(path: str | Path) -> bool:
    return AuditLog(path).verify_chain()
