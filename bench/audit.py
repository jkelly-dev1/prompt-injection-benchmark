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
import os
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

    def _lines(self) -> list[tuple[str, dict | None]]:
        """Every non-blank line, paired with its record or None if unreadable.

        Guarded per line, because one torn line must not take the file with
        it. A process killed mid-write leaves half a JSON object at the tail,
        and an unguarded json.loads there turns a single lost record into a
        log that raises on every later append and every verification, which
        is a log that can never be written to or trusted again. The torn
        line is kept in the file and reported as None: readers skip it,
        `verify_chain` returns False over it, and `append` chains from the
        last record that is intact.
        """
        if not self.path.exists():
            return []
        lines: list[tuple[str, dict | None]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    lines.append((line, None))
                    continue
                if not isinstance(record, dict) or "record_hash" not in record:
                    lines.append((line, None))
                    continue
                lines.append((line, record))
        return lines

    def _last_hash(self) -> str:
        last = GENESIS_HASH
        for _line, record in self._lines():
            if record is not None:
                last = record["record_hash"]
        return last

    def append(self, kind: str, run_id: str, payload: dict) -> dict:
        """Write one record and hand back exactly what was written.

        The hash is computed after `prev_hash` is filled in, never before, so a
        record can never be written with a hash that omits its own link.

        Flushed AND fsynced. The checkpoint is content to lose its tail on a
        power loss because a lost trial is simply rerun; an audit record is
        the thing that says a decision was made, and losing it after the
        caller was told it was written is the failure this log exists to
        rule out.
        """
        record = {
            "kind": kind,
            "run_id": run_id,
            "timestamp": utc_now(),
            "payload": payload,
            "prev_hash": self._last_hash(),
        }
        record["record_hash"] = compute_record_hash(record)
        # A torn tail has no newline. Writing straight after it would glue
        # this record onto the half-record before it and lose both, so the
        # write starts on a fresh line whenever the file does not end on one.
        separator = ""
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as handle:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    separator = "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(separator + json.dumps(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return record

    def read_all(self) -> list[dict]:
        """Every readable record. An unreadable line is skipped, not raised."""
        return [record for _line, record in self._lines() if record is not None]

    def verify_chain(self) -> bool:
        """True only if every record hashes correctly and links its predecessor.

        Both checks are required. The link check alone would miss an edited
        payload, and the hash check alone would miss a record that was moved,
        so a caller gets one answer that covers editing, reordering and
        excision together.

        A line that cannot be read is a broken chain, so the answer is False.
        It is an answer rather than an exception because the caller is a
        report printing "chain intact yes/no", and a verifier that raises
        instead of saying "no" is a verifier whose one job has been handed
        back to whoever reads the traceback.
        """
        previous = GENESIS_HASH
        for _line, record in self._lines():
            if record is None:
                return False
            if record.get("prev_hash") != previous:
                return False
            if compute_record_hash(record) != record.get("record_hash"):
                return False
            previous = record["record_hash"]
        return True


def verify_chain(path: str | Path) -> bool:
    return AuditLog(path).verify_chain()
