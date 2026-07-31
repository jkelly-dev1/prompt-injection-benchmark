"""Incremental persistence for a sweep, so an interruption costs one call.

WHY THIS EXISTS, and it is not hypothetical. A 240-call validation sweep against
claude-opus-5 died partway through on `anthropic.APIConnectionError` during a
transient DNS outage on the host. It produced NOTHING: `run_matrix` accumulated
attempts in memory and only the completed matrix was ever written, so an
interruption at any point discarded every call that had already been paid for.
At the full size that is 1,296 sequential calls and roughly 80 minutes of
exposure with no way to resume.

Retries are the wrong layer to fix this at. They shorten the odds of a blip
killing a run; they cannot bound the loss when an outage outlasts the backoff
window, and the loss is the whole run. This bounds it at one call.

WHAT IS WRITTEN. One JSON object per line. The first line is a HEADER
describing the shape of the sweep; every line after it is one `Attempt` exactly
as the scoring layer will read it. Append-only, flushed per line, because a
checkpoint that buffers is a checkpoint that loses the tail on the crash it
exists for.

THE HEADER IS A GUARD, NOT A COMMENT. Resuming one sweep into a differently
shaped one would silently mix observations from two experiments and report the
blend as a single measurement, which is a worse outcome than losing the run.
So the header records the fingerprint of the sweep that produced the file, and
a resume whose shape does not match is REFUSED with the difference named. The
provider and model are part of that fingerprint: attempts from the mock and
attempts from a real model are not interchangeable observations, and half a
file of each would be an unreadable number that looks like a clean one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bench.models import Attempt

#: A trial's identity: which payload, under which configuration, which repeat.
#: This is exactly the product `run_matrix` iterates, so a key is stable across
#: runs and independent of the order the file happens to be in.
TrialKey = tuple[str, tuple[str, ...], int]


def trial_key(attempt: Attempt) -> TrialKey:
    return (attempt.payload_id, tuple(attempt.defenses), attempt.repeat)


class ShapeMismatch(RuntimeError):
    """A resume was asked to continue a sweep it does not match.

    Deliberately not a warning. The failure this prevents is silent: the run
    completes, the report prints, and the numbers are a blend of two different
    experiments with nothing in the output saying so.
    """


def harness_digest(system_prompt: str, payloads) -> str:
    """A hash of the things a trial's OUTCOME depends on but its NAME does not.

    THIS EXISTS BECAUSE THE FIRST VERSION OF THIS FILE MISSED IT. The
    fingerprint recorded which payload ids and configurations a sweep covered,
    which catches "you resumed a 16-payload file into a 36-payload run". It did
    not record what those payloads SAID, or what the system prompt said. So
    editing the system prompt, which is exactly what removing the contaminated
    baseline required, left every id and label identical and a resume would
    have blended trials from two different harnesses into one report without a
    word of warning. Same payload_id, different experiment.

    Hashing the text closes that: change a payload, a tell or the system prompt
    and the digest moves, so the resume is refused like any other shape change.
    """
    parts = [system_prompt]
    for payload in sorted(payloads, key=lambda item: item.payload_id):
        parts.append(
            f"{payload.payload_id}|{payload.carrier}|{payload.injection}|"
            f"{payload.tell}|{payload.wants.value}|{payload.channel.value}"
        )
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def fingerprint(
    *,
    provider: str,
    model: str,
    payload_ids: tuple[str, ...],
    config_labels: tuple[tuple[str, ...], ...],
    repeats: int,
    harness: str = "",
) -> dict:
    """The shape of a sweep, reduced to something two runs can be compared on.

    Sorted throughout, so the fingerprint does not depend on the order the
    caller happened to pass things in. `harness` is the digest above: the ids
    say which trials ran, the digest says what they were.
    """
    return {
        "provider": provider,
        "model": model,
        "payload_ids": sorted(payload_ids),
        "configs": sorted(["+".join(label) if label else "(baseline)" for label in config_labels]),
        "repeats": repeats,
        "harness": harness,
    }


def describe_mismatch(expected: dict, found: dict) -> str:
    """Name what differs, in the words an operator can act on.

    A bare "shape mismatch" sends someone reading two JSON blobs by eye. The
    common cases are a forgotten --payloads flag and a switched provider, and
    both should be readable from the error alone.
    """
    parts: list[str] = []
    for field in ("provider", "model", "repeats"):
        if expected[field] != found[field]:
            parts.append(f"{field}: checkpoint has {found[field]!r}, this run wants {expected[field]!r}")
    if expected.get("harness") != found.get("harness"):
        parts.append(
            f"harness: the payload text or system prompt changed since that "
            f"file was written ({found.get('harness')!r} -> "
            f"{expected.get('harness')!r}). The trial ids match but they are "
            f"not the same trials"
        )
    for field in ("payload_ids", "configs"):
        if expected[field] != found[field]:
            parts.append(
                f"{field}: checkpoint has {len(found[field])}, this run wants "
                f"{len(expected[field])}"
            )
    return "; ".join(parts) or "the sweep shapes differ"


class Checkpoint:
    """Append-only record of completed trials, with a resume that verifies.

    Opening for a run that has no file yet writes the header. Opening for a run
    whose file exists checks the header and refuses on a mismatch.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    # --- writing ---------------------------------------------------------- #

    def open_for(self, shape: dict) -> None:
        """Start or continue a checkpoint for a sweep of this shape.

        Raises ShapeMismatch when a file exists and describes a different sweep.
        """
        if self.path.exists() and self.path.stat().st_size > 0:
            found = self._header()
            if found != shape:
                raise ShapeMismatch(
                    f"{self.path} records a different sweep -- "
                    f"{describe_mismatch(shape, found)}. Point --checkpoint at a "
                    f"new path, or drop the flags that changed the shape. "
                    f"Resuming across shapes would blend two experiments into "
                    f"one number."
                )
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"header": shape}, sort_keys=True) + "\n")

    def record(self, attempt: Attempt) -> None:
        """Append one completed trial and flush it.

        Flushed and fsync-free by design: the operating system buffer is enough
        to survive the process dying, which is the failure this is for. A power
        loss mid-sweep loses the tail, and losing the tail is what the whole
        file is here to make cheap.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(attempt.model_dump_json() + "\n")

    # --- reading ---------------------------------------------------------- #

    def _header(self) -> dict:
        with self.path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
        if not first.strip():
            raise ShapeMismatch(f"{self.path} is empty and carries no header")
        try:
            return json.loads(first)["header"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ShapeMismatch(
                f"{self.path} does not start with a checkpoint header. It was "
                f"probably not written by this tool."
            ) from exc

    def completed(self) -> dict[TrialKey, Attempt]:
        """Every trial already recorded, keyed by identity.

        A TRUNCATED FINAL LINE IS EXPECTED AND IS NOT AN ERROR. The process this
        protects against is one that died mid-write, so the last line can be
        half a JSON object. It is dropped and the trial is simply run again,
        which costs one call. Refusing to load the file instead would throw away
        the entire run over the one record that was always going to be redone.

        Later lines win over earlier ones for the same key, so a trial that was
        recorded twice across two resumes resolves to the most recent
        observation rather than raising.
        """
        if not self.path.exists():
            return {}
        recorded: dict[TrialKey, Attempt] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index == 0 or not line.strip():
                    continue
                try:
                    attempt = Attempt.model_validate_json(line)
                except ValueError:
                    # The torn tail. Anything unreadable after it would be too.
                    break
                recorded[trial_key(attempt)] = attempt
        return recorded
