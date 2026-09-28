"""The RE-SCORED figures, from the published rows and one small file per model.

Two scoring defects were fixed after the full sweeps finished. The published
rows under `results/` carry the verdicts each sweep PRINTED. This script
re-judges a raw checkpoint with `bench.harness.judge`, the function the
harness itself judges with, and writes the trials whose verdict moved:

    python3 scripts/rescore.py --from-checkpoint audit/full.openai.jsonl \\
        --model gpt-5.6-terra        writes results/rescored.gpt-5_6-terra.json

The checkpoints are not shipped, because they carry the models' prose. So
the file is what a reader gets, and `attempts(model, rescored=True)` is the
published rows with it applied: the rows that produce the RE-SCORED tables.

What a reader can check without the checkpoint: every moved trial whose
verdict does not read the answer text (a tool call, a GET or a memory write
under no output-level defense) is re-judged from the row alone by
tests/test_published_trials.py. The others are marked `needs_answer_text`,
because the published rows withhold the text they were judged on.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bench.attacks.corpus import CORPUS  # noqa: E402
from bench.defenses import OUTPUT_LEVEL  # noqa: E402
from bench.harness import judge  # noqa: E402
from bench.models import (  # noqa: E402
    REFUSED_WITHOUT_TEXT,
    Action,
    ActionKind,
    Attempt,
)

PAYLOADS = {p.payload_id: p for p in CORPUS}
SOURCES = {
    "claude-opus-5": "results/trials.claude-opus-5.jsonl",
    "gpt-5.6-terra": "results/trials.gpt-5_6-terra.jsonl",
}
VERDICT = ("complied", "contained", "contained_by")


def overlay_path(model: str) -> str:
    return os.path.join(ROOT, "results",
                        "rescored.%s.json" % model.replace(".", "_"))


def rows(model: str) -> list[dict]:
    """The published rows for `model`, header line dropped."""
    with open(os.path.join(ROOT, SOURCES[model]), encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    return [json.loads(ln) for ln in lines][1:]


def key(row: dict) -> tuple[str, tuple[str, ...], int]:
    return row["payload_id"], tuple(row["defenses"]), row["repeat"]


def needs_answer_text(payload_id: str, defenses) -> bool:
    """Whether judging this trial reads the answer text the rows withhold."""
    wants_answer = PAYLOADS[payload_id].wants is ActionKind.ANSWER
    return wants_answer or bool(frozenset(defenses) & OUTPUT_LEVEL)


def overlay(model: str) -> list[dict]:
    with open(overlay_path(model), encoding="utf-8") as fh:
        return json.load(fh)["moved"]


def _action(published: dict) -> Action:
    """An Action carrying what the row keeps: kind, target, and lengths.

    A textless refusal is the one case where the text matters to a count, and
    it is rebuilt exactly: an answer with no text and the provider's refusal
    marker as its `provoked_by`.
    """
    kind = ActionKind(published["kind"])
    refused = (kind is ActionKind.ANSWER and not published["answer_len"]
               and published["provoked_by_len"] == len(REFUSED_WITHOUT_TEXT))
    return Action(
        kind=kind,
        target=published["target"] or "",
        answer="x" * published["answer_len"],
        provoked_by=(REFUSED_WITHOUT_TEXT if refused
                     else "x" * published["provoked_by_len"]),
    )


def attempts(model: str, *, rescored: bool) -> list[Attempt]:
    """The published rows as Attempts, with the re-scored verdicts applied
    when `rescored` is true."""
    moved = ({(m["payload_id"], tuple(m["defenses"]), m["repeat"]):
              m["rescored"] for m in overlay(model)} if rescored else {})
    out = []
    for row in rows(model):
        verdict = moved.get(key(row), {f: row[f] for f in VERDICT})
        out.append(Attempt(
            payload_id=row["payload_id"],
            attack_class=row["attack_class"],
            channel=row["channel"],
            defenses=tuple(row["defenses"]),
            repeat=row["repeat"],
            complied=verdict["complied"],
            contained=verdict["contained"],
            contained_by=verdict["contained_by"],
            neutralized_by=row["neutralized_by"],
            action=_action(row["action"]),
        ))
    return out


def moved_in_checkpoint(path: str) -> list[dict]:
    """Every trial in a raw checkpoint whose verdict judge() changes."""
    with open(path, encoding="utf-8") as fh:
        records = [json.loads(ln) for ln in fh if ln.strip()][1:]
    moved = []
    for record in records:
        raw = record["action"]
        action = Action(kind=ActionKind(raw["kind"]), target=raw["target"],
                        provoked_by=raw["provoked_by"], answer=raw["answer"])
        got = judge(PAYLOADS[record["payload_id"]],
                    frozenset(record["defenses"]), action)
        was = tuple(record[f] for f in VERDICT)
        if got != was:
            moved.append({
                "payload_id": record["payload_id"],
                "defenses": list(record["defenses"]),
                "repeat": record["repeat"],
                "printed": dict(zip(VERDICT, was)),
                "rescored": dict(zip(VERDICT, got)),
                "needs_answer_text": needs_answer_text(record["payload_id"],
                                                       record["defenses"]),
            })
    return moved


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-checkpoint", required=True)
    ap.add_argument("--model", required=True, choices=sorted(SOURCES))
    args = ap.parse_args(argv)
    moved = moved_in_checkpoint(args.from_checkpoint)
    published = {key(r) for r in rows(args.model)}
    stray = [m for m in moved
             if (m["payload_id"], tuple(m["defenses"]), m["repeat"])
             not in published]
    if stray:
        print("REFUSED: %d moved trial(s) are not in %s, so this checkpoint "
              "is not the one that file was published from"
              % (len(stray), SOURCES[args.model]), file=sys.stderr)
        return 1
    with open(overlay_path(args.model), "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "moved": moved}, fh, indent=2,
                  sort_keys=True)
        fh.write("\n")
    print("%s: %d trial(s) moved; wrote %s"
          % (args.model, len(moved), os.path.relpath(overlay_path(args.model),
                                                      ROOT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
