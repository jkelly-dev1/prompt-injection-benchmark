#!/usr/bin/env python3
"""Write the publishable form of a real-model checkpoint.

    python scripts/publish_trials.py audit/full.jsonl results/trials.opus.jsonl

Why this exists. Every real-model figure in README.md is derived from a
checkpoint under audit/, and audit/ is not published. A reader therefore cannot
recompute those figures, which a benchmark must allow.

What is published is every field the scoring consumes: payload_id, defenses,
repeat, complied, contained, contained_by, neutralized_by, attack_class,
channel, and the action's kind and target. Running summarize() and
effect_over_baseline() over these rows reproduces what each sweep PRINTED,
because those functions read nothing else. The RE-SCORED tables in README.md
are these rows with results/rescored.<model>.json applied; see
scripts/rescore.py.

What is not published is the model's prose: the answer text and the stated
reason, each replaced by its LENGTH. summarize() and the effect table read
neither, and the repository does not redistribute provider output. Judging a
trial again does read the answer, so a re-scored verdict that depends on it
cannot be re-checked from these rows (see scripts/rescore.py). The lengths are
kept because they carry the one property the prose was evidence for: whether a
refusal came with words or without.

This is not a size measure. Stripping the prose removes about 41 percent of the
bytes, not an order of magnitude. The reason is provenance, not size.
"""
import json
import sys

SCORED = ("payload_id", "defenses", "repeat", "complied", "contained",
          "contained_by", "neutralized_by", "attack_class", "channel")


def publishable(trial: dict) -> dict:
    row = {k: trial[k] for k in SCORED if k in trial}
    action = trial.get("action") or {}
    row["action"] = {
        "kind": action.get("kind"),
        "target": action.get("target"),
        # The prose itself is withheld; its length is what the published
        # "refused with no answer text" figure is counted from.
        "answer_len": len(action.get("answer") or ""),
        "provoked_by_len": len(action.get("provoked_by") or ""),
    }
    return row


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.splitlines()[2].strip(), file=sys.stderr)
        return 2
    src, dst = argv[1], argv[2]
    rows = [json.loads(line) for line in open(src, encoding="utf-8") if line.strip()]
    header, trials = rows[0], rows[1:]
    with open(dst, "w", encoding="utf-8") as out:
        out.write(json.dumps({"header": header.get("header", header)},
                             separators=(",", ":")) + "\n")
        for trial in trials:
            out.write(json.dumps(publishable(trial), separators=(",", ":")) + "\n")
    print(f"{dst}: {len(trials):,} trials, header preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
