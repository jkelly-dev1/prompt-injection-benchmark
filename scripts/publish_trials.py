#!/usr/bin/env python3
"""Write the publishable form of a real-model checkpoint.

    python scripts/publish_trials.py audit/full.jsonl results/trials.opus.jsonl

WHY THIS EXISTS. Every real-model figure in README.md is derived from a
checkpoint under audit/, and audit/ is not published. A reader therefore cannot
recompute those figures, which is the one thing a benchmark must allow.

WHAT IS PUBLISHED IS EVERY FIELD THE SCORING CONSUMES: payload_id, defenses,
repeat, complied, contained, contained_by, neutralized_by, attack_class,
channel, and the action's kind and target. Running summarize() and
effect_over_baseline() over these rows reproduces the published tables exactly,
because those functions read nothing else.

WHAT IS NOT PUBLISHED IS THE MODEL'S PROSE -- the answer text and the stated
reason -- which is replaced by its LENGTH. The scoring never reads either, so
nothing measurable is lost, and the repository does not redistribute provider
output. The lengths are kept because they carry the one property the prose was
evidence for: whether a refusal came with words or without.

THIS IS NOT A SIZE MEASURE. Stripping the prose removes about 41 percent of the
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
