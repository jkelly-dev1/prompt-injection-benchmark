"""Re-derive the published real-model figures from results/*.jsonl and diff them.

A README is prose and drifts; the per-trial rows under `results/` are evidence
and do not. This script rebuilds each figure from those rows and asserts the
exact string appears in README.md, so a re-run that shifts a figure fails
loudly instead of leaving the document quietly wrong.

    python3 scripts/check_readme_numbers.py            check
    python3 scripts/check_readme_numbers.py --emit     print what it derives

IT READS ONLY WHAT A READER GETS. The raw checkpoints under `audit/` are not
shipped, so deriving from them would prove nothing about the published tree.
Every figure below comes from `results/`, which is tracked.

The count is printed whether or not anything is missing, so a version of this
script that quietly stopped deriving half of them is visible in its own output
rather than reported as clean.
"""

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCES = {
    "claude-opus-5": "results/trials.claude-opus-5.jsonl",
    "gpt-5.6-terra": "results/trials.gpt-5_6-terra.jsonl",
}


def rows(rel):
    path = os.path.join(ROOT, rel)
    with open(path, encoding="utf-8") as fh:
        lines = [ln for ln in fh if ln.strip()]
    return [json.loads(ln) for ln in lines][1:]


def normalize(text):
    """Collapse whitespace and emphasis so a reflow is not a false alarm."""
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip()


def derive():
    """(tag, exact string that must appear in README.md)."""
    out = []
    for model, rel in sorted(SOURCES.items()):
        trials = rows(rel)
        out.append((f"{model}:trials", f"{len(trials):,} trials"))
        textless = sum(1 for t in trials
                       if not t["action"]["answer_len"]
                       and t["action"]["kind"] == "answer")
        if model == "claude-opus-5":
            # The refused-without-text column, and the share the README quotes.
            out.append((f"{model}:textless", f"{textless:,}"))
            out.append((f"{model}:share", f"{textless / len(trials):.3f}"))
            carried = sum(1 for t in trials if t["action"]["answer_len"])
            out.append((f"{model}:with-text", str(carried)))
        complied = sum(1 for t in trials if t["complied"])
        out.append((f"{model}:complied", str(complied)))
    return out


def main():
    emit = "--emit" in sys.argv
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = normalize(fh.read())

    derived = derive()
    missing = [(tag, s) for tag, s in derived if normalize(s) not in readme]
    if emit:
        for tag, s in derived:
            print("%-28s %s" % (tag, s))
    for tag, s in missing:
        print("MISSING [%s]\n  %s" % (tag, s))
    print("\n%d of %d derived figures found verbatim in README.md"
          % (len(derived) - len(missing), len(derived)))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
