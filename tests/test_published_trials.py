"""The published per-trial rows must reproduce the published real-model figures.

A README that says its figures are reproducible, in a repository whose subject
is measurement, has to be able to fail. This is that test: it reads only the
files under `results/` -- the ones a reader actually gets -- and asserts they
carry every field the scoring consumes, and that scoring them gives the same
answer as scoring the unshipped checkpoint.

It is skipped when the checkpoint is absent, because the comparison needs both.
The first assertion does not need the checkpoint and never skips.
"""

from __future__ import annotations

import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
PUBLISHED = {
    ROOT / "results/trials.claude-opus-5.jsonl": ROOT / "audit/full.jsonl",
    ROOT / "results/trials.gpt-5_6-terra.jsonl": ROOT / "audit/full.openai.jsonl",
}
SCORED = ("payload_id", "defenses", "repeat", "complied", "contained",
          "contained_by", "neutralized_by", "attack_class", "channel")


def _rows(path: pathlib.Path) -> list[dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines][1:]


@pytest.mark.parametrize("published", sorted(PUBLISHED), ids=lambda p: p.name)
def test_every_scored_field_is_present_and_no_prose_is(published: pathlib.Path) -> None:
    rows = _rows(published)
    assert rows, f"{published.name} carries no trials"
    for row in rows:
        for field in SCORED:
            assert field in row, f"{published.name} drops {field}, which the scoring reads"
        action = row["action"]
        assert set(action) == {"kind", "target", "answer_len", "provoked_by_len"}, (
            f"{published.name} action carries {sorted(action)}; publishing the "
            "model's prose is the thing this file exists to avoid"
        )
        for length in ("answer_len", "provoked_by_len"):
            assert isinstance(action[length], int)


@pytest.mark.parametrize("published,checkpoint", sorted(PUBLISHED.items()),
                         ids=lambda p: getattr(p, "name", ""))
def test_the_published_rows_score_identically_to_the_checkpoint(
    published: pathlib.Path, checkpoint: pathlib.Path
) -> None:
    if not checkpoint.exists():
        pytest.skip(f"{checkpoint.name} is not present; nothing to compare against")
    theirs, ours = _rows(checkpoint), _rows(published)
    assert len(theirs) == len(ours)
    for a, b in zip(theirs, ours):
        for field in SCORED:
            assert a.get(field) == b.get(field), (
                f"{published.name} disagrees with the checkpoint on {field}: "
                f"{a.get(field)!r} against {b.get(field)!r}"
            )
